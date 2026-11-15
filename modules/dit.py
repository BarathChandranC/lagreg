"""
DiT (Peebles & Xie, 2023) for 2x2-tile composites, as a drop-in replacement for
modules/unet.py: same call signature model(x, t) -> eps, x (B,1,H,W), t (B,)
integer timesteps.

Standard DiT components:
  * patchify: non-overlapping p x p patches -> linear embedding (a stride-p conv)
  * fixed 2-D sin-cos positional embedding
  * timestep embedder: 256-dim sinusoidal -> MLP -> hidden
  * N transformer blocks with adaLN-Zero conditioning (shift/scale/gate for the
    attention and MLP branches, modulation layer zero-initialised so every block
    starts as the identity)
  * final layer: adaLN (shift/scale) + linear to p*p*C, zero-initialised, then
    unpatchify. Predicts eps only (no learned variance), 1 channel, pixel space.
No class conditioning (the model is unconditional, like the U-Net).

Sizes (Peebles & Xie, Table 1):   depth  hidden  heads
    S                               12     384     6      ~33M params
    B                               12     768    12      ~130M
Patch size should divide the TILE size (14 for 28x28 composites: p in {2, 7, 14})
so that no patch straddles two tiles; the constructor refuses one that doesn't
divide the image and warns if it straddles tiles.
"""
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DIT_SIZES = {"S": dict(depth=12, hidden=384, heads=6),
             "B": dict(depth=12, hidden=768, heads=12)}


def sincos_2d(dim, grid):
    """(grid*grid, dim) fixed 2-D sin-cos positional embedding (MAE / DiT style)."""
    assert dim % 4 == 0
    gh, gw = np.meshgrid(np.arange(grid, dtype=np.float64), np.arange(grid, dtype=np.float64),
                         indexing="ij")
    omega = 1.0 / 10000 ** (np.arange(dim // 4, dtype=np.float64) / (dim / 4.0))

    def emb1d(pos):
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    return np.concatenate([emb1d(gw), emb1d(gh)], axis=1).astype(np.float32)


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, t):
        return self.mlp(timestep_embedding(t, self.freq_dim))


class Attention(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        h = F.scaled_dot_product_attention(q, k, v)                    # (B, heads, L, d)
        return self.proj(h.transpose(1, 2).reshape(B, L, D))


class DiTBlock(nn.Module):
    def __init__(self, hidden, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden, heads)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        mh = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(hidden, mh), nn.GELU(approximate="tanh"), nn.Linear(mh, hidden))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))

    def forward(self, x, c):
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).chunk(6, dim=1)
        x = x + g1.unsqueeze(1) * self.attn(modulate(self.norm1(x), sh1, sc1))
        x = x + g2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), sh2, sc2))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden, patch, out_ch):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden, patch * patch * out_ch)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))

    def forward(self, x, c):
        sh, sc = self.ada(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), sh, sc))


class DiT(nn.Module):
    def __init__(self, img_size=28, patch=2, in_ch=1, size="S", depth=None, hidden=None,
                 heads=None, mlp_ratio=4.0, tile_size=None):
        super().__init__()
        cfg = dict(DIT_SIZES[size])
        cfg.update({k: v for k, v in dict(depth=depth, hidden=hidden, heads=heads).items() if v is not None})
        if img_size % patch:
            raise ValueError(f"patch {patch} does not divide image size {img_size}")
        tile = tile_size or img_size // 2
        if tile % patch:
            warnings.warn(f"patch {patch} does not divide the tile size {tile}: some patches "
                          f"straddle two tiles (tile-aligned choices: "
                          f"{[p for p in range(1, tile + 1) if tile % p == 0]})")
        self.img_size, self.patch, self.in_ch = img_size, patch, in_ch
        self.grid = img_size // patch
        H = cfg["hidden"]

        self.x_emb = nn.Conv2d(in_ch, H, kernel_size=patch, stride=patch)
        self.register_buffer("pos", torch.from_numpy(sincos_2d(H, self.grid)).unsqueeze(0),
                             persistent=False)
        self.t_emb = TimestepEmbedder(H)
        self.blocks = nn.ModuleList([DiTBlock(H, cfg["heads"], mlp_ratio) for _ in range(cfg["depth"])])
        self.final = FinalLayer(H, patch, in_ch)
        self._init()

    def _init(self):
        def basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(basic)
        w = self.x_emb.weight.data                                     # patch embed like a Linear
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.x_emb.bias)
        nn.init.normal_(self.t_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_emb.mlp[2].weight, std=0.02)
        for b in self.blocks:                                          # adaLN-Zero
            nn.init.zeros_(b.ada[-1].weight)
            nn.init.zeros_(b.ada[-1].bias)
        nn.init.zeros_(self.final.ada[-1].weight)
        nn.init.zeros_(self.final.ada[-1].bias)
        nn.init.zeros_(self.final.linear.weight)                       # output starts at 0
        nn.init.zeros_(self.final.linear.bias)

    def unpatchify(self, x):
        B, p, g, C = x.shape[0], self.patch, self.grid, self.in_ch
        x = x.view(B, g, g, p, p, C).permute(0, 5, 1, 3, 2, 4)        # (B,C,g,p,g,p)
        return x.reshape(B, C, g * p, g * p)

    def forward(self, x, t):
        h = self.x_emb(x).flatten(2).transpose(1, 2) + self.pos       # (B, L, H)
        c = self.t_emb(t)
        for blk in self.blocks:
            h = blk(h, c)
        return self.unpatchify(self.final(h, c))
