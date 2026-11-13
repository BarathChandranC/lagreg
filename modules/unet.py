"""
Small eps-predicting U-Net for 2x2-tile composites (28x28 or 56x56, 1 channel).

Self-attention at the two lowest resolutions gives every output pixel a
receptive field covering all four tiles -- without it the 4-way constraint is
architecturally unlearnable, and any 'lag' would be an artefact.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def gn(c):
    g = next(g for g in range(min(32, c), 0, -1) if c % g == 0)
    return nn.GroupNorm(g, c)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim, dropout=0.0):
        super().__init__()
        self.n1, self.c1 = gn(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2, self.c2 = gn(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(F.silu(temb))[:, :, None, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.n = gn(c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.out = nn.Conv2d(c, c, 1)
        self.heads = heads

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, self.heads, C // self.heads, H * W).unbind(1)
        h = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2),
                                           v.transpose(-1, -2))          # (B,h,HW,c)
        h = h.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.out(h)


class UNet(nn.Module):
    def __init__(self, img_size=28, base_ch=64, ch_mult=(1, 2, 2), n_res=2,
                 attn_res=(7, 14), dropout=0.0):
        super().__init__()
        self.img_size = img_size
        tdim = base_ch * 4
        self.temb = nn.Sequential(nn.Linear(base_ch, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.base_ch = base_ch
        self.inp = nn.Conv2d(1, base_ch, 3, padding=1)

        self.down = nn.ModuleList()
        chans = [base_ch]
        ch, res = base_ch, img_size
        for li, m in enumerate(ch_mult):
            for _ in range(n_res):
                blk = nn.ModuleList([ResBlock(ch, base_ch * m, tdim, dropout)])
                ch = base_ch * m
                if res in attn_res:
                    blk.append(Attn(ch))
                self.down.append(blk)
                chans.append(ch)
            if li != len(ch_mult) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                chans.append(ch)
                res //= 2

        self.mid1 = ResBlock(ch, ch, tdim, dropout)
        self.mid_attn = Attn(ch)
        self.mid2 = ResBlock(ch, ch, tdim, dropout)

        self.up = nn.ModuleList()
        for li, m in reversed(list(enumerate(ch_mult))):
            for _ in range(n_res + 1):
                blk = nn.ModuleList([ResBlock(ch + chans.pop(), base_ch * m, tdim, dropout)])
                ch = base_ch * m
                if res in attn_res:
                    blk.append(Attn(ch))
                self.up.append(blk)
            if li != 0:
                self.up.append(nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest"),
                                              nn.Conv2d(ch, ch, 3, padding=1)]))
                res *= 2
        self.out = nn.Sequential(gn(ch), nn.SiLU(), nn.Conv2d(ch, 1, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t):
        temb = self.temb(timestep_embedding(t, self.base_ch))
        h = self.inp(x)
        hs = [h]
        for blk in self.down:
            if isinstance(blk[0], ResBlock):
                h = blk[0](h, temb)
                for extra in blk[1:]:
                    h = extra(h)
            else:
                h = blk[0](h)
            hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)
        for blk in self.up:
            if isinstance(blk[0], ResBlock):
                h = blk[0](torch.cat([h, hs.pop()], 1), temb)
                for extra in blk[1:]:
                    h = extra(h)
            else:
                h = blk[1](blk[0](h))
        return self.out(h)
