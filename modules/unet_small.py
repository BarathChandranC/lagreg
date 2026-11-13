"""
Small U-Net denoiser (x0-prediction) for 32x32-ish single-channel images.

Deliberately small and plain: no attention, no fancy blocks. Two reasons.
(1) The object of study is the loss decomposition, not the architecture --
extra machinery adds confounds without adding anything to measure.
(2) Channel width is a KNOWN modulator of spectral bias in conv nets (wide
nets wash the spectral ordering out, narrow nets retain it), so `base_ch` is
exposed as a first-class knob: sweeping it is the cheapest available control
for separating "the model has a hierarchy-driven learning order" from "the
model has ordinary width-dependent spectral bias".
"""
import math

import torch
import torch.nn as nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding; t is (B,) 1-indexed."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class Block(nn.Module):
    """Conv-GN-SiLU x2 with additive timestep conditioning."""

    def __init__(self, cin: int, cout: int, tdim: int, groups: int = 8):
        super().__init__()
        g1 = math.gcd(groups, cin) or 1
        g2 = math.gcd(groups, cout) or 1
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(g1, cin)
        self.n2 = nn.GroupNorm(g2, cout)
        self.emb = nn.Linear(tdim, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, temb):
        h = self.c1(self.act(self.n1(x)))
        h = h + self.emb(self.act(temb))[:, :, None, None]
        h = self.c2(self.act(self.n2(h)))
        return h + self.skip(x)


class UNetSmall(nn.Module):
    """U-Net predicting x0 directly from (x_t, t).

    Args:
        in_ch: image channels.
        base_ch: width of the finest level. Sweep this to probe spectral bias.
        ch_mults: channel multiplier per resolution level.
        tdim: timestep-embedding width.
    """

    def __init__(self, in_ch: int = 1, base_ch: int = 32, ch_mults=(1, 2, 2), tdim: int = 128):
        super().__init__()
        self.tdim = tdim
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        chans = [base_ch * m for m in ch_mults]

        self.stem = nn.Conv2d(in_ch, chans[0], 3, padding=1)
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(len(chans) - 1):
            self.downs.append(Block(chans[i], chans[i], tdim))
            self.pools.append(nn.Conv2d(chans[i], chans[i + 1], 3, stride=2, padding=1))
        self.mid = Block(chans[-1], chans[-1], tdim)
        self.ups = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in reversed(range(len(chans) - 1)):
            self.upsamples.append(nn.ConvTranspose2d(chans[i + 1], chans[i], 4, stride=2, padding=1))
            self.ups.append(Block(chans[i] * 2, chans[i], tdim))
        self.out_norm = nn.GroupNorm(math.gcd(8, chans[0]) or 1, chans[0])
        self.out = nn.Conv2d(chans[0], in_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W), (B,) -> predicted x0 (B,C,H,W)."""
        temb = self.tmlp(timestep_embedding(t, self.tdim))
        h = self.stem(xt)
        skips = []
        for blk, pool in zip(self.downs, self.pools):
            h = blk(h, temb)
            skips.append(h)
            h = pool(h)
        h = self.mid(h, temb)
        for up, blk in zip(self.upsamples, self.ups):
            h = up(h)
            s = skips.pop()
            if h.shape[-2:] != s.shape[-2:]:      # guard odd sizes
                h = torch.nn.functional.interpolate(h, size=s.shape[-2:], mode="nearest")
            h = blk(torch.cat([h, s], dim=1), temb)
        return self.out(torch.nn.functional.silu(self.out_norm(h)))

    def get_params_dict(self):
        return {"class": "UNetSmall"}
