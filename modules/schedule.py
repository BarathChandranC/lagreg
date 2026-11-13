"""
Discrete-time VP diffusion, epsilon prediction.

    Z_t = a_t X + s_t eps,   a_t = sqrt(abar_t),  s_t = sqrt(1 - abar_t),  t in {1..T}

with the cosine abar schedule (Nichol & Dhariwal 2021). The Bayes-optimal
epsilon predictor is m*(z,t) = E[eps | Z_t = z] = (z - a_t E[X | Z_t = z]) / s_t.
"""
import math

import torch


class CosineSchedule:
    def __init__(self, T=1000, s=0.008, max_beta=0.999, device="cpu"):
        self.T = int(T)
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
        abar = f / f[0]
        betas = (1 - abar[1:] / abar[:-1]).clamp(max=max_beta)
        abar = torch.cumprod(1 - betas, 0)                     # index t-1 <-> time t
        self.abar = abar.to(device)
        self.betas = betas.to(device)
        self.device = torch.device(device)

    def a_s(self, t):
        """t: long tensor in {1..T} (any shape) -> (a_t, s_t) float64 of the same shape."""
        ab = self.abar[t - 1]
        return ab.sqrt(), (1 - ab).sqrt()

    def q_sample(self, x0, t, eps):
        a, s = self.a_s(t)
        shape = (-1,) + (1,) * (x0.dim() - 1)
        return a.float().view(shape) * x0 + s.float().view(shape) * eps

    def t_from_frac(self, frac):
        return max(1, min(self.T, int(round(frac * self.T))))
