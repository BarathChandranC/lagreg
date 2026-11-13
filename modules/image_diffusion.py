"""
Continuous Gaussian (DDPM) forward process for images, x0-prediction.

x0-prediction, not eps-prediction, deliberately: the band decomposition acts
on the residual r = x0 - x0_hat, and the floor-cancellation identity is
stated for that residual. With eps-prediction you would have to convert back
to an x0 estimate before decomposing, introducing a t-dependent rescaling
(1/sqrt(alpha_bar)) that blows up at high noise and would silently reweight
the bands. Keep the prediction target and the decomposition target the same
object.
"""
from typing import Tuple

import torch


def alpha_bars_schedule(t_final: int, schedule: str = "cosine", s: float = 0.008,
                        device=None) -> torch.Tensor:
    """alpha_bar_t for t = 1..t_final, strictly decreasing in t.

    Args:
        t_final: number of diffusion steps T.
        schedule: 'cosine' (Nichol & Dhariwal) or 'linear' (DDPM original).
        s: cosine schedule offset.
        device: torch device.

    Returns:
        (t_final,) tensor; index t-1 holds alpha_bar_t.
    """
    if schedule == "cosine":
        steps = torch.arange(t_final + 1, dtype=torch.float64, device=device) / t_final
        f = torch.cos((steps + s) / (1 + s) * torch.pi * 0.5) ** 2
        ab = (f / f[0])[1:]
    elif schedule == "linear":
        betas = torch.linspace(1e-4, 0.02, t_final, dtype=torch.float64, device=device)
        ab = torch.cumprod(1.0 - betas, dim=0)
    else:
        raise ValueError(f"unknown schedule {schedule!r}")
    return ab.clamp(1e-8, 1 - 1e-8).float()


def q_sample(x0: torch.Tensor, t: torch.Tensor, alpha_bars: torch.Tensor,
             noise: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward-noise x0 at per-sample timesteps t (1-indexed).

    Args:
        x0: (B, C, H, W).
        t: (B,) integer timesteps in [1, t_final].
        alpha_bars: (t_final,) schedule.
        noise: optional (B, C, H, W) pre-drawn noise -- pass a FIXED noise
            tensor when comparing checkpoints, otherwise checkpoint-to-
            checkpoint differences are contaminated by fresh sampling noise.

    Returns:
        (xt, noise).
    """
    if noise is None:
        noise = torch.randn_like(x0)
    ab = alpha_bars[t - 1].view(-1, 1, 1, 1)
    xt = ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise
    return xt, noise


def sample_timesteps(batch: int, t_final: int, device, generator=None) -> torch.Tensor:
    """Uniform t in [1, t_final]."""
    return torch.randint(1, t_final + 1, (batch,), device=device, generator=generator)
