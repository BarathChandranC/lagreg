"""
Orthonormal Haar 2D DWT and band bookkeeping -- the W_j operators.

The whole band-resolved risk vector rests on ONE property: W is orthonormal,
so by Parseval

    ||r||^2 = ||W r||^2 = sum_j ||W_j r||^2                (exact, no residual)

for any residual r whatsoever. This gives DISJOINT, EXACTLY-SUMMING band
terms as a fact of linear algebra -- it does not depend on the data, the
model, or any hierarchy assumption.

WHAT IT DOES NOT GIVE (read this before interpreting any output):
  - Band j is NOT a clean measurement of "hierarchy level j". A wavelet band
    is a spatial-SCALE projection; on real data, scale is correlated with,
    but does not isolate, whatever compositional level structure exists.
  - Bands have WILDLY unequal coefficient counts. For 32x32 with J=4:
    detail_1 has 768 coefficients, approx_4 has 4 -- a 192x spread. Raw
    ||W_j r||^2 is therefore NOT comparable across bands: a coarse band's
    smaller total can simply mean "fewer terms in the sum". Always use the
    per-coefficient (mean) form, or an explicit normalizer, when comparing
    bands to each other. band_sizes() is provided for exactly this.

Convention: J decomposition levels give J+1 bands, ordered FINE -> COARSE:
    detail_1 (finest, highest frequency, largest support in coeff count)
    detail_2
    ...
    detail_J
    approx_J (coarsest, the running 2^-J-scale average)
"""
from typing import Dict, List, Tuple

import torch


def _haar_step_1d(x: torch.Tensor, dim: int) -> torch.Tensor:
    """One orthonormal Haar lifting step along `dim`; length must be even.

    a = (x[0::2] + x[1::2]) / sqrt(2)      (approximation / low-pass)
    d = (x[0::2] - x[1::2]) / sqrt(2)      (detail / high-pass)

    The 1/sqrt(2) is what makes the 2x2 butterfly an orthonormal rotation,
    hence the whole cascade orthonormal, hence Parseval exact. Do not
    "simplify" it to 1/2 -- that breaks the energy identity everything here
    depends on.
    """
    n = x.shape[dim]
    if n % 2 != 0:
        raise ValueError(f"Haar step needs an even length along dim {dim}, got {n}")
    x = x.movedim(dim, -1)
    a = (x[..., 0::2] + x[..., 1::2]) / (2.0 ** 0.5)
    d = (x[..., 0::2] - x[..., 1::2]) / (2.0 ** 0.5)
    return torch.cat([a, d], dim=-1).movedim(-1, dim)


def dwt2(x: torch.Tensor, levels: int) -> Dict[str, torch.Tensor]:
    """Multi-level orthonormal Haar 2D DWT.

    Args:
        x: (B, C, H, W). H and W must each be divisible by 2**levels.
        levels: number of decomposition levels J.

    Returns:
        dict mapping band name -> coefficient tensor:
            'detail_1'..'detail_J' : (B, C, 3, H/2^j, W/2^j)   [LH, HL, HH]
            'approx_J'             : (B, C, H/2^J, W/2^J)
    """
    if x.dim() != 4:
        raise ValueError(f"dwt2 expects (B, C, H, W), got shape {tuple(x.shape)}")
    H, W = x.shape[-2:]
    if H % (2 ** levels) or W % (2 ** levels):
        raise ValueError(f"H={H}, W={W} must be divisible by 2**levels={2**levels}")

    out: Dict[str, torch.Tensor] = {}
    cur = x
    for j in range(1, levels + 1):
        cur = _haar_step_1d(cur, -2)
        cur = _haar_step_1d(cur, -1)
        h, w = cur.shape[-2] // 2, cur.shape[-1] // 2
        LL = cur[..., :h, :w]
        LH = cur[..., :h, w:]
        HL = cur[..., h:, :w]
        HH = cur[..., h:, w:]
        out[f"detail_{j}"] = torch.stack([LH, HL, HH], dim=2)
        cur = LL
    out[f"approx_{levels}"] = cur
    return out


def band_names(levels: int) -> List[str]:
    """Band names ordered FINE -> COARSE."""
    return [f"detail_{j}" for j in range(1, levels + 1)] + [f"approx_{levels}"]


def band_energies(r: torch.Tensor, levels: int, per_coefficient: bool = True) -> Dict[str, torch.Tensor]:
    """Per-band energy of a residual tensor.

    Args:
        r: (B, C, H, W) residual, e.g. x0 - x0_hat.
        levels: J.
        per_coefficient: if True (default) return MEAN squared coefficient per
            band; if False return the SUM. Only the sum satisfies
            sum_j energy_j == ||r||^2 exactly -- the mean form divides each
            band by its own coefficient count and is what you want for
            comparing bands to each other (see module docstring on the 192x
            count spread). Pick deliberately; they answer different questions.

    Returns:
        dict band name -> scalar tensor (averaged over the batch).
    """
    coeffs = dwt2(r, levels)
    out = {}
    for name, c in coeffs.items():
        sq = c.pow(2).flatten(1).sum(dim=1)                 # (B,) per-sample sum
        if per_coefficient:
            n = c[0].numel()
            out[name] = (sq / n).mean()
        else:
            out[name] = sq.mean()
    return out


def band_sizes(levels: int, shape: Tuple[int, int, int]) -> Dict[str, int]:
    """Coefficient count per band, for normalization / sanity checks.

    Args:
        levels: J.
        shape: (C, H, W) of a single image.

    Returns:
        dict band name -> int coefficient count. Sums to C*H*W.
    """
    C, H, W = shape
    sizes = {}
    for j in range(1, levels + 1):
        h, w = H // (2 ** j), W // (2 ** j)
        sizes[f"detail_{j}"] = C * 3 * h * w
    h, w = H // (2 ** levels), W // (2 ** levels)
    sizes[f"approx_{levels}"] = C * h * w
    return sizes


def check_parseval(x: torch.Tensor, levels: int, tol: float = 1e-4) -> float:
    """Assert sum_j ||W_j x||^2 == ||x||^2; returns the max relative error.

    Cheap and worth calling once at startup: if this fails, every band number
    downstream is meaningless, and the cause is almost always an odd
    dimension or a dropped 1/sqrt(2).
    """
    total = x.pow(2).flatten(1).sum(dim=1)
    coeffs = dwt2(x, levels)
    banded = sum(c.pow(2).flatten(1).sum(dim=1) for c in coeffs.values())
    rel = ((banded - total).abs() / total.clamp_min(1e-12)).max().item()
    if rel > tol:
        raise AssertionError(f"Parseval violated: max relative error {rel:.3e} > {tol:g}")
    return rel
