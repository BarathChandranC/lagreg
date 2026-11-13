"""
Band-resolved risk vector for image diffusion -- the analog of the RHM
ladder, built on wavelet bands instead of BP rungs.

    L_j       = E||W_j (x0 - x0_hat)||^2 / n_j        (model, per coefficient)
    L_j^base  = E||W_j (x0 - D*_sigma )||^2 / n_j     (closed form, no model)
    Delta_j   = L_j - L_j^base                        (floor cancels exactly)

STRUCTURAL DIFFERENCES FROM THE RHM LADDER -- these are not fixable here and
should be carried into any writeup:

  1. NO PERFECT-MODEL CHECK. On RHM you can substitute the exact oracle and
     confirm the corrected vector goes to flat zero. There is no oracle here,
     so that validation is simply unavailable. `Delta_j -> -||W_j h||^2`, not
     0, for a perfect model -- and ||W_j h||^2 is unknown.
  2. NOT LEVEL-ISOLATING. Bands are spatial-SCALE projections. Scale
     correlates with hierarchy level but does not isolate it.
  3. NO SIGN GUARANTEE, and for a different reason than RHM. There is no
     DPI-style argument for an arbitrary baseline. Delta_j < 0 simply means
     the model beat D*_sigma in that band -- expected and desirable, not an
     error. Plot on symlog.
  4. THE SUM IS SHORT BY THE BASELINE'S OWN GAP. sum_j Delta_j = (total model
     excess) - (total baseline excess), NOT the training loss. The raw
     per-band L_j DO sum to the training loss exactly (Parseval); the
     baseline-corrected Delta_j do not. Both are logged, deliberately.

WHAT IS TRUSTWORTHY: emergence ORDERING across checkpoints, and trajectory
SHAPE (monotone decay vs. flat). A flat Delta_j is ambiguous between
"resolved" and "never engaged" -- only the shape distinguishes them.
"""
from typing import Dict, List, Optional

import numpy as np
import torch

from .gaussian_baseline import GaussianBandBaseline
from .image_diffusion import q_sample
from .wavelet import band_energies, band_names, band_sizes, check_parseval


class BandEvaluator:
    """Evaluates the band-resolved risk vector at a fixed, frozen eval set.

    The (x_t, t, noise) triples are drawn ONCE at construction and reused at
    every checkpoint. This is the same reasoning as the RHM ladder: if you
    redraw noise each time, checkpoint-to-checkpoint differences mix genuine
    model change with fresh sampling noise, and the emergence times -- the
    one thing this vector is actually good for -- become unreadable.
    """

    def __init__(self, x0_eval: torch.Tensor, alpha_bars: torch.Tensor, levels: int,
                 baseline: GaussianBandBaseline, device, batch_size: int = 256,
                 n_t_samples: int = 4, seed: int = 0):
        """
        Args:
            x0_eval: (N, C, H, W) evaluation images. Pass the FULL split --
                subsampling to a fixed size independent of N reintroduces an
                N-independent Monte Carlo floor across an N-sweep.
            alpha_bars: (T,) schedule, the same one training uses.
            levels: J wavelet levels.
            baseline: a FITTED GaussianBandBaseline.
            n_t_samples: independent t draws per eval image. E_t must be taken
                over the same t-distribution training uses, or the per-band
                numbers describe a different objective than the one optimised.
        """
        self.levels = levels
        self.baseline = baseline
        self.device = device
        self.batch_size = int(batch_size)
        self.alpha_bars = alpha_bars.to(device)
        self.names = band_names(levels)
        self.sizes = band_sizes(levels, tuple(x0_eval.shape[1:]))

        check_parseval(x0_eval[: min(8, len(x0_eval))].to(device), levels)

        g = torch.Generator(device="cpu").manual_seed(int(seed))
        T = alpha_bars.shape[0]
        self.x0 = x0_eval.to(device)
        n = self.x0.shape[0]
        self.t = torch.randint(1, T + 1, (n_t_samples, n), generator=g).to(device)
        self.noise = torch.randn(n_t_samples, *self.x0.shape, generator=g).to(device)
        self.n_t_samples = int(n_t_samples)

        # baseline is model-free and t-only, so precompute the whole schedule once
        self._base_sched = baseline.band_losses_over_schedule(alpha_bars.cpu())

    @torch.no_grad()
    def _baseline_for(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Per-band baseline loss averaged over the timesteps actually drawn."""
        return {n: self._base_sched[n].to(t.device)[t - 1] for n in self.names}

    @torch.no_grad()
    def evaluate(self, model) -> Dict[str, float]:
        """Compute the band vector for one checkpoint.

        Returns dict with, per band:
            L_{band}        raw model per-coefficient band loss
            Lbase_{band}    closed-form baseline per-coefficient band loss
            Delta_{band}    L - Lbase (floor-cancelled)
            n_{band}        Delta / Lbase (baseline-normalised; see caveat)
        plus 'loss_total' (the true training loss) and
        'parseval_residual' (should be ~0; if not, bands are meaningless).
        """
        was_training = model.training
        model.eval()
        try:
            acc = {n: 0.0 for n in self.names}
            acc_base = {n: 0.0 for n in self.names}
            tot, seen = 0.0, 0
            for rep in range(self.n_t_samples):
                for i in range(0, self.x0.shape[0], self.batch_size):
                    sl = slice(i, min(i + self.batch_size, self.x0.shape[0]))
                    x0 = self.x0[sl]
                    t = self.t[rep, sl]
                    xt, _ = q_sample(x0, t, self.alpha_bars, noise=self.noise[rep, sl])
                    x0_hat = model(xt, t)
                    r = x0 - x0_hat
                    be = band_energies(r, self.levels, per_coefficient=True)
                    bb = self._baseline_for(t)
                    b = x0.shape[0]
                    for n in self.names:
                        acc[n] += float(be[n].item()) * b
                        acc_base[n] += float(bb[n].mean().item()) * b
                    tot += float(r.pow(2).mean().item()) * b
                    seen += b
        finally:
            if was_training:
                model.train()

        out: Dict[str, float] = {"loss_total": tot / seen}
        # raw band losses must reconstruct the total exactly (Parseval)
        recon = sum((acc[n] / seen) * self.sizes[n] for n in self.names) / sum(self.sizes.values())
        out["parseval_residual"] = abs(recon - out["loss_total"])
        for n in self.names:
            L = acc[n] / seen
            Lb = acc_base[n] / seen
            out[f"L_{n}"] = L
            out[f"Lbase_{n}"] = Lb
            out[f"Delta_{n}"] = L - Lb
            out[f"n_{n}"] = (L - Lb) / Lb if abs(Lb) > 1e-12 else float("nan")
        return out


def grouped_bands(row: Dict[str, float], levels: int, group_at: int = 2) -> Dict[str, float]:
    """Collapse J+1 bands into fine / coarse groups.

    Bands are ordered fine -> coarse, so detail_1..detail_{group_at} are the
    FINE (local, high-frequency) group and the rest are COARSE (global).

    Coefficient-count weighted, because bands differ in size by orders of
    magnitude (192x for 32x32 with J=4) -- a plain average would be dominated
    by whichever band happens to have the fewest coefficients.
    """
    names = band_names(levels)
    fine = names[:group_at]
    coarse = names[group_at:]
    sizes = {n: 1.0 for n in names}   # counts cancel in a per-coefficient mean
    def _w(keys, pre):
        num = sum(row[f"{pre}_{n}"] * sizes[n] for n in keys)
        return num / sum(sizes[n] for n in keys)
    return {
        "D_local": _w(fine, "Delta"), "D_global": _w(coarse, "Delta"),
        "L_local": _w(fine, "L"), "L_global": _w(coarse, "L"),
        "Lbase_local": _w(fine, "Lbase"), "Lbase_global": _w(coarse, "Lbase"),
    }


def emergence_steps(rows: List[Dict[str, float]], key: str, frac: float = 0.5) -> Optional[int]:
    """First step where `key` falls to `frac` of the way from its start to its min.

    Order-preserving in the series, so ANY fixed positive rescaling of the
    series gives the identical answer -- the emergence ORDER across bands is
    therefore independent of the normalizer choice, unlike cross-band
    magnitude comparisons at a fixed checkpoint.
    """
    y = np.array([r[key] for r in rows], dtype=float)
    s = np.array([r["step"] for r in rows])
    if len(y) == 0 or not np.isfinite(y).all():
        return None
    thr = y.min() + frac * (y[0] - y.min())
    hit = np.where(y <= thr)[0]
    return int(s[hit[0]]) if len(hit) else None
