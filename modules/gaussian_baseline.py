"""
Closed-form Gaussian-optimal denoiser D*_sigma, band-resolved -- the stand-in
for the BP oracle that does not exist on images.

WHY THIS EXISTS
---------------
On RHM, the floor E||n||^2 (n = x0 - E[x0|xt], irreducible even for a perfect
model) is removable because exact BP gives you E[x0|xt] directly. On images
there is no oracle, and E[x0|xt] is a full nonparametric regression -- not
computable. The only way to remove the floor is the two-predictor trick:
write the same identity for the model and for a SECOND predictor whose own
population loss is computable in closed form, and subtract, so the identical
floor term cancels algebraically:

    E||W_j(x0 - x0_hat)||^2 = E||W_j n||^2 + E||W_j e_model||^2
    E||W_j(x0 - g     )||^2 = E||W_j n||^2 + E||W_j h||^2
    ------------------------------------------------------------- subtract
    Delta_j = E||W_j e_model||^2 - E||W_j h||^2      (floor gone, exactly)

D*_sigma -- the best AFFINE predictor of x0 from xt -- is chosen as g purely
because its population loss has a closed form needing only (mu, Sigma), no
model and no training.

THE COST, STATED PLAINLY: the floor cancels exactly regardless of which g you
pick, but the LEFTOVER term h = E[x0|xt] - g(xt) does not vanish and is not
removed. D*_sigma is the best predictor within a RESTRICTED (affine) class,
so h is exactly "everything nonlinear that a Gaussian model cannot see" --
edges, texture, compositional structure. h is known to be largest at FINE
scales (natural-image statistics are strongly non-Gaussian at fine scale,
close to Gaussian at coarse scale after enough noise). So Delta_j is NOT a
clean per-band model-quality measure: fine bands get a permissive floor
(-||W_j h||^2 is large) while coarse bands are pinned near zero. Read
ORDERING and TRAJECTORY SHAPE across checkpoints; do not read a single
checkpoint's cross-band magnitudes as "which level is worse".

MATH (verified against Monte Carlo)
-----------------------------------
Under x0 ~ N(mu, Sigma) and xt = sqrt(ab) x0 + sqrt(1-ab) eps:

    D*_sigma(xt) = mu + sqrt(ab) Sigma M^-1 (xt - sqrt(ab) mu),
                   M = ab Sigma + (1-ab) I
    Cov(x0|xt)   = Sigma - ab Sigma M^-1 Sigma          (independent of xt)
    E||W_j(x0 - D*)||^2 = tr( [Cov(x0|xt)]_{jj} )       (block trace on band j)

Everything is done in the WAVELET domain: W is orthonormal, so the isotropic
Gaussian forward process is unchanged by it (W eps is still standard normal),
and the band-block traces are then just contiguous diagonal blocks of the
transformed conditional covariance. This is exact, not an approximation.
"""
from typing import Dict, Optional

import torch

from .wavelet import band_names, band_sizes, dwt2


def flatten_bands(coeffs: Dict[str, torch.Tensor], levels: int) -> torch.Tensor:
    """Concatenate band coefficients into one (B, D) vector, fine -> coarse."""
    return torch.cat([coeffs[n].flatten(1) for n in band_names(levels)], dim=1)


def band_slices(levels: int, shape) -> Dict[str, slice]:
    """Index ranges of each band inside the flatten_bands() vector."""
    sizes = band_sizes(levels, shape)
    out, off = {}, 0
    for n in band_names(levels):
        out[n] = slice(off, off + sizes[n])
        off += sizes[n]
    return out


class GaussianBandBaseline:
    """Fits (mu, Sigma) of the wavelet coefficients once, then serves exact
    per-band population losses of D*_sigma at any noise level.

    Model-free with respect to the diffusion network: nothing here ever sees
    x0_hat. Fit once on the training set, reuse for every checkpoint.
    """

    def __init__(self, levels: int, shape, diagonal: bool = False,
                 shrinkage: float = 1e-3, device=None):
        """
        Args:
            levels: J, must match whatever the diagnostics use.
            shape: (C, H, W) of one image.
            diagonal: if True keep only per-coefficient variances (D-vector
                instead of DxD). Much cheaper and numerically bulletproof, but
                a STRICTLY WEAKER baseline: it cannot exploit correlations
                between coefficients, so h is larger and the fine-band floor
                even more permissive. Use for large D or as a cross-check;
                prefer full covariance when D^2 fits in memory.
            shrinkage: ridge added to Sigma's diagonal (Ledoit-Wolf-style).
                With D comparable to N the sample covariance is singular and
                M^-1 is unusable; this is not cosmetic.
            device: torch device.
        """
        self.levels = levels
        self.shape = tuple(shape)
        self.diagonal = diagonal
        self.shrinkage = shrinkage
        self.device = device
        self.mu: Optional[torch.Tensor] = None
        self.Sigma: Optional[torch.Tensor] = None       # (D,D) or (D,) if diagonal
        self.slices = band_slices(levels, shape)
        self.D = sum(band_sizes(levels, shape).values())

    @torch.no_grad()
    def fit(self, x0: torch.Tensor, batch_size: int = 512) -> "GaussianBandBaseline":
        """Estimate (mu, Sigma) of wavelet coefficients from clean images.

        Args:
            x0: (N, C, H, W) clean training images.
            batch_size: streaming batch size for the DWT pass.
        """
        n = x0.shape[0]
        if not self.diagonal and self.D > 6000:
            raise ValueError(
                f"D={self.D} would need a {self.D}x{self.D} covariance "
                f"(~{self.D**2*4/1e9:.1f} GB). Use diagonal=True or smaller images."
            )
        if n < self.D and not self.diagonal:
            # Not fatal (shrinkage handles rank deficiency) but the estimate is
            # poor and every downstream Delta_j inherits that error, so say so.
            print(f"[GaussianBandBaseline] WARNING: N={n} < D={self.D}; sample covariance "
                  f"is rank-deficient and leans entirely on shrinkage={self.shrinkage:g}. "
                  f"Treat baseline losses as rough.")

        acc_sum = torch.zeros(self.D, dtype=torch.float64, device=self.device)
        acc_outer = (torch.zeros(self.D, dtype=torch.float64, device=self.device)
                     if self.diagonal else
                     torch.zeros(self.D, self.D, dtype=torch.float64, device=self.device))
        for i in range(0, n, batch_size):
            b = x0[i:i + batch_size].to(self.device)
            y = flatten_bands(dwt2(b, self.levels), self.levels).double()
            acc_sum += y.sum(0)
            if self.diagonal:
                acc_outer += y.pow(2).sum(0)
            else:
                acc_outer += y.T @ y

        mu = acc_sum / n
        if self.diagonal:
            var = acc_outer / n - mu.pow(2)
            self.Sigma = var.clamp_min(0) + self.shrinkage
        else:
            cov = acc_outer / n - torch.outer(mu, mu)
            cov = 0.5 * (cov + cov.T)                       # enforce symmetry
            self.Sigma = cov + self.shrinkage * torch.eye(self.D, dtype=torch.float64,
                                                          device=self.device)
        self.mu = mu
        return self

    @torch.no_grad()
    def band_losses(self, alpha_bar: float) -> Dict[str, float]:
        """Exact per-band population MSE of D*_sigma at one noise level.

        Args:
            alpha_bar: scalar alpha_bar_t.

        Returns:
            dict band -> PER-COEFFICIENT mean squared error (matching
            wavelet.band_energies(per_coefficient=True), so the two are
            directly subtractable).
        """
        if self.Sigma is None:
            raise RuntimeError("call fit() first")
        ab = float(alpha_bar)
        if self.diagonal:
            lam = self.Sigma                                    # (D,)
            cond = lam * (1 - ab) / (ab * lam + (1 - ab))       # exact, per coefficient
        else:
            S = self.Sigma
            M = ab * S + (1 - ab) * torch.eye(self.D, dtype=S.dtype, device=S.device)
            cond_full = S - ab * S @ torch.linalg.solve(M, S)
            cond = torch.diagonal(cond_full)
        out = {}
        for name, sl in self.slices.items():
            blk = cond[sl]
            out[name] = float(blk.mean().item())
        return out

    @torch.no_grad()
    def band_losses_over_schedule(self, alpha_bars: torch.Tensor) -> Dict[str, torch.Tensor]:
        """band_losses() for every t in the schedule. Computed once, reused."""
        names = band_names(self.levels)
        acc = {n: [] for n in names}
        for ab in alpha_bars.tolist():
            bl = self.band_losses(ab)
            for n in names:
                acc[n].append(bl[n])
        return {n: torch.tensor(v) for n, v in acc.items()}
