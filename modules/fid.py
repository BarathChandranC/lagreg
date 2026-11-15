"""
FID / KID for mod-K MNIST samples, in the tile classifier's feature space.

Features: penultimate layer of TileCNN (2*width dims per tile), the four
tiles' features concatenated into one vector per composite.
  fid_composite  Fréchet distance on those vectors
  kid_composite  unbiased MMD^2, cubic polynomial kernel, same vectors

IMPORTANT -- whether FID can see the rule DEPENDS ON THE RULE. Frechet distance
compares means and covariances (1st/2nd order). Under mod-K every set of <= 3
tiles is exactly independent, so rule-satisfying composites and independent ones
have identical feature means and covariances, and FID is blind to it. Under a
pairwise-correlated rule such as c0 + c1 = c2 + c3 that is NOT true: FID does
see the rule there.
`fid_norule_floor` (logged once) is the FID between those two real sets:
it should be ~ the noise floor `fid_real_floor`, which is the empirical
proof that FID is blind to hallucination here. Use FID for local quality,
valid|recognised and the decomposition for the rule.

Bias: FID is biased upward at finite n; compare checkpoints only at the same
n_gen, and read them against `fid_real_floor` (real vs real at that n).
KID is unbiased, so its floor is ~0.
"""
import numpy as np
import torch

from .data import to_tiles, from_tiles, sample_composites


# ------------------------------------------------------------ statistics

def _sqrtm_psd(c):
    w, v = np.linalg.eigh((c + c.T) / 2)
    return (v * np.sqrt(np.clip(w, 0, None))) @ v.T


def frechet(mu1, c1, mu2, c2):
    """||mu1-mu2||^2 + tr(c1) + tr(c2) - 2 tr((c1^1/2 c2 c1^1/2)^1/2), via eigh (no scipy)."""
    s1 = _sqrtm_psd(c1)
    m = _sqrtm_psd(s1 @ c2 @ s1)
    d = mu1 - mu2
    return float(d @ d + np.trace(c1) + np.trace(c2) - 2 * np.trace(m))


def stats(f):
    f = np.asarray(f, dtype=np.float64)
    return f.mean(0), np.cov(f, rowvar=False)


def kid(f1, f2, n_subsets=50, subset_size=1000, seed=0):
    """Unbiased MMD^2 with k(x,y) = (x.y/d + 1)^3 (Binkowski et al. 2018). Returns (mean, std)."""
    f1 = np.asarray(f1, dtype=np.float64)
    f2 = np.asarray(f2, dtype=np.float64)
    d = f1.shape[1]
    m = min(subset_size, len(f1), len(f2))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_subsets):
        x = f1[rng.choice(len(f1), m, replace=False)]
        y = f2[rng.choice(len(f2), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
                    + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    vals = np.array(vals)
    return float(vals.mean()), float(vals.std())


# --------------------------------------------------------------- features

@torch.no_grad()
def tile_features(clf, imgs, batch=4096):
    """imgs (n,1,2s,2s) -> (n, 4, D) penultimate TileCNN features (eval mode)."""
    clf.eval()
    n, s = imgs.shape[0], imgs.shape[-1] // 2
    tiles = to_tiles(imgs).reshape(n * 4, 1, s, s)
    body = clf.f[:-1]                              # everything except the final Linear
    feats = torch.cat([body(tiles[i:i + batch]).float() for i in range(0, tiles.shape[0], batch)])
    return feats.view(n, 4, -1).cpu().numpy()


def sample_composites_norule(pools, n, generator=None):
    """Real composites with 4 INDEPENDENT tiles drawn from the rule's own per-tile
    marginals -- same 1st-order statistics as the population, rule broken."""
    c = pools.rule.sample_independent(n, generator=generator, device=pools.device)
    idx = pools.sample_idx(c, generator=generator)
    return from_tiles(pools.tiles(idx), pools.tile_size)


# --------------------------------------------------------------- evaluator

class FIDEvaluator:
    """
    Reference = n_ref fresh real composites from the population (pool + rule).
    Floors (computed once, at n_gen samples):
      fid_real_floor     real rule composites vs reference      (finite-n bias)
      fid_norule_floor   real NO-rule composites vs reference   (~ real floor => FID blind to rule)
    """

    def __init__(self, clf, pools, n_gen, n_ref=10000, seed=0, log=print):
        self.clf, self.n_gen = clf, int(n_gen)
        g = torch.Generator(device=pools.device).manual_seed(60_000 + seed)
        ref, _, _ = sample_composites(pools, n_ref, generator=g)
        fr = tile_features(clf, ref)
        self.ref_comp = fr.reshape(len(fr), -1)
        self.mu_c, self.c_c = stats(self.ref_comp)

        real, _, _ = sample_composites(pools, self.n_gen, generator=g)
        norule = sample_composites_norule(pools, self.n_gen, generator=g)
        self.floors = {}
        for name, x in (("real", real), ("norule", norule)):
            m = self._metrics(x)
            for k, v in m.items():
                self.floors[f"{k}_{name}_floor"] = v
        log("FID floors at n=%d: " % self.n_gen + "  ".join(
            f"{k} {v:.4g}" for k, v in self.floors.items()))

    def _metrics(self, imgs):
        f = tile_features(self.clf, imgs)
        fc = f.reshape(len(f), -1)
        mu, c = stats(fc)
        k_mean, k_std = kid(fc, self.ref_comp, subset_size=min(1000, len(fc)))
        return {"fid_composite": frechet(mu, c, self.mu_c, self.c_c),
                "kid_composite": k_mean, "kid_composite_std": k_std}

    def __call__(self, gen):
        if gen.shape[0] != self.n_gen:
            raise ValueError(f"FID floors were computed at n={self.n_gen}, got {gen.shape[0]} samples")
        return {**self._metrics(gen), **self.floors}