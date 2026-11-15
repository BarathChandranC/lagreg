"""
Hierarchical-lag regularizer for mod-K MNIST using the SAME quadratic
decomposition the evaluator logs (modules/decomposition.py), estimated on
training batches:

    L = eps-MSE + lambda * ReLU( n_global - sg[n_local] - margin )^2

    n_local  = R_local  / G_loc,   R_local  = E||Pi_t f - m_loc||^2,   G_loc  = E||m_loc||^2
    n_global = R_global / G_glob,  R_global = E||(I-Pi_t) f - m_glob||^2, G_glob = E||m_glob||^2

Per call, on the composites passed in (the caller passes the first
--reg-batch rows of the random training batch), each at its own t:
  * oracle targets m_loc, m_glob and the tile posteriors eta (no grad);
  * M conditional completions per (composite, tile): tile i held at z_i, the
    other three redrawn from their exact conditional (C_i ~ eta_i, two residues
    uniform, the last fixed by the rule, fresh pool tiles, fresh noise at that
    composite's t) -- identical to the evaluator's sampler;
  * the model is run on the completions WITH gradient, and
        R_local  <- U-statistic   (||sum v||^2 - sum ||v||^2) / (M(M-1)),  v = f_i(completion) - m_loc_i
        R_global <- ||f_i(z) - m_glob_i - (m_loc_i + vbar)||^2 - S^2 / M
    exactly as in the evaluator; f_i(z) is the main forward pass's output.
These are unbiased for R_local, R_global at the current theta, so their
gradients are unbiased gradients of R_local, R_global. (ReLU(.)^2 of a noisy
estimate is not unbiased for ReLU(.)^2 of the truth -- noise adds a small
positive bias to the penalty; --reg-margin absorbs some of it.)

G_loc and G_glob are fixed constants computed once on reference composites with
t ~ Uniform{1..T} (the training distribution), so the per-batch n's are on the
same scale as the logged t-averaged emp_avg_n_local / emp_avg_n_global.

Because the quadratic n_global also charges leakage (n_global = (1-align)^2 +
spurious), this penalty pushes both toward learning the rule AND toward removing
cross-tile dependence unrelated to it. It is bounded below (n_global >= 0 in
expectation), so it cannot be driven to -infinity.

Cost: reg_batch * 4 * M extra forward+backward passes per regularized step
(defaults 32 * 4 * 2 = 256 images, about two extra batches of 128).
"""
import contextlib

import torch
import torch.nn.functional as F

from .data import to_tiles, from_tiles, sample_composites


def completions_batch(pools, K, z_tiles, eta, a, s, M):
    """
    z_tiles (nb,4,d) noisy tiles, eta (nb,4,K) tile posteriors, a, s (nb,) per composite.
    Returns images (nb*4*M, 1, H, W) ordered (composite, region i, completion m), with
    region i held at z_i and the other three tiles resampled from their exact conditional.
    """
    nb, _, d = z_tiles.shape
    dev = z_tiles.device
    ci = torch.multinomial(eta.reshape(nb * 4, K).float(), M, replacement=True).view(nb, 4, M)
    ar = torch.arange(4, device=dev)
    cls = pools.rule.complete(ci)                                     # (nb,4,M,4)
    x = pools.tiles(pools.sample_idx(cls))                             # (nb,4,M,4,d)
    a_ = a.float().view(nb, 1, 1, 1, 1)
    s_ = s.float().view(nb, 1, 1, 1, 1)
    zz = a_ * x + s_ * torch.randn_like(x)
    zz[:, ar, :, ar] = z_tiles.float().permute(1, 0, 2).unsqueeze(2).expand(4, nb, M, d)
    return from_tiles(zz.reshape(nb * 4 * M, 4, d), pools.tile_size)


class LagRegularizer:
    def __init__(self, oracle, sched, pools, margin=0.0, n_completions=2, n_ref=2048,
                 seed=0, log=print):
        M = int(n_completions)
        if M < 2:
            raise ValueError("--reg-completions must be >= 2 (U-statistic needs pairs)")
        self.oracle, self.sched, self.pools = oracle, sched, pools
        self.K = pools.K
        self.margin, self.M = float(margin), M

        g = torch.Generator(device=pools.device).manual_seed(70_000 + seed)
        x, _, _ = sample_composites(pools, n_ref, generator=g)
        t = torch.randint(1, sched.T + 1, (n_ref,), device=pools.device, generator=g)
        eps = torch.randn(x.shape, device=pools.device, generator=g)
        a, s = sched.a_s(t)
        tg = oracle.targets(to_tiles(sched.q_sample(x, t, eps)), a, s)
        self.G_loc = float((tg["m_loc"] ** 2).sum((1, 2)).mean())
        self.G_glob = float((tg["m_glob"] ** 2).sum((1, 2)).mean())
        log(f"lag regularizer: G_loc = {self.G_loc:.4g}, G_glob = {self.G_glob:.4g} "
            f"(per composite, t ~ U{{1..{sched.T}}}, n_ref={n_ref}); "
            f"M={self.M}, margin={self.margin}")

    def __call__(self, model, pred, zt, t, autocast_ctx=None):
        """
        model: the denoiser; pred: its output on zt (with grad); zt (n,1,H,W); t (n,).
        autocast_ctx: the training forward's autocast context (or None).
        Returns (reg, info dict of floats). Gradients flow through pred and through
        the completion forward passes.
        """
        nb = zt.shape[0]
        M, d = self.M, self.oracle.d
        z_sub, t_sub = zt, t
        a, s = self.sched.a_s(t_sub)
        with torch.no_grad():
            zt_t = to_tiles(z_sub)
            tg = self.oracle.targets(zt_t, a, s)
            ml, mg = tg["m_loc"].float(), tg["m_glob"].float()
            imgs = completions_batch(self.pools, self.K, zt_t, tg["eta"], a, s, M)
        tb = t_sub.repeat_interleave(4 * M)
        with (autocast_ctx if autocast_ctx is not None else contextlib.nullcontext()):
            out = model(imgs, tb)
        fo = to_tiles(out.float()).view(nb, 4, M, 4, d)
        ar = torch.arange(4, device=zt.device)
        fi = fo[:, ar, :, ar].permute(1, 0, 2, 3)                      # (nb,4,M,d): region-i output
        v = fi - ml.unsqueeze(2)
        Sv = v.sum(2)
        Q = (v ** 2).sum((2, 3))
        U = ((Sv ** 2).sum(-1) - Q) / (M * (M - 1))                     # (nb,4) -> R_local
        vbar = Sv / M
        S2 = (Q - M * (vbar ** 2).sum(-1)) / (M - 1)
        f0 = to_tiles(pred.float())
        G = ((f0 - mg - (ml + vbar)) ** 2).sum(-1) - S2 / M              # (nb,4) -> R_global

        n_loc = U.sum(1).mean() / self.G_loc
        n_glob = G.sum(1).mean() / self.G_glob
        lag = n_glob - n_loc.detach()
        reg = F.relu(lag - self.margin) ** 2
        return reg, {"n_global": float(n_glob.detach()), "n_local": float(n_loc.detach()),
                     "lag": float(lag.detach()), "reg": float(reg.detach())}