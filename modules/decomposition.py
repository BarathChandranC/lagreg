"""
Exact local/global decomposition of the eps-prediction risk, per noise level t.

Under the population measure, with (Pi_t f)_i(z_i) = E[f_i(Z_t) | Z_{t,i} = z_i]:

    E||eps - f||^2 = C_t + R_local + R_global
    C_t      = E||eps - m*||^2
    R_local  = E||Pi_t f - m^loc||^2
    R_global = E||(I - Pi_t) f - m^glob||^2
    R_local + R_global = E||f - m*||^2            (Pythagoras)

Estimating Pi_t f. Given z_i, the rest of the image is drawn from its exact
conditional: C_i ~ eta_i(. | z_i) (fresh draw per completion -- NOT the
example's own label), the other three residues uniform subject to the sum
constraint, tiles uniform from their residue pools, noised at the same t.
Z_{-i} is independent of Z_i given C_i, so this is an exact sampler of
Z_{-i} | Z_i = z_i.

With M completions and v_m = f_i(z_i, Z_{-i}^(m)) - m^loc_i(z_i):

    U        = (||sum_m v_m||^2 - sum_m ||v_m||^2) / (M (M-1))     E[U] = ||(Pi f)_i - m^loc_i||^2
    S2       = sum_m ||v_m - vbar||^2 / (M-1)                      E[S2] = tr Cov(f_i | z_i)
    G        = ||f_i(z) - m^glob_i(z) - (m^loc_i + vbar)||^2 - S2 / M
                                                                    E[G] = ||(f - Pi f)_i - m^glob_i||^2

(G uses that the original Z_{-i} and the completions are independent given z_i,
which holds on both eval sets since completions are drawn fresh.)

R_global is the direct estimator G -- unbiased for E||(I - Pi)f - m^glob||^2
on any eval set, and free of the large-term cancellation in the subtraction
form. R_global_sub = E||f - m*||^2 - R_local is reported alongside; on the
pop set the two agree in expectation (Pythagoras), so `pyth_resid` =
R_global_sub - R_global is a check of sampler + oracle. On the emp
(training) set f is evaluated at points it was fit to, the cross term
2E<Pi f - m^loc, (I - Pi)f - m^glob> need not vanish there, and pyth_resid
measures it.

Two further families of keys (added after the projection split above was
found NOT to decompose the training loss on the emp set):

  Sample-wise split ("sw_"), exact on ANY eval set, train included:
      loss - C = sw_D_base + sw_D_global          (per sample, by construction)
      sw_D_base   = E[ ||Pi f - eps||^2 - ||m^loc - eps||^2 ]
                  = R_local + 2 cross_loc,   cross_loc = E<Pi f - m^loc, m^loc - eps>
                    (estimated by E<vbar, m^loc - eps>; unbiased, since the
                    completions are independent of eps given z_i)
      sw_D_global = (loss - C) - sw_D_base
  On pop, cross_loc and the full cross term vanish in expectation, so
  sw_D_base = R_local and sw_D_global = R_global there. On the training set
  they differ: a memorizing model beats the population oracle on the points
  it was fit to, and the sw_ terms can go NEGATIVE (as the RHM ladder did).
  sw_n_local = sw_D_base / E||m^loc||^2,  sw_n_global = sw_D_global / E||m^glob||^2.

  Linear / alignment split ("lin_"), the RHM-ladder analogue (differences of
  distances to nested references, linear in f, so leakage cancels):
      lin_n_global = 1 - E<f - m^loc, m^glob> / E||m^glob||^2
      lin_n_local  = 1 - E<f - m^glob, m^loc> / E||m^loc||^2
  Bayes -> 0 exactly; local-only model f = m^loc -> lin_n_global = 1;
  zero predictor -> 1 + E<m^loc, m^glob>/E||.||^2 (= 1 on pop in expectation).
  Raw inner products ip_f_glob, ip_f_loc, ip_loc_glob are logged so these
  can be t-averaged exactly.

All quantities are reported per pixel (sums over the 4 tiles divided by D).
Normalisation follows the attached note: n_local = R_local / E||m^loc||^2,
n_global = R_global / E||m^glob||^2 (zero predictor -> 1, Bayes -> 0). At low
noise E||m^glob||^2 can be tiny; `glob_energy` is logged so those t can be
excluded downstream instead of trusted.
"""
import math

import torch

from .data import to_tiles, from_tiles


def _se(x):
    x = x.double()
    return float(x.std(unbiased=True) / math.sqrt(max(1, x.numel()))) if x.numel() > 1 else 0.0


class DecompositionEvaluator:
    def __init__(self, oracle, sched, x0, t_list, n_completions=16, seed=0,
                 batch_size=1024, label="pop"):
        if n_completions < 2:
            raise ValueError("need n_completions >= 2 for the unbiased U-statistic")
        self.oracle, self.sched = oracle, sched
        self.pools = oracle.pools
        self.K, self.s_tile, self.d = oracle.K, oracle.pools.tile_size, oracle.d
        self.D = 4 * self.d
        self.M = int(n_completions)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.label = label
        dev = x0.device
        self.device = dev

        g = torch.Generator(device=dev).manual_seed(self.seed)
        self.items = []
        for t in t_list:
            n = x0.shape[0]
            tt = torch.full((n,), int(t), dtype=torch.long, device=dev)
            eps = torch.randn(x0.shape, generator=g, device=dev)
            z = sched.q_sample(x0, tt, eps)
            a, s = sched.a_s(tt)
            tg = oracle.targets(to_tiles(z), a, s)
            zc = tg["Zcheck"]
            spread = float(((zc.max(1).values - zc.min(1).values) / zc.mean(1)).abs().max())
            if spread > 1e-6:
                raise AssertionError(f"[{label} t={t}] per-region normalisers disagree "
                                     f"(rel spread {spread:g}) -- joint posterior is wrong")
            f32 = lambda v: v.float()
            self.items.append(dict(
                t=int(t), tt=tt, z=z, eps_tiles=to_tiles(eps),
                a=float(a[0]), s=float(s[0]),
                eta=tg["eta"].float(),
                m_star=f32(tg["m_star"]), m_loc=f32(tg["m_loc"]), m_glob=f32(tg["m_glob"]),
            ))
        self.n = x0.shape[0]

    # ---------------------------------------------------------- completions

    def _completions(self, it, b_idx, gen):
        """
        For composites b_idx (nb,), every region i and M completions, build the
        full noisy images with region i held at z_i and the other three resampled.
        Returns images (nb*4*M, 1, H, W), ordered (b, i, m).
        """
        K, M, nb = self.K, self.M, b_idx.numel()
        dev = self.device
        z_t = to_tiles(it["z"][b_idx])                                   # (nb,4,d)
        eta = it["eta"][b_idx]                                           # (nb,4,K)

        # C_i ~ eta_i, fresh per completion
        ci = torch.multinomial(eta.reshape(nb * 4, K), M, replacement=True,
                               generator=gen).view(nb, 4, M)             # (nb,4,M)
        cls = torch.randint(0, K, (nb, 4, M, 4), device=dev, generator=gen)   # [b,i,m,slot]
        ar = torch.arange(4, device=dev)
        cls[:, ar, :, ar] = ci.permute(1, 0, 2)                         # slot i <- c_i
        # the 'last' free slot for region i is (i+3)%4; fix it by the constraint
        last = (ar + 3) % 4
        tot = cls.sum(-1) - cls[:, ar, :, last].permute(1, 0, 2)         # sum over the other 3
        cls[:, ar, :, last] = ((-tot) % K).permute(1, 0, 2)

        idx = self.pools.sample_idx(cls, generator=gen)                  # (nb,4,M,4)
        x = self.pools.tiles(idx)                                        # (nb,4,M,4,d)
        noise = torch.randn(x.shape, device=dev, generator=gen)
        zz = it["a"] * x + it["s"] * noise
        # hold region i at the observed z_i
        zz[:, ar, :, ar] = z_t.permute(1, 0, 2).unsqueeze(2).expand(4, nb, M, self.d)
        return from_tiles(zz.reshape(nb * 4 * M, 4, self.d), self.s_tile)

    # ------------------------------------------------------------- evaluate

    @torch.no_grad()
    def evaluate(self, model):
        was_training = model.training
        model.eval()
        out = {}
        try:
            for it in self.items:
                out.update(self._eval_one(model, it))
        finally:
            if was_training:
                model.train()
        return out

    def _forward(self, model, x, t):
        res = []
        for lo in range(0, x.shape[0], self.batch_size):
            xb = x[lo:lo + self.batch_size]
            tb = torch.full((xb.shape[0],), t, dtype=torch.long, device=x.device)
            res.append(model(xb, tb).float())
        return torch.cat(res)

    def _eval_one(self, model, it):
        t, D, M = it["t"], self.D, self.M
        f_full = to_tiles(self._forward(model, it["z"], t))              # (n,4,d)
        eps, ms, ml, mg = it["eps_tiles"], it["m_star"], it["m_loc"], it["m_glob"]

        per = lambda v: (v ** 2).sum((1, 2)) / D                         # (n,) per pixel
        loss_i = per(eps - f_full)
        C_i = per(eps - ms)
        exc_i = per(f_full - ms)
        Eloc = per(ml).mean()
        Eglob = per(mg).mean()
        align = ((f_full * mg).sum((1, 2)) / D).mean() / Eglob.clamp_min(1e-30)

        # ---- completions: U-statistic for R_local, debiased direct R_global
        gen = torch.Generator(device=self.device).manual_seed(self.seed + 7919 * t)
        nb_max = max(1, self.batch_size // (4 * M))
        U_b = torch.zeros(self.n, device=self.device)
        G_b = torch.zeros(self.n, device=self.device)
        S2_b = torch.zeros(self.n, device=self.device)
        CL_b = torch.zeros(self.n, device=self.device)                  # <vbar, m_loc - eps>
        ar = torch.arange(4, device=self.device)
        for lo in range(0, self.n, nb_max):
            b_idx = torch.arange(lo, min(self.n, lo + nb_max), device=self.device)
            nb = b_idx.numel()
            imgs = self._completions(it, b_idx, gen)
            fo = to_tiles(self._forward(model, imgs, t)).view(nb, 4, M, 4, self.d)
            fi = fo[:, ar, :, ar].permute(1, 0, 2, 3)                    # (nb,4,M,d): region i output
            v = fi - ml[b_idx].unsqueeze(2)                              # (nb,4,M,d)
            Sv = v.sum(2)                                                # (nb,4,d)
            Q = (v ** 2).sum((2, 3))                                     # (nb,4)
            U = ((Sv ** 2).sum(-1) - Q) / (M * (M - 1))                  # (nb,4)
            vbar = Sv / M
            S2 = (Q - M * (vbar ** 2).sum(-1)) / (M - 1)                 # (nb,4)
            resid = f_full[b_idx] - mg[b_idx] - (ml[b_idx] + vbar)       # (nb,4,d)
            G = (resid ** 2).sum(-1) - S2 / M                            # (nb,4)
            U_b[b_idx] = U.sum(1) / D
            G_b[b_idx] = G.sum(1) / D
            S2_b[b_idx] = S2.sum(1) / D
            CL_b[b_idx] = (vbar * (ml[b_idx] - eps[b_idx])).sum((1, 2)) / D

        ip = lambda u, w: ((u * w).sum((1, 2)) / D).mean()               # per pixel
        ip_fg, ip_fl, ip_lg = ip(f_full, mg), ip(f_full, ml), ip(ml, mg)
        # sample-wise split: exact on this set, by construction
        swb_i = U_b + 2 * CL_b                                           # (n,)
        swg_i = (loss_i - C_i) - swb_i

        R_loc = U_b.mean()
        R_glob_sub = exc_i.mean() - R_loc
        R_glob_dir = G_b.mean()
        key = lambda k: f"t{t}_{k}"
        return {
            key("loss"): float(loss_i.mean()), key("loss_se"): _se(loss_i),
            key("C"): float(C_i.mean()),
            key("excess"): float(exc_i.mean()), key("excess_se"): _se(exc_i),
            # identity check: loss - C - excess = 2E<eps - m*, m* - f>, zero in expectation
            key("loss_identity_resid"): float(loss_i.mean() - C_i.mean() - exc_i.mean()),
            key("R_local"): float(R_loc), key("R_local_se"): _se(U_b),
            key("R_global"): float(R_glob_dir), key("R_global_se"): _se(G_b),
            key("R_global_sub"): float(R_glob_sub),
            key("pyth_resid"): float(R_glob_sub - R_glob_dir),
            key("cond_var"): float(S2_b.mean()),                 # E tr Cov(f_i | z_i): how much f uses other tiles
            key("loc_energy"): float(Eloc), key("glob_energy"): float(Eglob),
            key("n_local"): float(R_loc / Eloc),
            key("n_global"): float(R_glob_dir / Eglob.clamp_min(1e-30)),
            key("n_global_sub"): float(R_glob_sub / Eglob.clamp_min(1e-30)),
            key("align_global"): float(align),
            # ---- sample-wise split (sums exactly to loss - C on this set)
            key("cross_loc"): float(CL_b.mean()), key("cross_loc_se"): _se(CL_b),
            key("sw_D_base"): float(swb_i.mean()), key("sw_D_base_se"): _se(swb_i),
            key("sw_D_global"): float(swg_i.mean()), key("sw_D_global_se"): _se(swg_i),
            key("sw_n_local"): float(swb_i.mean() / Eloc),
            key("sw_n_global"): float(swg_i.mean() / Eglob.clamp_min(1e-30)),
            # ---- linear / alignment split (RHM-ladder analogue)
            key("ip_f_glob"): float(ip_fg), key("ip_f_loc"): float(ip_fl),
            key("ip_loc_glob"): float(ip_lg),
            key("lin_n_global"): float(1 - (ip_fg - ip_lg) / Eglob.clamp_min(1e-30)),
            key("lin_n_local"): float(1 - (ip_fl - ip_lg) / Eloc),
        }