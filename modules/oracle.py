"""
Exact Bayes oracle for composite MNIST, relative to the finite tile pool and a
composition rule (modules/rules.py). The rule supplies the per-tile marginal
prior and the joint class posterior; everything else here is rule-agnostic.

Per tile i and residue c (pool D_c, tile x in D_c):

    L_i(c)        = p(z_i | C_i = c) = |D_c|^-1 sum_{x in D_c} N(z_i; a x, s^2 I)
    eta_{i,c}     = P(C_i = c | Z_i = z_i)      = softmax_c log L_i(c)     (C_i marginally uniform)
    mu_{i,c}      = E[X_i | Z_i = z_i, C_i = c] = within-pool softmax mean

Joint class posterior. With S_{-i}(r) = P_indep(sum_{j != i} C_j = r mod K)
under the product of the other three eta's (a circular convolution),

    pi_{i,c}      = P(C_i = c | Z = z) = eta_{i,c} S_{-i}(-c) / Z,
    Z             = sum_c eta_{i,c} S_{-i}(-c) = A_t(z) / K       (same for every i)

Conditional means (X_i is independent of everything else given C_i):

    E[X_i | Z = z]     = sum_c pi_{i,c}  mu_{i,c}
    E[X_i | Z_i = z_i] = sum_c eta_{i,c} mu_{i,c}

Epsilon-prediction targets:

    m*_i      = (z_i - a E[X_i | z]) / s
    m^loc_i   = (z_i - a E[X_i | z_i]) / s                = (Pi_t m*)_i
    m^glob_i  = m*_i - m^loc_i = -(a/s) sum_c (pi_{i,c} - eta_{i,c}) mu_{i,c}

All computed in float64 by default: m^glob is a difference of nearly-equal
quantities multiplied by a/s, which is large at low noise.
"""
import math

import torch


class ModKOracle:
    def __init__(self, pools, dtype=torch.float64, chunk=256):
        self.pools = pools
        self.K = pools.K
        self.d = pools.d
        self.dtype = dtype
        self.chunk = int(chunk)
        X = pools.flat.to(dtype)                                  # (P, d)
        self.X = X
        self.sqn = (X * X).sum(-1)                                # (P,)
        # residue id of every pool row, and log|D_c|
        self.res = torch.repeat_interleave(
            torch.arange(self.K, device=X.device), pools.sizes)   # (P,)
        self.log_sizes = pools.sizes.to(dtype).log()              # (K,)
        # circular-convolution index: conv(p,q)[r] = sum_a p[a] q[(r-a) mod K]
        r = torch.arange(self.K, device=X.device)
        self._conv_idx = (r[:, None] - r[None, :]) % self.K       # [r, a] -> (r-a) mod K

    # ---------------------------------------------------- per-tile quantities

    @torch.no_grad()
    def tile_posteriors(self, z, a, s):
        """
        z: (n, d) noisy tiles; a, s: (n,) scale/std (any float dtype).
        Returns eta (n,K), mu (n,K,d), logL (n,K) (log L up to a per-row constant).
        """
        z = z.to(self.dtype)
        a = a.to(self.dtype)
        s = s.to(self.dtype)
        n = z.shape[0]
        K, P = self.K, self.X.shape[0]
        eta = torch.empty(n, K, dtype=self.dtype, device=z.device)
        logL = torch.empty(n, K, dtype=self.dtype, device=z.device)
        mu = torch.empty(n, K, self.d, dtype=self.dtype, device=z.device)
        onehot = torch.nn.functional.one_hot(self.res, K).to(self.dtype)   # (P,K)

        for lo in range(0, n, self.chunk):
            hi = min(n, lo + self.chunk)
            zc, ac, sc = z[lo:hi], a[lo:hi, None], s[lo:hi, None]
            # log N(z; a x, s^2 I) up to terms constant in x:
            #   (a z.x - a^2 |x|^2 / 2) / s^2
            logw = (ac * (zc @ self.X.T) - 0.5 * ac ** 2 * self.sqn[None, :]) / sc ** 2   # (m,P)
            # per-residue logsumexp via a max-shift shared across residues
            mx = torch.full((hi - lo, K), -math.inf, dtype=self.dtype, device=z.device)
            mx = mx.scatter_reduce(1, self.res.expand(hi - lo, P), logw, reduce="amax")
            w = torch.exp(logw - mx[:, self.res])                                        # (m,P) in (0,1]
            Zc = w @ onehot                                                              # (m,K)
            lL = mx + Zc.log() - self.log_sizes[None, :]
            logL[lo:hi] = lL
            eta[lo:hi] = torch.softmax(lL, dim=-1)
            # mu_c = sum_{x in D_c} w_x x / sum_{x in D_c} w_x
            for c in range(K):
                o = int(self.pools.offsets[c]); sz = int(self.pools.sizes[c])
                wc = w[:, o:o + sz]
                mu[lo:hi, c] = (wc @ self.X[o:o + sz]) / Zc[:, c:c + 1]
        return eta, mu, logL

    def circ_conv(self, p, q):
        """p, q: (..., K) -> (..., K), conv[r] = sum_a p[a] q[r-a]."""
        return (p.unsqueeze(-2) * q[..., self._conv_idx]).sum(-1)

    def joint_class_posterior(self, eta):
        """eta: (n,4,C) -> pi (n,4,C), A (n,), Zs (n,4). Delegated to the rule."""
        return self.pools.rule.joint_posterior(eta)

    # ------------------------------------------------------------ targets

    @torch.no_grad()
    def targets(self, z_tiles, a, s):
        """
        z_tiles: (n,4,d); a, s: (n,).
        Returns dict with m_star, m_loc, m_glob (n,4,d), eta, pi (n,4,K), A (n,),
        and Zcheck (n,4) -- the four per-region normalisers, which must agree.
        """
        n = z_tiles.shape[0]
        zf = z_tiles.reshape(n * 4, self.d)
        af = a.repeat_interleave(4)
        sf = s.repeat_interleave(4)
        _, mu, logL = self.tile_posteriors(zf, af, sf)
        # eta_i(c) = P(C_i = c | z_i) uses the rule's own per-slot marginal prior,
        # which is uniform for mod-K but NOT for e.g. c0+c1 = c2+c3.
        eta = torch.softmax(logL.view(n, 4, self.K)
                            + self.pools.rule.log_marg.to(self.dtype)[None], dim=-1)
        mu = mu.view(n, 4, self.K, self.d)
        pi, A, Zs = self.joint_class_posterior(eta)

        a_ = a.to(self.dtype).view(n, 1, 1)
        s_ = s.to(self.dtype).view(n, 1, 1)
        z = z_tiles.to(self.dtype)
        Ex_loc = (eta.unsqueeze(-1) * mu).sum(-2)
        Ex_full = (pi.unsqueeze(-1) * mu).sum(-2)
        m_loc = (z - a_ * Ex_loc) / s_
        m_glob = -(a_ / s_) * ((pi - eta).unsqueeze(-1) * mu).sum(-2)
        m_star = m_loc + m_glob
        return dict(m_star=m_star, m_loc=m_loc, m_glob=m_glob,
                    eta=eta, pi=pi, A=A, Zcheck=Zs,
                    Ex_loc=Ex_loc, Ex_full=Ex_full)


class OracleDenoiser(torch.nn.Module):
    """
    Wraps the oracle as an eps-predicting 'model' f(x_img, t). Used only by
    check_oracle.py to validate the evaluator: mode='star' must give
    R_local = R_global = 0; mode='loc' must give R_local = 0, n_global = 1;
    mode='zero' must give n_local = n_global = 1.
    """

    def __init__(self, oracle, sched, mode="star"):
        super().__init__()
        self.oracle, self.sched, self.mode = oracle, sched, mode
        self._dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x, t):
        from .data import to_tiles, from_tiles
        if self.mode == "zero":
            return torch.zeros_like(x)
        a, s = self.sched.a_s(t)
        tg = self.oracle.targets(to_tiles(x), a, s)
        key = {"star": "m_star", "loc": "m_loc"}[self.mode]
        return from_tiles(tg[key].to(x.dtype), self.oracle.pools.tile_size)