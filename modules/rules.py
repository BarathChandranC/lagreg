"""
Composition rules on the four tile classes, as an explicit validity mask.

A rule is any subset V of C^4 (C = classes per tile). The data distribution is
uniform on V, so

    P(c)        = 1_V(c) / |V|
    marg_i(c)   = P(C_i = c)                     -- NOT uniform in general
    chance      = sum_{c in V} prod_i marg_i(c_i)
                  = P(rule holds) for a model that emits tiles independently
                    from the correct per-tile marginals.  This is the baseline
                    `valid_given_recognised` must beat.  (mod-K: exactly 1/K.)

Bayes quantities. With L_i(c) = p(z_i | C_i = c) and
eta_i(c) = P(C_i = c | z_i) proportional to marg_i(c) L_i(c), write
w_i = eta_i / marg_i (proportional to L_i).  Then

    P(c | z)    proportional to  P(c) prod_i w_i(c_i)
    pi_i(c)     = sum_{c': c'_i = c} P(c') prod_j w_j(c'_j) / A
    A_t(z)      = sum_c P(c) prod_i w_i(c_i)
                = p_rule(z) / p_independent(z)

For the mod-K rule this reproduces the old circular-convolution oracle exactly
(there marg is uniform, w = K eta, and A = K * P_indep(sum C = 0)).

Rules provided:
    modk(K)        sum_i c_i = 0 (mod K).  Every TRIPLE of classes is exactly
                   independent, so there is no pairwise or three-way signal:
                   the constraint lives only in the full four-way joint.
    sum_pair()     c_0 + c_1 = c_2 + c_3 on digits 0..9 (Lu et al., 670 valid
                   tuples).  Pairwise correlated -- a large c_0 forces a large
                   c_2 + c_3 -- so gradient descent has a low-order path to it,
                   and FID is NOT blind to this rule.
    pairwise(C)    c_3 = c_0, with c_1, c_2 free.  Control that isolates
                   whether four-way parity structure is the obstacle.
    checksum(C)    c_3 = (c_0 + c_1 + c_2) mod C.  Identical to modk(C) up to
                   relabelling; kept only to make that explicit.
"""
import torch


class TupleRule:
    """Rule on 4 tile classes given by a boolean mask over C^4."""

    def __init__(self, mask, name, dtype=torch.float64):
        if mask.ndim != 4 or len(set(mask.shape)) != 1:
            raise ValueError(f"mask must be (C,C,C,C), got {tuple(mask.shape)}")
        self.name = name
        self.C = int(mask.shape[0])
        self.dtype = dtype
        self.mask = mask.bool()
        self.valid = self.mask.nonzero()                       # (V,4) long
        self.n_valid = int(self.valid.shape[0])
        if self.n_valid == 0:
            raise ValueError(f"rule {name!r} has no valid tuples")
        self.prior = self.mask.to(dtype) / self.n_valid        # (C,C,C,C)
        self.marg = torch.stack([                              # (4,C)
            self.prior.sum(dim=tuple(j for j in range(4) if j != i)) for i in range(4)])
        if (self.marg <= 0).any():
            raise ValueError(f"rule {name!r}: some class never occurs; drop it from C")
        self.log_marg = self.marg.log()
        self._build_conditional()

    # ---- P(rule) under independent tiles drawn from the correct marginals
    @property
    def chance(self):
        p = self.marg[0][:, None, None, None] * self.marg[1][None, :, None, None] \
            * self.marg[2][None, None, :, None] * self.marg[3][None, None, None, :]
        return float((p * self.mask).sum())

    def _build_conditional(self):
        """cond[i][c] lists the rows of `valid` whose slot i equals c (zero-padded)."""
        cnt = torch.zeros(4, self.C, dtype=torch.long)
        cond = torch.zeros(4, self.C, 1, dtype=torch.long)
        rows = []
        for i in range(4):
            per_c = [torch.nonzero(self.valid[:, i] == c, as_tuple=True)[0] for c in range(self.C)]
            cnt[i] = torch.tensor([len(x) for x in per_c])
            rows.append(per_c)
        width = int(cnt.max())
        cond = torch.zeros(4, self.C, width, dtype=torch.long)
        for i in range(4):
            for c in range(self.C):
                r = rows[i][c]
                if len(r):
                    cond[i, c, :len(r)] = r
        self.cnt, self.cond = cnt, cond

    def to(self, device):
        for k in ("mask", "valid", "prior", "marg", "log_marg", "cnt", "cond"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    # ------------------------------------------------------------- sampling

    def sample(self, n, generator=None, device=None):
        """(n,4) class tuples drawn uniformly from the valid set."""
        dev = device or self.valid.device
        i = torch.randint(0, self.n_valid, (n,), device=dev, generator=generator)
        return self.valid.to(dev)[i]

    def sample_independent(self, n, generator=None, device=None):
        """(n,4) tiles drawn INDEPENDENTLY from the correct per-tile marginals
        (the rule ignored) -- the FID 'no rule' control."""
        dev = device or self.marg.device
        return torch.stack([torch.multinomial(self.marg[i].to(dev).float(), n,
                                              replacement=True, generator=generator)
                            for i in range(4)], dim=1)

    def satisfies(self, c):
        """c: (...,4) long -> (...) bool."""
        return self.mask[c[..., 0], c[..., 1], c[..., 2], c[..., 3]]

    def complete(self, ci, generator=None):
        """
        ci: (nb,4,M) -- for each region i (dim 1), M sampled classes for slot i.
        Returns (nb,4,M,4): valid tuples with slot i set to ci[:, i, :], the
        other three drawn uniformly from the rule's conditional given slot i.
        """
        nb, _, M = ci.shape
        out = torch.empty(nb, 4, M, 4, dtype=torch.long, device=ci.device)
        for i in range(4):
            c = ci[:, i]                                               # (nb,M)
            n_c = self.cnt[i][c]                                       # (nb,M)
            u = torch.rand(c.shape, device=c.device, generator=generator)
            j = torch.minimum((u * n_c).long(), n_c - 1)
            out[:, i] = self.valid[self.cond[i][c, j]]                 # (nb,M,4)
        return out

    # ------------------------------------------------------------ posterior

    def joint_posterior(self, eta):
        """
        eta: (n,4,C) per-tile posteriors P(C_i | z_i).
        Returns pi (n,4,C), A (n,), Zs (n,4) -- the four per-region normalisers,
        which agree up to round-off and are logged as a consistency check.
        """
        w = eta / self.marg.to(eta.dtype)[None]                        # proportional to L_i
        pr = self.prior.to(eta.dtype)
        slots = "abcd"
        pi, Zs = [], []
        for i in range(4):
            others = [j for j in range(4) if j != i]
            # S_i(c) = sum over the other three of prior * prod_{j != i} w_j
            spec = "abcd," + ",".join(f"n{slots[j]}" for j in others) + f"->n{slots[i]}"
            S = torch.einsum(spec, pr, *[w[:, j] for j in others])     # (n,C)
            un = w[:, i] * S
            Z = un.sum(-1, keepdim=True)
            pi.append(un / Z)
            Zs.append(Z.squeeze(-1))
        return torch.stack(pi, 1), Zs[0], torch.stack(Zs, 1)

    def __repr__(self):
        return (f"TupleRule({self.name!r}, C={self.C}, valid={self.n_valid}/{self.C ** 4}, "
                f"chance={self.chance:.4f})")


# ----------------------------------------------------------------- builders

def _grid(C):
    r = torch.arange(C)
    return torch.meshgrid(r, r, r, r, indexing="ij")


def modk(K):
    a, b, c, d = _grid(K)
    return TupleRule(((a + b + c + d) % K) == 0, f"modk{K}")


def checksum(C):
    a, b, c, d = _grid(C)
    return TupleRule(d == ((a + b + c) % C), f"checksum{C}")


def sum_pair(C=10):
    """c_0 + c_1 = c_2 + c_3 (top row sum = bottom row sum). Lu et al.'s rule."""
    a, b, c, d = _grid(C)
    return TupleRule((a + b) == (c + d), f"sumpair{C}")


def pairwise(C):
    """c_3 = c_0; c_1 and c_2 unconstrained."""
    a, b, c, d = _grid(C)
    return TupleRule(d == a, f"pairwise{C}")


RULES = {"modk": modk, "checksum": checksum, "sumpair": sum_pair, "pairwise": pairwise}


def build(name, K):
    if name not in RULES:
        raise ValueError(f"unknown rule {name!r}; choose from {sorted(RULES)}")
    return RULES[name](K)