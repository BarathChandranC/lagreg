import contextlib

import numpy as np
import torch

from .bp_torch import run_BP_diffusion_torch
from .diffusion import generate_noisy_sequences_single
from .losses import get_loss_coefficients_continuous_new

_EPS = 1e-12


def compute_bp_soft_targets(xt, t, alpha_bars, rho, k, vocab_size, factorized_layers=0):
    alpha_bars_t = alpha_bars[t - 1]
    field_intensity = torch.sqrt(alpha_bars_t) / (1 - alpha_bars_t + 1e-8)
    field = (xt * field_intensity[:, None, None]).double()   # bp_torch runs its internal
                                                               # probs in double precision
                                                               # regardless of input dtype --
                                                               # match it here, or every
                                                               # factorized_layers>0 call fails.
    if not isinstance(rho, torch.Tensor):
        rho = torch.as_tensor(np.asarray(rho))
    rho = rho.to(xt.device).float()

    bp = run_BP_diffusion_torch(rho, k, vocab_size, field, factorized_layers=int(factorized_layers))
    return bp.float()   # back to float32 for the KL/log_softmax math downstream


@torch.no_grad()
def compute_oracle_ladder_batch(xt, t, alpha_bars, rho, k, vocab_size,
                                factorized_layers_list):
    if 0 not in [int(f) for f in factorized_layers_list]:
        raise ValueError("oracle ladder needs factorized_layers=0 (BP_0 is the reference)")

    bp0 = compute_bp_soft_targets(xt, t, alpha_bars, rho, k, vocab_size,
                                  factorized_layers=0).float()
    log_bp0 = bp0.clamp_min(_EPS).log()

    oracle_sums = {}
    for f in factorized_layers_list:
        f = int(f)
        if f == 0:
            # D_KL(BP_0 || BP_0) == 0 by construction. Computed explicitly
            # anyway (rather than hard-coded) so that a nonzero value here
            # surfaces nondeterminism in run_BP_diffusion_torch instead of
            # being masked -- see the assertion in LadderEvaluator.__init__.
            bpf = bp0
        else:
            bpf = compute_bp_soft_targets(xt, t, alpha_bars, rho, k, vocab_size,
                                          factorized_layers=f).float()
        neg_ent = torch.xlogy(bpf, bpf).sum(-1)          # = -H(BP_f), 0*log0 == 0
        kl = neg_ent - (bpf * log_bp0).sum(-1)           # (B, L)
        oracle_sums[f] = kl.sum()

    n_tokens = xt.shape[0] * xt.shape[1]
    return oracle_sums, n_tokens


@torch.no_grad()
def compute_ladder_kl_batch(model, xt, t, x0_hard, alpha_bars, alphas, rho, k, vocab_size,
                            factorized_layers_list, rescale_loss=False):

    use_cuda = xt.is_cuda
    autocast_ctx = (torch.amp.autocast('cuda', enabled=False) if use_cuda
                    else contextlib.nullcontext())

    with autocast_ctx:
        logits = model(xt.float(), t)
        log_probs = torch.log_softmax(logits.float(), dim=-1)          # (B, L, Q)

        ce_per_token = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, vocab_size), x0_hard.reshape(-1), reduction='none'
        ).view(x0_hard.shape)                                          # (B, L)

        if rescale_loss:
            coeffs = get_loss_coefficients_continuous_new(alpha_bars, alphas, t)   # (B,)
            ce_per_token = ce_per_token * coeffs.unsqueeze(1)           # matches compute_cross_entropy_loss
        loss_total_sum = ce_per_token.sum()

        kl_sums, ent_sum = {}, None
        for f in factorized_layers_list:
            bp = compute_bp_soft_targets(xt, t, alpha_bars, rho, k, vocab_size,
                                         factorized_layers=f).float()
            neg_ent = torch.xlogy(bp, bp).sum(-1)                       # 0*log0 == 0
            kl = neg_ent - (bp * log_probs).sum(-1)                     # (B, L)
            kl_sums[f] = kl.sum()
            if int(f) == 0:
                ent_sum = (-neg_ent).sum()

    n_tokens = xt.shape[0] * xt.shape[1]
    return kl_sums, ent_sum, loss_total_sum, n_tokens


def level_excess(bp_ladder, ell, oracle_ladder=None):

    missing = [k for k in range(ell + 1) if k not in bp_ladder]
    if missing:
        raise ValueError(f"level_excess needs a contiguous ladder 0..{ell}; missing {missing}")

    b = np.stack([np.atleast_1d(np.asarray(bp_ladder[k], dtype=float))
                  for k in range(ell + 1)], axis=0)                     # (ell+1, T)

    if oracle_ladder is not None:
        missing_o = [k for k in range(ell + 1) if k not in oracle_ladder]
        if missing_o:
            raise ValueError(f"oracle_ladder needs a contiguous ladder 0..{ell}; missing {missing_o}")
        o = np.stack([np.atleast_1d(np.asarray(oracle_ladder[k], dtype=float))
                      for k in range(ell + 1)], axis=0)                 # (ell+1, 1) broadcasts
        g = b - o
    else:
        g = b

    c = g[::-1]                                             # c[d] = g[ell-d]
    R = np.diff(c, axis=0, prepend=np.zeros_like(c[:1]))

    resid = np.abs(R.sum(axis=0) - b[0]).max()
    if resid > 1e-6 * max(1.0, np.abs(b[0]).max()):
        raise AssertionError(f"telescoping broken: max |sum_d R_d - bp_0| = {resid:g}")
    R = R[::-1]  
    return R


def level_excess_grouped(bp_ladder, ell, oracle_ladder, group_at=2):
    """
    Collapse the ell+1-rung ladder into 3 terms -- global, local, root -- by
    merging g_k differences on either side of a single split point
    `group_at` (default 2, giving {0,1,2} vs {2,3,4} for ell=4). This is an
    EXACT re-partition of the same telescoped sum level_excess() computes --
    it merges consecutive R_d terms, it does not introduce a new quantity:

        D_global   = g_0          - g_{group_at}
        D_local = g_{group_at} - g_ell
        D_root   = g_ell

        D_global + D_local + D_root == g_0 == bp_0^emp   (exact, asserted below)

    Individual terms (grouped or not) carry no sign guarantee -- see
    level_excess()'s module notes; this function does not change that.

    NORMALIZATION (oracle-gap): D_global and D_local are additionally divided
    by the corresponding gap in the MODEL-FREE oracle ladder,
        n_global   = D_global   / |bp_oracle[group_at] - bp_oracle[0]|
        n_local = D_local / |bp_oracle[ell] - bp_oracle[group_at]|
    bp_k^oracle spans roughly two orders of magnitude across k (a property
    of the grammar/BP factorization, not of the model), so raw D_global vs
    D_local are NOT comparable in magnitude across groups without this --
    a larger raw D_local can simply reflect a larger available gap at that
    rung pair, not worse model performance. n_root is not defined (there is
    no rung above ell to take a gap against); D_root is returned
    un-normalized, same object as g_ell.

    n_global / n_local are NOT guaranteed to lie in [0, 1] and are not
    "fraction resolved" -- they are excess divergence expressed IN UNITS OF
    the oracle's own gap at that group. Values > 1 mean the model's excess
    divergence between those two rungs exceeds the divergence separating
    the rungs themselves; this is expected and not an error.

    Args:
        bp_ladder: {k: scalar or 1-D array over checkpoints} for k=0..ell,
            same shape convention as level_excess().
        ell: tree depth.
        oracle_ladder: {k: scalar} model-free bp_k^oracle for k=0..ell.
            REQUIRED (unlike level_excess(), where it's optional): grouping
            without the oracle correction reintroduces the non-flat-under-
            a-perfect-model problem the correction exists to fix, and
            grouping without the gap normalization reintroduces the raw-
            magnitude scale confound this function exists to avoid.
        group_at: rung index splitting fine (0..group_at) from coarse
            (group_at..ell). Must satisfy 0 < group_at < ell.

    Returns:
        dict with keys 'D_global', 'D_local', 'D_root' (raw, exact-summing
        terms) and 'n_global', 'n_local' (oracle-gap-normalized). Each value
        has the same shape as the inputs (scalar or (T,) array).
    """
    if not (0 < group_at < ell):
        raise ValueError(f"group_at must satisfy 0 < group_at < ell; got group_at={group_at}, ell={ell}")
    missing = [k for k in range(ell + 1) if k not in bp_ladder]
    if missing:
        raise ValueError(f"level_excess_grouped needs a contiguous ladder 0..{ell}; missing {missing}")
    missing_o = [k for k in range(ell + 1) if k not in oracle_ladder]
    if missing_o:
        raise ValueError(f"level_excess_grouped needs a contiguous oracle_ladder 0..{ell}; missing {missing_o}")

    bp = {k: np.atleast_1d(np.asarray(bp_ladder[k], dtype=float)) for k in range(ell + 1)}
    orc = {k: float(np.asarray(oracle_ladder[k], dtype=float)) for k in range(ell + 1)}
    g = {k: bp[k] - orc[k] for k in range(ell + 1)}

    D_global = g[0] - g[group_at]
    D_local = g[group_at] - g[ell]
    D_root = g[ell]

    resid = np.abs((D_global + D_local + D_root) - bp[0]).max()
    if resid > 1e-6 * max(1.0, np.abs(bp[0]).max()):
        raise AssertionError(f"grouped telescoping broken: max |D_global+D_local+D_root - bp_0| = {resid:g}")

    gap_global = abs(orc[group_at] - orc[0])
    gap_local = abs(orc[ell] - orc[group_at])
    if gap_global <= 0 or gap_local <= 0:
        raise ValueError(
            f"oracle gaps must be strictly positive to normalize (gap_global={gap_global:g}, "
            f"gap_local={gap_local:g}) -- check oracle_ladder is monotone in k"
        )

    return {
        'D_global': D_global, 'D_local': D_local, 'D_root': D_root,
        'n_global': D_global / gap_global, 'n_local': D_local / gap_local,
    }


class LadderEvaluator:


    def __init__(self, x0_oh_eval, t_final, alpha_bars, alphas, rho, k, vocab_size,
                 factorized_layers_list, device, batch_size,
                 reweight=False, rescale_loss=False, seed=0, compute_oracle=True,
                 group_at=2):
        for f in factorized_layers_list:
            if not (0 <= int(f) <= int(k)):
                raise ValueError(f"factorized_layers={f} invalid for tree depth ell={k}")

        torch.manual_seed(int(seed))   # generate_noisy_sequences_single has no generator arg
        self.x0_oh_eval = x0_oh_eval.to(device)
        self.x0_hard = self.x0_oh_eval.argmax(dim=-1).long()

        xt_list, t_list = [], []
        for i in range(len(self.x0_oh_eval)):
            # returns (xt, x0, t) -- x0 is echoed back unchanged, discarded here
            xt_i, _, t_i = generate_noisy_sequences_single(
                x0=self.x0_oh_eval[i], t_final=t_final, alpha_bars=alpha_bars,
                device=device, reweight=reweight,
            )
            xt_list.append(xt_i)
            t_list.append(t_i)
        self.xt = torch.stack(xt_list, dim=0)
        self.t = torch.stack(t_list, dim=0).long()

        self.alpha_bars, self.alphas = alpha_bars, alphas
        self.rho, self.k, self.vocab_size = rho, k, vocab_size
        self.factorized_layers_list = [int(f) for f in factorized_layers_list]
        self.batch_size = int(batch_size)
        self.rescale_loss = bool(rescale_loss)
        self._full_ladder = set(self.factorized_layers_list) == set(range(self.k + 1))
        self.group_at = int(group_at)
        if self._full_ladder and not (0 < self.group_at < self.k):
            raise ValueError(f"group_at must satisfy 0 < group_at < ell; got group_at={group_at}, ell={self.k}")
        self.bp_oracle = None
        if compute_oracle:
            if 0 not in self.factorized_layers_list:
                raise ValueError("compute_oracle=True needs factorized_layers=0 in the ladder")
            self.bp_oracle = self._compute_oracle_ladder()
            # bp_0^oracle = D_KL(BP_0 || BP_0) must be 0. A nonzero value here
            # means run_BP_diffusion_torch(factorized_layers=0) is not returning
            # the same tensor on two calls with identical inputs -- catch that
            # now rather than letting it leak into every Delta_d downstream.
            z = abs(self.bp_oracle[0])
            if z > 1e-6:
                raise AssertionError(
                    f"bp_0^oracle = D_KL(BP_0||BP_0) = {z:g}, expected 0 -- "
                    "BP at factorized_layers=0 is not deterministic across calls"
                )

    @torch.no_grad()
    def _compute_oracle_ladder(self):
        """bp_k^oracle for every rung, accumulated over the frozen eval set."""
        sums = {f: torch.zeros((), device=self.xt.device) for f in self.factorized_layers_list}
        n_tok = 0
        for i in range(0, len(self.xt), self.batch_size):
            sl = slice(i, min(i + self.batch_size, len(self.xt)))
            o_sums, nt = compute_oracle_ladder_batch(
                self.xt[sl], self.t[sl], self.alpha_bars, self.rho, self.k,
                self.vocab_size, self.factorized_layers_list,
            )
            for f, v in o_sums.items():
                sums[f] += v
            n_tok += nt
        return {f: (sums[f] / n_tok).item() for f in self.factorized_layers_list}

    @torch.no_grad()
    def evaluate(self, model):
        was_training = model.training
        model.eval()
        try:
            sums = {f: torch.zeros((), device=self.xt.device) for f in self.factorized_layers_list}
            ent = torch.zeros((), device=self.xt.device)
            loss_total = torch.zeros((), device=self.xt.device)
            n_tok = 0
            for i in range(0, len(self.xt), self.batch_size):
                sl = slice(i, min(i + self.batch_size, len(self.xt)))
                kl_sums, ent_sum, lt_sum, nt = compute_ladder_kl_batch(
                    model, self.xt[sl], self.t[sl], self.x0_hard[sl],
                    self.alpha_bars, self.alphas, self.rho, self.k, self.vocab_size,
                    self.factorized_layers_list, rescale_loss=self.rescale_loss,
                )
                for f, v in kl_sums.items():
                    sums[f] += v
                if ent_sum is not None:
                    ent += ent_sum
                loss_total += lt_sum
                n_tok += nt
        finally:
            if was_training:
                model.train()

        out = {f'bp_{f}': (sums[f] / n_tok).item() for f in self.factorized_layers_list}
        out['loss_total'] = (loss_total / n_tok).item()
        if 0 in self.factorized_layers_list:
            out['bp0_entropy'] = (ent / n_tok).item()
            out['residual'] = out['loss_total'] - (out['bp_0'] + out['bp0_entropy'])
        if self.bp_oracle is not None:
            # constant across checkpoints; emitted every call so a single
            # ladder_live.json row stays self-contained.
            for f in self.factorized_layers_list:
                out[f'bp_{f}_oracle'] = self.bp_oracle[f]
                out[f'g_{f}'] = out[f'bp_{f}'] - self.bp_oracle[f]
        if self._full_ladder:
            bp_row = {f: out[f'bp_{f}'] for f in range(self.k + 1)}
            R_raw = level_excess(bp_row, self.k)
            for kk in range(self.k + 1):
                out[f'R_{kk}'] = float(R_raw[kk, 0])
 
            if self.bp_oracle is not None:
                R_cor = level_excess(bp_row, self.k, oracle_ladder=self.bp_oracle)
                for kk in range(self.k + 1):
                    out[f'delta_{kk}'] = float(R_cor[kk, 0])

                # 3-term grouped view (global / local / root), oracle-gap
                # normalized. Exact re-partition of the same telescoped sum
                # above -- see level_excess_grouped()'s docstring. Additive
                # to the existing keys; nothing above is changed by this.
                grouped = level_excess_grouped(bp_row, self.k, self.bp_oracle,
                                               group_at=self.group_at)
                for key, val in grouped.items():
                    out[key] = float(np.asarray(val).reshape(-1)[0])
        return out