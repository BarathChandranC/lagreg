"""
train_and_eval.py

Single script combining train_transformer.py and sequential_learning_reduced.py:
trains the denoiser and evaluates it IN-PROCESS at the same log-spaced steps
that used to trigger checkpointing, so no per-checkpoint weights need to be
written to disk.

At every eval step it writes three JSON files under the run's save_path,
FILENAME-tagged by reg_lambda (save_path itself is shared across reg_lambda
values at the same N/seed -- see reg_tag in main()):
  - train_eval_log_reg{tag}.json: one combined row per eval step -- both
    emp_*/pop_* ladders (prefixed), pop_loss (the held-out block's native
    loss_total from LadderEvaluator -- full-batch, marginal-t evaluation, in
    ordinary per-token loss units, distinct from the raw g_0/KL(BP_0) scale),
    lag_emp/lag_pop, reg diagnostics, and sample classification (memorized /
    valid / local / global) from freshly generated samples.
    NOTE: this no longer includes Omega_k = pop - emp. Omega_n_global alone
    is a poor memorization signal -- a real generalization gap and
    train-block-specific exploitation both surface as the same negative
    number, with no way to tell them apart from Omega alone (see the
    lag_pop divergence observed empirically). pop_loss is used instead:
    a directly interpretable single-block quantity whose absolute level and
    trend mean something without needing a paired emp value.
  - ladder_emp_seed{seed}_N{L}_reg{tag}.json / ladder_pop_seed{seed}_N{L}_reg{tag}.json:
    the per-block g_k/R_k/delta_k/D_global/D_local/D_root/n_global/n_local
    ladder, UNPREFIXED, one file per block. NOTE: the _reg{tag} suffix means
    these do NOT match combine_loss_runs.py's TAGGED_RE
    (ladder_(emp|pop)_seed(\\d+)_N(\\d+)\\.json exactly, no trailing suffix)
    -- that script's discover() will silently skip these unless its regex is
    widened to accept the suffix, or you point it at a single-lambda subset.

and optionally adds a lag-closing regularizer to the training loss (see below).

Usage::

    python train_and_eval.py --seed 0 --reduced-length 4096 --batch-size 512 \
        --use-cross-entropy-loss --n-evals 25 --eval-block-i 10 --n-gen 1000 \
        --reg-lambda 0.1 --reg-delta-margin 0.08 --reg-delta-huber 1.0

WHY THE REGULARIZER TARGETS THE LAG n_global - sg[n_local], NOT n_global ALONE:
a raw `lambda * n_global` penalty has no lower bound: D_global = kl0 - klg has
no floor because KL(bp_g || p_theta) has no ceiling, so gradient descent can
drive n_global to -infinity by making the model diverge from the group_at
soft target, with zero counter-pressure from the main (hard-label) loss. This
was observed directly (n_global fell from ~1.3 to -18.6 over ~4500 steps)
while sample-level `valid` stayed flat at ~0.18-0.24 the whole time, and the
emp/pop gap (Omega) blew up from ~0.07 to +3.0 in the same window -- textbook
proxy-gaming: the surrogate collapses on train-block-specific structure while
buying nothing on the actual quantity of interest.

Penalizing the LAG instead -- lag = n_global - sg[n_local], margin-clipped and
Huber'd -- fixes this structurally, not just numerically:
  - it is bounded below by construction: once n_global <= n_local - delta,
    relu(...) = 0 and the gradient vanishes, so there is no free direction
    left to exploit past the target;
  - it is self-scheduling: early in training n_global ~= n_local (no lag), so
    the term is inert without needing --reg-warmup-steps as a hand-tuned gate;
  - stop-gradient on n_local means the optimizer cannot cheat by dragging
    local back down to close the gap -- local resolving before global is the
    predicted healthy trajectory (risk additive over levels, validity
    conjunctive) and must stay untouched by this term.

CAVEATS STILL WORTH MONITORING:
  (1) n_local is computed from the SAME forward pass / shared backbone as
      n_global, so stop-gradient blocks the explicit chain-rule path through
      n_local but not contamination via shared representation capacity.
      Compare n_local under reg_lambda>0 against a matched reg_lambda=0
      control at the same steps; if the reg run's n_local drifts outside the
      lambda=0 run's noise band, that is shared-backbone drift, not something
      this loss term can fix.
  (2) This closes lag on the TRAINING block only. Track lag_pop = pop_n_global
      - pop_n_local against lag_emp the same way Omega was used before: if
      lag_emp closes while lag_pop stays wide, the gaming target has just
      moved from an absolute value to a relative one.
  (3) delta_margin should be set from the pre-regularizer Omega_n_global noise
      floor (observed ~0.06-0.08 here), not guessed. delta_huber should be set
      from the per-STEP (not per-eval) noise of the lag itself under a
      reg_lambda=0 control run, since that is the noise scale the gradient
      actually sees -- the eval-time ladder is smoothed over a full block and
      will understate it.
"""
import json
import logging
import math
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.noise_schedules import alpha_bars_schedule, get_alpha_beta_from_alpha_bar
from modules.ladder_diagnostics import LadderEvaluator, compute_bp_soft_targets
from modules.losses import (compute_mse_loss, compute_cross_entropy_loss,
                            get_loss_coefficients_continuous_new)
from modules.training import DiffusionDataset
from utils import load_data, create_model, setup_device, create_save_path

# Classification helpers are reused as-is; that module guards its argparse
# behind __main__, so importing it is side-effect free apart from its own
# imports and matplotlib rcParams.
from sequential_learning import (build_parent_array, generate_samples,
                                          classification_stats, safe_block_indices)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Regularizer
# --------------------------------------------------------------------------

def lag_reg_from_logits(logits, xt, t, alpha_bars, rho, k, vocab_size,
                        group_at, ell, gaps, oracle_offsets,
                        delta_margin, delta_huber):
    """
    Differentiable lag-closing regularizer for one training batch.

        n_global = [KL(BP_0||model) - KL(BP_g||model) - oracle_offsets["global"]] / gaps["global"]
        n_local  = [KL(BP_g||model) - KL(BP_ell||model) - oracle_offsets["local"]] / gaps["local"]
        lag      = n_global - stop_grad(n_local)
        reg_loss = Huber_{delta_huber}( relu(lag - delta_margin) )

    `logits` is the SAME forward pass the main loss uses -- no second model
    call. n_local is detached before the lag is formed: the term must never
    supply gradient that could drag n_local back up to close the gap from the
    wrong side, since local resolving first is the predicted healthy
    trajectory, not the pathology being corrected.

    Costs three extra BP-oracle calls (rungs 0, group_at, ell) per invocation
    -- one more than the previous n_global-only version, since n_local now
    needs the ell-rung target too. See --reg-every for amortising this.

    Returns (reg_loss, n_global.detach(), n_local.detach(), lag.detach()) --
    the detached scalars are for logging only, not further use in the graph.
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)          # (B, L, Q)
    with torch.no_grad():
        bp0 = compute_bp_soft_targets(xt, t, alpha_bars, rho, k, vocab_size,
                                      factorized_layers=0).float()
        bpg = compute_bp_soft_targets(xt, t, alpha_bars, rho, k, vocab_size,
                                      factorized_layers=group_at).float()
        bpe = compute_bp_soft_targets(xt, t, alpha_bars, rho, k, vocab_size,
                                      factorized_layers=ell).float()
        ent0 = torch.xlogy(bp0, bp0).sum(-1)        # = -H(BP_0), constant wrt theta
        entg = torch.xlogy(bpg, bpg).sum(-1)
        ente = torch.xlogy(bpe, bpe).sum(-1)
    kl0 = (ent0 - (bp0 * log_probs).sum(-1)).mean()
    klg = (entg - (bpg * log_probs).sum(-1)).mean()
    kle = (ente - (bpe * log_probs).sum(-1)).mean()

    n_global = (kl0 - klg - oracle_offsets["global"]) / gaps["global"]
    n_local = (klg - kle - oracle_offsets["local"]) / gaps["local"]

    lag = n_global - n_local.detach()
    residual = F.relu(lag - delta_margin)
    reg_loss = F.huber_loss(residual, torch.zeros_like(residual), delta=delta_huber)

    return reg_loss, n_global.detach(), n_local.detach(), lag.detach()


# --------------------------------------------------------------------------
# KL_FACT-style columnar accumulator (matches sequential_learning's own
# output shape so existing downstream plotting code can load it unmodified)
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Eval at one step
# --------------------------------------------------------------------------

def run_eval(step, model, ev_emp, ev_pop, f_list, group_at, gaps, gen_cfg):
    """One evaluation row: emp ladder, pop ladder, Omega, grouped terms, samples."""
    row = {"step": int(step)}

    for prefix, ev in (("emp", ev_emp), ("pop", ev_pop)):
        out = ev.evaluate(model)
        for key, val in out.items():
            row[f"{prefix}_{key}"] = val

        # grouped terms, computed here rather than relying on a particular
        # level_excess_grouped() being present in ladder_diagnostics
        g = {f: out[f"g_{f}"] for f in f_list}
        ell = max(f_list)
        D_global = g[0] - g[group_at]
        D_local = g[group_at] - g[ell]
        row[f"{prefix}_D_global"] = D_global
        row[f"{prefix}_D_local"] = D_local
        row[f"{prefix}_D_root"] = g[ell]
        row[f"{prefix}_n_global"] = D_global / gaps["global"]
        row[f"{prefix}_n_local"] = D_local / gaps["local"]

    # pop_loss: the held-out block's native loss_total from LadderEvaluator
    # (full-batch, marginal-t evaluation) -- the population-side counterpart
    # to a normal training loss, in ordinary per-token units. Distinct from
    # g_0 (raw KL(BP_0||model) against the coarsest oracle target, which sits
    # on a different, much larger scale). Used in place of the emp/pop Omega
    # gap: Omega_n_global on its own is a poor memorization signal -- a real
    # generalization gap and train-block exploitation both show up as the
    # same negative number, with no way to tell them apart from Omega alone
    # (see the lag_pop divergence observed earlier), whereas pop_loss is a
    # directly interpretable single-block quantity: absolute level and trend
    # are meaningful without needing a paired emp value to compare against.
    row["pop_loss"] = row["pop_loss_total"]

    # lag at both emp and pop -- if lag_emp closes while lag_pop stays wide,
    # the regularizer has moved the gaming target from an absolute value to
    # a relative one.
    row["lag_emp"] = row["emp_n_global"] - row["emp_n_local"]
    row["lag_pop"] = row["pop_n_global"] - row["pop_n_local"]

    if gen_cfg is not None:
        gen = generate_samples(
            model, gen_cfg["n_gen"], gen_cfg["seq_len"], gen_cfg["vocab_size"],
            gen_cfg["t_final"], gen_cfg["sched"], gen_cfg["device"],
        )
        cls, depth_hist = classification_stats(
            gen, gen_cfg["parent_arr"], gen_cfg["ell"],
            gen_cfg["train_set"], gen_cfg["train_arr"],
            local_max_depth=gen_cfg["local_max_depth"],
        )
        row.update(cls)
        row["depth_hist"] = {str(k): v for k, v in depth_hist.items()}

    return row


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = setup_device()
    torch.set_float32_matmul_precision("high")

    if args.use_compute_mse_loss == args.use_cross_entropy_loss:
        raise ValueError("Specify exactly one of --use-compute-mse-loss / --use-cross-entropy-loss.")
    loss_name = "mse_loss" if args.use_compute_mse_loss else "cross_entropy_loss"

    # ---------------- data ----------------
    vocab_size, sequences, seq_len, k, rho, _ = load_data(args.data_path, device)
    L = args.reduced_length
    tr_lo = args.pick_i_for_training * L
    tr_hi = tr_lo + L
    if tr_hi > len(sequences):
        raise ValueError(f"train block [{tr_lo}:{tr_hi}] out of bounds ({len(sequences)} seqs)")
    train_seqs = sequences[tr_lo:tr_hi]

    ev_lo = args.eval_block_i * L
    ev_hi = ev_lo + L
    if ev_hi > len(sequences):
        raise ValueError(
            f"eval block [{ev_lo}:{ev_hi}] out of bounds ({len(sequences)} seqs). "
            f"--eval-block-i {args.eval_block_i} needs >= {ev_hi} sequences at "
            f"reduced_length={L}."
        )
    if args.eval_block_i == args.pick_i_for_training:
        raise ValueError("--eval-block-i must differ from --pick-i-for-training, "
                         "otherwise 'pop' is just 'emp' again.")
    eval_seqs = sequences[ev_lo:ev_hi]

    x0s = torch.nn.functional.one_hot(train_seqs.long(), vocab_size).float()
    x0s_eval = torch.nn.functional.one_hot(eval_seqs.long(), vocab_size).float()
    logger.info("train block %d: %s | eval block %d: %s",
                args.pick_i_for_training, tuple(x0s.shape), args.eval_block_i, tuple(x0s_eval.shape))

    # ---------------- model / schedule ----------------
    model_args = {"hidden_size_encoder": args.hidden_size_encoder, "nhead": args.nhead,
                  "num_layers": args.num_layers, "model_channels": args.model_channels}
    model, _, actual_hidden = create_model(model_args, vocab_size, seq_len, device)
    # NOT compiled: generate_samples() and the eval path call the model with
    # many different batch shapes, which forces repeated recompilation and can
    # cost more than the compiled speedup on the training loop itself.
    logger.info("model params: %d", sum(p.numel() for p in model.parameters()))

    alpha_bars = alpha_bars_schedule(args.t_final, device=device, s=args.s, schedule="linear")
    alphas, _ = get_alpha_beta_from_alpha_bar(alpha_bars)
    sched = {"alpha_bars": alpha_bars, "alphas": alphas}

    f_list = ([int(x) for x in args.bp_factorized_layers.split(",")]
              if args.bp_factorized_layers else list(range(k + 1)))
    if 0 not in f_list or max(f_list) != k:
        raise ValueError(f"f_list must be the full contiguous ladder 0..{k}; got {f_list}")
    if not (0 < args.group_at < k):
        raise ValueError(f"--group-at must satisfy 0 < group_at < {k}")

    # ---------------- evaluators (frozen (x_t,t), oracle computed once) ------
    common = dict(t_final=args.t_final, alpha_bars=alpha_bars, alphas=alphas, rho=rho,
                  k=k, vocab_size=vocab_size, factorized_layers_list=f_list,
                  device=device, batch_size=args.ladder_batch_size,
                  reweight=args.reweighting, rescale_loss=args.rescale_loss,
                  seed=args.seed, compute_oracle=True)
    logger.info("building emp evaluator (train block)...")
    ev_emp = LadderEvaluator(x0_oh_eval=x0s[:args.n_ladder_eval], **common)
    logger.info("building pop evaluator (held-out block %d)...", args.eval_block_i)
    ev_pop = LadderEvaluator(x0_oh_eval=x0s_eval[:args.n_ladder_eval], **common)

    ell = k
    orc = ev_emp.bp_oracle
    gaps = {"global": abs(orc[args.group_at] - orc[0]),
            "local": abs(orc[ell] - orc[args.group_at])}
    oracle_offsets = {"global": orc[0] - orc[args.group_at],
                      "local": orc[args.group_at] - orc[ell]}
    logger.info("oracle ladder: %s", {f: round(v, 6) for f, v in orc.items()})
    logger.info("gaps: global=%.6g local=%.6g", gaps["global"], gaps["local"])

    # ---------------- sample-classification setup ----------------
    gen_cfg = None
    if args.n_gen > 0:
        train_arr = train_seqs.detach().cpu().numpy().astype(np.int64)
        gen_cfg = dict(
            n_gen=args.n_gen, seq_len=int(seq_len), vocab_size=int(vocab_size),
            t_final=args.t_final, sched=sched, device=device,
            parent_arr=build_parent_array(rho), ell=int(k),
            train_set=set(map(tuple, train_arr)), train_arr=train_arr,
            local_max_depth=args.local_max_depth,
        )
        logger.info("classification on: %d samples/eval, local/global split at depth %d",
                    args.n_gen, args.local_max_depth)

    # ---------------- output dir ----------------
    save_path = create_save_path(
        args.data_path, args.n_iter, args.reduced_length, loss_name, args.num_layers,
        actual_hidden, False, args.rescale_loss, args.seed,
        model_type=args.model_name + f"_pick_{args.pick_i_for_training}_for_training",
        add_flag=args.add_flag, results_dir=args.results_dir,
    )
    # reg_tag disambiguates FILENAMES within a shared save_path across
    # reg_lambda values at the same (N, seed) -- save_path itself is no
    # longer varied by reg_lambda (that's handled at the sbatch level via a
    # constant --add-flag, not a lambda-dependent one). NOTE: this means
    # ladder_emp_*/ladder_pop_* no longer match combine_loss_runs.py's
    # TAGGED_RE (ladder_(emp|pop)_seed(\d+)_N(\d+)\.json exactly, no
    # trailing suffix) -- that script's discover() will silently skip these
    # unless its regex is widened to accept the _reg{tag} suffix.
    reg_tag = str(args.reg_lambda).replace(".", "p")

    with open(os.path.join(save_path, f"full_params_reg{reg_tag}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    log_path = os.path.join(save_path, f"train_eval_log_reg{reg_tag}.json")
    # Per-block ladder diagnostics, written separately in the same
    # step/g_k/R_k/delta_k/D_global/D_local/D_root/n_global/n_local shape
    # sequential_learning_reduced.py writes for a single model/block -- so an
    # emp or pop file from here can be loaded with the same downstream code
    # as a standalone sequential-learning run, no prefix-stripping needed.
    ladder_emp_path = os.path.join(save_path, f"ladder_emp_seed{args.seed}_N{L}_reg{reg_tag}.json")
    ladder_pop_path = os.path.join(save_path, f"ladder_pop_seed{args.seed}_N{L}_reg{reg_tag}.json")

    # ---------------- eval steps (same log-spacing checkpointing used) -------
    steps_per_epoch = -(-len(x0s) // args.batch_size)
    total_steps = args.n_iter * steps_per_epoch
    eval_steps = set(torch.logspace(2, math.log10(total_steps), args.n_evals).long().tolist())
    logger.info("total_steps=%d (spe=%d), %d eval points: %s",
                total_steps, steps_per_epoch, len(eval_steps), sorted(eval_steps))

    # ---------------- training ----------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    WARMUP = 50
    if args.lr_schedule == "default":
        def lr_lambda(s):
            if s <= WARMUP:
                return 1.0 - (s / WARMUP) * 0.5
            T = total_steps - WARMUP
            return 0.1 + 0.4 * 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / T))
    else:
        def lr_lambda(s):
            return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loader = torch.utils.data.DataLoader(
        DiffusionDataset(x0s, args.t_final, alpha_bars, reweight=args.reweighting, device="cpu"),
        batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=max(1, os.cpu_count() // 4), pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
    )

    rows, rows_emp, rows_pop, gstep = [], [], [], 0
    for epoch in range(args.n_iter):
        for batch in loader:
            gstep += 1
            xt, x0b, tb = (b.to(device, non_blocking=True) for b in batch)

            # One forward pass shared by the main loss and the regularizer.
            # Autocast is left OFF here: the regularizer's KL against BP
            # targets is numerically delicate (log of a near-one-hot softmax),
            # and BP itself runs in double internally.
            logits = model(xt, tb)

            coeffs = (get_loss_coefficients_continuous_new(alpha_bars, alphas, tb)
                      if args.rescale_loss else None)
            if args.use_compute_mse_loss:
                loss = compute_mse_loss(x0b, logits, rescale_coeffs=coeffs).mean()
            else:
                loss = compute_cross_entropy_loss(
                    torch.argmax(x0b, dim=-1).long(), logits, rescale_coeffs=coeffs).mean()

            reg_n_glob = reg_n_loc = reg_lag = reg_loss_val = None
            use_reg = (args.reg_lambda > 0
                       and gstep > args.reg_warmup_steps
                       and gstep % args.reg_every == 0)
            if use_reg:
                reg_loss, n_glob_d, n_loc_d, lag_d = lag_reg_from_logits(
                    logits, xt, tb, alpha_bars, rho, k, vocab_size,
                    args.group_at, ell, gaps, oracle_offsets,
                    args.reg_delta_margin, args.reg_delta_huber)
                loss = loss + args.reg_lambda * reg_loss
                reg_n_glob, reg_n_loc, reg_lag = (float(n_glob_d), float(n_loc_d), float(lag_d))
                reg_loss_val = float(reg_loss.detach())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if gstep in eval_steps:
                row = run_eval(gstep, model, ev_emp, ev_pop, f_list,
                               args.group_at, gaps, gen_cfg)
                row["train_loss_batch"] = float(loss.detach())
                row["reg_n_global_batch"] = reg_n_glob
                row["reg_n_local_batch"] = reg_n_loc
                row["reg_lag_batch"] = reg_lag
                row["reg_loss_batch"] = reg_loss_val
                row["lr"] = optimizer.param_groups[0]["lr"]
                rows.append(row)
                with open(log_path, "w") as f:
                    json.dump(rows, f, indent=2)

                # Split the combined row into unprefixed emp/pop diagnostics,
                # matching sequential_learning_reduced.py's single-block
                # output shape. omega_*/lag_*/reg_*/classification fields are
                # cross-block by construction and stay in train_eval_log.json
                # only, not duplicated into either per-block file.
                prefix_len = len("emp_")
                rows_emp.append({"step": row["step"],
                                 **{key[prefix_len:]: val for key, val in row.items()
                                    if key.startswith("emp_")}})
                rows_pop.append({"step": row["step"],
                                 **{key[prefix_len:]: val for key, val in row.items()
                                    if key.startswith("pop_")}})
                with open(ladder_emp_path, "w") as f:
                    json.dump(rows_emp, f, indent=2)
                with open(ladder_pop_path, "w") as f:
                    json.dump(rows_pop, f, indent=2)

                logger.info(
                    "step %d | emp n_glob %.4f n_loc %.4f lag %.4f | pop n_glob %.4f lag %.4f | "
                    "pop_loss %.4f%s",
                    gstep, row["emp_n_global"], row["emp_n_local"], row["lag_emp"],
                    row["pop_n_global"], row["lag_pop"],
                    row["pop_loss"],
                    (f" | valid {row['valid']:.3f} local {row['local']:.3f} "
                     f"global {row['global']:.3f} memorized {row['memorized']:.3f} "
                     f"distinct {int(row['n_distinct'])}"
                     if gen_cfg else ""),
                )

    if args.save_final_model:
        torch.save({"model_state_dict": model.state_dict(), "global_step": gstep},
                   os.path.join(save_path, f"final_step_{gstep}_reg{reg_tag}.pt"))
        logger.info("saved final weights only (per-step checkpoints intentionally not written)")
    logger.info("done -> %s", log_path)


if __name__ == "__main__":
    p = ArgumentParser(description="Train + in-process ladder/sample evaluation.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-name", type=str, default="transformer")
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--add-flag", type=str, default=None)

    p.add_argument("--data-path", type=str,
                   default="../data/labeled_data_restrictedfixed_6_4_1.0_4_0.npy")
    p.add_argument("--reduced-length", type=int, default=4096)
    p.add_argument("--pick-i-for-training", type=int, default=0)
    p.add_argument("--eval-block-i", type=int, default=10,
                   help="Held-out block for the 'pop' ladder. Needs the dataset to hold "
                        "at least (eval_block_i+1)*reduced_length sequences.")
    p.add_argument("--batch-size", type=int, default=512)

    p.add_argument("--hidden-size-encoder", type=int, default=512)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--model-channels", type=int, default=32)

    p.add_argument("--t-final", type=int, default=500)
    p.add_argument("--s", type=float, default=1e-4)

    p.add_argument("--n-iter", type=int, default=10000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-schedule", type=str, default="default", choices=["default", "constant"])
    p.add_argument("--use-compute-mse-loss", action="store_true")
    p.add_argument("--use-cross-entropy-loss", action="store_true")
    p.add_argument("--reweighting", action="store_true")
    p.add_argument("--rescale-loss", action="store_true")
    p.add_argument("--save-final-model", action="store_true",
                   help="Save ONLY the final weights. Per-step checkpoints are never "
                        "written -- that is the point of this script.")

    # evaluation
    p.add_argument("--n-evals", type=int, default=45,
                   help="Number of log-spaced eval points (same spacing the old "
                        "--checkpointing-period-steps used).")
    p.add_argument("--bp-factorized-layers", type=str, default=None)
    p.add_argument("--ladder-batch-size", type=int, default=512)
    p.add_argument("--n-ladder-eval", type=int, default=1024,
                   help="Sequences per block for the ladder. NOTE: a FIXED value here "
                        "puts an N-independent Monte Carlo floor on bp_k across an "
                        "N-sweep; set it >= your largest reduced_length to evaluate "
                        "on the full block instead.")
    p.add_argument("--group-at", type=int, default=2,
                   help="Rung splitting global (0..group_at) from local (group_at..ell).")
    p.add_argument("--n-gen", type=int, default=1000,
                   help="Unconditional samples per eval for classification. 0 disables.")
    p.add_argument("--local-max-depth", type=int, default=2)

    # regularizer
    p.add_argument("--reg-lambda", type=float, default=0.0,
                   help="Coefficient on the lag Huber loss added to the training loss. "
                        "0 disables.")
    p.add_argument("--reg-warmup-steps", type=int, default=500,
                   help="Do not apply the regularizer before this step. Early training "
                        "is dominated by fitting the unconditional marginal, where "
                        "n_global and n_local haven't yet diverged and the term is "
                        "inert anyway -- this mainly guards against the extra BP-oracle "
                        "cost before it's needed.")
    p.add_argument("--reg-every", type=int, default=1,
                   help="Apply the regularizer every k steps, amortising its three extra "
                        "BP calls (rungs 0, group_at, ell). Prefer every-step with small "
                        "lambda over rare-and-large.")
    p.add_argument("--reg-delta-margin", type=float, default=0.08,
                   help="Dead-zone on the lag (n_global - n_local): no penalty below "
                        "this. Set from the pre-regularizer Omega_n_global noise floor "
                        "(observed ~0.06-0.08 in a lambda=0 control), not guessed.")
    p.add_argument("--reg-delta-huber", type=float, default=1.0,
                   help="Huber transition point on relu(lag - delta_margin): quadratic "
                        "(proportional gradient) below this, linear-capped above it. "
                        "Calibrate from the per-STEP (not per-eval) std of the lag under "
                        "a reg_lambda=0 control run -- the eval-time ladder is smoothed "
                        "over a full block and will understate the noise the gradient "
                        "actually sees each step. The default of 1.0 happens to match "
                        "the O(1) scale n_global/n_local sit at in the healthy regime, "
                        "but that's coincidence, not derivation -- measure it.")

    main(p.parse_args())