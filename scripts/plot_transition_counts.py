"""
plot_transition_counts.py

Pooled transition-COUNT matrix between the four sample-classification states
(memorized / valid / local / global), across consecutive checkpoints within a
given TRAINING-STEP range -- following arXiv:2605.10019 Section C.4.1 /
Figure 11's methodology:

  1. Fix N initial noise tensors ONCE; reuse the exact same tensor, unchanged,
     across every checkpoint's generation.
  2. At each checkpoint, generate a sample from EACH of those fixed noises
     using a DETERMINISTIC reverse pass (posterior mean only, no injected
     noise). generate_samples() in sequential_learning_reduced.py already
     supports this via deterministic=True -- this script's
     generate_samples_from_noise() is that same update rule, just taking the
     starting noise as an argument instead of drawing a fresh one internally,
     so "sample i at checkpoint t" and "sample i at checkpoint t+1" differ
     ONLY because the model weights changed, never because of sampler
     randomness. (The paper achieves the equivalent property with a literal
     deterministic ODE sampler, Heun's 2nd order; zeroing the ancestral
     sampler's injected-noise term is the cheaper way to get the same
     property without writing a new integrator -- see the conversation this
     script came out of for that discussion.)
  3. Classify EVERY generated sample into one of the four states, per-sample
     (not aggregated into rates -- the transition matrix needs to know which
     INDIVIDUAL sample moved between which states, unlike
     classification_stats()'s batch fractions).
  4. For every pair of CONSECUTIVE checkpoints (in step order) within the
     requested [--step-min, --step-max] range, count how many samples moved
     from state i to state j. Sum these counts over every consecutive pair
     in the range -> ONE pooled 4x4 transition-count matrix.
  5. Plot it as an annotated heatmap (rows = "from" state, columns = "to"
     state), alongside its row-normalized transition-PROBABILITY version.

Reuses sequential_learning_reduced.py's own building blocks (build_parent_array,
violation_depth_batch, safe_block_indices, build_experiment, load_epoch_weights,
get_schedule) rather than reimplementing classification -- only the
fixed-noise sampler wrapper, the per-sample classifier, and the
checkpoint-scanning-restricted-to-a-range + counting/plotting are new.

Usage::

    python plot_transition_counts.py \
        --models-root /scratch/work/chandrb1/biased-generalization \
        --subpath results_transformer_pick_0_for_training/models_restricted_continuous_0/test_model_script_12500_4096_cross_entropy_loss/8_512 \
        --step-min 1000 --step-max 60000 --n-gen 1000 --seed 0 \
        --out transition_counts.png
"""
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sequential_learning import (
    build_parent_array, violation_depth_batch, safe_block_indices,
    build_experiment, load_epoch_weights, get_schedule,
)
from modules.noise_schedules import get_alpha_beta_from_alpha_bar
from utils import setup_device

STATES = ["memorized", "valid", "local", "global"]
STATE_IDX = {s: i for i, s in enumerate(STATES)}


@torch.inference_mode()
def generate_samples_from_noise(model, x_init, t_final, sched, device,
                                temperature=1.0, clamp=False, deterministic=True):
    """
    Same reverse-process update as generate_samples() in
    sequential_learning_reduced.py, but takes the STARTING NOISE x_init as an
    argument instead of drawing a fresh torch.randn(...) internally -- this is
    what lets the SAME initial noise be reused across every checkpoint (see
    module docstring for why that matters). deterministic=True (opposite of
    the original function's default) drops the injected-noise term at every
    step, so the ONLY thing that differs between two checkpoints' generations
    from the same x_init is the model weights.
    """
    model.eval()
    ab = sched["alpha_bars"]
    alpha, beta = get_alpha_beta_from_alpha_bar(ab)

    x = x_init.clone()
    n_gen = x.shape[0]
    for step in range(int(t_final), 0, -1):
        t = torch.full((n_gen,), step, device=device, dtype=torch.long)
        logits = model(x, t)
        if clamp:
            vocab_size = logits.shape[-1]
            x0_hat = torch.nn.functional.one_hot(
                logits.argmax(dim=-1), num_classes=vocab_size).float()
        else:
            x0_hat = torch.softmax(logits / temperature, dim=-1)

        a, ab_t, b = alpha[step - 1], ab[step - 1], beta[step - 1]
        ab_prev = ab[step - 2] if step > 1 else torch.tensor(1.0, device=device)
        mean = (ab_prev.sqrt() * b / (1 - ab_t)) * x0_hat \
             + (a.sqrt() * (1 - ab_prev) / (1 - ab_t)) * x
        if step > 1 and not deterministic:
            var = b * (1 - ab_prev) / (1 - ab_t)
            x = mean + var.sqrt() * torch.randn_like(x)
        else:
            x = mean
    return x.argmax(dim=-1).detach().cpu().numpy().astype(np.int64)


def classify_per_sample(gen, parent_arr, ell, train_set, local_max_depth=2):
    """
    Per-SAMPLE state labels (a (n,) array of strings), not aggregated
    fractions -- same precedence rule as classification_stats() in
    sequential_learning_reduced.py (memorized checked first, since a
    memorized sample is by construction also a valid parse and must not
    double-count).
    """
    n = len(gen)
    mem_flags = np.array([tuple(s) in train_set for s in gen])
    depths = violation_depth_batch(gen, parent_arr, ell)
    states = np.full(n, "", dtype=object)
    states[mem_flags] = "memorized"
    novel = ~mem_flags
    states[novel & (depths == -1)] = "valid"
    states[novel & (depths >= 1) & (depths <= local_max_depth)] = "local"
    states[novel & (depths > local_max_depth)] = "global"
    assert (states != "").all(), "unreachable: every sample must land in exactly one state"
    return states


def find_checkpoints_in_range(model_dir, ckpt_tag, step_min, step_max):
    """Same glob+parse logic as sequential_learning_reduced.py's main(),
    restricted to a [step_min, step_max] inclusive range, sorted ascending."""
    model_dir = Path(model_dir)
    if ckpt_tag == "epoch":
        ckpts = list(model_dir.glob("test_model_script_epoch_*.pt"))
    else:
        ckpts = list(model_dir.glob("test_model_script_step_*.pt"))

    def _parse(c):
        stem = c.stem
        if "_epoch_" in stem:
            tag, num_str = "epoch", stem.split("_epoch_")[1]
        elif "_step_" in stem:
            tag, num_str = "step", stem.split("_step_")[1]
        else:
            return None
        if not num_str.isdigit():
            return None
        return tag, int(num_str), c

    parsed = [r for r in (_parse(c) for c in ckpts) if r is not None]
    if not parsed:
        raise RuntimeError(f"No checkpoints found in {model_dir}")
    tags_present = {t for t, _, _ in parsed}
    if len(tags_present) > 1:
        raise RuntimeError(f"Mixed epoch/step checkpoints in {model_dir}: {tags_present}")
    by_num = {num: path for _, num, path in parsed}
    all_steps = sorted(by_num.keys())
    kept = [s for s in all_steps if step_min <= s <= step_max]
    if len(kept) < 2:
        raise RuntimeError(
            f"Only {len(kept)} checkpoint(s) found in step range "
            f"[{step_min}, {step_max}] under {model_dir} (need >= 2 to form any "
            f"transition). Checkpoints actually available: {all_steps}")
    print(f"{len(all_steps)} checkpoints total under {model_dir}; "
          f"{len(kept)} fall inside [{step_min}, {step_max}]: {kept}")
    return [(s, by_num[s]) for s in kept]


def main(args):
    device = setup_device()

    model_dir = Path(args.models_root) / args.subpath
    exp = build_experiment(model_dir, device, do_compile=(not args.no_compile))
    reduced_length = int(exp["params"]["reduced_length"])
    tree_depth = int(exp["k"])

    parent_arr = build_parent_array(exp["rho"])
    train_idx, _, _ = safe_block_indices(
        num_seqs=exp["sequences"].shape[0], reduced_length=reduced_length,
        train_block_i=int(exp["params"].get("pick_i_for_training", 0) or 0),
        test_block_i=args.test_block_i, device=exp["sequences"].device)
    train_arr = exp["sequences"][train_idx].detach().cpu().numpy().astype(np.int64)
    train_set = set(map(tuple, train_arr))

    t_final = int(exp["params"]["t_final"])
    schedule_cache = {}
    sched = get_schedule(schedule_cache, t_test=t_final, s=float(exp["params"]["s"]), device=device)

    checkpoints = find_checkpoints_in_range(model_dir, args.ckpt_tag, args.step_min, args.step_max)

    # Fix the initial noise ONCE -- reused, byte-for-byte, across every
    # checkpoint's generation below. This is the crux of the whole analysis
    # (see module docstring): without it, differences between checkpoints
    # would be confounded by sampler randomness rather than isolating the
    # model's own change.
    torch.manual_seed(int(args.seed))
    x_init = torch.randn(args.n_gen, int(exp["seq_len"]), int(exp["vocab_size"]), device=device)

    state_grid = []  # one (n_gen,) state array per kept checkpoint, in step order
    for step, ckpt_path in checkpoints:
        print(f"--- checkpoint step={step}: {ckpt_path} ---")
        load_epoch_weights(exp, ckpt_path, device)
        gen = generate_samples_from_noise(
            exp["model"], x_init, t_final, sched, device,
            deterministic=not args.stochastic)
        states = classify_per_sample(gen, parent_arr, tree_depth, train_set,
                                     local_max_depth=args.local_max_depth)
        counts = {s: int((states == s).sum()) for s in STATES}
        print(f"  state counts: {counts}")
        state_grid.append(states)

    # Pooled transition COUNT matrix over every consecutive checkpoint pair
    # in the kept range.
    T_mat = np.zeros((4, 4), dtype=np.int64)
    for i in range(len(state_grid) - 1):
        s_from, s_to = state_grid[i], state_grid[i + 1]
        for a_state in STATES:
            mask_from = (s_from == a_state)
            if not mask_from.any():
                continue
            for b_state in STATES:
                T_mat[STATE_IDX[a_state], STATE_IDX[b_state]] += int(
                    (mask_from & (s_to == b_state)).sum())

    print()
    print(f"Pooled transition COUNT matrix over {len(state_grid) - 1} consecutive "
          f"checkpoint pair(s), steps {checkpoints[0][0]}..{checkpoints[-1][0]}:")
    for i, s in enumerate(STATES):
        print(f"  from {s:>10s}: {dict(zip(STATES, T_mat[i].tolist()))}")

    row_sums = T_mat.sum(axis=1, keepdims=True)
    P_mat = np.divide(T_mat, row_sums, out=np.zeros_like(T_mat, dtype=float),
                      where=row_sums > 0)

    fig, (ax_c, ax_p) = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, mat, title, fmt in (
        (ax_c, T_mat, "Transition counts", "d"),
        (ax_p, P_mat, "Transition probabilities (row-normalized)", ".2f"),
    ):
        im = ax.imshow(mat, cmap="viridis", vmin=0)
        ax.set_xticks(range(4)); ax.set_xticklabels(STATES, rotation=30, ha="right")
        ax.set_yticks(range(4)); ax.set_yticklabels(STATES)
        ax.set_xlabel("to state"); ax.set_ylabel("from state")
        ax.set_title(title, fontsize=11)
        for i in range(4):
            for j in range(4):
                val = mat[i, j]
                text = f"{val:{fmt}}"
                ax.text(j, i, text, ha="center", va="center",
                       color="white" if val < mat.max() * 0.6 else "black", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Pooled state-transition matrix, steps {checkpoints[0][0]}"
        f"–{checkpoints[-1][0]} ({len(checkpoints)} checkpoints, "
        f"{len(state_grid) - 1} consecutive pairs, n_gen={args.n_gen}, "
        f"{'stochastic' if args.stochastic else 'deterministic'} sampler)",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--models-root", type=str, required=True)
    p.add_argument("--subpath", type=str, required=True)
    p.add_argument("--ckpt-tag", type=str, choices=["epoch", "step"], default="step")
    p.add_argument("--step-min", type=int, required=True,
                   help="Lowest checkpoint step/epoch (inclusive) in the pooled range.")
    p.add_argument("--step-max", type=int, required=True,
                   help="Highest checkpoint step/epoch (inclusive) in the pooled range.")
    p.add_argument("--n-gen", type=int, default=1000,
                   help="Number of fixed initial-noise samples (N), reused across every "
                        "checkpoint. arXiv:2605.10019 used 2048; 1000 matches this "
                        "project's existing --n-gen convention elsewhere.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the ONE fixed batch of initial noise, reused unchanged "
                        "across every checkpoint in the range.")
    p.add_argument("--stochastic", action="store_true",
                   help="Use the ORIGINAL stochastic ancestral sampler (injected noise "
                        "every step) instead of the deterministic posterior-mean-only "
                        "variant. Default is deterministic -- see module docstring for "
                        "why that's the right choice for this analysis.")
    p.add_argument("--test-block-i", type=int, default=0)
    p.add_argument("--local-max-depth", type=int, default=2)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--out", type=str, default="transition_counts.png")
    main(p.parse_args())