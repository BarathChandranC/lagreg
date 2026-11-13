"""
plot_transition_probability_matrix.py

The 4x4 state-transition probability matrix from "The Two Clocks"
(arXiv:2605.10019), Section C.4.1 / Figure 11 -- adapted to THIS project's
own gen_samples_step_*.npz dumps and its own established
memorized/valid/local/global classifier (classification_stats), rather
than the paper's own four states (their states are invalid-quantization /
invalid-rule-violating / valid-novel / valid-memorized; this project
already has an equally-exclusive 4-way split -- memorized / valid
(generalized) / local (shallow break) / global (hallucinated) -- via
sequential_learning.classification_stats, so that's what's reused here
rather than reimplementing the paper's own state definitions from
scratch, which would need details of their quantization-ambiguity check
this project doesn't have).

WHAT THE PAPER DOES (quoted from Section C.4.1):
  "At each checkpoint, each generated sample is assigned to one of four
  mutually exclusive states... From consecutive checkpoint pairs (t, t+1)
  we compute a 4x4 transition count tensor T(t), where T[t, i, j] is the
  number of samples in state i at step t that move to state j at step
  t+1. Row-normalizing gives the empirical transition probability
  matrix." They then aggregate counts by SUMMING across all consecutive
  pairs within a training-phase window (pre-rule-learning/rule-learning/
  memorization), and row-normalize ONCE at the end -- not one matrix per
  pair, averaged.

WHAT THIS SCRIPT DOES, exactly mirroring that:
  1. discover_gen_sample_files(): gen_samples_step_<N>.npz files under
     --gen-samples-dir with step in [--step-min, --step-max] (default
     2000..20000, per your request), sorted by step.
  2. classify_samples_per_sample(): runs EVERY individual generated
     sample through this project's own classification_stats, ONE SAMPLE
     AT A TIME (batch size 1) -- classification_stats returns fractions
     over its batch, and a batch of 1 makes exactly one of
     memorized/valid/local/global equal to 1.0, i.e. that sample's state.
     This is the same one-at-a-time pattern
     plot_rhm_attractor_field_real.classify_candidate() already uses for
     a single candidate; I don't have a batched-per-sample variant of
     classification_stats, so this reuses the established idiom rather
     than guessing at one. SLOW for many samples x many checkpoints --
     see --max-samples-per-step.
  3. For every CONSECUTIVE pair of kept checkpoints (by step, not
     necessarily adjacent original checkpoints if some in between fall
     outside [step_min, step_max]), accumulates the 4x4 count matrix
     T[i,j] += 1 for each generation index whose state at step t was i
     and at step t+1 is j.
  4. Sums those counts across every consecutive pair in the window (one
     running 4x4 count tensor for the whole window, exactly as the paper
     aggregates "pre-rule-learning"/"rule-learning"/"memorization" phases
     separately), then row-normalizes ONCE at the end.
  5. Plots the row-normalized matrix as an annotated heatmap (probability
     in each cell, raw count in parentheses) plus writes the raw and
     normalized matrices to a .csv next to the plot.

CRITICAL ASSUMPTION -- SAMPLE ALIGNMENT ACROSS CHECKPOINTS:
  Transition counting requires that generation index i at step t and
  generation index i at step t+1 are "the same" sample lineage (e.g. the
  same eval seed / same initial noise re-used every checkpoint) -- this
  is what makes "sample i moved from state A to state B" meaningful
  rather than comparing two unrelated draws. I don't know whether your
  gen_samples_step_*.npz dumps actually preserve that alignment (fixed
  eval seeds reused every checkpoint) -- if generation index i is
  RESAMPLED independently at every checkpoint, this script will still
  run and produce a matrix, but it will just be measuring the marginal
  state distribution's product, not real transitions. Sanity-check this
  before trusting the figure: if consecutive-checkpoint self-transition
  probabilities (the diagonal) come out close to what you'd get from
  random pairing of two independent marginals, alignment is probably not
  preserved and this script's core assumption doesn't hold for your data.
  If array lengths differ between two consecutive files, this script
  truncates to the shorter length and PRINTS A WARNING -- it does not
  try to re-align them by content.

CAVEATS (same npz-key uncertainty as plot_gen_samples_state_transition.py):
  - Sample array key is guessed the same way (see load_gen_samples()) --
    check the printed keys/shapes, pass --samples-key if the guess is
    wrong.
  - classification_stats()'s call signature is taken from
    train_and_eval.py's run_eval() usage; if its returned dict's keys
    aren't literally 'memorized'/'valid'/'local'/'global', fix STATE_KEYS
    below.

Usage::

    python plot_transition_probability_matrix.py \\
        --data-path ../data/labeled_data_restrictedfixed_6_4_1.0_4_0.npy \\
        --reduced-length 4096 --pick-i-for-training 0 \\
        --gen-samples-dir ../results_lagreg_field_sweep/results_transformer_pick_0_for_training/models_restricted_continuous_0/test_model_script_12500_4096_cross_entropy_loss/8_512_lagreg_field_lam0p0/gen_samples \\
        --step-min 2000 --step-max 20000 \\
        --out transition_matrix_2000_20000.png
"""
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # matches train_and_eval.py

from utils import load_data, setup_device
from sequential_learning import build_parent_array, classification_stats


# The four mutually-exclusive states this project already classifies
# generated samples into (classification_stats' own vocabulary) -- the
# direct analog of the paper's four states, just this project's own
# names for them. Order fixed here so matrix rows/cols are always in
# this order everywhere below.
STATE_KEYS = ["memorized", "valid", "local", "global"]
STATE_LABELS = {"memorized": "memorized", "valid": "valid (generalized)",
               "local": "local (bad)", "global": "global (hallucinated)"}


# ---------------------------------------------------------------------
# 1. gen_samples_step_<N>.npz discovery -- identical logic/regex to
#    plot_gen_samples_state_transition.py's discover_gen_sample_files().
# ---------------------------------------------------------------------

_GEN_STEP_RE = re.compile(r"gen_samples_step_(\d+)\.npz$")


def discover_gen_sample_files(gen_samples_dir, step_min, step_max):
    gen_samples_dir = Path(gen_samples_dir)
    all_npz = sorted(gen_samples_dir.glob("gen_samples_step_*.npz"))
    if not all_npz:
        raise RuntimeError(f"no gen_samples_step_*.npz files found under {gen_samples_dir}")
    kept = []
    for p in all_npz:
        m = _GEN_STEP_RE.search(p.name)
        if m is None:
            continue
        step = int(m.group(1))
        if step_min <= step <= step_max:
            kept.append((step, p))
    kept.sort(key=lambda sp: sp[0])
    if not kept:
        all_steps = sorted(int(_GEN_STEP_RE.search(p.name).group(1)) for p in all_npz
                           if _GEN_STEP_RE.search(p.name))
        raise RuntimeError(
            f"no gen_samples_step_*.npz file under {gen_samples_dir} has a step in "
            f"[{step_min}, {step_max}]. Steps actually present ({len(all_steps)} total): "
            f"{all_steps}")
    if len(kept) < 2:
        raise RuntimeError(
            f"only {len(kept)} checkpoint(s) found in [{step_min}, {step_max}] -- need "
            f"at least 2 (a consecutive pair) to compute any transition. Widen the "
            f"step range.")
    print(f"found {len(kept)} gen_samples file(s) in step range [{step_min}, {step_max}]: "
          f"steps={[s for s, _ in kept]}")
    return kept


_GUESS_KEYS = ["samples", "gen_samples", "generated", "sequences", "seqs", "x", "x0"]


def load_gen_samples(npz_path, seq_len, samples_key=None):
    data = np.load(npz_path)
    print(f"  {npz_path.name}: keys={list(data.keys())}  "
          f"shapes={ {k: data[k].shape for k in data.keys()} }")
    if samples_key is not None:
        return data[samples_key]
    for k in _GUESS_KEYS:
        if k in data:
            arr = data[k]
            if arr.ndim >= 2:
                print(f"  -> using key {k!r} (guessed; pass --samples-key to override)")
                return arr
    first_key = list(data.keys())[0]
    print(f"  WARNING: couldn't confidently identify the samples array -- "
          f"falling back to the first key {first_key!r}. Pass --samples-key explicitly.")
    return data[first_key]


# ---------------------------------------------------------------------
# 2. Per-sample classification -- one sample at a time (batch size 1),
#    same idiom as plot_rhm_attractor_field_real.classify_candidate().
# ---------------------------------------------------------------------

def classify_samples_per_sample(samples, parent_arr, k, train_set, train_arr,
                                local_max_depth, max_samples=None):
    """samples: (n_gen, seq_len) discrete token ids. Returns a length-n_gen
    list of state strings (one of STATE_KEYS), or None for any sample
    classification_stats didn't put fully into exactly one of the four
    (a warning is printed and it's skipped from the transition count)."""
    n = len(samples) if max_samples is None else min(len(samples), max_samples)
    if max_samples is not None and len(samples) > max_samples:
        print(f"  (capping to first {max_samples} of {len(samples)} samples via "
              f"--max-samples-per-step)")
    states = []
    n_ambiguous = 0
    for i in range(n):
        probe = np.asarray(samples[i:i + 1], dtype=np.int64)
        cls, _ = classification_stats(probe, parent_arr, ell=k, train_set=train_set,
                                      train_arr=train_arr, local_max_depth=local_max_depth)
        hit = [key for key in STATE_KEYS if float(cls.get(key, 0.0)) > 0.5]
        if len(hit) != 1:
            n_ambiguous += 1
            states.append(None)
        else:
            states.append(hit[0])
    if n_ambiguous:
        print(f"  WARNING: {n_ambiguous}/{n} sample(s) didn't land in exactly one of "
              f"{STATE_KEYS} (classification_stats keys may not match STATE_KEYS -- "
              f"check them) -- excluded from the transition count.")
    return states


# ---------------------------------------------------------------------
# 3. Transition counting -- accumulate one 4x4 count matrix across every
#    consecutive checkpoint pair in the kept range, THEN row-normalize
#    once (matches the paper's own aggregation order, see module
#    docstring point 4).
# ---------------------------------------------------------------------

def accumulate_transition_counts(states_by_step):
    """states_by_step: list of (step, states_list) in increasing step
    order. Returns (counts, n_pairs_used, n_samples_used) where counts is
    a (4,4) int array indexed by STATE_KEYS.index(state)."""
    idx = {s: i for i, s in enumerate(STATE_KEYS)}
    counts = np.zeros((4, 4), dtype=np.int64)
    n_pairs_used = 0
    n_samples_used = 0
    for (step_t, states_t), (step_tp1, states_tp1) in zip(states_by_step, states_by_step[1:]):
        n = min(len(states_t), len(states_tp1))
        if len(states_t) != len(states_tp1):
            print(f"  WARNING: step {step_t} has {len(states_t)} samples but step "
                  f"{step_tp1} has {len(states_tp1)} -- truncating both to {n} for "
                  f"this pair (index-aligned, not content-realigned).")
        pair_used = 0
        for i in range(n):
            s_t, s_tp1 = states_t[i], states_tp1[i]
            if s_t is None or s_tp1 is None:
                continue
            counts[idx[s_t], idx[s_tp1]] += 1
            pair_used += 1
        print(f"  pair (step {step_t} -> step {step_tp1}): {pair_used}/{n} samples used")
        n_pairs_used += 1
        n_samples_used += pair_used
    return counts, n_pairs_used, n_samples_used


def row_normalize(counts):
    counts = counts.astype(np.float64)
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
    return probs


# ---------------------------------------------------------------------
# 4. Plotting + CSV dump.
# ---------------------------------------------------------------------

def plot_transition_matrix(probs, counts, step_min, step_max, n_pairs_used,
                           n_samples_used, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    im = ax.imshow(probs, cmap="viridis", vmin=0, vmax=1)
    labels = [STATE_LABELS[k] for k in STATE_KEYS]
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("state at $t+1$", fontsize=11)
    ax.set_ylabel("state at $t$", fontsize=11)
    for i in range(4):
        for j in range(4):
            txt_color = "white" if probs[i, j] < 0.6 else "black"
            ax.text(j, i, f"{probs[i, j]:.2f}\n({int(counts[i, j])})",
                    ha="center", va="center", fontsize=9, color=txt_color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06, label="transition probability")
    ax.set_title(f"State transition probability matrix\n"
                f"steps {step_min}-{step_max}  "
                f"({n_pairs_used} consecutive-checkpoint pair(s), "
                f"{n_samples_used} sample-transitions)", fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")

    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w") as f:
        f.write("," + ",".join(f"{k}_tplus1" for k in STATE_KEYS) + "\n")
        for i, key in enumerate(STATE_KEYS):
            row = ",".join(f"{probs[i, j]:.6f} ({int(counts[i, j])})" for j in range(4))
            f.write(f"{key}_t,{row}\n")
    print(f"wrote {csv_path}")


# ---------------------------------------------------------------------
# 5. Main.
# ---------------------------------------------------------------------

def main(args):
    device = setup_device()
    vocab_size, sequences, seq_len, k, rho, _ = load_data(args.data_path, device)
    import torch
    if not isinstance(rho, torch.Tensor):
        rho = torch.as_tensor(np.asarray(rho))
    rho = rho.to(device).float()

    L = args.reduced_length
    tr_lo = args.pick_i_for_training * L
    tr_hi = tr_lo + L
    train_block = sequences[tr_lo:tr_hi].detach().cpu().numpy().astype(np.int64)

    parent_arr = build_parent_array(rho)
    train_set = set(tuple(int(v) for v in row) for row in train_block)

    kept = discover_gen_sample_files(args.gen_samples_dir, args.step_min, args.step_max)

    states_by_step = []
    for step, npz_path in kept:
        samples = load_gen_samples(npz_path, seq_len, samples_key=args.samples_key)
        print(f"classifying step {step} ({len(samples)} samples)...")
        states = classify_samples_per_sample(
            samples, parent_arr, k, train_set, train_block, args.local_max_depth,
            max_samples=args.max_samples_per_step)
        counts_this_step = {s: states.count(s) for s in STATE_KEYS}
        print(f"  step {step} marginal state counts: {counts_this_step}")
        states_by_step.append((step, states))

    counts, n_pairs_used, n_samples_used = accumulate_transition_counts(states_by_step)
    probs = row_normalize(counts)

    print()
    print("aggregated 4x4 transition COUNT matrix (rows=state at t, cols=state at t+1):")
    print("        " + "  ".join(f"{k:>9s}" for k in STATE_KEYS))
    for i, key in enumerate(STATE_KEYS):
        print(f"{key:>8s}" + "  ".join(f"{counts[i, j]:>9d}" for j in range(4)))
    print()
    print("row-normalized transition PROBABILITY matrix:")
    print("        " + "  ".join(f"{k:>9s}" for k in STATE_KEYS))
    for i, key in enumerate(STATE_KEYS):
        print(f"{key:>8s}" + "  ".join(f"{probs[i, j]:>9.4f}" for j in range(4)))

    plot_transition_matrix(probs, counts, args.step_min, args.step_max,
                           n_pairs_used, n_samples_used, args.out)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--data-path", type=str,
                   default="../data/labeled_data_restrictedfixed_6_4_1.0_4_0.npy")
    p.add_argument("--reduced-length", type=int, default=4096)
    p.add_argument("--pick-i-for-training", type=int, default=0)
    p.add_argument("--local-max-depth", type=int, default=2)

    p.add_argument("--gen-samples-dir", type=str, required=True,
                   help="Directory containing gen_samples_step_<N>.npz files.")
    p.add_argument("--step-min", type=int, default=2000)
    p.add_argument("--step-max", type=int, default=20000)
    p.add_argument("--samples-key", type=str, default=None,
                   help="Exact npz key holding the generated sequences. Omit to "
                        "auto-guess (printed on every file) and pass this once you've "
                        "seen the real key name.")
    p.add_argument("--max-samples-per-step", type=int, default=None,
                   help="Cap how many samples per checkpoint are classified -- "
                        "classification is one-sample-at-a-time (see module "
                        "docstring), so this bounds runtime. Default: classify all.")
    p.add_argument("--out", type=str, default="transition_probability_matrix.png")
    main(p.parse_args())