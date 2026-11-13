"""
combine_runs.py

Combine the per-N outputs of sequential_learning_reduced.py into one figure,
averaging across seeds.

Layout: one horizontal strip per training-set size N, x-axis = gradient steps
(log). The colour of the strip at a given step is an additive RGB mix of the
sample composition at that checkpoint:

    R = Bad generation (local violations)   G = valid (novel + grammatical)
    B = hallucinated (global violations)

so a strip that is pure red is entirely bad-generation output, pure green is
entirely valid, and intermediate mixtures read directly as their blend --
yellow = bad-generation+valid, cyan = valid+hallucinated, magenta =
bad-generation+hallucinated, white = all three in equal measure.

Each colour is normalised by its largest channel, so hue encodes composition
independently of brightness. Brightness is then used for memorization:
value = 1 - memorized, so a fully memorizing model goes black regardless of
hue. Hue therefore describes the mix among non-memorized samples.

Alpha encodes generation health: alpha = n_distinct / n_gen. A collapsed model
emitting one sequence 256 times fades to nearly invisible, because its
composition is an artifact and should not be read as a measurement.

SEED AVERAGING. Runs with different seeds but the same N land in the SAME
output directory (the path is built from reduced_length / N_test / ratio /
backend only -- no seed component), so several npz can match one N. All of
them are collected and averaged elementwise per checkpoint before any colour
is computed. This is a change from the earlier behaviour, which kept only the
most recently modified file and silently discarded every other seed.

Strips are contiguous on a log y-axis, positioned and sized by their true
N value, so the vertical gap between rows reflects the actual numeric
difference in training-set size. Only three quantities fit in RGB; to
foreground memorization instead of one of the defaults, pass e.g.
--channels memorized,valid,global.

Every cell of the composition grid can additionally be labelled with the
percentage of generalized samples (valid = novel + grammatical) at that
checkpoint, in small text, via --annotate-valid-pct (on by default).

Usage (run from scripts/)::

    python combine_runs.py --plots-root ../plots --batch-size 512
    python combine_runs.py --plots-root ../plots --channels local,valid,global
    python combine_runs.py --plots-root ../plots --require-seeds 2
"""
from argparse import ArgumentParser
from pathlib import Path
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe


# Only files tagged with BOTH a seed and an N are accepted, i.e. produced with
#     sequential_learning_reduced.py --job-id seed<S>_N<M>
# so the filename ends in "..._seed0_N1024.npz". Untagged files (bare
# "..._latest.npz", written when --job-id was omitted) are ignored: they are
# overwritten by every subsequent untagged run, so which run they came from
# is unknowable, and stale ones from older sweeps otherwise get averaged in.
TAGGED_RE = re.compile(r"_seed(\d+)_N(\d+)$")


def discover(plots_root: Path, n_test=None, ratio=None, backend=None,
             require_tag=True):
    """
    Find ALL npz per N under
        <plots_root>/exp6_factorized/N_train_<N>/N_test_<M>/ratio_<r>/backend_<b>/*.npz

    n_test / ratio / backend restrict which sub-branch is accepted, so runs
    evaluated under different settings are not silently mixed into one figure.
    require_tag additionally demands a "_seed<S>_N<M>" filename suffix.

    The N used for grouping comes from the directory (N_train_<N>); when a file
    is tagged, the N in its name must agree, or it is rejected as misfiled.

    Returns {N: [paths]}. Multiple seeds for the same N share a directory and
    are ALL returned, to be averaged downstream.
    """
    root = Path(plots_root)
    runs = {}
    skipped_filter, skipped_untagged, mismatched = 0, [], []
    for npz in sorted(root.rglob("N_train_*/**/*.npz")):
        s = str(npz)
        m = re.search(r"N_train_(\d+)", s)
        if not m:
            continue
        if n_test is not None and f"N_test_{n_test}/" not in s + "/":
            skipped_filter += 1
            continue
        if ratio is not None and f"ratio_{ratio}/" not in s + "/":
            skipped_filter += 1
            continue
        if backend is not None and f"backend_{backend}/" not in s + "/":
            skipped_filter += 1
            continue

        N = int(m.group(1))

        tag = TAGGED_RE.search(npz.stem)
        if tag is None:
            if require_tag:
                skipped_untagged.append(npz)
                continue
        else:
            n_in_name = int(tag.group(2))
            if n_in_name != N:
                mismatched.append((npz, N, n_in_name))
                continue

        runs.setdefault(N, []).append(npz)

    if skipped_filter:
        print(f"skipped {skipped_filter} npz not matching the requested filters")
    if skipped_untagged:
        print(f"skipped {len(skipped_untagged)} untagged npz "
              f"(no _seed<S>_N<M> suffix); pass --allow-untagged to include them:")
        for p in skipped_untagged:
            print(f"    {p}")
    if mismatched:
        print(f"skipped {len(mismatched)} npz whose filename N disagrees with "
              f"its N_train_ directory:")
        for p, n_dir, n_name in mismatched:
            print(f"    {p}  (dir N={n_dir}, name N={n_name})")
    return dict(sorted(runs.items()))


def geometric_edges(x):
    """Cell edges for non-uniformly spaced points on a log axis."""
    x = np.asarray(x, dtype=float)
    if x.size == 1:
        return np.array([x[0] * 0.9, x[0] * 1.1])
    mid = np.sqrt(x[:-1] * x[1:])
    first = x[0] ** 2 / mid[0]
    last = x[-1] ** 2 / mid[-1]
    return np.concatenate([[first], mid, [last]])


def linear_edges(x):
    """Cell edges for non-uniformly spaced points on a linear axis (arithmetic
    midpoints, rather than the geometric midpoints used for the log x-axis)."""
    x = np.asarray(x, dtype=float)
    if x.size == 1:
        return np.array([x[0] - 1.0, x[0] + 1.0])
    mid = (x[:-1] + x[1:]) / 2.0
    first = 2.0 * x[0] - mid[0]
    last = 2.0 * x[-1] - mid[-1]
    return np.concatenate([[first], mid, [last]])


def read_ckpt_tag(npz_path: Path):
    """
    Return 'step', 'epoch', or None for a run's checkpoint convention.

    sequential_learning_reduced.py stores its x-values under the key "epochs"
    whatever the convention, so the values are GLOBAL STEPS when the run was
    evaluated with --ckpt-tag step and EPOCHS otherwise. The distinction is
    recoverable from the saved args, either in the npz (needs allow_pickle) or
    in the .json sibling, which is plain text and cheaper to trust.
    """
    js = npz_path.with_suffix(".json")
    if js.exists():
        try:
            import json
            with open(js) as f:
                args = json.load(f).get("args", {})
            tag = args.get("ckpt_tag")
            if tag in ("step", "epoch"):
                return tag
        except Exception:
            pass
    try:
        d = np.load(npz_path, allow_pickle=True)
        if "args" in d:
            args = d["args"].item()
            tag = args.get("ckpt_tag")
            if tag in ("step", "epoch"):
                return tag
    except Exception:
        pass
    return None


def load_run(npz_paths, N: int, batch_size: int, channels, mem_gain=1.0,
             x_axis="auto", collapse_agg="min", exclude_mem=False):
    """
    Load every npz for this N (one per seed), average the raw per-checkpoint
    fields across seeds, then derive colour and alpha from the seed-mean.

    Averaging happens on the RAW fractions, before the peak-normalisation and
    the memorization/alpha mappings, so the plotted colour is the colour of
    the mean composition -- not the mean of per-seed colours, which would not
    be the same thing once normalisation is applied.

    x_axis controls how the stored x-values are interpreted:
      'auto'   read ckpt_tag from the run's args and decide
      'step'   values are already gradient steps -- use as-is
      'epoch'  values are epochs -- multiply by ceil(N / batch_size)
    Getting this wrong rescales each row by a DIFFERENT factor (steps_per_epoch
    varies with N), which misaligns runs that in fact stopped at the same step.
    """
    if isinstance(npz_paths, (str, Path)):
        npz_paths = [npz_paths]

    dsets = [np.load(p) for p in npz_paths]
    needed = list(channels) + ["epochs"]
    for p, d in zip(npz_paths, dsets):
        missing = [k for k in needed if k not in d]
        if missing:
            raise KeyError(
                f"{p} is missing {missing}. Was it produced with --n-gen > 0?"
            )

    # All seeds must share the same checkpoint grid; they do when the runs came
    # from one sweep (same TARGET_STEPS and --checkpointing-period-steps).
    epochs_ref = np.asarray(dsets[0]["epochs"], dtype=float)
    for p, d in zip(npz_paths[1:], dsets[1:]):
        e = np.asarray(d["epochs"], dtype=float)
        if e.shape != epochs_ref.shape or not np.allclose(e, epochs_ref):
            raise ValueError(
                f"{p} has a different checkpoint grid than {npz_paths[0]} "
                f"({e.shape} vs {epochs_ref.shape}) -- cannot average "
                f"mismatched seeds without interpolation."
            )

    steps_per_epoch = int(np.ceil(N / batch_size))

    mode = x_axis
    if mode == "auto":
        tags = {read_ckpt_tag(p) for p in npz_paths}
        tags.discard(None)
        if len(tags) > 1:
            raise ValueError(
                f"N={N}: seeds disagree about the checkpoint convention "
                f"({tags}). Re-evaluate them the same way, or force one with "
                f"--x-axis."
            )
        mode = tags.pop() if tags else "epoch"

    if mode == "step":
        steps = epochs_ref
    elif mode == "epoch":
        steps = epochs_ref * steps_per_epoch
    else:
        raise ValueError(f"bad x_axis mode: {mode!r}")

    def avg_field(key):
        return np.mean(
            np.stack([np.asarray(d[key], dtype=float) for d in dsets], axis=0),
            axis=0,
        )

    # Optionally drop memorized samples entirely and renormalise the rest.
    # local/valid/global/memorized are disjoint fractions of n_gen summing to
    # 1, so conditioning on "not memorized" is a division by (1 - memorized).
    # The composition then describes the NOVEL output only, and memorization
    # stops competing with the other three for the same unit of mass.
    fields = {k: avg_field(k) for k in ("local", "valid", "global", "memorized")
              if k in dsets[0]}
    for k in ("local", "valid", "global", "memorized"):
        fields.setdefault(k, np.zeros(len(steps)))

    if exclude_mem:
        keep = 1.0 - fields["memorized"]
        # where everything was memorized there is no novel output to describe;
        # leave the row at zero rather than dividing by ~0 and amplifying noise
        safe = keep > 1e-6
        for k in ("local", "valid", "global"):
            fields[k] = np.divide(fields[k], keep, out=np.zeros_like(fields[k]),
                                  where=safe)
        n_dropped = int((~safe).sum())
        fields["memorized"] = np.zeros_like(fields["memorized"])
    else:
        n_dropped = 0

    rgb = np.stack([fields[c] if c in fields else avg_field(c)
                    for c in channels], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    # The three fractions sum to (at most) 1, so feeding them straight to RGB
    # would confine every mixture to the dark corner of the cube: an equal
    # three-way split would render (1/3, 1/3, 1/3), near-black, rather than
    # white. Normalising by the largest channel puts every colour on the bright
    # face, so hue alone encodes composition and brightness carries no signal
    # -- leaving alpha as the only intensity-like channel.
    peak = rgb.max(axis=-1, keepdims=True)
    rgb = np.divide(rgb, peak, out=np.zeros_like(rgb), where=peak > 0)

    # Brightness is now free (normalisation removed it as a signal), so give it
    # to memorization: value = 1 - memorized. Hue therefore describes the
    # composition among NON-memorized samples, and how dark it is says how much
    # of the output was a verbatim training sequence. mem_gain exaggerates a
    # rate that is often small in absolute terms.
    mem = fields["memorized"]
    value = np.clip(1.0 - mem_gain * mem, 0.0, 1.0)
    rgb = rgb * value[:, None]

    if "n_distinct" in dsets[0]:
        # n_gen is not stored; infer the cap from the largest value any seed
        # reached, not from the seed mean, which is already shrunk by collapse.
        nd_all = np.stack([np.asarray(d["n_distinct"], dtype=float)
                           for d in dsets], axis=0)
        n_gen = max(float(nd_all.max()), 1.0)
        # Collapse disqualifies rather than averages: if ANY seed emitted a
        # near-constant batch at this checkpoint, the seed-mean composition is
        # part artifact and should not be read. Taking the min across seeds
        # propagates that, where the mean would hide one collapsed seed behind
        # a healthy one (1/256 and 256/256 average to a reassuring 0.5).
        nd = nd_all.min(axis=0) if collapse_agg == "min" else nd_all.mean(axis=0)
        alpha = nd / n_gen
        n_collapsed_seeds = int(((nd_all / n_gen) < 0.10).any(axis=0).sum())
    else:
        alpha = np.ones(len(steps))
        n_collapsed_seeds = 0
    alpha_raw = alpha.copy()

    raw = dict(fields)
    # keep the pre-renormalisation memorization rate so it can be annotated
    # even when it is excluded from the colour
    raw["mem_orig"] = (avg_field("memorized") if "memorized" in dsets[0]
                       else np.zeros(len(steps)))
    raw["alpha_raw"] = alpha_raw
    raw["n_all_memorized"] = n_dropped
    raw["n_collapsed_ckpts"] = n_collapsed_seeds

    return (steps, rgb, np.clip(alpha, 0.05, 1.0), mem, steps_per_epoch,
            len(dsets), mode, raw)


# Maps raw npz field names to the label shown in the figure (colour key,
# top annotation). Keys not listed here are displayed as-is. This only
# affects display text -- the npz field actually loaded is still the raw
# name passed via --channels.
DISPLAY_NAMES = {
    "local": "Bad generation",
    "global": "hallucinated",
}


def disp(name):
    return DISPLAY_NAMES.get(name, name)


def strip(ax, y0, height, edges, colors):
    for i in range(len(colors)):
        ax.add_patch(Rectangle(
            (edges[i], y0), edges[i + 1] - edges[i], height,
            facecolor=colors[i], edgecolor="none",
        ))


def annotate_valid_pct(ax, y0, height, edges, valid, fontsize=5,
                       min_px_frac=0.0):
    """
    Write the percentage of generalized samples (valid = novel + grammatical)
    inside every individual cell of the composition grid, in small text.

    Unlike annotate_regions/annotate_memorization, which only label the
    widest contiguous run of a state per row, this labels EVERY cell -- the
    grid is per-(N, checkpoint), and each of those cells gets its own number,
    since valid% is a continuous quantity rather than a small set of states.
    No minimum-width gate is applied by default (min_px_frac=0); pass a value
    in (0, 1) -- a fraction of the row's total x-range -- to skip cells too
    narrow to hold text legibly at high checkpoint density.
    """
    valid = np.asarray(valid)
    total_width = np.log10(edges[-1]) - np.log10(edges[0])
    for i in range(len(valid)):
        cell_width = np.log10(edges[i + 1]) - np.log10(edges[i])
        if total_width > 0 and cell_width / total_width < min_px_frac:
            continue
        xmid = 10 ** ((np.log10(edges[i]) + np.log10(edges[i + 1])) / 2)
        ax.text(xmid, y0 + height / 2, f"{valid[i]:.0%}",
                ha="center", va="center", fontsize=fontsize, color="white",
                zorder=7,
                path_effects=[pe.withStroke(linewidth=1.2,
                                            foreground=(0, 0, 0, 0.6))])

def annotate_composition_pct(ax, y0, height, edges,
                             valid, hallu, local,
                             fontsize=4.5, min_px_frac=0.0):
    """
    Annotate every composition cell with:
        valid%
        H hallu%
        M memorized%

    H = hallucinated (global)
    M = memorized
    """
    valid = np.asarray(valid)
    hallu = np.asarray(hallu)
    local = np.asarray(local)

    total_width = np.log10(edges[-1]) - np.log10(edges[0])

    for i in range(len(valid)):
        cell_width = np.log10(edges[i + 1]) - np.log10(edges[i])

        if total_width > 0 and cell_width / total_width < min_px_frac:
            continue

        xmid = 10 ** (
            (np.log10(edges[i]) + np.log10(edges[i + 1])) / 2
        )

        label = (
            f"{valid[i] * 100:.0f}\n"
            f"H {hallu[i] * 100:.0f}\n"
            f"B {local[i] * 100:.0f}" )

        ax.text(
            xmid,
            y0 + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="white",
            zorder=7,
            linespacing=0.9,
            path_effects=[
                pe.withStroke(
                    linewidth=1.2,
                    foreground=(0, 0, 0, 0.6)
                )
            ],
        )

# ---------------------------------------------------------------- categorical
#
# A companion view to the RGB mix: every checkpoint is forced into ONE of a
# handful of states. Hard categories make TRANSITIONS easy to locate -- the eye
# finds edges far more reliably than it estimates a hue mixture -- at the cost
# of magnitude, since a 34/33/33 split and a 90/5/5 split render identically.

CAT_VALID, CAT_LOCAL, CAT_HALLU, CAT_MEM, CAT_COLLAPSE = 0, 1, 2, 3, 4

CAT_COLORS = {
    CAT_VALID:    "#2ca02c",   # green
    CAT_LOCAL:    "#d62728",   # red
    CAT_HALLU:    "#1f77b4",   # blue
    CAT_MEM:      "#000000",   # black
    CAT_COLLAPSE: "#b0b0b0",   # grey -- degenerate, composition is an artifact
}

CAT_LABELS = {
    CAT_VALID:    "valid",
    CAT_LOCAL:    "bad generation",
    CAT_HALLU:    "hallucinated",
    CAT_MEM:      "memorizing",
    CAT_COLLAPSE: "collapsed (unreliable)",
}


RULE_TEXT = {
    CAT_VALID:    "valid > {valid_thr:.0%}",
    CAT_LOCAL:    "local > hallu",
    CAT_HALLU:    "hallu > local",
    CAT_MEM:      "mem >= {mem_thr:.0%}",
    CAT_COLLAPSE: "distinct < {collapse_thr:.0%}",
}

RULE_TEXT_THRESHOLD = {
    CAT_VALID:    "else",
    CAT_LOCAL:    "local > {local_thr:.0%}",
    CAT_HALLU:    "hallu > {hallu_thr:.0%}",
    CAT_MEM:      "mem >= {mem_thr:.0%}",
    CAT_COLLAPSE: "distinct < {collapse_thr:.0%}",
}


def annotate_memorization(ax, y0, height, edges, mem, thr):
    """
    Write the memorization rate inside each contiguous run of checkpoints
    where it exceeds thr. The colour is left alone -- this only reports how
    much of the output was verbatim training data in that region, so a region
    can read 'hallucinated' and still carry a visible memorization figure.
    """
    hi = np.asarray(mem) >= thr
    i = 0
    while i < len(hi):
        if not hi[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(hi) and hi[j + 1]:
            j += 1
        width = np.log10(edges[j + 1]) - np.log10(edges[i])
        if width >= 0.10:
            xmid = 10 ** ((np.log10(edges[i]) + np.log10(edges[j + 1])) / 2)
            ax.text(xmid, y0 + height * 0.78,
                    f"mem {np.mean(mem[i:j + 1]):.0%}",
                    ha="center", va="center", fontsize=6.5, color="white",
                    zorder=6, style="italic",
                    path_effects=[pe.withStroke(linewidth=1.8,
                                                foreground=(0, 0, 0, 0.55))])
        i = j + 1


def annotate_regions(ax, y0, height, edges, cats, fmt, min_decades=0.16):
    """
    Write the deciding rule inside each contiguous run of a state.

    Labels go on the widest run of each state per row only, and only if that
    run is wide enough on the log axis to hold text, so the plot does not turn
    into a wall of repeated captions.
    """
    cats = np.asarray(cats)
    runs = {}
    i = 0
    while i < len(cats):
        j = i
        while j + 1 < len(cats) and cats[j + 1] == cats[i]:
            j += 1
        width = np.log10(edges[j + 1]) - np.log10(edges[i])
        if cats[i] not in runs or width > runs[cats[i]][2]:
            runs[cats[i]] = (i, j, width)
        i = j + 1

    for c, (i, j, width) in runs.items():
        if width < min_decades or c not in fmt:
            continue
        xmid = 10 ** ((np.log10(edges[i]) + np.log10(edges[j + 1])) / 2)
        ax.text(xmid, y0 + height / 2, fmt[c], ha="center", va="center",
                fontsize=7, color="white" if c != CAT_COLLAPSE else "0.25",
                zorder=5,
                path_effects=[pe.withStroke(linewidth=1.6,
                                            foreground=(0, 0, 0, 0.35))]
                if c != CAT_COLLAPSE else None)


def debounce(labels, k=2):
    """
    Commit a state change only once it holds for k consecutive checkpoints.

    An isolated checkpoint keeps the previous state instead of forming a
    one-cell island. This is what suppresses speckle at the local/hallucinated
    argmax crossing, where the ordering is inside sampling noise, and it does
    so without inventing a category for it. A change at the very last
    checkpoint cannot be confirmed and is therefore not committed.
    """
    if k <= 1 or len(labels) == 0:
        return labels
    out = labels.copy()
    cur = labels[0]
    for i in range(1, len(labels)):
        if labels[i] == cur:
            out[i] = cur
            continue
        run = labels[i:i + k]
        if len(run) == k and np.all(run == labels[i]):
            cur = labels[i]
        out[i] = cur
    return out


def classify(local, valid, hallu, mem, alpha,
             rule="threshold", local_thr=0.33, hallu_thr=0.33,
             valid_thr=0.67, mem_thr=0.05, collapse_thr=None,
             debounce_k=2):
    """
    One state per checkpoint.

    Precedence is fixed and matters: collapse first (the composition is an
    artifact of a degenerate model and should not be read at all), then
    memorization (a memorized sample is by construction a valid parse, so it
    would otherwise hide inside `valid`), then the composition rule.

    rule='threshold'  local > local_thr -> bad generation
                      else hallu > hallu_thr -> hallucinated
                      else valid
                      The elif chain is what keeps the states exclusive; a
                      literal "hallu > t AND local < t" leaves a gap exactly at
                      local == t.

    rule='argmax'     whichever of (valid, local, hallu) is largest. Since the
                      four fractions sum to 1, a 1/3 threshold is already close
                      to argmax, but argmax has no arbitrary constant and does
                      not flicker when a curve sits on the boundary.
    """
    n = len(local)
    out = np.full(n, CAT_VALID, dtype=int)

    if rule == "staged":
        # local vs hallucinated decided by argmax, so neither is a residual
        # bucket: each has to beat the other rather than inherit what is left.
        # Generalization still has to clear an absolute bar, keeping the
        # asymmetric burden -- easy to be called broken, hard to be called good.
        out = np.where(local > hallu, CAT_LOCAL, CAT_HALLU)
        out[valid > valid_thr] = CAT_VALID
    elif rule == "threshold":
        out[hallu > hallu_thr] = CAT_HALLU
        out[local > local_thr] = CAT_LOCAL
    elif rule == "argmax":
        stack = np.stack([valid, local, hallu], axis=-1)
        out = np.array([CAT_VALID, CAT_LOCAL, CAT_HALLU])[stack.argmax(-1)]
    else:
        raise ValueError(f"bad rule: {rule!r}")

    out[mem >= mem_thr] = CAT_MEM
    if collapse_thr is not None:
        out[alpha < collapse_thr] = CAT_COLLAPSE

    # Every state change -- not just memorization -- must persist for
    # debounce_k checkpoints before the colour changes.
    raw_labels = out.copy()
    out = debounce(out, debounce_k)
    return out, int((out != raw_labels).sum())


def categorical_key(ax, rule, local_thr, hallu_thr, valid_thr, mem_thr,
                    collapse_thr, exclude_mem=False):
    order = [CAT_VALID, CAT_LOCAL, CAT_HALLU]
    if not exclude_mem:
        order.append(CAT_MEM)
    if collapse_thr is not None:
        order.append(CAT_COLLAPSE)
    for i, c in enumerate(order):
        ax.add_patch(Rectangle((0, -i), 0.8, 0.8, facecolor=CAT_COLORS[c],
                               edgecolor="0.75", lw=0.6))
        ax.text(1.0, -i + 0.4, CAT_LABELS[c], va="center", fontsize=8)

    if rule == "staged":
        rules = [f"valid > {valid_thr:.0%}", "else argmax(local, hallu)"]
    elif rule == "threshold":
        rules = [f"local > {local_thr:.0%}", f"else hallu > {hallu_thr:.0%}",
                 "else valid"]
    else:
        rules = ["argmax(valid, local, hallu)"]
    if not exclude_mem:
        rules = [f"mem >= {mem_thr:.0%} overrides"] + rules
    if collapse_thr is not None:
        rules = [f"distinct < {collapse_thr:.0%} overrides all"] + rules

    y = -len(order) - 0.8
    ax.text(0, y, "priority", fontsize=8, va="center", color="0.3")
    for i, r in enumerate(rules):
        ax.text(0, y - 0.7 * (i + 1), r, fontsize=7, va="center", color="0.45")

    ax.set_xlim(0, 5)
    ax.set_ylim(y - 0.7 * (len(rules) + 1), 1)
    ax.axis("off")
    ax.set_title("key", fontsize=9, loc="left", pad=22)


def colour_key(ax, channels):
    """Small additive-mix reference: the three primaries and their blends."""
    labels = [disp(c) for c in channels]
    combos = [
        ((1, 0, 0), labels[0]),
        ((0, 1, 0), labels[1]),
        ((0, 0, 1), labels[2]),
        ((1, 1, 0), f"{labels[0]}+{labels[1]}"),
        ((0, 1, 1), f"{labels[1]}+{labels[2]}"),
        ((1, 0, 1), f"{labels[0]}+{labels[2]}"),
        ((1, 1, 1), "all three (equal)"),
    ]
    for i, (c, lab) in enumerate(combos):
        ax.add_patch(Rectangle((0, -i), 0.8, 0.8, facecolor=c,
                               edgecolor="0.75", lw=0.6))
        ax.text(1.0, -i + 0.4, lab, va="center", fontsize=8)
    n = len(combos)
    # memorization ramp: the same hue at decreasing value
    ax.text(0, -n - 0.5, "memorization", fontsize=8, va="center")
    for j, v in enumerate([1.0, 0.66, 0.33, 0.0]):
        ax.add_patch(Rectangle((j * 0.85, -n - 2.0), 0.8, 0.8,
                               facecolor=(0.2 * v, 1.0 * v, 0.2 * v),
                               edgecolor="0.75", lw=0.6))
    ax.text(0, -n - 2.6, "none -> all", fontsize=7, va="center", color="0.4")

    ax.set_xlim(0, 5)
    ax.set_ylim(-n - 3.2, 1)
    ax.axis("off")
    ax.set_title("key", fontsize=9, loc="left", pad=22)


def main(args):
    channels = [c.strip() for c in args.channels.split(",")]
    if len(channels) != 3:
        raise ValueError(f"--channels needs exactly 3 keys, got {channels}")
    if args.exclude_memorized and "memorized" in channels:
        raise SystemExit(
            "--exclude-memorized removes memorized samples, so it cannot also "
            "be one of --channels (it would be identically zero).")

    runs = discover(Path(args.plots_root), n_test=args.n_test,
                    ratio=args.ratio, backend=args.backend,
                    require_tag=not args.allow_untagged)
    if not runs:
        raise SystemExit(
            f"No npz found under {args.plots_root} matching "
            f"N_test={args.n_test} ratio={args.ratio} backend={args.backend}"
        )

    print(f"found {len(runs)} N values: {sorted(runs)}")
    for N, paths in runs.items():
        print(f"  N={N:>6}  {len(paths)} seed file(s)")
        for p in paths:
            print(f"           {p.name}")

    if args.require_seeds is not None:
        short = {N: len(ps) for N, ps in runs.items() if len(ps) < args.require_seeds}
        if short:
            raise SystemExit(
                f"--require-seeds {args.require_seeds} but these N have fewer: "
                f"{short}. Rerun the missing evaluations, or lower the "
                f"requirement to average over what is present."
            )

    fig = plt.figure(figsize=(13, 0.85 * len(runs) + 2.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[5, 1], wspace=0.02)
    ax = fig.add_subplot(gs[0])
    axk = fig.add_subplot(gs[1])

    figc = plt.figure(figsize=(13, 0.85 * len(runs) + 2.4))
    gsc = figc.add_gridspec(1, 2, width_ratios=[5, 1], wspace=0.02)
    axc = figc.add_subplot(gsc[0])
    axck = figc.add_subplot(gsc[1])

    # Row boundaries are the cell edges of the actual N values, not a uniform
    # per-row height, so a strip's vertical extent and position reflect the
    # true numeric gap between consecutive N's.
    Ns = np.array(sorted(runs.keys()), dtype=float)
    y_edges = geometric_edges(Ns)

    yticks, ylabels = [], []
    xmin, xmax = np.inf, -np.inf

    collapse_thr = None if args.collapse_thresh <= 0 else args.collapse_thresh

    modes_used = set()
    for row, (N, npz_paths) in enumerate(runs.items()):
        steps, rgb, alpha, mem, spe, n_seeds, mode, raw = load_run(
            npz_paths, N, args.batch_size, channels, mem_gain=args.mem_gain,
            x_axis=args.x_axis, collapse_agg=args.collapse_agg,
            exclude_mem=args.exclude_memorized)
        modes_used.add(mode)
        print(f"  N={N:>6}  x-axis from '{mode}' values, "
              f"range {steps[0]:.0f} .. {steps[-1]:.0f}")
        if args.exclude_memorized and raw.get("n_all_memorized"):
            print(f"           {raw['n_all_memorized']} checkpoint(s) were "
                  f"~100% memorized; left blank (nothing novel to renormalise)")
        if raw["n_collapsed_ckpts"]:
            print(f"           {raw['n_collapsed_ckpts']} checkpoint(s) had at "
                  f"least one collapsed seed (agg='{args.collapse_agg}')")
        edges = geometric_edges(steps)
        xmin, xmax = min(xmin, edges[0]), max(xmax, edges[-1])

        y0, y1 = y_edges[row], y_edges[row + 1]
        strip(ax, y0, y1 - y0, edges,
              [(*rgb[i], alpha[i]) for i in range(len(steps))])
        if args.annotate_valid_pct:
            annotate_composition_pct(
                ax,
                y0,
                y1 - y0,
                edges,
                raw["valid"],
                raw["global"],
                raw["local"],
                fontsize=args.valid_pct_fontsize,
                min_px_frac=args.valid_pct_min_width,
            )

        cats, n_deb = classify(
            raw["local"], raw["valid"], raw["global"], raw["memorized"], alpha,
            rule=args.cat_rule, local_thr=args.local_thresh,
            hallu_thr=args.hallu_thresh, valid_thr=args.valid_thresh,
            mem_thr=args.mem_thresh, collapse_thr=collapse_thr,
            debounce_k=args.debounce)
        if n_deb:
            print(f"           debounce suppressed {n_deb} unconfirmed "
                  f"state change(s)")

        # The local/hallucinated edge is an argmax crossing; within this band
        # the ordering is not resolvable at n_gen*seeds samples, so the exact
        # transition step should not be quoted.
        if args.cat_rule == "staged" and args.report_band > 0:
            gap = np.abs(raw["local"] - raw["global"])
            near = np.flatnonzero((gap < args.report_band) &
                                  (cats != CAT_MEM) & (cats != CAT_VALID) &
                                  (cats != CAT_COLLAPSE))
            if near.size:
                print(f"           |local-hal| < {args.report_band:.2f} at steps "
                      f"{', '.join(f'{steps[i]:.0f}' for i in near)}"
                      f"  <- edge position unreliable here")
        strip(axc, y0, y1 - y0, edges, [CAT_COLORS[c] for c in cats])
        if args.annotate_regions:
            base = (RULE_TEXT_THRESHOLD if args.cat_rule == "threshold"
                    else RULE_TEXT)
            fmt = {c: t.format(valid_thr=args.valid_thresh,
                               local_thr=args.local_thresh,
                               hallu_thr=args.hallu_thresh,
                               mem_thr=args.mem_thresh,
                               collapse_thr=(collapse_thr or 0))
                   for c, t in base.items()}
            annotate_regions(axc, y0, y1 - y0, edges, cats, fmt)
        if args.exclude_memorized and args.annotate_memorization:
            annotate_memorization(axc, y0, y1 - y0, edges,
                                  raw["mem_orig"], args.mem_thresh)

        # first step at which each state is entered -- the transition itself
        trans = []
        report_cats = ((CAT_LOCAL, CAT_HALLU) if args.exclude_memorized
                       else (CAT_LOCAL, CAT_HALLU, CAT_MEM))
        for c in report_cats:
            hit = np.flatnonzero(cats == c)
            if hit.size:
                trans.append(f"{CAT_LABELS[c]}@{steps[hit[0]]:.0f}")
        if trans:
            print(f"           first entry: {'  '.join(trans)}")

        yticks.append(N)
        ylabels.append(f"N={N:g}" + (f"  ({n_seeds}s)" if n_seeds > 1 else ""))

    if len(modes_used) > 1:
        print(f"WARNING: rows were built from mixed conventions {modes_used}; "
              f"their x-axes are not comparable. Force one with --x-axis.")

    for a in (ax, axc):
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlim(xmin, xmax)
        a.set_ylim(y_edges[0], y_edges[-1])
        a.set_yticks(yticks)
        a.set_yticklabels(ylabels, fontsize=9)
        a.set_xlabel("gradient steps", fontsize=12)
        a.grid(axis="x", which="both", alpha=0.25, lw=0.5)
        a.set_axisbelow(True)
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)

    axc.set_title("Dominant state vs gradient steps (seed-averaged)",
                  fontsize=12, loc="left", pad=22)
    axc.text(
        0.0, 1.012,
        f"rule = {args.cat_rule}"
        + (f"   |   valid > {args.valid_thresh:.0%}, else argmax(local, hallu)"
           if args.cat_rule == "staged" else
           f"   |   local > {args.local_thresh:.0%}, hallu > {args.hallu_thresh:.0%}"
           if args.cat_rule == "threshold" else "   |   argmax(valid, local, hallu)")
        + ("" if args.exclude_memorized
           else f"   |   mem >= {args.mem_thresh:.0%} overrides")
        + f"   |   debounce {args.debounce}",
        transform=axc.transAxes, fontsize=9, color="0.35", va="bottom",
    )
    categorical_key(axck, args.cat_rule, args.local_thresh, args.hallu_thresh,
                    args.valid_thresh, args.mem_thresh, collapse_thr,
                    exclude_mem=args.exclude_memorized)

    outc = Path(args.cat_out)
    if args.tag_filenames:
        if args.cat_rule == "staged":
            tag = f"_{args.cat_rule}_valid{args.valid_thresh:.2f}"
        elif args.cat_rule == "threshold":
            tag = (f"_{args.cat_rule}_local{args.local_thresh:.2f}"
                   f"_hallu{args.hallu_thresh:.2f}")
        else:
            tag = f"_{args.cat_rule}"
        tag += f"_mem{args.mem_thresh:.2f}_deb{args.debounce}"
        outc = outc.with_name(outc.stem + tag + outc.suffix)
    outc.parent.mkdir(parents=True, exist_ok=True)
    figc.savefig(outc, dpi=200, bbox_inches="tight")
    print(f"wrote {outc}")

    ax.set_title("Sample composition vs gradient steps (seed-averaged)",
                 fontsize=12, loc="left", pad=22)
    ax.text(
        0.0, 1.012,
        f"colour = additive mix of {disp(channels[0])} (R) / {disp(channels[1])} (G) / "
        f"{disp(channels[2])} (B)   |   darkness = memorization   |   "
        "opacity = distinct / n_gen"
        + ("   |   text = % generalized (valid)" if args.annotate_valid_pct
           else ""),
        transform=ax.transAxes, fontsize=9, color="0.35", va="bottom",
    )
    colour_key(axk, channels)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--plots-root", type=str, default="../plots")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Used to convert epochs to gradient steps.")
    p.add_argument("--n-test", type=int, default=1000,
                   help="Only use runs evaluated with this N_test. None = any.")
    p.add_argument("--ratio", type=str, default=None,
                   help="Only use runs with this t_denoising_ratio, e.g. 0.15.")
    p.add_argument("--backend", type=str, default=None,
                   help="Only use runs with this BP backend, e.g. torch.")
    p.add_argument("--channels", type=str, default="local,valid,global",
                   help="Three npz keys mapped to R,G,B in that order.")
    p.add_argument("--exclude-memorized", action="store_true",
                   help="Drop memorized samples and renormalise local/valid/"
                        "global over the novel output only. Memorization then "
                        "no longer darkens the RGB plot nor overrides the "
                        "categorical state.")
    p.add_argument("--annotate-memorization", action="store_true", default=True,
                   help="With --exclude-memorized, still print the memorization "
                        "rate inside regions where it exceeds --mem-thresh.")
    p.add_argument("--no-annotate-memorization", dest="annotate_memorization",
                   action="store_false")
    p.add_argument("--annotate-valid-pct", action="store_true", default=True,
                   help="Label every cell of the composition grid (the RGB "
                        "figure, not the categorical one) with the percentage "
                        "of generalized samples (valid = novel + grammatical) "
                        "at that checkpoint, in small text.")
    p.add_argument("--no-annotate-valid-pct", dest="annotate_valid_pct",
                   action="store_false")
    p.add_argument("--valid-pct-fontsize", type=float, default=5.0,
                   help="Font size for the per-cell valid%% labels.")
    p.add_argument("--valid-pct-min-width", type=float, default=0.0,
                   help="Skip the per-cell valid%% label in cells narrower "
                        "than this fraction of the row's total log-x range "
                        "(0 = label every cell regardless of width).")
    p.add_argument("--mem-gain", type=float, default=1.0,
                   help="Scales how strongly memorization darkens the colour. "
                        "Use >1 when memorization rates are small but of interest.")
    p.add_argument("--x-axis", type=str, default="auto",
                   choices=["auto", "step", "epoch"],
                   help="How to read the stored x-values. 'auto' (default) "
                        "reads ckpt_tag from each run's args. 'step' uses them "
                        "as-is; 'epoch' multiplies by ceil(N/batch_size). "
                        "Runs checkpointed with --checkpointing-period-steps "
                        "store STEPS under the 'epochs' key, so multiplying "
                        "them again misaligns rows by ceil(N/batch_size).")
    p.add_argument("--allow-untagged", action="store_true",
                   help="Also accept npz without a _seed<S>_N<M> filename "
                        "suffix. Off by default: untagged files are "
                        "overwritten by every untagged run, so stale results "
                        "from older sweeps get averaged in silently.")
    p.add_argument("--require-seeds", type=int, default=None,
                   help="Fail if any N has fewer than this many npz files, "
                        "instead of silently averaging over fewer seeds.")
    p.add_argument("--cat-rule", type=str, default="staged",
                   choices=["staged", "threshold", "argmax"],
                   help="How to pick the single state per checkpoint.")
    p.add_argument("--local-thresh", type=float, default=0.33)
    p.add_argument("--valid-thresh", type=float, default=0.67)
    p.add_argument("--annotate-regions", action="store_true", default=True,
                   help="Write the deciding rule inside each region.")
    p.add_argument("--no-annotate-regions", dest="annotate_regions",
                   action="store_false")
    p.add_argument("--debounce", type=int, default=2,
                   help="A state change is committed only after it holds for "
                        "this many consecutive checkpoints. 1 disables.")
    p.add_argument("--report-band", type=float, default=0.05,
                   help="Print which checkpoints have |local-hal| below this, "
                        "where the argmax edge is inside sampling noise. "
                        "0 disables. Does not change any colour.")
    p.add_argument("--hallu-thresh", type=float, default=0.33)
    p.add_argument("--mem-thresh", type=float, default=0.05)
    p.add_argument("--collapse-agg", type=str, default="min",
                   choices=["min", "mean"],
                   help="How distinct/n_gen is combined across seeds. 'min' "
                        "(default) flags a checkpoint if ANY seed collapsed; "
                        "'mean' lets a healthy seed mask a collapsed one.")
    p.add_argument("--collapse-thresh", type=float, default=0.10,
                   help="distinct/n_gen below this is drawn grey, since the "
                        "composition of a collapsed model is an artifact. "
                        "0 disables.")
    p.add_argument("--out", type=str, default="../plots/composition_combined.png")
    p.add_argument("--tag-filenames", action="store_true", default=True,
                   help="Append the active thresholds to the categorical "
                        "filename, so a threshold sweep does not overwrite "
                        "itself.")
    p.add_argument("--no-tag-filenames", dest="tag_filenames",
                   action="store_false")
    p.add_argument("--cat-out", type=str,
                   default="../plots/composition_categorical.png")
    main(p.parse_args())