"""
combine_lagreg_composition.py

Adapted from combine_runs.py for train_and_eval.py's output instead of
sequential_learning_reduced.py's. Same RGB-strip composition plot (local=R,
valid=G, global=B by default; darkness=memorization; alpha=distinct/n_gen),
same seed-averaging-before-colour philosophy. Colour IS peak-normalized,
same as combine_runs.py, so a clearly-dominated cell reaches full
saturation rather than looking washed out; colour_key's legend swatches
are correspondingly softened off pure primaries to match what's actually
attainable in the strip, rather than showing colours the plot itself
never produces. Same plotting primitives
(geometric_edges/strip/annotate_composition_pct/colour_key) -- copied
verbatim from combine_runs.py rather than re-derived, so the visual
conventions match exactly.
A
WHAT'S DIFFERENT FROM combine_runs.py, AND WHY:
  - Source data: train_eval_log_reg{tag}.json (a flat JSON list of per-step
    dicts), not .npz. Fields are read directly by name (valid/local/global/
    memorized/n_distinct/step) -- no epoch/step ambiguity since
    train_and_eval.py always logs gradient steps under "step".
  - N and seed come from the PATH, not the filename, matching how
    combine_runs.py already does it for its N_train_<N> directories:
    test_model_script_{n_iter}_{N}_{loss_name} gives N, and
    models_restricted_continuous_{seed} gives seed.
    train_eval_log_reg{tag}.json has no N/seed in its OWN filename (unlike
    ladder_emp_*/ladder_pop_*) since save_path is already N/seed-specific --
    this script does not need one either, for the same reason.
  - reg_lambda is selected via the filename tag (train_eval_log_reg{tag}.json
    where tag = str(reg_lambda).replace(".","p")), and is REQUIRED (one
    lambda per figure) -- there is no analog of combine_runs.py's --channels
    doing double duty as a lambda selector.

MULTI-GENERATION COLLISION (same issue as combine_lagreg_ladder_beforeafter.py,
fixed here the same way): n_iter is baked into save_path's directory name
(test_model_script_{n_iter}_{N}_...) via create_save_path()'s keying, but is
NOT part of train_eval_log_reg{tag}.json's own filename. If more than one
sweep generation (e.g. a TARGET_STEPS change, 100000 -> 200000) has ever
written into the same --results-root, PATH_N_RE previously captured n_iter
as group(1) and discarded it, using only group(2) (N) -- so discover() would
silently find BOTH generations' files under the same (N, seed) key. This is
now caught explicitly: --target-steps + --batch compute the expected n_iter
per N (reproducing run_lagreg_nsweep_full.sh's own STEPS_PER_EPOCH/N_ITER
arithmetic exactly) and filter to one generation; a duplicate-seed check
raises a clear error either way, rather than silently averaging two
different runs together.

THIS REVISION (styling, single-lambda mode):
  - No plot title and no caption/subtitle line above the axes either --
    colour_key (below the plot) carries that information on its own now.
  - Cell composition-%% labels now prefix the generalized/valid figure with
    "G" (matching H/B/M's letter-prefix convention), and "valid" is now
    displayed as "generalized" everywhere (DISPLAY_NAMES). "local" is now
    displayed as just "Bad" (was "Bad generation").
  - colour_key is laid out horizontally below the main plot (a second,
    short gridspec row) instead of in a narrow column beside it; the
    "all three (equal)" swatch is removed; memorization is no longer its
    own multi-swatch gradient row -- it's now a single grey swatch
    appended to the end of the same row as the colour combos.
  - Cell composition-%% labels are now LEFT-aligned (near the left edge of
    each cell) and drawn in BLACK, instead of centered white text -- the
    stroke outline was flipped from a dark semi-transparent stroke to a
    light one so black text stays legible against the strip's own
    (often dark/saturated) cell colours.

Usage (run from the results root, or pass --results-root explicitly)::

    python combine_lagreg_composition.py \\
        --results-root /scratch/work/chandrb1/biased-generalization/results_lagreg_sweep \\
        --reg-lambda 0.1 --target-steps 200000 --batch 512
    python combine_lagreg_composition.py --results-root ... --reg-lambda 0.1 \\
        --target-steps 200000 --batch 512 \\
        --n-vals 4096,8192,16384,65536 --require-seeds 2
"""
from argparse import ArgumentParser
from pathlib import Path
import json
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
from matplotlib.transforms import offset_copy


# test_model_script_{n_iter}_{N}_{loss_name} -- n_iter is group(1), N is group(2).
PATH_N_RE = re.compile(r"test_model_script_(\d+)_(\d+)_")
# models_restricted_continuous_{seed} (or _uniform_/_standard_, matching
# create_save_path's dataset_tag branches -- seed is always the trailing int).
PATH_SEED_RE = re.compile(r"models_\w+_continuous(?:_rescaled)?_(\d+)")


def reg_tag(reg_lambda: float) -> str:
    return str(reg_lambda).replace(".", "p")


def expected_n_iter(N: int, target_steps: int, batch: int) -> int:
    """OPTIONAL CONVENIENCE ONLY -- reproduces run_lagreg_nsweep_full.sh's own
    arithmetic to auto-generate an --n-iter-map without typing one N at a
    time. NOT the authoritative check: n_iter is already captured directly
    from the path by PATH_N_RE's group(1), so matching it literally (see
    --n-iter-map) is the real mechanism and cannot go stale if this formula
    or BATCH ever changes in a future sweep script."""
    steps_per_epoch = -(-N // batch)   # ceil division, matches bash's (N+BATCH-1)/BATCH
    return target_steps // steps_per_epoch


def discover(results_root: Path, reg_lambda: float, n_vals=None, n_iter_map=None, seeds=None):
    """
    Find every train_eval_log_reg{tag}.json under results_root, extract
    (N, seed, n_iter) from its path, and group by N.

    Returns {N: [(seed, path), ...]}, one entry per seed found for that N.
    Files whose path doesn't yield both N and seed are skipped and reported,
    same reasoning as combine_runs.py's TAGGED_RE rejection: an unparseable
    path could be a run from an unrelated experiment sharing --results-root.

    n_iter_map, if given ({N: n_iter}), additionally requires the path's OWN
    n_iter (PATH_N_RE's group 1, read directly -- not predicted from a
    formula) to equal n_iter_map[N] -- disambiguates multiple sweep
    generations sharing one results_root (see module docstring).
    """
    tag = reg_tag(reg_lambda)
    pattern = f"train_eval_log_reg{tag}.json"
    runs = {}
    skipped = []
    filtered_out_n_iter = []
    for f in sorted(Path(results_root).rglob(pattern)):
        s = str(f)
        m_n = PATH_N_RE.search(s)
        m_seed = PATH_SEED_RE.search(s)
        if not (m_n and m_seed):
            skipped.append(f)
            continue
        n_iter_found = int(m_n.group(1))
        N = int(m_n.group(2))
        seed = int(m_seed.group(1))
        if n_vals is not None and N not in n_vals:
            continue
        if seeds is not None and seed not in seeds:
            continue
        if n_iter_map is not None and N in n_iter_map and n_iter_found != n_iter_map[N]:
            filtered_out_n_iter.append((f, n_iter_found, n_iter_map[N]))
            continue
        runs.setdefault(N, []).append((seed, f))
    if skipped:
        print(f"skipped {len(skipped)} '{pattern}' file(s) whose path didn't "
              f"match the expected N/seed pattern:")
        for p in skipped:
            print(f"    {p}")
    if filtered_out_n_iter:
        print(f"filtered out {len(filtered_out_n_iter)} file(s) whose own n_iter "
              f"(read from the path) didn't match --n-iter-map:")
        for p, found, want in filtered_out_n_iter:
            print(f"    {p}  (n_iter={found}, wanted {want})")

    for N, seed_paths in runs.items():
        seen = {}
        for seed, p in seed_paths:
            seen.setdefault(seed, []).append(p)
        dupes = {seed: paths for seed, paths in seen.items() if len(paths) > 1}
        if dupes:
            lines = "\n".join(
                f"  seed={seed}:\n" + "\n".join(f"    {p}" for p in paths)
                for seed, paths in dupes.items()
            )
            raise ValueError(
                f"N={N}, reg_lambda={reg_lambda}: duplicate seed(s) found under "
                f"{results_root} -- more than one sweep generation (different "
                f"n_iter/save_path) has written into this results_root:\n{lines}\n"
                f"Resolve with --n-iter-map (read the correct n_iter for N={N} "
                f"straight off one of the test_model_script_<n_iter>_{N}_ paths "
                f"above), or remove the stale generation's directory."
            )
    return dict(sorted(runs.items()))


# Uniform brightness factor applied to every cell colour (and to the legend
# swatches, so they still match). 1.0 = original tone; lower = darker.
CELL_DARKEN = 1.0


def load_run(seed_paths, N: int, mem_gain: float, channels, collapse_agg="min"):
    """
    Load every seed's train_eval_log for this N, average local/valid/global/
    memorized across seeds (raw fractions, before colour is derived -- same
    reasoning as combine_runs.py: the plotted colour must be the colour of
    the mean composition, not the mean of per-seed colours), then compute
    RGB/alpha/value exactly as combine_runs.py does. Colour IS peak-
    normalized (a clearly-dominated cell reaches full saturation on its
    dominant channel(s) rather than looking washed out) -- the legend
    (colour_key) is softened to match this attainable gamut instead of
    showing unreachable pure primaries.
    """
    all_rows = []
    for seed, p in seed_paths:
        with open(p) as fh:
            rows = json.load(fh)
        if not rows:
            raise ValueError(f"{p} is empty")
        all_rows.append(sorted(rows, key=lambda r: r["step"]))

    steps_ref = np.array([r["step"] for r in all_rows[0]], dtype=float)
    for (seed, p), rows in zip(seed_paths[1:], all_rows[1:]):
        s = np.array([r["step"] for r in rows], dtype=float)
        if s.shape != steps_ref.shape or not np.allclose(s, steps_ref):
            raise ValueError(
                f"{p} has a different step grid than {seed_paths[0][1]} "
                f"({s.shape} vs {steps_ref.shape}) -- cannot average "
                f"mismatched seeds without interpolation."
            )

    def avg_field(key):
        return np.mean(
            np.stack([[r[key] for r in rows] for rows in all_rows], axis=0),
            axis=0,
        )

    fields = {k: avg_field(k) for k in ("local", "valid", "global", "memorized")}

    rgb = np.stack([fields[c] for c in channels], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    peak = rgb.max(axis=-1, keepdims=True)
    rgb = np.divide(rgb, peak, out=np.zeros_like(rgb), where=peak > 0)

    mem = fields["memorized"]
    value = np.clip(1.0 - mem_gain * mem, 0.0, 1.0)
    rgb = rgb * value[:, None] * CELL_DARKEN

    nd_all = np.stack([[r["n_distinct"] for r in rows] for rows in all_rows], axis=0)
    n_gen = max(float(nd_all.max()), 1.0)
    nd = nd_all.min(axis=0) if collapse_agg == "min" else nd_all.mean(axis=0)
    alpha = nd / n_gen

    return steps_ref, rgb, np.clip(alpha, 0.05, 1.0), mem, len(all_rows), fields


# --- plotting primitives -----------------------------------------------

def geometric_edges(x):
    x = np.asarray(x, dtype=float)
    if x.size == 1:
        return np.array([x[0] * 0.9, x[0] * 1.1])
    mid = np.sqrt(x[:-1] * x[1:])
    first = x[0] ** 2 / mid[0]
    last = x[-1] ** 2 / mid[-1]
    return np.concatenate([[first], mid, [last]])


NON_DISTINCT_ALPHA_CUTOFF = 0.15


def strip(ax, y0, height, edges, colors):
    for i in range(len(colors)):
        # Collapsed-sample cells (alpha at its floor) are left blank instead
        # of drawn as a faint, greyish wash.
        if len(colors[i]) == 4 and colors[i][3] <= NON_DISTINCT_ALPHA_CUTOFF:
            continue
        ax.add_patch(Rectangle(
            (edges[i], y0), edges[i + 1] - edges[i], height,
            facecolor=colors[i], edgecolor="none",
        ))


# alpha is clipped to a [0.05, 1.0] floor (see load_run) -- a cell sitting
# right at that floor has (near-)zero actually-distinct generations, i.e.
# it's in the "non distinct" / collapsed-sample zone, where a composition
# %% label would be reporting on a handful (or zero) of real samples. Those
# labels are skipped rather than drawn.

# How far into the cell (as a fraction of the cell's own log-width) the
# left-aligned label sits, so it doesn't touch the cell's left border.
LEFT_LABEL_INSET_FRAC = 0.04

# Extra leftward shift of every cell label, in characters (one character is
# taken as ~0.6 x fontsize points, the average glyph width of the default
# sans-serif font). Applied as a screen-space offset, so it is the same
# visual distance in every cell regardless of the cell's width on the log axis.
LABEL_SHIFT_CHARS = 2
CHAR_WIDTH_EM = 0.6
# The first and last labelled columns get their own shift, so the leftmost
# labels stay inside the strip (negative = move right) and the rightmost
# labels move further left, away from the strip's right end.
FIRST_LABEL_SHIFT_CHARS = -1
LAST_LABEL_SHIFT_CHARS = 4

# Vertical spacing between the G/H/B/M lines (1.0 = lines touching).
LABEL_LINESPACING = 0.95


def annotate_composition_pct(ax, y0, height, edges, valid, hallu, local, mem,
                             alpha=None, fontsize=7.5, max_labels=12):
    """Label at most max_labels points, evenly subsampled by INDEX -- not a
    per-cell width-fraction threshold, which is all-or-nothing when eval
    points are uniformly log-spaced (every cell then has the same width
    fraction, so there's nothing for a threshold to selectively skip).
    "G" prefixes the generalized/valid figure, matching H/B/M's own
    letter-prefix convention (previously shown bare, with no letter).
    Cells in the non-distinct/collapsed-sample zone (alpha at its clipped
    floor -- see NON_DISTINCT_ALPHA_CUTOFF) are skipped: their generations
    aren't distinct enough for the %% breakdown to mean anything.

    Labels are LEFT-aligned near each cell's left edge (was: centered on
    the cell's geometric midpoint) and drawn in BLACK (was: white) -- the
    outline stroke was correspondingly flipped to a light, semi-transparent
    white (was: dark) so the now-black text still reads clearly against the
    strip's own (often dark or saturated) cell colours."""
    valid, hallu, local, mem = map(np.asarray, (valid, hallu, local, mem))
    def shifted(chars):
        return offset_copy(ax.transData, fig=ax.figure,
                           x=-chars * CHAR_WIDTH_EM * fontsize, y=0, units="points")
    n = len(valid)
    if n <= max_labels:
        idxs = range(n)
    else:
        idxs = np.unique(np.linspace(0, n - 1, max_labels).astype(int))
    shown = [i for i in idxs
             if alpha is None or alpha[i] > NON_DISTINCT_ALPHA_CUTOFF]
    for i in shown:
        if i == shown[0]:
            chars = FIRST_LABEL_SHIFT_CHARS
        elif i == shown[-1]:
            chars = LAST_LABEL_SHIFT_CHARS
        else:
            chars = LABEL_SHIFT_CHARS
        # Left edge of the cell (in log space, matching the strip's log
        # x-scale) plus a small inset, rather than the geometric-mean
        # midpoint used for the old centered layout.
        log_lo, log_hi = np.log10(edges[i]), np.log10(edges[i + 1])
        x_left = 10 ** (log_lo + LEFT_LABEL_INSET_FRAC * (log_hi - log_lo))
        label = (f"G {valid[i]*100:.0f}\nH {hallu[i]*100:.0f}\n"
                f"B {local[i]*100:.0f}\nM {mem[i]*100:.0f}")
        ax.text(x_left, y0 + height / 2, label, ha="left", va="center",
                transform=shifted(chars),
                fontsize=fontsize, fontweight="bold", color="white", zorder=7,
                linespacing=LABEL_LINESPACING,
                path_effects=[pe.withStroke(linewidth=1.2, foreground="black")])


# "valid" is displayed as "generalized" everywhere a channel name is shown
# (legend swatches, the caption line) -- the underlying JSON field/CLI
# channel key is still "valid", only the display text changed.
DISPLAY_NAMES = {"local": "Bad", "global": "hallucinated", "valid": "generalized"}


def disp(name):
    return DISPLAY_NAMES.get(name, name)


# A cell that's "clearly dominated" by one or two channels still has a
# nonzero peak-normalized remainder on the others -- it never actually
# hits a literal 0. Softening the legend's zero entries up to this floor
# makes the swatches match what the strip can really produce, instead of
# showing pure primaries the plot itself never renders.
LEGEND_SOFTEN_FLOOR = 0.28


def soften(c):
    return tuple((v if v == 1 else LEGEND_SOFTEN_FLOOR) * CELL_DARKEN for v in c)


def colour_key(ax, channels):
    """Horizontal layout: one single row of swatches+labels -- the colour
    combos plus a single grey "memorization" swatch appended at the end of
    the same row (was: memorization shown as its own 4-swatch white->black
    gradient on a separate row below the combos)."""
    labels = [disp(c) for c in channels]
    combos = [
        (soften((1, 0, 0)), labels[0]), (soften((0, 1, 0)), labels[1]), (soften((0, 0, 1)), labels[2]),
        (soften((1, 1, 0)), f"{labels[0]}+{labels[1]}"),
        (soften((0, 1, 1)), f"{labels[1]}+{labels[2]}"),
        (soften((1, 0, 1)), f"{labels[0]}+{labels[2]}"),
        ((0.5, 0.5, 0.5), "memorization"),
    ]
    # Slot width scales with each label's own length (was: one fixed
    # slot_w for all entries, which overlapped the longer combo labels like
    # "generalized+hallucinated") -- a rough chars-to-width estimate at
    # fontsize 8, plus fixed room for the swatch itself.
    slot_ws = [max(2.6, 0.145 * len(lab) + 1.3) for _, lab in combos]
    xs = np.cumsum([0.0] + slot_ws[:-1])
    for x, (c, lab) in zip(xs, combos):
        ax.add_patch(Rectangle((x, 1.15), 0.7, 0.7, facecolor=c, edgecolor="0.75", lw=0.6))
        ax.text(x + 0.9, 1.5, lab, va="center", fontsize=8)
    total_w = xs[-1] + slot_ws[-1]

    ax.set_xlim(-0.3, total_w)
    ax.set_ylim(0.9, 2.1)
    ax.axis("off")


# --- main --------------------------------------------------------------

def parse_seeds(s):
    """'2,3,4' -> {2, 3, 4}; None -> None (all seeds found)."""
    return None if s is None else set(int(x) for x in s.split(","))


def parse_n_iter_map(s):
    if s is None:
        return None
    out = {}
    for pair in s.split(","):
        n_str, iter_str = pair.split(":")
        out[int(n_str)] = int(iter_str)
    return out


def render_seed_condition_grid(N_to_path, N: int, args, title, out_path):
    """One combined grid for ONE seed and ONE condition (before OR after):
    rows = N values (distinct here, since seed is fixed -- no duplicate-row
    problem the way --per-seed's mixed-N-and-seed rows had). Reuses the same
    geometric_edges/strip pipeline as main_single, just scoped to a single
    (seed, condition)'s worth of files."""
    channels = [c.strip() for c in args.channels.split(",")]
    Ns_sorted = sorted(N_to_path.keys())
    loaded = {}
    max_n_steps = 1
    for N in Ns_sorted:
        seed, p = N_to_path[N]
        steps, rgb, alpha, mem, n_seeds, fields = load_run([(seed, p)], N, args.mem_gain,
                                                           channels, args.collapse_agg)
        loaded[N] = (steps, rgb, alpha, fields)
        max_n_steps = max(max_n_steps, len(steps))

    fig_width = args.fig_width or max(10, 0.7 * max_n_steps)
    Ns_arr = np.array(Ns_sorted, dtype=float)
    y_edges = geometric_edges(Ns_arr)
    fig, ax = plt.subplots(figsize=(fig_width, 0.9 * len(Ns_sorted) + 1.6))

    xmin, xmax = np.inf, -np.inf
    yticks, ylabels = [], []
    for row, N in enumerate(Ns_sorted):
        steps, rgb, alpha, fields = loaded[N]
        edges = geometric_edges(steps)
        xmin, xmax = min(xmin, edges[0]), max(xmax, edges[-1])
        y0, y1 = y_edges[row], y_edges[row + 1]
        strip(ax, y0, y1 - y0, edges, [(*rgb[i], alpha[i]) for i in range(len(steps))])
        if args.annotate_valid_pct:
            annotate_composition_pct(ax, y0, y1 - y0, edges,
                                     fields["valid"], fields["global"], fields["local"], fields["memorized"],
                                     alpha=alpha,
                                     fontsize=args.valid_pct_fontsize, max_labels=len(steps))
        yticks.append(N)
        ylabels.append(f"N={N:g}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("gradient steps", fontsize=11)
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(1.0)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main_split_by_seed(args):
    """4 plots total (2 seeds x 2 conditions), each combining every N as
    stacked rows -- the transpose of main_beforeafter_per_seed's split-output
    (which splits by (N, seed) with before/after as columns)."""
    n_vals = (set(int(x) for x in args.n_vals.split(",")) if args.n_vals else None)
    results_root = Path(args.results_root)

    def build_n_iter_map(explicit, target_steps, label):
        m = parse_n_iter_map(explicit)
        if m is None and target_steps is not None:
            if not n_vals:
                raise SystemExit(f"--target-steps-{label} also needs --n-vals -- "
                                 f"or use --n-iter-map-{label} instead.")
            m = {N: expected_n_iter(N, target_steps, args.batch) for N in n_vals}
        return m

    n_iter_map_before = build_n_iter_map(args.n_iter_map_before, args.target_steps_before, "before")
    n_iter_map_after = build_n_iter_map(args.n_iter_map_after, args.target_steps_after, "after")
    runs_before = discover(results_root, args.reg_lambda_before, n_vals, n_iter_map=n_iter_map_before,
                           seeds=parse_seeds(args.seeds))
    runs_after = discover(results_root, args.reg_lambda_after, n_vals, n_iter_map=n_iter_map_after,
                          seeds=parse_seeds(args.seeds))

    all_seeds = sorted({seed for runs in (runs_before, runs_after)
                        for seed_paths in runs.values() for seed, _ in seed_paths})
    if not all_seeds:
        raise SystemExit(f"No train_eval_log files found under {results_root}")

    stem, suffix = Path(args.out).stem, Path(args.out).suffix
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in all_seeds:
        for cond_label, runs, lam in (("before", runs_before, args.reg_lambda_before),
                                      ("after", runs_after, args.reg_lambda_after)):
            N_to_path = {}
            for N, seed_paths in runs.items():
                for s, p in seed_paths:
                    if s == seed:
                        N_to_path[N] = (seed, p)
            if not N_to_path:
                print(f"seed {seed}, {cond_label} (reg_lambda={lam}): no data, skipping")
                continue
            title = f"seed {seed}, {cond_label} (reg_lambda={lam})"
            out_path = out_dir / f"{stem}_seed{seed}_{cond_label}{suffix}"
            render_seed_condition_grid(N_to_path, seed, args, title, out_path)


def main(args):
    if bool(args.reg_lambda_before is not None) != bool(args.reg_lambda_after is not None):
        raise SystemExit("--reg-lambda-before and --reg-lambda-after must be given together "
                         "(before/after modes) -- for a single lambda, use --reg-lambda "
                         "instead (with --per-seed if you want individual rows).")
    if args.reg_lambda_before is not None:
        if args.split_by_seed:
            main_split_by_seed(args)
        else:
            main_beforeafter_per_seed(args)
    else:
        if args.reg_lambda is None:
            raise SystemExit("Need either --reg-lambda (single-lambda mode) or both "
                             "--reg-lambda-before/-after (before/after modes).")
        main_single(args)


def main_beforeafter_per_seed(args):
    """N-wise before/after-regularization comparison, one row per (N, seed) --
    combines --per-seed's row granularity with combine_lagreg_ladder_beforeafter.py's
    before/after column layout, applied to the composition strip instead of a
    line plot. Needs a genuine subplot grid (one ax per cell) rather than the
    single shared-axis + row-offset approach main_single() uses, since strips
    at different (N, seed) rows are now independent cells, not stacked bands
    of one continuous axis."""
    channels = [c.strip() for c in args.channels.split(",")]
    if len(channels) != 3:
        raise ValueError(f"--channels needs exactly 3 keys, got {channels}")
    n_vals = (set(int(x) for x in args.n_vals.split(",")) if args.n_vals else None)
    results_root = Path(args.results_root)

    def build_n_iter_map(explicit, target_steps, label):
        m = parse_n_iter_map(explicit)
        if m is None and target_steps is not None:
            if not n_vals:
                raise SystemExit(f"--target-steps-{label} also needs --n-vals -- "
                                 f"or use --n-iter-map-{label} instead.")
            m = {N: expected_n_iter(N, target_steps, args.batch) for N in n_vals}
        return m

    n_iter_map_before = build_n_iter_map(args.n_iter_map_before, args.target_steps_before, "before")
    n_iter_map_after = build_n_iter_map(args.n_iter_map_after, args.target_steps_after, "after")

    runs_before = discover(results_root, args.reg_lambda_before, n_vals, n_iter_map=n_iter_map_before,
                           seeds=parse_seeds(args.seeds))
    runs_after = discover(results_root, args.reg_lambda_after, n_vals, n_iter_map=n_iter_map_after,
                          seeds=parse_seeds(args.seeds))

    all_Ns = set(runs_before) | set(runs_after)
    if not all_Ns:
        raise SystemExit(f"No train_eval_log files found for reg_lambda in "
                         f"{{{args.reg_lambda_before}, {args.reg_lambda_after}}} under {results_root}")

    print(f"before=reg_lambda={args.reg_lambda_before}  after=reg_lambda={args.reg_lambda_after}")
    for label, runs in (("before", runs_before), ("after", runs_after)):
        print(f"  {label}: found N = {sorted(runs)}")
        for N, seed_paths in runs.items():
            print(f"    N={N:>6}  {[(s, p.name) for s, p in seed_paths]}")

    # rows: every (N, seed) present in EITHER column, N descending, seed ascending
    seeds_by_n = {}
    for runs in (runs_before, runs_after):
        for N, seed_paths in runs.items():
            for seed, _ in seed_paths:
                seeds_by_n.setdefault(N, set()).add(seed)
    row_keys = [(N, seed) for N in sorted(seeds_by_n, reverse=True)
               for seed in sorted(seeds_by_n[N])]

    def find(runs, N, seed):
        for s, p in runs.get(N, []):
            if s == seed:
                return p
        return None

    xmin, xmax = np.inf, -np.inf
    loaded = {}
    for (N, seed) in row_keys:
        for label, runs in (("before", runs_before), ("after", runs_after)):
            p = find(runs, N, seed)
            if p is None:
                loaded[(N, seed, label)] = None
                continue
            steps, rgb, alpha, mem, n_seeds, fields = load_run([(seed, p)], N, args.mem_gain,
                                                               channels, args.collapse_agg)
            loaded[(N, seed, label)] = (steps, rgb, alpha, fields)
            xmin, xmax = min(xmin, steps.min()), max(xmax, steps.max())

    if args.split_output:
        stem, suffix = Path(args.out).stem, Path(args.out).suffix
        out_dir = Path(args.out).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        seeds_per_n = {}
        for (N, seed) in row_keys:
            seeds_per_n.setdefault(N, []).append(seed)

        for N in sorted(seeds_per_n, reverse=True):
            seeds = sorted(seeds_per_n[N])
            n_rows = len(seeds)
            max_n_steps = max((len(loaded[(N, s, l)][0]) for s in seeds for l in ("before", "after")
                              if loaded[(N, s, l)] is not None), default=1)
            auto_width = max(10, 0.7 * max_n_steps)   # ~0.35"/label/column x 2 columns
            fig_width = args.fig_width or auto_width
            fig, axes = plt.subplots(n_rows, 2, sharex=True, figsize=(fig_width, 3.0 * n_rows + 0.6))
            if n_rows == 1:
                axes = axes.reshape(1, 2)

            for row, seed in enumerate(seeds):
                for col, label in enumerate(("before", "after")):
                    ax = axes[row, col]
                    entry = loaded[(N, seed, label)]
                    lam = args.reg_lambda_before if label == "before" else args.reg_lambda_after
                    if row == 0:
                        ax.set_title(f"{label} (reg_lambda={lam})", fontsize=10)
                    if entry is None:
                        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                                transform=ax.transAxes, fontsize=9, color="0.5")
                    else:
                        steps, rgb, alpha, fields = entry
                        edges = geometric_edges(steps)
                        strip(ax, 0.0, 1.0, edges, [(*rgb[i], alpha[i]) for i in range(len(steps))])
                        if args.annotate_valid_pct:
                            annotate_composition_pct(ax, 0.0, 1.0, edges,
                                                     fields["valid"], fields["global"], fields["local"], fields["memorized"],
                                                     alpha=alpha,
                                                     fontsize=args.valid_pct_fontsize,
                                                     max_labels=len(steps))
                        ax.set_xlim(steps.min() * 0.9, steps.max() * 1.1)
                    ax.set_xscale("log")
                    ax.set_ylim(0, 1)
                    ax.set_yticks([])
                    for s in ("top", "right", "bottom", "left"):
                        ax.spines[s].set_visible(True)
                        ax.spines[s].set_linewidth(1.0)
                    if col == 0:
                        ax.set_ylabel(f"seed {seed}", fontsize=9, rotation=0, ha="right", va="center")
                    if row == n_rows - 1:
                        ax.set_xlabel("gradient steps", fontsize=10)

            fig.suptitle(f"N={N:g} -- reg_lambda={args.reg_lambda_before} vs "
                        f"{args.reg_lambda_after}", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            out_path = out_dir / f"{stem}_N{N}{suffix}"
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {out_path}")
        return

    fig, axes = plt.subplots(len(row_keys), 2, sharex=True,
                             figsize=(args.fig_width or 20, 1.1 * len(row_keys) + 1.4))
    if len(row_keys) == 1:
        axes = axes.reshape(1, 2)

    for row, (N, seed) in enumerate(row_keys):
        for col, label in enumerate(("before", "after")):
            ax = axes[row, col]
            entry = loaded[(N, seed, label)]
            lam = args.reg_lambda_before if label == "before" else args.reg_lambda_after
            title = f"{label} (reg_lambda={lam})" if row == 0 else ""
            if entry is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="0.5")
            else:
                steps, rgb, alpha, fields = entry
                edges = geometric_edges(steps)
                strip(ax, 0.0, 1.0, edges, [(*rgb[i], alpha[i]) for i in range(len(steps))])
                if args.annotate_valid_pct:
                    annotate_composition_pct(ax, 0.0, 1.0, edges,
                                             fields["valid"], fields["global"], fields["local"], fields["memorized"],
                                             alpha=alpha,
                                             fontsize=args.valid_pct_fontsize,
                                             max_labels=args.max_labels)
            ax.set_xscale("log")
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=9)
            for s in ("top", "right", "bottom", "left"):
                ax.spines[s].set_visible(True)
                ax.spines[s].set_linewidth(1.0)
            if col == 0:
                ax.set_ylabel(f"N={N:g} (seed {seed})", fontsize=9, rotation=0, ha="right", va="center")

    for ax in axes.flat:
        ax.set_xlim(xmin, xmax)
    for col in range(2):
        axes[-1, col].set_xlabel("gradient steps", fontsize=11)

    fig.suptitle(f"Sample composition, reg_lambda={args.reg_lambda_before} vs "
                f"{args.reg_lambda_after}, per seed", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


def main_single(args):
    channels = [c.strip() for c in args.channels.split(",")]
    if len(channels) != 3:
        raise ValueError(f"--channels needs exactly 3 keys, got {channels}")

    n_vals = (set(int(x) for x in args.n_vals.split(",")) if args.n_vals else None)
    n_iter_map = parse_n_iter_map(args.n_iter_map)
    if n_iter_map is None and args.target_steps is not None:
        if not n_vals:
            raise SystemExit("--target-steps also needs --n-vals (to know which N "
                             "to compute n_iter for) -- or use --n-iter-map instead.")
        n_iter_map = {N: expected_n_iter(N, args.target_steps, args.batch) for N in n_vals}

    runs = discover(Path(args.results_root), args.reg_lambda, n_vals, n_iter_map=n_iter_map,
                    seeds=parse_seeds(args.seeds))
    if not runs:
        raise SystemExit(
            f"No train_eval_log_reg{reg_tag(args.reg_lambda)}.json found under "
            f"{args.results_root}" + (f" for N in {sorted(n_vals)}" if n_vals else "")
        )

    print(f"reg_lambda={args.reg_lambda} -- found {len(runs)} N value(s): {sorted(runs)}")
    for N, seed_paths in runs.items():
        print(f"  N={N:>6}  {len(seed_paths)} seed file(s): "
              f"{[(s, p.name) for s, p in seed_paths]}")

    if args.require_seeds is not None:
        short = {N: len(sp) for N, sp in runs.items() if len(sp) < args.require_seeds}
        if short:
            raise SystemExit(
                f"--require-seeds {args.require_seeds} but these N have fewer: "
                f"{short}."
            )
    if n_vals is not None:
        missing = n_vals - set(runs)
        if missing:
            print(f"WARNING: requested N values with NO data found: {sorted(missing)} "
                  f"-- these rows will simply be absent from the plot, not blank rows.")

    # Build the list of plot rows. Default: one row per N, all its seeds
    # averaged inside load_run (unchanged behaviour). --per-seed: one row
    # per (N, seed), each passed to load_run as a single-element list --
    # load_run's own averaging is then a no-op (mean of one thing is itself),
    # so no separate single-seed code path is needed.
    if args.per_seed:
        panel_rows = []
        for N in sorted(runs, reverse=True):          # largest N first, seeds in seed order within
            for seed, p in sorted(runs[N]):
                panel_rows.append((f"N={N:g} (seed {seed})", N, [(seed, p)]))
    else:
        panel_rows = [(None, N, seed_paths) for N, seed_paths in runs.items()]  # label set below

    # Two-row gridspec: main strip on top, colour key below (was: a narrow
    # side column next to the strip) -- height_ratios keeps the key short
    # relative to the main plot regardless of how many N rows it has.
    fig = plt.figure(figsize=(args.fig_width or 13, 0.8 * (0.85 * len(panel_rows) + 2.3)))
    gs = fig.add_gridspec(2, 1, height_ratios=[max(len(panel_rows), 1), 0.7], hspace=0.2)
    ax = fig.add_subplot(gs[0])
    axk = fig.add_subplot(gs[1])

    # Row edges: geometric-by-N-value only makes sense when every row has a
    # DISTINCT N (default mode) -- with --per-seed, multiple rows share the
    # same N, and geometric_edges would produce zero-width strips for
    # duplicates. Use uniform ordinal spacing (row i occupies [i, i+1])
    # whenever rows aren't guaranteed distinct.
    if args.per_seed:
        n_rows = len(panel_rows)
        y_edges = np.arange(n_rows + 1, dtype=float)
    else:
        Ns = np.array(sorted(runs.keys()), dtype=float)
        y_edges = geometric_edges(Ns)

    yticks, ylabels = [], []
    xmin, xmax = np.inf, -np.inf

    for row, (label, N, seed_paths) in enumerate(panel_rows):
        steps, rgb, alpha, mem, n_seeds, fields = load_run(
            seed_paths, N, args.mem_gain, channels, args.collapse_agg)
        print(f"  N={N:>6}  step range {steps[0]:.0f} .. {steps[-1]:.0f}  "
              f"({n_seeds} seed(s) averaged)" if not args.per_seed else
              f"  {label}  step range {steps[0]:.0f} .. {steps[-1]:.0f}")
        edges = geometric_edges(steps)
        xmin, xmax = min(xmin, edges[0]), max(xmax, edges[-1])

        y0, y1 = y_edges[row], y_edges[row + 1]
        strip(ax, y0, y1 - y0, edges, [(*rgb[i], alpha[i]) for i in range(len(steps))])
        if args.annotate_valid_pct:
            annotate_composition_pct(ax, y0, y1 - y0, edges,
                                     fields["valid"], fields["global"], fields["local"], fields["memorized"],
                                     alpha=alpha,
                                     fontsize=args.valid_pct_fontsize,
                                     max_labels=args.max_labels)

        if args.per_seed:
            yticks.append((y0 + y1) / 2)
            ylabels.append(label)
        else:
            yticks.append(N)
            ylabels.append(f"{N:g}")

    ax.set_xscale("log")
    if not args.per_seed:
        ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_yticks([], minor=True)  # no minor-tick dashes between the N labels
    ax.set_ylabel("training size", fontsize=12)
    ax.set_xlabel("gradient steps", fontsize=12)
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(1.0)

    # No title, and no caption/subtitle line above the axes either (was:
    # ax.set_title(...) plus a descriptive "colour = additive mix of..."
    # caption) -- the colour_key below the plot now carries that
    # information on its own.
    colour_key(axk, channels)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--results-root", type=str, required=True,
                   help="Root to search (e.g. results_lagreg_sweep).")
    p.add_argument("--reg-lambda", type=float, default=None,
                   help="Single-lambda mode: which reg_lambda's runs to plot, e.g. 0.1. "
                        "Mutually exclusive with --reg-lambda-before/-after.")
    p.add_argument("--reg-lambda-before", type=float, default=None,
                   help="Before/after-per-seed grid mode: the 'before' column's "
                        "reg_lambda. Must be given together with --reg-lambda-after.")
    p.add_argument("--reg-lambda-after", type=float, default=None,
                   help="Before/after-per-seed grid mode: the 'after' column's "
                        "reg_lambda. Must be given together with --reg-lambda-before.")
    p.add_argument("--n-vals", type=str, default=None,
                   help="Comma-separated N values to include, e.g. "
                        "4096,8192,16384,65536. Default: all N found.")
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated seeds to include, e.g. 2,3,4. Default: all seeds "
                        "found. Applies in every mode (single, before/after, split).")
    p.add_argument("--n-iter-map", type=str, default=None,
                   help="Single-lambda mode: 'N:n_iter,N:n_iter,...' pinning the exact "
                        "n_iter to accept per N, read straight off the paths printed "
                        "when a duplicate-seed error fires (e.g. '512:200000,1024:100000,"
                        "...'). The authoritative, formula-free disambiguator when "
                        "results_root has multiple sweep generations.")
    p.add_argument("--target-steps", type=int, default=None,
                   help="Single-lambda mode: OPTIONAL CONVENIENCE, auto-generates "
                        "--n-iter-map via run_lagreg_nsweep_full.sh's own arithmetic. "
                        "Needs --n-vals too.")
    p.add_argument("--n-iter-map-before", type=str, default=None,
                   help="Before/after mode analog of --n-iter-map, for the 'before' column.")
    p.add_argument("--n-iter-map-after", type=str, default=None,
                   help="Before/after mode analog of --n-iter-map, for the 'after' column.")
    p.add_argument("--target-steps-before", type=int, default=None,
                   help="Before/after mode analog of --target-steps, for the 'before' column.")
    p.add_argument("--target-steps-after", type=int, default=None,
                   help="Before/after mode analog of --target-steps, for the 'after' column.")
    p.add_argument("--batch", type=int, default=512,
                   help="BATCH from the sweep script (default 512) -- only used "
                        "with --target-steps.")
    p.add_argument("--channels", type=str, default="local,valid,global")
    p.add_argument("--mem-gain", type=float, default=1.0)
    p.add_argument("--collapse-agg", type=str, default="min", choices=["min", "mean"])
    p.add_argument("--require-seeds", type=int, default=None)
    p.add_argument("--annotate-valid-pct", action="store_true", default=True)
    p.add_argument("--no-annotate-valid-pct", dest="annotate_valid_pct", action="store_false")
    p.add_argument("--fig-width", type=float, default=None,
                   help="Figure width in inches. Default: 13 in single-lambda mode, "
                        "20 in before/after mode (which needs more since each strip "
                        "only gets half the width, split across two columns).")
    p.add_argument("--valid-pct-fontsize", type=float, default=7.5)
    p.add_argument("--max-labels", type=int, default=12,
                   help="Show at most this many composition-%% labels per strip, "
                        "evenly subsampled by index -- avoids illegible overlapping "
                        "text when eval points are densely, uniformly log-spaced "
                        "(a width-fraction threshold can't selectively thin those, "
                        "since every cell has the same width fraction). Increase for "
                        "more detail if the figure is wide enough to fit it.")
    p.add_argument("--split-by-seed", action="store_true",
                   help="Before/after mode: write one file per (seed, condition) -- "
                        "e.g. 2 seeds x before/after = 4 files -- each combining every "
                        "N as stacked rows (the transpose of --split-output, which "
                        "splits by (N, seed) with before/after as columns).")
    p.add_argument("--split-output", action="store_true",
                   help="Before/after mode: write ONE FILE PER N instead of a single "
                        "combined grid -- each file stacks that N's seeds as rows, "
                        "before/after as columns, full width, EVERY eval-point label "
                        "shown (no --max-labels subsampling). Output filenames are "
                        "derived from --out by inserting _N<N> before the extension.")
    p.add_argument("--per-seed", action="store_true",
                   help="One row per (N, seed) instead of seed-averaged rows per N. "
                        "Row spacing switches from log-by-N to uniform ordinal, since "
                        "multiple rows can share the same N.")
    p.add_argument("--out", type=str, default="../plots/lagreg_composition.png")
    main(p.parse_args())