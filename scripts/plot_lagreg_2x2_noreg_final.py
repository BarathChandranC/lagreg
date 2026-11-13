"""
plot_lagreg_2x2_noreg_final.py

n_global/n_local vs gradient steps, NO-REGULARIZATION case only
(reg_lambda=0.0), laid out as a FIXED 1x4 ROW:

    N=65536   N=1024   N=16384   N=512

(Previously a 2x2 grid -- N=65536/1024 on top, N=16384/512 below --
changed to a single row of 4 side by side per request; FIXED_LAYOUT is
now a single-row list-of-lists so the rest of the plotting loop below
only had to change from axes[row, col] to axes[col], nothing about the
data loading/discovery changed.)

Built directly off the attached combine_lagreg_ladder_beforeafter.py --
discover()/expected_n_iter()/load_run() are copied verbatim from that file
(same target_steps+batch disambiguation mechanism, same duplicate-seed
detection, same rglob-based file discovery), just called ONCE (one
reg_lambda, not a before/after pair) and rendered into the fixed 1x4 layout
above instead of an N-rows x 2-columns grid.

NEW: test loss overlay, elbow marker, and local-onset lines for N=1024/512.
  --overlay-pop-loss (on by default) draws each cell's seed-averaged test
  loss ("loss_total" from the ladder_pop_*.json rows -- always the pop
  file, i.e. the held-out/test-set loss) on a second y-axis, as a gray
  dashed line labeled "test loss". This is always read from the *_pop_*
  files regardless of --source -- n_global/n_local can be plotted from
  either source, but "test loss" only means one thing, so it is not
  re-derived from --source=emp. If --source is already "pop", the same
  already-discovered/loaded files are reused (no duplicate disk reads);
  if --source is "emp", the pop files are discovered and loaded
  separately, once.

  The elbow of that test loss curve (same max-distance-from-chord method
  as combine_loss_runs.py's find_elbow) is marked in every cell with a
  gray dotted vertical line labeled "T_elb" just above the axes.

  n_local's onset-of-decline (per onset_step/draw_onset_lines, already
  used for N=65536/16384) is now also marked for N=1024 and N=512 -- just
  the red n_local line there, not n_global's.

Styling, per the running list of requests in this conversation:
  - no background gridlines
  - full black box (all four spines, thick)
  - sparse, evenly-spaced markers (not one per eval point)
  - legend labels are real LaTeX n_global / n_local
  - y-axis labeled "rescaled risk"
  - the region BETWEEN the two curves (the lag) is shaded
  - minor ticks off the log x-axis
  - bigger, black (not gray) text throughout

Usage (run from scripts/, no GPU needed)::

    python plot_lagreg_2x2_noreg_final.py \\
        --results-root ../results_lagreg_sweep \\
        --reg-lambda 0.0 --source pop --require-seeds 2 \\
        --target-steps 200000 --batch 512 \\
        --out ../plots/lagreg_2x2_noreg.png
"""
from argparse import ArgumentParser
from pathlib import Path
import json
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


FIXED_LAYOUT = [[65536, 16384, 1024, 512]]  # single row, descending N left to right


def reg_tag(reg_lambda: float) -> str:
    return str(reg_lambda).replace(".", "p")


def build_tagged_re(source: str, tag: str) -> re.Pattern:
    return re.compile(rf"^ladder_{source}_seed(\d+)_N(\d+)_reg{re.escape(tag)}\.json$")


def expected_n_iter(N: int, target_steps: int, batch: int) -> int:
    """Reproduces run_lagreg_nsweep_full.sh's own arithmetic exactly:
    STEPS_PER_EPOCH=$(( (N + BATCH - 1) / BATCH )); N_ITER=$(( TARGET_STEPS / STEPS_PER_EPOCH )).
    n_iter is baked into save_path's directory name (test_model_script_<n_iter>_<N>_...)
    and differs PER N even within one sweep generation -- a single fixed
    --path-filter substring cannot disambiguate generations across multiple N
    the way it could for one N in isolation."""
    steps_per_epoch = -(-N // batch)   # ceil division, matches bash's (N+BATCH-1)/BATCH
    return target_steps // steps_per_epoch


def discover(results_root: Path, source: str, reg_lambda: float, n_vals=None,
            path_filter=None, target_steps=None, batch=None):
    """Find every ladder_{source}_seed<S>_N<M>_reg{tag}.json under
    results_root for one reg_lambda. Returns {N: [(seed, path), ...]}.

    Copied verbatim from the attached combine_lagreg_ladder_beforeafter.py
    (same rglob-collision handling, same target_steps+batch disambiguation),
    just called once here since there's only one condition (no before/after
    pair) in this script.
    """
    tag = reg_tag(reg_lambda)
    tagged_re = build_tagged_re(source, tag)
    runs = {}
    skipped = []
    filtered_out_n_iter = []
    for f in sorted(results_root.rglob(f"ladder_{source}_seed*_N*_reg{tag}.json")):
        if path_filter is not None and path_filter not in str(f):
            continue
        m = tagged_re.match(f.name)
        if m is None:
            skipped.append(f)
            continue
        seed, N = int(m.group(1)), int(m.group(2))
        if n_vals is not None and N not in n_vals:
            continue
        if target_steps is not None and batch is not None:
            n_iter = expected_n_iter(N, target_steps, batch)
            needle = f"test_model_script_{n_iter}_{N}_"
            if needle not in str(f):
                filtered_out_n_iter.append((f, needle))
                continue
        runs.setdefault(N, []).append((seed, f))
    if skipped:
        print(f"skipped {len(skipped)} file(s) not matching "
              f"'ladder_{source}_seed<S>_N<M>_reg{tag}.json':")
        for p in skipped:
            print(f"    {p}")
    if filtered_out_n_iter:
        print(f"filtered out {len(filtered_out_n_iter)} file(s) not matching the "
              f"expected n_iter for their N (target_steps={target_steps}, batch={batch}):")
        for p, needle in filtered_out_n_iter:
            print(f"    {p}  (expected '{needle}' in path)")

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
                f"{results_root} -- more than one sweep generation has written "
                f"into this results_root and rglob is picking up both:\n{lines}\n"
                f"Resolve with --target-steps + --batch (computes the right "
                f"n_iter per N automatically), --path-filter (only works for a "
                f"single N), or remove the stale generation's directory."
            )
    return dict(sorted(runs.items()))


def load_run(seed_paths, N: int):
    """Load every seed's ladder file for this N, align on the shared step
    grid, return seed-mean/std of n_global and n_local. Copied verbatim from
    the attachment."""
    all_rows = []
    for seed, p in seed_paths:
        with open(p) as fh:
            rows = json.load(fh)
        if not rows:
            raise ValueError(f"{p} is empty")
        for key in ("n_global", "n_local"):
            if key not in rows[0]:
                raise KeyError(f"{p}: missing {key!r}; available keys: {sorted(rows[0].keys())}")
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

    out = {}
    for key in ("n_global", "n_local"):
        vals = np.stack([[r[key] for r in rows] for rows in all_rows], axis=0)
        out[key] = (vals.mean(axis=0), vals.std(axis=0))
    return steps_ref, out, len(all_rows)


def load_pop_loss(seed_paths, N: int):
    """Load every seed's ladder_pop file for this N, align on the shared
    step grid, return seed-mean/std of population loss ("loss_total").
    Mirrors load_run's alignment logic, but for the single loss_total key --
    kept separate from load_run (which is copied verbatim from the
    attachment and left untouched) rather than folding loss_total into it.
    Returns None if any file is missing "loss_total" (so the overlay can be
    skipped gracefully for that N rather than crashing the whole plot)."""
    all_rows = []
    for seed, p in seed_paths:
        with open(p) as fh:
            rows = json.load(fh)
        if not rows:
            raise ValueError(f"{p} is empty")
        if "loss_total" not in rows[0]:
            print(f"  N={N}: {p} has no 'loss_total' key -- skipping population "
                  f"loss overlay for this N (available keys: {sorted(rows[0].keys())})")
            return None
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

    vals = np.stack([[r["loss_total"] for r in rows] for rows in all_rows], axis=0)
    return steps_ref, vals.mean(axis=0), vals.std(axis=0)


# Validated via the dataviz skill's validate_palette.js (light mode: CVD dE
# 21.6, normal-vision dE 32.3, both clear of the floors) -- n_global always
# blue, n_local always red.
COLOR_GLOBAL = "#2a78d6"
COLOR_LOCAL = "#e34948"
COLOR_TEST_LOSS = "#444444"


def find_elbow(x, y, log_x=True, log_y=True):
    """Classic max-distance-from-chord elbow/knee detector (same method used
    in combine_loss_runs.py): normalize x/y to [0,1] (in log space matching
    the plotted scale), draw the chord between the first/last points, return
    the x of the point farthest (perpendicular) from that chord."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xt = np.log10(x) if log_x else x
    yt = np.log10(y) if (log_y and np.all(y > 0)) else y
    x_span = xt.max() - xt.min()
    y_span = yt.max() - yt.min()
    xn = (xt - xt.min()) / x_span if x_span > 0 else np.zeros_like(xt)
    yn = (yt - yt.min()) / y_span if y_span > 0 else np.zeros_like(yt)
    x1, y1, x2, y2 = xn[0], yn[0], xn[-1], yn[-1]
    num = np.abs((x2 - x1) * (yn - y1) - (y2 - y1) * (xn - x1))
    den = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + 1e-12
    dist = num / den
    return float(x[int(np.argmax(dist))])


def onset_step(steps, values, frac=0.1):
    """First step at which `values` has dropped `frac` of the way from its
    initial (early-training plateau) level toward its own minimum -- used
    as the operational definition of "onset of minimization" for the
    two-headed-arrow annotation. frac=0.1 means: the point is flagged as
    soon as the curve has fallen 10% of its own total drop, not at the
    steepest point or the eventual floor -- deliberately an EARLY marker of
    "started decreasing", not "finished decreasing" or "decreasing fastest".
    Values are averaged over the first 2 points (not just steps[0]) as a
    small guard against a noisy single first reading setting v0."""
    v0 = float(np.mean(values[:2]))
    vmin = float(values.min())
    if v0 <= vmin:
        return float(steps[0])
    thresh = v0 - frac * (v0 - vmin)
    below = np.where(values <= thresh)[0]
    idx = int(below[0]) if len(below) else len(values) - 1
    return float(steps[idx])


def draw_onset_lines(ax, steps, g_mean, l_mean, frac=0.1):
    """Vertical dotted lines marking the onset of each curve's decline (per
    onset_step above) -- red at n_local's onset, blue at n_global's, using
    the same COLOR_LOCAL/COLOR_GLOBAL as the curves themselves."""
    x_local = onset_step(steps, l_mean, frac)
    x_global = onset_step(steps, g_mean, frac)
    ax.axvline(x_local, color=COLOR_LOCAL, ls=":", lw=1.8, zorder=4)
    ax.axvline(x_global, color=COLOR_GLOBAL, ls=":", lw=1.8, zorder=4)


def draw_local_onset_line(ax, steps, l_mean, frac=0.1):
    """Just the n_local onset line (red dotted) -- used for N=1024/512 where
    only n_local's onset is marked, not n_global's."""
    x_local = onset_step(steps, l_mean, frac)
    ax.axvline(x_local, color=COLOR_LOCAL, ls=":", lw=1.8, zorder=4)


def draw_elbow_line(ax, steps, values, label=r"$T_{\mathrm{elb}}$", log_x=True, log_y=True):
    """Vertical dotted line at the elbow of `values` (find_elbow above),
    labeled just above the axes so it doesn't collide with the curves or
    legend. Drawn on `ax` (not the twin axis) so it spans the full subplot
    height regardless of which axis the labeled curve is plotted on."""
    x_elb = find_elbow(steps, values, log_x=log_x, log_y=log_y)
    ax.axvline(x_elb, color=COLOR_TEST_LOSS, ls=":", lw=1.6, zorder=2)
    trans = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
    # Below the x-axis itself (y<0 in axes fraction), between the tick
    # numbers and the "gradient steps" label -- clip_on=False so it isn't
    # cut off by the axes bounding box.
    ax.text(x_elb, -0.13, label, transform=trans, ha="center", va="top",
             fontsize=15, color=COLOR_TEST_LOSS, style="italic", zorder=5,
             clip_on=False)


def plot_cell(ax, steps, out, n_seeds, title, n_markers=9,
              test_loss=None, show_legend=True, mark_elbow=False, test_loss_ylim=None):
    g_mean, g_std = out["n_global"]
    l_mean, l_std = out["n_local"]
    markevery = max(1, len(steps) // n_markers)

    # Shade the region BETWEEN the two curves (the lag) -- the only shading
    # in this plot now. Per-seed std ribbons in blue/red were removed on
    # request (g_std/l_std are still computed and returned by load_run, just
    # not drawn here).
    ax.fill_between(steps, l_mean, g_mean, color="0.55", alpha=0.18, zorder=0)

    ax.plot(steps, g_mean, color=COLOR_GLOBAL, marker="o", ms=7.5, lw=2.2,
            markevery=markevery, label=r"$n_{\mathrm{global}}$", zorder=3)
    ax.plot(steps, l_mean, color=COLOR_LOCAL, marker="s", ms=7.5, lw=2.2,
            markevery=markevery, label=r"$n_{\mathrm{local}}$", zorder=3)

    ax.axhline(0.0, color="0.75", lw=0.8, ls=":", zorder=0)
    ax.set_xscale("log")
    ax.minorticks_off()   # no small unlabeled ticks between the decades
    ax.set_title(title, fontsize=13, color="black")
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color("black")
        ax.spines[s].set_linewidth(1.6)
    ax.tick_params(colors="black", labelsize=11, width=1.3)

    ax2 = None
    if test_loss is not None:
        tl_steps, tl_mean, tl_std = test_loss
        ax2 = ax.twinx()
        ax2.plot(tl_steps, tl_mean, color=COLOR_TEST_LOSS, ls="--", lw=1.8,
                 marker="^", ms=5.5, markevery=max(1, len(tl_steps) // n_markers),
                 label="test loss", zorder=2)
        if test_loss_ylim is not None:
            ax2.set_ylim(*test_loss_ylim)
        ax2.tick_params(colors="black", labelsize=10, width=1.1)
        for s in ("top", "right", "left", "bottom"):
            ax2.spines[s].set_visible(s == "right")
            if s == "right":
                ax2.spines[s].set_color("black")
                ax2.spines[s].set_linewidth(1.6)
        if mark_elbow:
            draw_elbow_line(ax, tl_steps, tl_mean)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if ax2 is not None:
            h2, l2 = ax2.get_legend_handles_labels()
            handles, labels = handles + h2, labels + l2
        ax.legend(handles, labels, fontsize=10.5, loc="upper right", frameon=False)

    return ax2


def main(args):
    results_root = Path(args.results_root)
    n_vals_needed = {N for row in FIXED_LAYOUT for N in row}
    runs = discover(results_root, args.source, args.reg_lambda, n_vals_needed,
                    path_filter=args.path_filter,
                    target_steps=args.target_steps, batch=args.batch)

    missing = n_vals_needed - set(runs)
    if missing:
        raise SystemExit(f"No ladder_{args.source} files found for N={sorted(missing)} "
                         f"(reg_lambda={args.reg_lambda}) under {results_root} -- "
                         f"needed for the fixed 1x4 layout {FIXED_LAYOUT}.")

    print(f"source={args.source}  results_root={results_root}  reg_lambda={args.reg_lambda}")
    for N in sorted(n_vals_needed):
        print(f"  N={N:>6}  {len(runs[N])} seed file(s): "
              f"{[(s, p.name) for s, p in runs[N]]}")

    if args.require_seeds is not None:
        short = {N: len(runs[N]) for N in n_vals_needed if len(runs[N]) < args.require_seeds}
        if short:
            raise SystemExit(f"--require-seeds {args.require_seeds} but short seeds for: {short}.")

    # Population loss is always read from the *_pop_* files regardless of
    # --source -- reuse `runs` when --source is already "pop" (no duplicate
    # discovery/reads), otherwise discover the pop files separately once.
    pop_runs = None
    if args.overlay_pop_loss:
        if args.source == "pop":
            pop_runs = runs
        else:
            pop_runs = discover(results_root, "pop", args.reg_lambda, n_vals_needed,
                                path_filter=args.path_filter,
                                target_steps=args.target_steps, batch=args.batch)
            missing_pop = n_vals_needed - set(pop_runs)
            if missing_pop:
                print(f"WARNING: --overlay-pop-loss requested but no ladder_pop files "
                      f"found for N={sorted(missing_pop)} -- test loss will be "
                      f"omitted for those cells.")

    loaded = {}
    xmin, xmax = np.inf, -np.inf
    ymin, ymax = np.inf, -np.inf
    for N in n_vals_needed:
        steps, out, n_seeds = load_run(runs[N], N)
        loaded[N] = (steps, out, n_seeds)
        xmin, xmax = min(xmin, steps.min()), max(xmax, steps.max())
        for key in ("n_global", "n_local"):
            mean, std = out[key]
            lo, hi = mean.min(), mean.max()
            ymin, ymax = min(ymin, lo), max(ymax, hi)
    pad = 0.05 * (ymax - ymin)
    ylim = (ymin - pad, ymax + pad)

    test_loss_loaded = {}
    tl_ymin, tl_ymax = np.inf, -np.inf
    if args.overlay_pop_loss and pop_runs is not None:
        for N in n_vals_needed:
            if N not in pop_runs:
                test_loss_loaded[N] = None
                continue
            test_loss_loaded[N] = load_pop_loss(pop_runs[N], N)
            if test_loss_loaded[N] is not None:
                _, tl_mean, _ = test_loss_loaded[N]
                tl_ymin = min(tl_ymin, tl_mean.min())
                tl_ymax = max(tl_ymax, tl_mean.max())
    # Same y-range on every cell's test-loss (right) axis, matching how the
    # rescaled-risk (left) axis is already shared -- otherwise each cell
    # auto-scales independently and the four right axes aren't comparable.
    test_loss_ylim = None
    if np.isfinite(tl_ymin) and np.isfinite(tl_ymax):
        tl_pad = 0.05 * (tl_ymax - tl_ymin)
        test_loss_ylim = (tl_ymin - tl_pad, tl_ymax + tl_pad)

    # Single row of 4 -- was 2x2 (see module docstring). len(FIXED_LAYOUT)==1
    # always, so `axes` is a flat length-4 array (squeeze=False avoided on
    # purpose: with exactly 1 row, plt.subplots already returns a 1-D array
    # for the 4 columns, which is exactly the indexing used below).
    n_cols = len(FIXED_LAYOUT[0])
    fig, axes = plt.subplots(1, n_cols, sharex=True, figsize=(4.6 * n_cols, 5.2))

    # NOTE: onset lines (draw_onset_lines/draw_local_onset_line) and the
    # elbow marker (draw_elbow_line, via mark_elbow) are intentionally not
    # drawn here -- user asked for the dashed population-loss overlay with
    # no dotted vertical lines. mark_elbow is hardcoded False below.
    for col, N in enumerate(FIXED_LAYOUT[0]):
        ax = axes[col]
        steps, out, n_seeds = loaded[N]
        test_loss = test_loss_loaded.get(N) if args.overlay_pop_loss else None
        ax2 = plot_cell(ax, steps, out, n_seeds, f"N={N:g}",
                       n_markers=args.n_markers, test_loss=test_loss,
                       test_loss_ylim=test_loss_ylim, mark_elbow=False)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(*ylim)

        ax.set_xlabel("gradient steps", fontsize=13, color="black")

    fig.text(0.005, 0.5, "rescaled risk", fontsize=14, color="black",
             rotation=90, va="center", ha="left")
    if args.overlay_pop_loss:
        fig.text(0.995, 0.5, "test loss", fontsize=14, color=COLOR_TEST_LOSS,
                 rotation=90, va="center", ha="right")
    fig.suptitle(f"n_global / n_local, reg_lambda={args.reg_lambda} "
                f"(source={args.source}, seed-averaged, no regularization)",
                fontsize=15, color="black")
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.90))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=str, required=True,
                   help="Root of the lagreg sweep (run_lagreg_nsweep_full.sh's "
                        "--results-dir).")
    p.add_argument("--reg-lambda", type=float, default=0.0,
                   help="Single condition to plot. Default 0.0 (no regularization), "
                        "which is this script's entire purpose -- there is "
                        "deliberately no before/after pair here.")
    p.add_argument("--path-filter", type=str, default=None,
                   help="Keep only paths containing this substring -- only "
                        "disambiguates a single N; for multiple N (this script "
                        "always needs all 4) use --target-steps + --batch instead.")
    p.add_argument("--target-steps", type=int, default=None,
                   help="This sweep generation's TARGET_STEPS, e.g. 200000 -- "
                        "computes the expected n_iter per N (matching "
                        "run_lagreg_nsweep_full.sh's own arithmetic) and filters "
                        "out any other generation's files sharing this "
                        "results_root. Requires --batch. Needed as soon as "
                        "results_root has more than one sweep generation.")
    p.add_argument("--batch", type=int, default=512,
                   help="BATCH from the sweep script -- used with --target-steps.")
    p.add_argument("--source", type=str, default="pop", choices=["emp", "pop"])
    p.add_argument("--require-seeds", type=int, default=None)
    p.add_argument("--n-markers", type=int, default=9,
                   help="Roughly how many markers to draw per line, evenly spaced "
                        "by index via markevery.")
    p.add_argument("--overlay-pop-loss", dest="overlay_pop_loss", action="store_true",
                   default=True,
                   help="Overlay each cell's seed-averaged test loss "
                        "(loss_total from ladder_pop_*.json) on a second y-axis, "
                        "as a gray dashed line, with its elbow marked as T_elb. "
                        "On by default.")
    p.add_argument("--no-overlay-pop-loss", dest="overlay_pop_loss", action="store_false",
                   help="Disable the test loss overlay (and its T_elb marker).")
    p.add_argument("--out", type=str, default="../plots/lagreg_1x4_noreg.png")
    main(p.parse_args())