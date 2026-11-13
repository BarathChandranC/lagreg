"""
combine_lagreg_ladder_beforeafter.py

n_global/n_local vs gradient steps, seed-averaged, arranged as a 2-rows x
N-columns grid: columns are training-set size N, side by side left to
right (largest N on the left, same convention as combine_loss_runs.py);
rows are "before regularization" (top) and "after regularization"
(bottom). Each cell plots both n_global and n_local as two lines, same as
combine_lagreg_ladder.py.

LAYOUT (this revision): transposed from the previous N-rows x 2-columns
grid (which stacked N values vertically, getting tall fast) to 2-rows x
N-columns instead, so N values sit side by side horizontally and the
figure stays wide/short rather than narrow/tall regardless of how many N
values are in --n-vals.

Run once with --source pop for the population-ladder comparison, once with
--source emp for the training-block one -- --source is a single selector, not
a third grid dimension, matching how combine_lagreg_ladder.py already does it.

CHANGED FROM THE PREVIOUS VERSION: both columns now come from the SAME
sweep's output. run_lagreg_nsweep_full.sh's LAMBDAS array includes 0.0
alongside 0.1 -- reg_lambda=0.0 through train_and_eval.py already IS the
unregularized run, tagged _reg0p0.json exactly like any other lambda,
written into --results-dir by the SAME script. There is no longer a
separate untagged pipeline to point at: --before-root and discover_before()
are gone, replaced by --reg-lambda-before (default 0.0), read via the same
discover() call as --reg-lambda-after against the same --results-root.

This also resolves the previous CAVEAT about whether the "before" column's
rows would even contain n_global/n_local: since both columns are now
produced by the identical train_and_eval.py -> LadderEvaluator.evaluate()
-> run_eval() code path (just called with a different --reg-lambda), the
grouped-computation question that applied to analyze_checkpoints.py's
separate, unverified output no longer applies here.

Usage (run from scripts/, no GPU needed)::

    python combine_lagreg_ladder_beforeafter.py \\
        --results-root ../results_lagreg_sweep \\
        --reg-lambda-before 0.0 --reg-lambda-after 0.1 \\
        --source pop --n-vals 512,1024,2048,4096,8192,16384,65536 --require-seeds 2
"""
from argparse import ArgumentParser
from pathlib import Path
import json
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def reg_tag(reg_lambda: float) -> str:
    return str(reg_lambda).replace(".", "p")


def build_tagged_re(source: str, tag: str) -> re.Pattern:
    return re.compile(rf"^ladder_{source}_seed(\d+)_N(\d+)_reg{re.escape(tag)}\.json$")


def expected_n_iter(N: int, target_steps: int, batch: int) -> int:
    """OPTIONAL CONVENIENCE ONLY -- reproduces run_lagreg_nsweep_full.sh's
    own arithmetic to auto-generate an --n-iter-map without typing one N at
    a time. NOT the authoritative check: n_iter is already legible directly
    in the path (test_model_script_<n_iter>_<N>_...), so matching it
    literally (see --n-iter-map) is the real mechanism and cannot go stale
    if this formula or BATCH ever changes in a future sweep script."""
    steps_per_epoch = -(-N // batch)   # ceil division, matches bash's (N+BATCH-1)/BATCH
    return target_steps // steps_per_epoch


PATH_N_ITER_RE = re.compile(r"test_model_script_(\d+)_\d+_")


def discover(results_root: Path, source: str, reg_lambda: float, n_vals=None,
            path_filter=None, n_iter_map=None, seeds=None):
    """Find every ladder_{source}_seed<S>_N<M>_reg{tag}.json under
    results_root for one reg_lambda. Returns {N: [(seed, path), ...]}.
    Used for BOTH the before and after columns -- they differ only in
    which reg_lambda tag is being searched for, not in file layout.

    rglob searches the ENTIRE tree under results_root, so if more than one
    sweep generation has ever written into this results_root (different
    n_iter -- e.g. a step-count change like TARGET_STEPS 100000 -> 200000
    -- changes create_save_path()'s directory but NOT the leaf filename,
    since n_iter isn't part of ladder_{source}_seed<S>_N<M>_reg{tag}.json),
    multiple genuinely different files can collide under the same (N, seed)
    key here. This is caught explicitly below (duplicate seed -> error
    naming both paths) rather than silently appended or averaged.

    path_filter, if given, keeps only paths whose str() contains it -- a
    single fixed substring, fine when only ONE N is in play.

    n_iter_map, if given ({N: n_iter}), keeps only files whose OWN n_iter
    (read directly from the path via PATH_N_ITER_RE, not predicted) equals
    n_iter_map[N] for that file's N -- the general, formula-free fix for
    multiple N, since n_iter differs per N even within one generation.
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
        if seeds is not None and seed not in seeds:
            continue
        if n_iter_map is not None and N in n_iter_map:
            m_iter = PATH_N_ITER_RE.search(str(f))
            if m_iter is None:
                raise ValueError(f"{f}: --n-iter-map given but no test_model_script_<n_iter>_ "
                                 f"segment found in path -- can't verify n_iter.")
            n_iter_found = int(m_iter.group(1))
            if n_iter_found != n_iter_map[N]:
                filtered_out_n_iter.append((f, n_iter_found, n_iter_map[N]))
                continue
        runs.setdefault(N, []).append((seed, f))
    if skipped:
        print(f"skipped {len(skipped)} file(s) not matching "
              f"'ladder_{source}_seed<S>_N<M>_reg{tag}.json':")
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
                f"{results_root} -- more than one sweep generation has written "
                f"into this results_root and rglob is picking up both:\n{lines}\n"
                f"Resolve with --n-iter-map (read the correct n_iter for N={N} "
                f"straight off one of the test_model_script_<n_iter>_{N}_ paths "
                f"above), --path-filter (only works for a single N), or remove "
                f"the stale generation's directory."
            )
    return dict(sorted(runs.items()))


def load_run(seed_paths, N: int):
    """Load every seed's ladder file for this N, align on the shared step
    grid, return seed-mean/std of n_global and n_local."""
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



# Categorical pair validated via the dataviz skill's validate_palette.js
# (light mode: CVD dE 21.6, normal-vision dE 32.3, both well clear of the
# floors) -- blue/red, kept as a fixed assignment (n_global always blue,
# n_local always red) rather than re-cycled per plot.
COLOR_GLOBAL = "#2a78d6"
COLOR_LOCAL = "#e34948"


def onset_step(steps, values, frac=0.1):
    """First step at which `values` has fallen `frac` of the way from its
    early-plateau level down toward its own minimum -- the operational
    definition of "onset of minimization" used for the dotted marker lines."""
    v0 = float(np.mean(values[:2]))
    vmin = float(values.min())
    if v0 <= vmin:
        return float(steps[0])
    thresh = v0 - frac * (v0 - vmin)
    below = np.where(values <= thresh)[0]
    idx = int(below[0]) if len(below) else len(values) - 1
    return float(steps[idx])


def draw_onset_lines(ax, steps, g_mean, l_mean, frac=0.1):
    x_local = onset_step(steps, l_mean, frac)
    x_global = onset_step(steps, g_mean, frac)
    ax.axvline(x_local, color=COLOR_LOCAL, ls=":", lw=1.8, zorder=4)
    ax.axvline(x_global, color=COLOR_GLOBAL, ls=":", lw=1.8, zorder=4)


def plot_cell(ax, steps, out, n_seeds, show_std, title, n_markers=9, show_onset=True):
    g_mean, g_std = out["n_global"]
    l_mean, l_std = out["n_local"]

    # Sparse, evenly-spaced markers regardless of how many eval points there
    # are (was: a marker at every one of ~45 points, which is what made the
    # earlier version look cluttered/dense) -- markevery takes a stride, so
    # convert the desired marker COUNT into one.
    markevery = max(1, len(steps) // n_markers)

    # Shade the region BETWEEN the two curves (the lag, n_global - n_local)
    # first, so it sits behind the std ribbons and the lines themselves.
    ax.fill_between(steps, l_mean, g_mean, color="0.55", alpha=0.18, zorder=0)

    ax.plot(steps, g_mean, color=COLOR_GLOBAL, marker="o", ms=7.5, lw=2.2,
            markevery=markevery, label=r"$n_{\mathrm{global}}$", zorder=3)
    ax.plot(steps, l_mean, color=COLOR_LOCAL, marker="s", ms=7.5, lw=2.2,
            markevery=markevery, label=r"$n_{\mathrm{local}}$", zorder=3)
    if show_std and n_seeds > 1:
        ax.fill_between(steps, g_mean - g_std, g_mean + g_std, color=COLOR_GLOBAL, alpha=0.15, zorder=2)
        ax.fill_between(steps, l_mean - l_std, l_mean + l_std, color=COLOR_LOCAL, alpha=0.15, zorder=2)
    if show_onset:
        draw_onset_lines(ax, steps, g_mean, l_mean)

    ax.axhline(0.0, color="0.75", lw=0.8, ls=":", zorder=0)
    ax.set_xscale("log")
    # No background grid; full box (all four spines) instead of the
    # left/bottom-only minimalist look -- matches the reference style.
    ax.legend(fontsize=11, loc="upper right", frameon=False)
    ax.set_title(title, fontsize=12)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color("black")
        ax.spines[s].set_linewidth(1.6)
    ax.tick_params(colors="black", labelsize=11, width=1.3)


def parse_n_iter_map(s):
    if s is None:
        return None
    out = {}
    for pair in s.split(","):
        n_str, iter_str = pair.split(":")
        out[int(n_str)] = int(iter_str)
    return out


def main(args):
    n_vals = (set(int(x) for x in args.n_vals.split(",")) if args.n_vals else None)
    results_root = Path(args.results_root)
    n_iter_map_before = parse_n_iter_map(args.n_iter_map_before)
    n_iter_map_after = parse_n_iter_map(args.n_iter_map_after)
    if n_iter_map_before is None and args.target_steps_before is not None:
        if not n_vals:
            raise SystemExit("--target-steps-before also needs --n-vals (to know which "
                             "N to compute n_iter for) -- or use --n-iter-map-before instead.")
        n_iter_map_before = {N: expected_n_iter(N, args.target_steps_before, args.batch)
                             for N in n_vals}
    if n_iter_map_after is None and args.target_steps_after is not None:
        if not n_vals:
            raise SystemExit("--target-steps-after also needs --n-vals (to know which "
                             "N to compute n_iter for) -- or use --n-iter-map-after instead.")
        n_iter_map_after = {N: expected_n_iter(N, args.target_steps_after, args.batch)
                            for N in n_vals}

    seeds = set(int(x) for x in args.seeds.split(",")) if args.seeds else None
    runs_before = discover(results_root, args.source, args.reg_lambda_before, n_vals,
                          path_filter=args.path_filter_before, n_iter_map=n_iter_map_before,
                          seeds=seeds)
    runs_after = discover(results_root, args.source, args.reg_lambda_after, n_vals,
                         path_filter=args.path_filter_after, n_iter_map=n_iter_map_after,
                         seeds=seeds)

    all_Ns = set(runs_before) | set(runs_after)
    if not all_Ns:
        raise SystemExit(
            f"No ladder_{args.source} files found for reg_lambda in "
            f"{{{args.reg_lambda_before}, {args.reg_lambda_after}}} under {results_root}"
        )

    print(f"source={args.source}  results_root={results_root}  "
          f"before=reg_lambda={args.reg_lambda_before}  after=reg_lambda={args.reg_lambda_after}")
    for label, runs in (("before", runs_before), ("after", runs_after)):
        print(f"  {label}: found N = {sorted(runs)}")
        for N, seed_paths in runs.items():
            print(f"    N={N:>6}  {len(seed_paths)} seed file(s): "
                  f"{[(s, p.name) for s, p in seed_paths]}")

    if args.require_seeds is not None:
        for label, runs in (("before", runs_before), ("after", runs_after)):
            short = {N: len(sp) for N, sp in runs.items() if len(sp) < args.require_seeds}
            if short:
                raise SystemExit(
                    f"--require-seeds {args.require_seeds} but {label} "
                    f"(reg_lambda={args.reg_lambda_before if label == 'before' else args.reg_lambda_after}) "
                    f"is short seeds for: {short}."
                )

    if n_vals is not None:
        missing_either = n_vals - all_Ns
        if missing_either:
            print(f"WARNING: requested N with NO data in EITHER column: "
                  f"{sorted(missing_either)} -- absent from the grid entirely.")
        for label, runs in (("before", runs_before), ("after", runs_after)):
            missing = (n_vals & all_Ns) - set(runs)
            if missing:
                lam = args.reg_lambda_before if label == "before" else args.reg_lambda_after
                print(f"WARNING: N present in the other column but missing from "
                      f"'{label}' (reg_lambda={lam}, under {results_root}): "
                      f"{sorted(missing)} -- that cell left blank.")

    # LAYOUT: 2 ROWS (before, after) x len(Ns) COLUMNS (N values, largest on
    # the left) -- transposed from the previous len(Ns)-rows x 2-columns
    # grid, so N values sit side by side horizontally instead of stacking
    # vertically (was: len(Ns) rows x 2 cols, tall; now: 2 rows x len(Ns)
    # cols, wide). Everything below is the same computation, just addressed
    # as axes[row_for_label, col_for_N] instead of axes[row_for_N, col_for_label].
    Ns = sorted(all_Ns, reverse=True)  # largest N on the left
    fig, axes = plt.subplots(2, len(Ns), sharex=True, sharey=True,
                             figsize=(3.4 * len(Ns) + 1.2, 7.0))
    if len(Ns) == 1:
        axes = axes.reshape(2, 1)

    xmin, xmax = np.inf, -np.inf
    ymin, ymax = np.inf, -np.inf
    loaded = {}  # (N, 'before'|'after') -> (steps, out, n_seeds) or None
    for N in Ns:
        for label, runs in (("before", runs_before), ("after", runs_after)):
            if N not in runs:
                loaded[(N, label)] = None
                continue
            steps, out, n_seeds = load_run(runs[N], N)
            loaded[(N, label)] = (steps, out, n_seeds)
            xmin, xmax = min(xmin, steps.min()), max(xmax, steps.max())
            for key in ("n_global", "n_local"):
                mean, std = out[key]
                lo, hi = mean.min(), mean.max()
                if args.show_std and n_seeds > 1:
                    lo, hi = min(lo, (mean - std).min()), max(hi, (mean + std).max())
                ymin, ymax = min(ymin, lo), max(ymax, hi)

    pad = 0.05 * (ymax - ymin)
    ylim = (ymin - pad, ymax + pad)

    for col, N in enumerate(Ns):
        for row, label in enumerate(("before", "after")):
            ax = axes[row, col]
            entry = loaded[(N, label)]
            lam = args.reg_lambda_before if label == "before" else args.reg_lambda_after
            row_title = ("without regularization" if lam == 0.0
                        else rf"with regularization ($\lambda={lam:g}$)")
            title = f"N={N:g}" if row == 0 else ""
            if entry is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11, color="0.5")
                ax.set_xscale("log")
                ax.set_ylim(*ylim)
                if row == 0:
                    ax.set_title(title, fontsize=12)
                for s in ("top", "right", "left", "bottom"):
                    ax.spines[s].set_visible(True)
                    ax.spines[s].set_color("black")
                    ax.spines[s].set_linewidth(1.6)
                ax.tick_params(colors="black", labelsize=11, width=1.3)
            else:
                steps, out, n_seeds = entry
                plot_cell(ax, steps, out, n_seeds, args.show_std, title,
                         n_markers=args.n_markers, show_onset=args.show_onset)
                ax.set_ylim(*ylim)
            if col == 0:
                ax.set_ylabel(row_title, fontsize=12, color="black",
                             rotation=90, ha="center", va="center", labelpad=14)

    for ax in axes.flat:
        ax.set_xlim(xmin, xmax)
    for col in range(len(Ns)):
        axes[-1, col].set_xlabel("gradient steps", fontsize=13, color="black")

    # Shared y-axis title, distinct from the per-column "N=..." facet
    # labels (top row titles) and the per-row before/after ylabels.
    fig.text(0.005, 0.5, "rescaled risk", fontsize=14,
             color="black", rotation=90, va="center", ha="left")

    fig.suptitle(f"n_global / n_local, reg_lambda={args.reg_lambda_before} vs "
                f"{args.reg_lambda_after} (source={args.source}, seed-averaged)",
                fontsize=15, color="black")
    fig.tight_layout(rect=(0.02, 0, 1, 0.96))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--results-root", type=str, required=True,
                   help="Root of the lagreg sweep (run_lagreg_nsweep_full.sh's "
                        "--results-dir) -- BOTH before and after columns are read "
                        "from here, distinguished only by --reg-lambda-before/-after.")
    p.add_argument("--reg-lambda-before", type=float, default=0.0,
                   help="The 'before' row's reg_lambda, e.g. 0.0 (unregularized). "
                        "Must be one of the LAMBDAS values the sweep actually ran.")
    p.add_argument("--reg-lambda-after", type=float, required=True,
                   help="The 'after' row's reg_lambda, e.g. 0.1.")
    p.add_argument("--path-filter-before", type=str, default=None,
                   help="Keep only 'before' paths containing this substring -- "
                        "only disambiguates a single N; for multiple N use "
                        "--n-iter-map-before instead.")
    p.add_argument("--path-filter-after", type=str, default=None,
                   help="Same as --path-filter-before, for the 'after' row.")
    p.add_argument("--n-iter-map-before", type=str, default=None,
                   help="'N:n_iter,N:n_iter,...' pinning the exact n_iter to accept "
                        "per N for the 'before' row, read straight off the paths "
                        "printed when a duplicate-seed error fires (e.g. "
                        "'512:200000,1024:100000,...'). The authoritative, formula-"
                        "free disambiguator when results_root has multiple sweep "
                        "generations.")
    p.add_argument("--n-iter-map-after", type=str, default=None,
                   help="Same as --n-iter-map-before, for the 'after' row.")
    p.add_argument("--target-steps-before", type=int, default=None,
                   help="OPTIONAL CONVENIENCE: auto-generates --n-iter-map-before "
                        "via run_lagreg_nsweep_full.sh's own arithmetic instead of "
                        "typing the map by hand. Needs --n-vals too. Prefer "
                        "--n-iter-map-before directly if unsure this formula still "
                        "matches the sweep script that produced your data.")
    p.add_argument("--target-steps-after", type=int, default=None,
                   help="Same as --target-steps-before, for the 'after' row.")
    p.add_argument("--batch", type=int, default=512,
                   help="BATCH from the sweep script (default 512) -- only used "
                        "with --target-steps-before/-after.")
    p.add_argument("--source", type=str, default="pop", choices=["emp", "pop"])
    p.add_argument("--n-vals", type=str, default=None,
                   help="Comma-separated N values, e.g. 512,1024,2048,4096,8192,16384,65536. "
                        "Default: all N found in either row.")
    p.add_argument("--require-seeds", type=int, default=None)
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated seeds to include, e.g. 2,3,4. Default: all seeds found.")
    p.add_argument("--show-std", action="store_true", default=True)
    p.add_argument("--no-show-std", dest="show_std", action="store_false")
    p.add_argument("--n-markers", type=int, default=9,
                   help="Roughly how many markers to draw per line, evenly spaced by "
                        "index (via markevery) regardless of how many eval points the "
                        "run actually has -- keeps dense ~45-point runs from looking "
                        "cluttered without subsampling the line itself.")
    p.add_argument("--show-onset", action="store_true", default=True,
                   help="Draw dotted vertical marker lines (red for n_local, blue "
                        "for n_global) at each curve's 'onset of minimization' -- "
                        "the first step where it has dropped 10%% of the way from "
                        "its early plateau toward its own minimum.")
    p.add_argument("--no-show-onset", dest="show_onset", action="store_false")
    p.add_argument("--out", type=str, default="../plots/lagreg_ladder_beforeafter.png")
    main(p.parse_args())