"""
combine_loss_runs.py

Combine the per-N, per-seed ladder outputs (ladder_{emp,pop}_seed<S>_N<M>.json,
as written by the analyze_checkpoints.py N-sweep script) into ONE overlaid
figure: population loss vs training time, ALL N values on a single panel,
seed-averaged, colored by N.

AESTHETIC (matches combine_runs.py's N-sweep convention, not stacked
subplots): every N gets a color from a continuous colormap keyed to
log10(N) -- combine_runs.py's own docstring describes its composition rows
as "colour strips positioned by log(N)"; this script carries that same
color-by-log(N) idea over to a line plot, on one shared axes, rather than
stacking one subplot per N. Line STYLE (not color) then encodes emp vs pop
when --overlay-emp is set: solid = pop, dashed = emp, both drawn in that
N's own color, exactly as the previous stacked-panel version's caption said
("solid = pop, dashed = emp") -- only the layout changed, not that
convention.

WHY THIS NEEDS ITS OWN SCRIPT RATHER THAN A FLAG ON combine_runs.py: the two
inputs are different objects. sequential_learning_reduced.py's npz store
generation-composition FRACTIONS (local/valid/global/memorized), naturally
mapped to RGB. analyze_checkpoints.py's JSON rows store a scalar LOSS per
checkpoint (loss_total, bp_0, ...) -- there is no 3-channel structure to mix,
so the natural rendering is a line, not a colour fill.

UPDATED: all axis/legend labels are now properly capitalized and bold, per
request, for readability.

RESTYLED: publication style matching the Fig. 1 schematic -- sequential
(viridis) colours for N, linear loss axis, muted phase bands behind the
curves, N legend in one row above the plot, no title, no seed-spread
shading, every other checkpoint only. All N are plotted by default
(--exclude-n can drop some). Also writes a vector .pdf next to the .png.

UPDATED AGAIN: --gen-field can now select 'global' (the hallucinated-sample
fraction) instead of just 'valid' (the generalized-sample fraction) for the
second-axis overlay next to the loss curves -- default is now 'global'
("Hallucinated"), and 'valid' ("Generalized"), 'local' ("Bad"), and
'memorized' remain available. Loss curves (test, and train if --overlay-emp)
are unaffected by this choice; --gen-field only swaps which composition
curve rides the second axis, e.g. --gen-field valid to go back to plotting
the generalized percentage instead of the hallucinated one.

Usage (run from scripts/)::

    python combine_loss_runs.py --results-root ../results_ladder_posthoc
    python combine_loss_runs.py --results-root ../results_ladder_posthoc \
        --loss-type pop --overlay-emp --metric loss_total
    python combine_loss_runs.py --results-root ../results_ladder_posthoc \
        --require-seeds 2 --y-log
    python combine_loss_runs.py --results-root ../results_ladder_posthoc \
        --overlay-emp --overlay-generalized --gen-root ../results_gen_sweep \
        --gen-field global
"""
import json
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
from matplotlib.colors import Normalize


# Matches exactly what the N-sweep script writes:
#     ladder_emp_seed0_N1024.json   ladder_pop_seed1_N65536.json
# Files not matching this (no seed/N tag, or the wrong loss-type prefix) are
# ignored -- same reasoning as combine_runs.py's TAGGED_RE: an untagged or
# ambiguously-named file could be silently overwritten by a later run, so
# only explicitly tagged files are trusted enough to average together.
TAGGED_RE = re.compile(r"^ladder_(emp|pop)_seed(\d+)_N(\d+)\.json$")


def discover(results_root: Path, loss_type: str):
    """
    Find every ladder_<loss_type>_seed<S>_N<M>.json directly under
    results_root (the sweep script writes them flat, not nested by N).

    Returns {N: [paths]}, one path per seed found for that N.
    """
    root = Path(results_root)
    runs = {}
    skipped = []
    for f in sorted(root.glob("ladder_*_seed*_N*.json")):
        m = TAGGED_RE.match(f.name)
        if m is None:
            skipped.append(f)
            continue
        kind, seed, N = m.group(1), int(m.group(2)), int(m.group(3))
        if kind != loss_type:
            continue
        runs.setdefault(N, []).append(f)
    if skipped:
        print(f"skipped {len(skipped)} file(s) not matching "
              f"'ladder_<emp|pop>_seed<S>_N<M>.json':")
        for p in skipped:
            print(f"    {p}")
    return dict(sorted(runs.items()))


def load_run(paths, N: int, metric: str, batch_size: int, x_axis: str = "auto"):
    """
    Load every seed's JSON for this N, align on a common checkpoint grid,
    and return the seed-mean (and seed-std) of `metric` vs training step.

    Each JSON is a list of per-checkpoint dicts (analyze_checkpoints.py's
    `rows`), each containing either a 'step' or 'epoch' key (whichever
    find_checkpoints() detected) plus whatever LadderEvaluator.evaluate()
    returned that checkpoint -- 'loss_total', 'bp_0', 'residual', etc.

    x_axis mirrors combine_runs.py's convention: 'auto' reads whichever key
    is present ('step' or 'epoch') and converts epoch -> gradient step via
    ceil(N/batch_size) if needed, so rows checkpointed on different
    conventions still land on a genuine gradient-step axis rather than
    silently comparing epoch counts to step counts.
    """
    all_rows = []
    for p in paths:
        with open(p) as fh:
            rows = json.load(fh)
        if not rows:
            raise ValueError(f"{p} is empty")
        if metric not in rows[0]:
            raise KeyError(
                f"{p}: metric {metric!r} not found in a row; available keys: "
                f"{sorted(rows[0].keys())}"
            )
        all_rows.append(sorted(rows, key=lambda r: r.get("step", r.get("epoch"))))

    tag_keys = {("step" if "step" in r[0] else "epoch") for r in all_rows}
    if len(tag_keys) > 1:
        raise ValueError(
            f"N={N}: seeds disagree on checkpoint convention {tag_keys} -- "
            f"re-evaluate them the same way."
        )
    tag_key = tag_keys.pop()

    ref_x = np.array([r[tag_key] for r in all_rows[0]], dtype=float)
    for p, rows in zip(paths[1:], all_rows[1:]):
        x = np.array([r[tag_key] for r in rows], dtype=float)
        if x.shape != ref_x.shape or not np.allclose(x, ref_x):
            raise ValueError(
                f"{p} has a different checkpoint grid than {paths[0]} "
                f"({x.shape} vs {ref_x.shape}) -- cannot average mismatched "
                f"seeds without interpolation."
            )

    mode = x_axis
    if mode == "auto":
        mode = "step" if tag_key == "step" else "epoch"
    if mode == "step":
        if tag_key != "step":
            raise ValueError(f"N={N}: rows are epoch-tagged but --x-axis step was forced")
        steps = ref_x
    elif mode == "epoch":
        steps_per_epoch = int(np.ceil(N / batch_size))
        steps = ref_x * steps_per_epoch if tag_key == "epoch" else ref_x
    else:
        raise ValueError(f"bad x_axis mode: {mode!r}")

    vals = np.stack([[r[metric] for r in rows] for rows in all_rows], axis=0)  # (n_seeds, T)
    return steps, vals.mean(axis=0), vals.std(axis=0), len(all_rows)


# --------------------------------------------------------------------------
# Generalized/hallucinated-sample-percentage overlay (from a DIFFERENT file
# family -- the analyze_checkpoints.py-style KL_FACT_*.json produced per
# (N, seed), NOT the ladder_{emp,pop}_*.json files above). Each file is a
# single dict (not a list of rows): "epochs" is the shared step axis, and
# "memorized"/"valid"/"local"/"global" are per-step FRACTIONS (0-1) of
# generated samples in each class. "valid" = novel AND grammar-consistent,
# i.e. genuinely generalized; "global" = a global-hallucination (grammar
# violated at a level not locally confinable, i.e. "hallucinated"); "local"
# = a local hallucination ("bad" but grammar-consistent nearby). --gen-field
# picks which one is plotted (default 'global', i.e. hallucinated).
# --------------------------------------------------------------------------

# Matches e.g. KL_FACT_train_size_4096_reps_5_Ntest_1000_ratio_0.15_latest_seed0_N4096.json
GEN_RE = re.compile(
    r"^KL_FACT_train_size_(\d+)_reps_(\d+)_Ntest_(\d+)_ratio_([\d.]+)_"
    r"latest_seed(\d+)_N(\d+)\.json$"
)

# Display name for each composition field, used in axis labels, the legend,
# and the figure title -- so the human-readable word ("Hallucinated",
# "Bad", "Generalized") shows up instead of the raw JSON field name
# ("global", "local", "valid").
GEN_FIELD_DISPLAY = {
    "global": "Hallucinated",
    "local": "Bad",
    "valid": "Generalized",
    "memorized": "Memorized",
}


def gen_field_display(field: str) -> str:
    return GEN_FIELD_DISPLAY.get(field, field.capitalize())


def discover_generalized(gen_root: Path, N: int, seed: int = None):
    """
    Find every KL_FACT_train_size_<N>_..._N<N>.json under gen_root, at any
    depth (these live nested as N_train_<N>/N_test_<T>/ratio_<r>/backend_<b>/,
    unlike the flat ladder_*.json files) -- one path per seed found for
    this N, or only `seed`'s path if given. train_size and the trailing N
    are required to match (both equal the reduced_length N in every file
    seen so far); reps/Ntest/ratio/backend are left free since those are
    run-config, not sweep axes here.
    """
    matches = sorted(Path(gen_root).rglob(f"KL_FACT_train_size_{N}_*_N{N}.json"))
    paths = []
    for f in matches:
        m = GEN_RE.match(f.name)
        if m is None:
            continue
        train_size, _reps, _ntest, _ratio, file_seed, n_tag = m.groups()
        if int(train_size) != N or int(n_tag) != N:
            continue
        if seed is not None and int(file_seed) != seed:
            continue
        paths.append(f)
    return paths


def load_generalized(paths, field, scale: float = 100.0, require_distinct: bool = True):
    """
    Load every seed's KL_FACT_*.json for one N, align on the shared
    'epochs' grid, and return the seed-mean (and seed-std) of `field`
    (scaled, e.g. x100 for a 0-1 fraction -> percent).

    If require_distinct, steps are dropped ENTIRELY (from the returned x
    grid too, not just masked to NaN) unless every seed's n_distinct at
    that step equals its own n_gen -- i.e. only once all n_gen generated
    samples are pairwise distinct. Early in training the model can collapse
    to a handful of repeated outputs (n_distinct << n_gen, as seen in this
    file's own example: n_distinct=1 for the first several steps before
    jumping to 1000); the fraction at those steps is computed over that
    degenerate, non-representative sample and would be misleading to plot
    as if it were a real percentage estimate.
    """
    all_steps, all_vals, all_masks = [], [], []
    for p in paths:
        with open(p) as fh:
            d = json.load(fh)
        if "epochs" not in d:
            raise KeyError(f"{p}: no 'epochs' key; available: {sorted(d.keys())}")
        if field not in d:
            raise KeyError(f"{p}: field {field!r} not found; available: {sorted(d.keys())}")
        all_steps.append(np.array(d["epochs"], dtype=float))
        all_vals.append(np.array(d[field], dtype=float) * scale)

        if require_distinct:
            if "n_distinct" not in d:
                raise KeyError(f"{p}: --require-all-distinct needs 'n_distinct'; "
                               f"available: {sorted(d.keys())}")
            n_distinct = np.array(d["n_distinct"], dtype=float)
            n_gen = int(d["args"]["n_gen"]) if "args" in d and "n_gen" in d["args"] \
                else int(round(n_distinct.max()))
            all_masks.append(n_distinct >= n_gen - 1e-9)

    ref = all_steps[0]
    for p, s in zip(paths[1:], all_steps[1:]):
        if s.shape != ref.shape or not np.allclose(s, ref):
            raise ValueError(
                f"{p} has a different 'epochs' grid than {paths[0]} "
                f"({s.shape} vs {ref.shape}) -- cannot average mismatched seeds."
            )

    vals = np.stack(all_vals, axis=0)  # (n_seeds, T)

    if require_distinct:
        # Keep a step only if EVERY seed had all n_gen samples distinct
        # there -- a step where one seed is still degenerate isn't a fair
        # average even if another seed already diversified.
        keep = np.logical_and.reduce(all_masks)
        ref, vals = ref[keep], vals[:, keep]

    return ref, vals.mean(axis=0), vals.std(axis=0), len(paths)


def find_elbow(x, y, log_x=True, log_y=True):
    """
    Classic max-distance-from-chord elbow finder: normalize x/y to [0,1]
    (in log space if requested, matching how the curve is actually
    plotted), draw the straight line ("chord") between the curve's first
    and last point, and return the x value of the point that sits
    FARTHEST (perpendicular distance) from that chord. For a loss curve
    that drops steeply then flattens, this lands on the knee between the
    two regimes -- no extra dependency (e.g. the `kneed` package) needed.
    """
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
    return x[int(np.argmax(dist))]


# Muted phase colours, shared with the Fig. 1 schematic.
C_BAD = "#f6d5d2"      # before the elbow: bad generation
C_WINDOW = "#d6def5"   # hallucination window
C_GEN = "#d3eedb"      # after T_global: generalization


def main(args):
    if args.overlay_generalized and not args.gen_root:
        raise SystemExit("--overlay-generalized requires --gen-root")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 9,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.linewidth": 0.8, "mathtext.default": "regular",
    })

    root = Path(args.results_root)
    runs = discover(root, args.loss_type)
    if not runs:
        raise SystemExit(
            f"No 'ladder_{args.loss_type}_seed*_N*.json' files found under {root}"
        )
    print(f"found {len(runs)} N values: {sorted(runs)}")

    exclude = {int(x) for x in args.exclude_n.split(",") if x.strip()} if args.exclude_n else set()
    if exclude:
        print(f"--exclude-n: dropping {sorted(exclude & set(runs))}")
        runs = {N: ps for N, ps in runs.items() if N not in exclude}
    if args.min_n is not None:
        runs = {N: ps for N, ps in runs.items() if N >= args.min_n}
    if not runs:
        raise SystemExit("no N values left to plot")
    for N, paths in runs.items():
        print(f"  N={N:>6}  {len(paths)} seed file(s)")

    if args.require_seeds is not None:
        short = {N: len(ps) for N, ps in runs.items() if len(ps) < args.require_seeds}
        if short:
            raise SystemExit(f"--require-seeds {args.require_seeds} but these N have fewer: {short}")

    emp_runs = discover(root, "emp") if args.overlay_emp else {}
    Ns = sorted(runs.keys())

    # Sequential colours: N is ordered, so small N = dark, large N = light.
    cmap = matplotlib.colormaps[args.cmap]
    positions = np.linspace(0.05, 0.85, len(Ns)) if len(Ns) > 1 else np.array([0.5])
    color_of = {N: cmap(pos) for N, pos in zip(Ns, positions)}

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    ax2 = ax.twinx() if args.overlay_generalized else None

    ymin, ymax = np.inf, -np.inf
    elbow_steps, x_lo, x_hi = [], np.inf, -np.inf
    curves = {}
    for N in Ns:
        steps, mean, std, n_seeds = load_run(runs[N], N, args.metric, args.batch_size, args.x_axis)
        # Elbow on the full-resolution curve, before thinning.
        elbow_steps.append(find_elbow(steps, mean, log_x=True, log_y=args.y_log))
        if args.alternate_checkpoints:
            steps, mean, std = steps[::2], mean[::2], std[::2]
        c = color_of[N]
        curves[N] = (steps, mean)
        ax.plot(steps, mean, color=c, lw=1.4, zorder=3)
        if args.show_std and n_seeds > 1:
            ax.fill_between(steps, mean - std, mean + std, color=c, alpha=0.15, lw=0)
        ymin, ymax = min(ymin, mean.min()), max(ymax, mean.max())
        x_lo, x_hi = min(x_lo, steps.min()), max(x_hi, steps.max())

        if args.overlay_emp and N in emp_runs:
            e_steps, e_mean, e_std, e_n = load_run(emp_runs[N], N, args.metric,
                                                   args.batch_size, args.x_axis)
            if args.alternate_checkpoints:
                e_steps, e_mean, e_std = e_steps[::2], e_mean[::2], e_std[::2]
            ax.plot(e_steps, e_mean, color=c, lw=1.1, ls="--", zorder=3)
            if args.show_std and e_n > 1:
                ax.fill_between(e_steps, e_mean - e_std, e_mean + e_std, color=c, alpha=0.1, lw=0)
            ymin, ymax = min(ymin, e_mean.min()), max(ymax, e_mean.max())

    if ax2 is not None:
        for N in Ns:
            gen_paths = discover_generalized(Path(args.gen_root), N, seed=args.gen_seed)
            if not gen_paths:
                print(f"N={N}: no KL_FACT file found -- skipping composition overlay")
                continue
            if args.gen_seed is None and len(gen_paths) > 1:
                gen_paths = gen_paths[:1]
            g_steps, g_mean, g_std, g_n = load_generalized(
                gen_paths, args.gen_field, scale=100.0 if args.gen_percent else 1.0,
                require_distinct=args.require_all_distinct)
            if len(g_steps) == 0:
                continue
            if args.alternate_checkpoints:
                g_steps, g_mean, g_std = g_steps[::2], g_mean[::2], g_std[::2]
            c = color_of[N]
            ax2.plot(g_steps, g_mean, color=c, lw=0.9, ls=":", marker="^", ms=3,
                     markevery=args.markevery, zorder=2)
            if args.show_std and g_n > 1:
                ax2.fill_between(g_steps, g_mean - g_std, g_mean + g_std, color=c, alpha=0.08, lw=0)

    avg_elbow = float(np.mean(elbow_steps))
    print(f"per-N elbow steps: {dict(zip(Ns, [round(e) for e in elbow_steps]))}")
    print(f"average elbow step: {avg_elbow:.1f}")

    # Phase bands behind everything (drawn on the back axis so the
    # composition curves on it stay visible).
    back = ax2 if ax2 is not None else ax
    if args.shade_phases:
        back.axvspan(x_lo, avg_elbow, color=C_BAD, lw=0, zorder=0)
        if args.t_global is not None:
            back.axvspan(avg_elbow, args.t_global, color=C_WINDOW, lw=0, zorder=0)
            back.axvspan(args.t_global, x_hi, color=C_GEN, lw=0, zorder=0)
        else:
            back.axvspan(avg_elbow, x_hi, color=C_WINDOW, lw=0, zorder=0)
    if ax2 is not None:
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)
    if args.y_log:
        ax.set_yscale("log")
    pad = 0.05 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)

    if args.mark_elbow:
        ax.axvline(avg_elbow, color="0.25", lw=0.8, ls=":", zorder=4)
        ax.text(avg_elbow * 1.08, ymax + 0.2 * pad, r"$T_{elb}$", fontsize=8, va="top")
    if args.annotate_elbow:
        # Arrow tip on the largest-N test-loss curve at the elbow.
        st, mn = curves[Ns[-1]]
        y_curve = float(np.interp(np.log10(avg_elbow), np.log10(st), mn))
        ax.annotate("elbow", xy=(avg_elbow * 1.05, y_curve),
                    xytext=(avg_elbow * 3.2, y_curve + 0.22 * (ymax - ymin)), fontsize=8,
                    arrowprops=dict(arrowstyle="->", lw=0.8))

    ax.set_xlabel("gradient steps")
    ax.set_ylabel("loss")
    if ax2 is not None:
        name = gen_field_display(args.gen_field).lower()
        ax2.set_ylabel(f"{name} samples (%)" if args.gen_percent else f"{name} fraction")
        if args.gen_percent:
            ax2.set_ylim(0, 100)

    # N legend: one row above the plot, out of the data.
    handles = [mlines.Line2D([], [], color=color_of[N], lw=1.5) for N in Ns]
    labels = [f"N = {Ns[0]:,}"] + [f"{N:,}" for N in Ns[1:]]
    leg1 = ax.legend(handles, labels, ncol=len(Ns), loc="lower center",
                     bbox_to_anchor=(0.5, 1.02), frameon=False,
                     handlelength=1.4, columnspacing=0.9)
    ax.add_artist(leg1)

    style = [mlines.Line2D([], [], color="0.3", lw=1.4, label="test loss")]
    if args.overlay_emp:
        style.append(mlines.Line2D([], [], color="0.3", lw=1.1, ls="--", label="training loss"))
    if ax2 is not None:
        name = gen_field_display(args.gen_field).lower()
        style.append(mlines.Line2D([], [], color="0.3", lw=0.9, ls=":", marker="^", ms=3,
                                   label=f"{name} (%)"))
    if len(style) > 1:
        ax.legend(handles=style, loc=args.style_legend_loc, frameon=True,
                  framealpha=1, edgecolor="0.8")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    if out.suffix.lower() != ".pdf":
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--results-root", type=str, required=True)
    p.add_argument("--loss-type", type=str, default="pop", choices=["emp", "pop"])
    p.add_argument("--metric", type=str, default="loss_total")
    p.add_argument("--overlay-emp", action="store_true",
                   help="Also plot the training loss (dashed).")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--x-axis", type=str, default="auto", choices=["auto", "step", "epoch"])
    p.add_argument("--y-log", action="store_true", default=False,
                   help="Log-scale the loss axis (default off: linear axis).")
    p.add_argument("--show-std", action="store_true", default=False,
                   help="Shade the across-seed spread (default off).")
    p.add_argument("--require-seeds", type=int, default=None)
    p.add_argument("--cmap", type=str, default="viridis",
                   help="Sequential colormap for N (default viridis).")
    p.add_argument("--exclude-n", type=str, default="",
                   help="Comma-separated N values to leave out, e.g. "
                        "1024,4096,16384. Default: keep all N.")
    p.add_argument("--min-n", type=int, default=None)
    p.add_argument("--alternate-checkpoints", action="store_true", default=True,
                   help="Plot every other checkpoint (default on).")
    p.add_argument("--all-checkpoints", dest="alternate_checkpoints", action="store_false")
    p.add_argument("--markevery", type=int, default=2,
                   help="Marker spacing on the composition curve.")
    p.add_argument("--shade-phases", action="store_true", default=False,
                   help="Shade bad-generation / window / generalization bands.")
    p.add_argument("--no-shade-phases", dest="shade_phases", action="store_false")
    p.add_argument("--t-global", type=float, default=None,
                   help="Step at which the window closes (T_global). If given, "
                        "the band after it is shaded as generalization.")
    p.add_argument("--annotate-elbow", action="store_true", default=False,
                   help="Draw an 'elbow' arrow (default off).")
    p.add_argument("--mark-elbow", action="store_true", default=False,
                   help="Draw the dotted T_elb line and label (default off).")
    p.add_argument("--no-annotate-elbow", dest="annotate_elbow", action="store_false")
    p.add_argument("--style-legend-loc", type=str, default="upper right")
    p.add_argument("--fig-width", type=float, default=4.6)
    p.add_argument("--fig-height", type=float, default=2.9)
    p.add_argument("--overlay-generalized", "--overlay-hallucinated",
                   dest="overlay_generalized", action="store_true",
                   help="Overlay a sample-composition percentage on a right axis.")
    p.add_argument("--gen-root", type=str, default=None)
    p.add_argument("--gen-seed", type=int, default=None)
    p.add_argument("--gen-field", type=str, default="global",
                   choices=["valid", "global", "local", "memorized"],
                   help="'global' = hallucinated (default), 'valid' = generalized, "
                        "'local' = bad, 'memorized'.")
    p.add_argument("--gen-percent", action="store_true", default=True)
    p.add_argument("--no-gen-percent", dest="gen_percent", action="store_false")
    p.add_argument("--require-all-distinct", action="store_true", default=True)
    p.add_argument("--no-require-all-distinct", dest="require_all_distinct", action="store_false")
    p.add_argument("--out", type=str, default="../plots/population_loss_combined.png")
    main(p.parse_args())
    