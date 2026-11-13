"""
plot_rhm_attractor_field_from_npz.py

Plots the trained-model score field DIRECTLY from the .npz files
train_and_eval.py already saved during a run with --attractor-save-npz --
NO checkpoint reload, NO model forward pass, NO BP computation. This is
for the case where the field at each eval step was already computed and
dumped to disk during training (see setup_attractor_field()/
plot_attractor_step()/_write_meta_once() in train_and_eval.py), and you
just want to re-render specific steps quickly.

Directory layout expected (per train_and_eval.py's attractor-plot output):
    <attractor_dir>/
        meta.json            -- written once (companions, config)
        plane_basis.npz       -- written once (ea, v1, v2, and/or the
                                  companion points' plane coordinates)
        oracle_t{t:04d}.npz    -- written once per t (oracle field, unused
                                  here -- this script only plots the model
                                  column, per your request)
        step_{gstep:07d}_t{t:04d}.npz  -- written per eval step per t
                                          (the MODEL field: alphas, betas,
                                          U, V -- exact key names are a
                                          best guess, see load_step_npz())

IMPORTANT CAVEAT: I do not have train_and_eval.py's exact
_write_meta_once()/plot_attractor_step() source in front of me right now,
only my own earlier description of what they do. The key names below
(alphas/betas/U/V for the per-step npz; ea/v1/v2/x_a/x_b/x_c/coord_ab for
plane_basis.npz) are my best guess, consistent with the naming used
throughout plot_rhm_attractor_field_real.py. Both loaders PRINT every key
they actually find in the file, so if the guess is wrong you'll see
immediately what's really in there and I can fix the key names in one
pass rather than you having to guess too.

Usage::

    python plot_rhm_attractor_field_from_npz.py \
        --attractor-dir ../results_lagreg_field_sweep/results_transformer_pick_0_for_training/models_restricted_continuous_0/test_model_script_12500_4096_cross_entropy_loss/8_512_lagreg_field/attractor_plots \
        --steps 1098,9102,56898 --t 50 --out attractor_from_npz.png
"""
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import matplotlib.lines as mlines


# Same marker-shape convention as plot_rhm_attractor_field_real.py, so a
# side-by-side row here looks consistent with that script's output.
# x_c and x_d are BOTH confirmed 'global' hallucinations (see
# classification_stats() results earlier) -- share the same marker shape
# so both read as "hallucinated" at a glance, distinct from x_a
# (memorized) and x_b (valid).
POINT_MARKERS = {
    "x_a": ("o", "memorized"),
    "x_b": ("s", "valid synonym"),
    "x_c": ("^", "hallucinated"),
    "x_d": ("^", "hallucinated (x_c+x_b-x_a)"),
}


def load_plane_basis(attractor_dir, coord_a, coord_b, coord_c, coord_d):
    """Companion-point plane coordinates, taken directly from CLI args
    rather than parsed out of plane_basis.npz/meta.json -- confirmed that
    this training run's automatic find_companions_nn() search landed on
    the SAME x_a/x_b/x_c as the fixed-companion run earlier, so the plane
    coordinates are already known exactly: x_a=(0,0), x_b=(sqrt(2),0),
    x_c=(0,sqrt(2)), x_d=(sqrt(2),sqrt(2)) -- these are this script's
    defaults (see the --coord-* CLI args), overridable if a different run
    ever needs different values."""
    coord_ab = {"x_a": coord_a, "x_b": coord_b, "x_c": coord_c, "x_d": coord_d}
    print(f"companion plane coordinates (fixed, not parsed from disk): {coord_ab}")
    return coord_ab, None


def load_step_npz(attractor_dir, step, t):
    """Load ONE per-step .npz -- filename convention
    step_{gstep:07d}_t{t:04d}.npz, matching your example
    step_0049417_t0050.npz. Tries the exact zero-padded name first, then
    falls back to a glob in case the padding width differs."""
    attractor_dir = Path(attractor_dir)
    exact = attractor_dir / f"step_{int(step):07d}_t{int(t):04d}.npz"
    if exact.exists():
        path = exact
    else:
        matches = sorted(attractor_dir.glob(f"step_*{int(step)}*_t{int(t):04d}.npz"))
        if not matches:
            all_npz = sorted(p.name for p in attractor_dir.glob("step_*.npz"))
            raise RuntimeError(
                f"no npz found for step={step}, t={t} under {attractor_dir} "
                f"(tried exact name {exact.name!r}). Files actually present: "
                f"{all_npz[:20]}{'...' if len(all_npz) > 20 else ''}")
        path = matches[0]
        print(f"  (exact filename {exact.name!r} not found; using closest glob "
              f"match {path.name!r})")

    data = np.load(path)
    print(f"  {path.name} keys: {list(data.keys())}")

    # Best-guess key names, matching the variable names used consistently
    # throughout plot_rhm_attractor_field_real.py's score_field_on_grid_*().
    def pick(*names):
        for n in names:
            if n in data:
                return data[n]
        raise KeyError(f"none of {names} found in {path.name}; keys present: {list(data.keys())}")

    alphas = pick("alphas", "alpha", "a")
    betas = pick("betas", "beta", "b")
    U = pick("U", "u")
    V = pick("V", "v")
    return alphas, betas, U, V, path


# Short in-plot text label per point -- separate from POINT_MARKERS's
# longer legend text.
POINT_SHORT_LABEL = {
    "x_a": r"$x_a$",
    "x_b": r"$x_b$",
    "x_c": r"$x_c$",
    "x_d": r"$x_d$",
}


def plot_vector_only_panel(fig, ax, alphas, betas, U, V, title, coord_ab, quiver_step=None):
    """Quiver + magnitude-heatmap background ('hot' colormap). Each
    companion point gets a distinct marker SHAPE (see POINT_MARKERS) AND
    a short text label (x_a/x_b/x_c/x_d) right next to it. The
    figure-level legend mapping shapes to full names is added ONCE by
    the caller (see add_point_legend()), not per-panel."""
    mag = np.sqrt(U ** 2 + V ** 2)
    im = ax.pcolormesh(alphas, betas, mag, shading="auto", cmap="hot")
    step = quiver_step if quiver_step is not None else max(1, len(alphas) // 20)
    ax.quiver(alphas[::step], betas[::step],
              U[::step, ::step], V[::step, ::step],
              color="cyan", scale=None, width=0.005)
    for name, (a, b) in coord_ab.items():
        marker, _ = POINT_MARKERS.get(name, ("o", name))
        ax.plot(a, b, marker=marker, ms=11, mfc="white", mec="blue", mew=2.0, zorder=5)
        short = POINT_SHORT_LABEL.get(name, name)
        ax.annotate(short, (a, b), textcoords="offset points", xytext=(9, 9),
                   fontsize=11, color="white", fontweight="bold", zorder=6,
                   path_effects=[patheffects.withStroke(linewidth=2, foreground="black")])
    ax.set_title(title, fontsize=11, color="black")
    ax.set_xlabel(r"$\alpha$", fontsize=10)
    ax.set_ylabel(r"$\beta$", fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\|s_\theta\|$")


def add_point_legend(fig):
    """ONE shared legend for the whole figure, placed to the right,
    mapping marker shape -> point meaning. x_c and x_d share a marker
    (both confirmed 'global' hallucinations), so they're combined into a
    single legend entry."""
    handles = [
        mlines.Line2D([], [], marker="o", mfc="white", mec="blue", mew=2.0, ms=11,
                     linestyle="None", label=r"$x_a$ (memorized)"),
        mlines.Line2D([], [], marker="s", mfc="white", mec="blue", mew=2.0, ms=11,
                     linestyle="None", label=r"$x_b$ (valid synonym)"),
        mlines.Line2D([], [], marker="^", mfc="white", mec="blue", mew=2.0, ms=11,
                     linestyle="None", label=r"$x_c$ / $x_d$ (hallucinated)"),
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.92, 0.5),
              fontsize=10, frameon=True, title="Key")


def _parse_coord(s):
    a, b = s.split(",")
    return (float(a), float(b))


def main(args):
    coord_ab, meta = load_plane_basis(
        args.attractor_dir,
        _parse_coord(args.coord_a), _parse_coord(args.coord_b),
        _parse_coord(args.coord_c), _parse_coord(args.coord_d))

    steps = [int(s) for s in args.steps.split(",")]
    fig, axes = plt.subplots(1, len(steps), figsize=(5.5 * len(steps), 5.2))
    if len(steps) == 1:
        axes = [axes]

    for ax, step in zip(axes, steps):
        print(f"--- step {step}, t={args.t} ---")
        alphas, betas, U, V, path = load_step_npz(args.attractor_dir, step, args.t)
        print(f"  field summary: |U| mean/max = {np.abs(U).mean():.6g}/{np.abs(U).max():.6g}, "
              f"|V| mean/max = {np.abs(V).mean():.6g}/{np.abs(V).max():.6g}")
        plot_vector_only_panel(fig, ax, alphas, betas, U, V, f"step {step}", coord_ab)

    fig.suptitle(rf"Trained model score field with regularization $\lambda=0.1$, $t={args.t}$",
                fontsize=12, color="black")
    add_point_legend(fig)
    fig.tight_layout(rect=[0, 0, 0.93, 1])  # leave a bit of room on the right for the legend
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--attractor-dir", type=str, required=True,
                   help="Directory containing meta.json, plane_basis.npz, and "
                        "step_{gstep:07d}_t{t:04d}.npz files (train_and_eval.py's "
                        "--attractor-save-npz output).")
    p.add_argument("--steps", type=str, required=True,
                   help="Comma-separated global steps to plot, e.g. '1098,9102,56898'.")
    p.add_argument("--t", type=int, required=True,
                   help="Diffusion timestep (matches the t{t:04d} in the filenames, "
                        "e.g. 50 for step_..._t0050.npz).")
    p.add_argument("--out", type=str, default="attractor_from_npz.png")
    p.add_argument("--coord-a", type=str, default="0.0,0.0",
                   help="Fixed plane coordinate 'alpha,beta' for x_a. Default is the "
                        "origin (always exact, by construction of build_plane).")
    p.add_argument("--coord-b", type=str, default="1.4142135623730951,0.0",
                   help="Fixed plane coordinate 'alpha,beta' for x_b. Default is "
                        "(sqrt(2), 0), confirmed to match the fixed-companion run's "
                        "own x_b for this same x_a/x_b/x_c (Hamming-1 edits at "
                        "different leaves).")
    p.add_argument("--coord-c", type=str, default="0.0,1.4142135623730951",
                   help="Fixed plane coordinate 'alpha,beta' for x_c. Default (0, sqrt(2)).")
    p.add_argument("--coord-d", type=str, default="1.4142135623730951,1.4142135623730951",
                   help="Fixed plane coordinate 'alpha,beta' for x_d = x_c+x_b-x_a. "
                        "Default (sqrt(2), sqrt(2)) -- the sum of x_b's and x_c's own "
                        "coords, closing the parallelogram/rectangle.")
    main(p.parse_args())