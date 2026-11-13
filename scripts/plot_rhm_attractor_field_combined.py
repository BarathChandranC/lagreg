"""
plot_rhm_attractor_field_combined.py

Combines TWO separate sources into ONE 4-panel row:

  - the REGULARIZED run's precomputed score fields, read directly from the
    .npz files train_and_eval.py already dumped during training (same
    loading logic as plot_rhm_attractor_field_from_npz.py -- no model
    reload, no BP computation, just np.load)
  - the UNREGULARIZED run's score field at one checkpoint, computed LIVE
    from the actual trained network (same logic as
    plot_rhm_attractor_field_real.py's --model-dir/--checkpoint-steps path
    -- load_data + build_parent_array + find_companions_nn + build_plane +
    load_checkpoint_model + score_field_on_grid_model)

Panel order (per request): npz-early, npz-mid, real-mid (no-reg), npz-late.
The middle two panels use the SAME alphas/betas grid (the npz-mid file's
own grid) so the regularized-vs-unregularized comparison at that slot is
visually apples-to-apples; the real-mid panel is evaluated ON that exact
grid via score_field_on_grid_model rather than building its own.

Companion-point markers are drawn LARGER than the two source scripts
(MARKER_SIZE below) per request, to be more readable when there are now 4
panels crammed into one row instead of 2 or 3.

CAVEAT: the npz files' plane coordinates are taken from the same fixed
--coord-a/--coord-b/--coord-c/--coord-d defaults as
plot_rhm_attractor_field_from_npz.py (confirmed matching a known
fixed-companion run, per that script's own docstring) -- NOT re-derived
from the (possibly different) x_a/x_b/x_c actually used for --model-dir's
unregularized run. If the unregularized run's find_companions_nn() lands
on a genuinely different x_a/x_b/x_c than the regularized run used, the
real-mid panel's marker positions (and the plane itself) will not
correspond to the SAME physical points as the npz panels' markers, even
though they're drawn at the same numeric (alpha, beta) coordinates. Pass
--x-a/--x-b/--x-c explicitly if you know the regularized run's exact
companions and want the real-mid panel to use the identical ones (matching
plane, not just matching coordinate labels).

Usage::

    python plot_rhm_attractor_field_combined.py \
        --attractor-dir ../results_lagreg_field_sweep/results_transformer_pick_0_for_training/models_restricted_continuous_0/test_model_script_12500_4096_cross_entropy_loss/8_512_lagreg_field/attractor_plots \
        --npz-early-step 1098 --npz-mid-step 24420 --npz-late-step 56898 --t 50 \
        --model-dir ../results_transformer_pick_0_for_training/models_restricted_continuous_0/test_model_script_12500_4096_cross_entropy_loss/8_512 \
        --real-step 24342 \
        --data-path ../data/labeled_data_restrictedfixed_6_4_1.0_4_0.npy \
        --reduced-length 4096 --pick-i-for-training 0 \
        --out attractor_combined.png
"""
import json
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import matplotlib.lines as mlines

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # matches train_and_eval.py

from modules.noise_schedules import alpha_bars_schedule, get_alpha_beta_from_alpha_bar
from modules.ladder_diagnostics import compute_bp_soft_targets
from utils import load_data, setup_device, create_model
from sequential_learning import build_parent_array, classification_stats


# Increased from the source scripts' ms=11/mew=2.0, per request ("increase
# the marker size") -- applies ONLY to the companion-point markers drawn
# IN each panel. The shared legend below intentionally keeps the ORIGINAL
# (smaller) size, per a follow-up request ("dont increase marker size in
# the legend"), so LEGEND_MARKER_SIZE/LEGEND_MARKER_EDGE_WIDTH are tracked
# separately from the in-panel ones.
MARKER_SIZE = 17
MARKER_EDGE_WIDTH = 2.6
LEGEND_MARKER_SIZE = 11
LEGEND_MARKER_EDGE_WIDTH = 2.0

# Larger title/axis-label font, per request ("increase font size just in
# the plot label") -- kept separate from MARKER_SIZE so it's clear this is
# purely a text-size change, not a marker change.
TITLE_FONTSIZE = 15
AXIS_LABEL_FONTSIZE = 13

POINT_MARKERS = {
    "x_a": ("o", "memorized"),
    "x_b": ("s", "valid synonym"),
    "x_c": ("^", "hallucinated"),
    "x_d": ("^", "hallucinated (x_c+x_b-x_a)"),
}
POINT_SHORT_LABEL = {
    "x_a": r"$x_a$", "x_b": r"$x_b$", "x_c": r"$x_c$", "x_d": r"$x_d$",
}

_STEP_RE = re.compile(r"step_(\d+)\.pt$")


# ---------------------------------------------------------------------
# npz loading (regularized fields) -- verbatim logic from
# plot_rhm_attractor_field_from_npz.py's load_step_npz().
# ---------------------------------------------------------------------

def load_step_npz(attractor_dir, step, t):
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


# ---------------------------------------------------------------------
# Real/unregularized checkpoint loading + companion search + score field
# -- verbatim logic from plot_rhm_attractor_field_real.py.
# ---------------------------------------------------------------------

def load_checkpoint_model(model_path, model_params_json, vocab_size, seq_len, device,
                          hidden_size_encoder, nhead, num_layers, model_channels):
    if model_params_json is not None:
        with open(model_params_json) as f:
            full_params = json.load(f)
        model_args = {
            "hidden_size_encoder": full_params["hidden_size_encoder"],
            "nhead": full_params["nhead"],
            "num_layers": full_params["num_layers"],
            "model_channels": full_params["model_channels"],
        }
        print(f"loaded model architecture from {model_params_json}: {model_args}")
    else:
        model_args = {
            "hidden_size_encoder": hidden_size_encoder, "nhead": nhead,
            "num_layers": num_layers, "model_channels": model_channels,
        }
        print(f"WARNING: no --model-params-json given -- using CLI/default "
              f"architecture {model_args}. If this doesn't match how the "
              f"checkpoint was trained, load_state_dict will either fail "
              f"loudly (shape mismatch) or -- worse -- succeed with the "
              f"wrong values.")

    model, _, actual_hidden = create_model(model_args, vocab_size, seq_len, device)

    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    if any(key.startswith("_orig_mod.") for key in state_dict):
        n_prefixed = sum(key.startswith("_orig_mod.") for key in state_dict)
        print(f"stripping torch.compile's '_orig_mod.' prefix from {n_prefixed} "
              f"state_dict keys (checkpoint was saved from a compiled model)")
        state_dict = {(key[len("_orig_mod."):] if key.startswith("_orig_mod.") else key): val
                      for key, val in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    step = ckpt.get("global_step", "unknown") if isinstance(ckpt, dict) else "unknown"
    print(f"loaded checkpoint {model_path} (global_step={step}); "
          f"actual_hidden={actual_hidden}")
    model.eval()
    return model, step


def find_checkpoint_by_step(model_dir, step):
    """EXACT match on the trailing 'step_<digits>.pt' -- see
    plot_rhm_attractor_field_real.py's own version for the rationale
    (substring matching on step numbers is fragile)."""
    model_dir = Path(model_dir)
    all_pt = sorted(model_dir.rglob("*.pt"))
    exact = []
    for p in all_pt:
        m = _STEP_RE.search(p.name)
        if m is not None and int(m.group(1)) == int(step):
            exact.append(p)
    if not exact:
        all_steps = sorted({int(_STEP_RE.search(p.name).group(1)) for p in all_pt
                            if _STEP_RE.search(p.name)})
        raise RuntimeError(
            f"no checkpoint with EXACT step={step} found under {model_dir}. "
            f"Steps actually present ({len(all_steps)} total): {all_steps}")
    if len(exact) > 1:
        print(f"WARNING: {len(exact)} files matched step={step} EXACTLY under "
              f"{model_dir} (should be unique) -- using the first: {exact[0]}")
    ckpt = exact[0]
    params_json = None
    for d in (ckpt.parent, ckpt.parent.parent):
        candidates = sorted(d.glob("full_params_reg*.json"))
        if candidates:
            params_json = candidates[0]
            break
    return ckpt, params_json


def hamming1_neighbors(seq, vocab_size):
    seq = tuple(int(s) for s in seq)
    for pos in range(len(seq)):
        for sym in range(vocab_size):
            if sym == seq[pos]:
                continue
            yield seq[:pos] + (sym,) + seq[pos + 1:]


def classify_candidate(seq, parent_arr, k, train_set, train_arr, local_max_depth):
    gen_probe = np.array([seq], dtype=np.int64)
    return classification_stats(
        gen_probe, parent_arr, ell=k, train_set=train_set, train_arr=train_arr,
        local_max_depth=local_max_depth,
    )


def find_companions_nn(anchor_candidates, full_support_sequences, vocab_size,
                       classifier_fn, max_anchor_candidates=None,
                       max_c_candidates_per_anchor=None):
    """Same construction as plot_rhm_attractor_field_real.py's version --
    see that script's docstring for the full rationale (separate anchor
    vs. full-grammar validity pools; x_c must be classifier-confirmed
    'global', not merely off-grammar)."""
    support_set = set(full_support_sequences)
    candidates = (anchor_candidates[:max_anchor_candidates] if max_anchor_candidates
                 else anchor_candidates)
    for x_a in candidates:
        x_b = next((n for n in hamming1_neighbors(x_a, vocab_size) if n in support_set), None)
        if x_b is None:
            continue
        x_c = None
        n_tried = 0
        for cand in hamming1_neighbors(x_a, vocab_size):
            if cand in support_set:
                continue
            n_tried += 1
            if max_c_candidates_per_anchor and n_tried > max_c_candidates_per_anchor:
                break
            cls, _depth_hist = classifier_fn(cand)
            if float(cls.get("global", 0.0)) > 0.5:
                x_c = cand
                break
        if x_c is None:
            continue
        return x_a, x_b, x_c
    raise RuntimeError(
        "no anchor among the searched candidates has both (a) a Hamming-1 "
        "valid neighbor in the FULL grammar, and (b) a Hamming-1 off-grammar "
        "neighbor classified 'global'. Try --max-anchor-candidates None, a "
        "smaller --local-max-depth, or pass --x-a/--x-b/--x-c explicitly."
    )


def onehot_embed(seq, seq_len, vocab_size):
    seq = np.asarray(seq, dtype=np.int64)
    emb = np.zeros((seq_len, vocab_size), dtype=np.float64)
    emb[np.arange(seq_len), seq] = 1.0
    return emb.reshape(-1)


def build_plane(x_a, x_b, x_c, seq_len, vocab_size):
    ea = onehot_embed(x_a, seq_len, vocab_size)
    eb = onehot_embed(x_b, seq_len, vocab_size)
    ec = onehot_embed(x_c, seq_len, vocab_size)
    v1 = eb - ea
    v1 /= np.linalg.norm(v1)
    v2 = ec - ea
    v2 -= (v2 @ v1) * v1
    v2 /= np.linalg.norm(v2)
    return ea, v1, v2


@torch.no_grad()
def score_field_on_grid_model_fixed_grid(ea, v1, v2, t, alpha_bars, model, seq_len,
                                         vocab_size, device, alphas, betas,
                                         batch_size=256):
    """Same Tweedie/model-forward logic as
    plot_rhm_attractor_field_real.py's score_field_on_grid_model(), but
    evaluated on a GIVEN alphas/betas grid (read from the npz-mid file)
    rather than building its own linspace from an extent -- so the
    real-mid panel's grid matches its npz neighbors exactly, point for
    point, rather than merely covering a similar range."""
    AA, BB = np.meshgrid(alphas, betas)
    flat_a, flat_b = AA.reshape(-1), BB.reshape(-1)
    pts = ea[None, :] + flat_a[:, None] * v1[None, :] + flat_b[:, None] * v2[None, :]
    N = pts.shape[0]
    grid_shape = (len(betas), len(alphas))  # matches np.meshgrid's default 'xy' indexing

    ab_t = alpha_bars[t - 1].item()
    x0_hats = np.zeros_like(pts)
    was_training = model.training
    model.eval()
    for i in range(0, N, batch_size):
        sl = slice(i, min(i + batch_size, N))
        xt = torch.as_tensor(pts[sl], dtype=torch.float32, device=device).reshape(-1, seq_len, vocab_size)
        t_tensor = torch.full((xt.shape[0],), int(t), device=device, dtype=torch.long)
        logits = model(xt, t_tensor)
        probs = torch.softmax(logits.float(), dim=-1)
        x0_hats[sl] = probs.reshape(xt.shape[0], -1).double().cpu().numpy()
    if was_training:
        model.train()

    scores = (np.sqrt(ab_t) * x0_hats - pts) / (1.0 - ab_t)
    U = (scores @ v1).reshape(grid_shape)
    V = (scores @ v2).reshape(grid_shape)
    return U, V


# ---------------------------------------------------------------------
# Plotting -- shared by all 4 panels regardless of source, verbatim from
# the two source scripts except for the larger MARKER_SIZE/MARKER_EDGE_WIDTH.
# ---------------------------------------------------------------------

def plot_vector_only_panel(fig, ax, alphas, betas, U, V, title, coord_ab, quiver_step=None):
    mag = np.sqrt(U ** 2 + V ** 2)
    im = ax.pcolormesh(alphas, betas, mag, shading="auto", cmap="hot")
    step = quiver_step if quiver_step is not None else max(1, len(alphas) // 20)
    ax.quiver(alphas[::step], betas[::step],
              U[::step, ::step], V[::step, ::step],
              color="black", scale=None, width=0.005)
    for name, (a, b) in coord_ab.items():
        marker, _ = POINT_MARKERS.get(name, ("o", name))
        ax.plot(a, b, marker=marker, ms=MARKER_SIZE, mfc="white", mec="blue",
                mew=MARKER_EDGE_WIDTH, zorder=5)
        short = POINT_SHORT_LABEL.get(name, name)
        ax.annotate(short, (a, b), textcoords="offset points", xytext=(9, 9),
                   fontsize=11, color="white", fontweight="bold", zorder=6,
                   path_effects=[patheffects.withStroke(linewidth=2, foreground="black")])
    ax.set_title(title, fontsize=TITLE_FONTSIZE, color="black")
    ax.set_xlabel(r"$\alpha$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"$\beta$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\|s_\theta\|$")


def add_point_legend(fig):
    handles = [
        mlines.Line2D([], [], marker="o", mfc="white", mec="blue", mew=LEGEND_MARKER_EDGE_WIDTH,
                     ms=LEGEND_MARKER_SIZE, linestyle="None", label=r"$x_a$ (memorized)"),
        mlines.Line2D([], [], marker="s", mfc="white", mec="blue", mew=LEGEND_MARKER_EDGE_WIDTH,
                     ms=LEGEND_MARKER_SIZE, linestyle="None", label=r"$x_b$ (valid synonym)"),
        mlines.Line2D([], [], marker="^", mfc="white", mec="blue", mew=LEGEND_MARKER_EDGE_WIDTH,
                     ms=LEGEND_MARKER_SIZE, linestyle="None", label=r"$x_c$ / $x_d$ (hallucinated)"),
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.92, 0.5),
              fontsize=10, frameon=True, title="Key")


def _parse_coord(s):
    a, b = s.split(",")
    return (float(a), float(b))


def main(args):
    device = setup_device()

    # ---- fixed plane coordinates shared by ALL 4 panels (see module
    # docstring's CAVEAT re: whether these truly correspond to the same
    # physical points for the real-mid panel) ----
    coord_ab = {
        "x_a": _parse_coord(args.coord_a), "x_b": _parse_coord(args.coord_b),
        "x_c": _parse_coord(args.coord_c), "x_d": _parse_coord(args.coord_d),
    }
    print(f"companion plane coordinates (fixed, shared by all 4 panels): {coord_ab}")

    # ---- npz-sourced panels (early, mid, late; regularized run) ----
    print(f"--- npz EARLY step {args.npz_early_step}, t={args.t} ---")
    alphas_e, betas_e, U_e, V_e, path_e = load_step_npz(args.attractor_dir, args.npz_early_step, args.t)

    print(f"--- npz MID step {args.npz_mid_step}, t={args.t} ---")
    alphas_m, betas_m, U_m, V_m, path_m = load_step_npz(args.attractor_dir, args.npz_mid_step, args.t)

    print(f"--- npz LATE step {args.npz_late_step}, t={args.t} ---")
    alphas_l, betas_l, U_l, V_l, path_l = load_step_npz(args.attractor_dir, args.npz_late_step, args.t)

    # ---- real/unregularized-checkpoint panel (same middle "slot", its own step) ----
    print(f"--- REAL (no-reg) step {args.real_step}, t={args.t} ---")
    vocab_size, sequences, seq_len, k, rho, _ = load_data(args.data_path, device)
    if not isinstance(rho, torch.Tensor):
        rho = torch.as_tensor(np.asarray(rho))
    rho = rho.to(device).float()

    alpha_bars = alpha_bars_schedule(args.t_final, device=device, s=args.s, schedule="linear")

    L = args.reduced_length
    tr_lo = args.pick_i_for_training * L
    tr_hi = tr_lo + L
    if tr_hi > len(sequences):
        raise ValueError(f"train block [{tr_lo}:{tr_hi}] out of bounds ({len(sequences)} seqs)")
    train_block = sequences[tr_lo:tr_hi].detach().cpu().numpy().astype(np.int64)
    train_block_tuples = [tuple(int(v) for v in row) for row in train_block]
    full_sequences_np = sequences.detach().cpu().numpy().astype(np.int64)
    full_support_tuples = [tuple(int(v) for v in row) for row in full_sequences_np]

    if args.x_a and args.x_b and args.x_c:
        x_a = tuple(int(v) for v in args.x_a.split(","))
        x_b = tuple(int(v) for v in args.x_b.split(","))
        x_c = tuple(int(v) for v in args.x_c.split(","))
        if not (len(x_a) == len(x_b) == len(x_c) == seq_len):
            raise ValueError(f"--x-a/--x-b/--x-c must each have seq_len={seq_len} entries")
        print(f"x_a/x_b/x_c FIXED from CLI: {x_a}, {x_b}, {x_c}")
    else:
        parent_arr = build_parent_array(rho)
        train_set = set(train_block_tuples)
        classifier_fn = lambda seq: classify_candidate(
            seq, parent_arr, k, train_set, train_block, args.local_max_depth)
        x_a, x_b, x_c = find_companions_nn(
            train_block_tuples, full_support_tuples, vocab_size, classifier_fn,
            max_anchor_candidates=args.max_anchor_candidates)
        print(f"auto-found companions (find_companions_nn): x_a={x_a}, x_b={x_b}, x_c={x_c}")

    ea, v1, v2 = build_plane(x_a, x_b, x_c, seq_len, vocab_size)

    ckpt_path, params_json = find_checkpoint_by_step(args.model_dir, args.real_step)
    print(f"real checkpoint: {ckpt_path} (params_json={params_json})")
    model, ckpt_step = load_checkpoint_model(
        str(ckpt_path), str(params_json) if params_json else args.model_params_json,
        vocab_size, seq_len, device,
        args.hidden_size_encoder, args.nhead, args.num_layers, args.model_channels)

    # Evaluate on EXACTLY the npz-mid file's alphas/betas grid, so the
    # real-mid panel lines up visually with its npz neighbors point for
    # point (see score_field_on_grid_model_fixed_grid's docstring).
    U_r, V_r = score_field_on_grid_model_fixed_grid(
        ea, v1, v2, args.t, alpha_bars, model, seq_len, vocab_size, device,
        alphas_m, betas_m, batch_size=args.batch_size)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- assemble the combined 4-panel figure: npz-early, npz-mid,
    # real-mid (no-reg), npz-late -- exact order requested ----
    fig, axes = plt.subplots(1, 4, figsize=(5.5 * 4, 5.2))
    plot_vector_only_panel(fig, axes[0], alphas_e, betas_e, U_e, V_e,
                           f"step {args.npz_early_step}", coord_ab)
    plot_vector_only_panel(fig, axes[1], alphas_m, betas_m, U_m, V_m,
                           f"step {args.npz_mid_step}", coord_ab)
    plot_vector_only_panel(fig, axes[2], alphas_m, betas_m, U_r, V_r,
                           f"step {args.real_step}", coord_ab)
    plot_vector_only_panel(fig, axes[3], alphas_l, betas_l, U_l, V_l,
                           f"step {args.npz_late_step}", coord_ab)

    fig.suptitle(rf"Attractor field across training, $t={args.t}$",
                fontsize=TITLE_FONTSIZE, color="black")
    add_point_legend(fig)
    fig.tight_layout(rect=[0, 0, 0.93, 1])  # leave room on the right for the legend
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)

    # --- npz (regularized) source ---
    p.add_argument("--attractor-dir", type=str, required=True,
                   help="Directory containing step_{gstep:07d}_t{t:04d}.npz files "
                        "(train_and_eval.py's --attractor-save-npz output, regularized run).")
    p.add_argument("--npz-early-step", type=int, required=True)
    p.add_argument("--npz-mid-step", type=int, required=True)
    p.add_argument("--npz-late-step", type=int, required=True)
    p.add_argument("--t", type=int, required=True,
                   help="Diffusion timestep, shared by ALL 4 panels (matches t{t:04d} "
                        "in the npz filenames and is also the t passed to the real "
                        "model's Tweedie-score evaluation).")
    p.add_argument("--coord-a", type=str, default="0.0,0.0")
    p.add_argument("--coord-b", type=str, default="1.4142135623730951,0.0")
    p.add_argument("--coord-c", type=str, default="0.0,1.4142135623730951")
    p.add_argument("--coord-d", type=str, default="1.4142135623730951,1.4142135623730951")

    # --- real/unregularized checkpoint source ---
    p.add_argument("--data-path", type=str,
                   default="../data/labeled_data_restrictedfixed_6_4_1.0_4_0.npy")
    p.add_argument("--reduced-length", type=int, default=4096)
    p.add_argument("--pick-i-for-training", type=int, default=0)
    p.add_argument("--t-final", type=int, default=500)
    p.add_argument("--s", type=float, default=1e-4)
    p.add_argument("--local-max-depth", type=int, default=2)
    p.add_argument("--max-anchor-candidates", type=int, default=None)
    p.add_argument("--x-a", type=str, default=None,
                   help="Comma-separated FIXED x_a for the real-mid panel. If --x-a/"
                        "--x-b/--x-c are ALL given, find_companions_nn() is skipped.")
    p.add_argument("--x-b", type=str, default=None)
    p.add_argument("--x-c", type=str, default=None)
    p.add_argument("--model-dir", type=str, required=True,
                   help="Directory of the UNREGULARIZED run to load the --real-step "
                        "checkpoint from (e.g. the '8_512' directory).")
    p.add_argument("--real-step", type=int, required=True,
                   help="Exact global step of the checkpoint to load from --model-dir "
                        "for the real-mid panel.")
    p.add_argument("--model-params-json", type=str, default=None,
                   help="Fallback architecture json if no full_params_reg*.json is "
                        "auto-discovered next to the checkpoint.")
    p.add_argument("--hidden-size-encoder", type=int, default=512)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--model-channels", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=256)

    p.add_argument("--out", type=str, default="attractor_combined.png")
    main(p.parse_args())