"""
plot_rhm_attractor_field_real.py

Attractor-field visualization on the ACTUAL RHM/BP-oracle pipeline -- not
the standalone toy grammar in plot_rhm_attractor_field.py. Reuses your real
modules with the exact same import pattern as train_and_eval.py:
    modules.noise_schedules.alpha_bars_schedule, get_alpha_beta_from_alpha_bar
    modules.ladder_diagnostics.compute_bp_soft_targets
    utils.load_data, setup_device, create_model
    sequential_learning.build_parent_array, classification_stats

CHECKPOINT SUPPORT: pass --model-path to a .pt file saved by
train_and_eval.py's --save-final-model (a dict with a "model_state_dict"
key) and the second ("model proxy") column is computed from that TRAINED
NETWORK's actual Tweedie score instead of a factorized-BP approximation --
see score_field_on_grid_model() and load_checkpoint_model(). Without
--model-path, the script falls back to the factorized_layers proxy as
before (no behavior change for existing usage).

Reconstructing the model needs its architecture hyperparameters
(hidden_size_encoder, nhead, num_layers, model_channels), which
train_and_eval.py does NOT save inside the checkpoint itself -- they're in
the separate full_params_reg{tag}.json it writes alongside it. Point
--model-params-json at that file and this script reads them from there;
otherwise it falls back to the --hidden-size-encoder/--nhead/--num-layers/
--model-channels CLI flags (defaults matching train_and_eval.py's own
defaults), which you must set correctly yourself if you don't pass the
json -- a mismatched architecture will load the state dict incorrectly
(wrong shapes error loudly; RIGHT shapes with WRONG values load silently
and give a garbage field, so prefer --model-params-json whenever you have
it).

WHAT THIS DOES (same skeleton as the toy script, real machinery underneath):

  1. Loads your real (vocab_size, sequences, seq_len, k, rho) via load_data.
     `sequences` is treated as your enumerated valid-sequence pool -- the
     real-data analog of the toy script's VALID_SEQUENCES.

  2. Picks companions by the SAME nearest-neighbor construction as the toy
     script -- this part needed no change, since it only ever used set
     membership in the valid pool, never the grammar's internals:
         x_a = an anchor sequence (from a chosen train block, so it is
               genuinely "memorized" relative to that block)
         x_b = a Hamming-1 neighbor of x_a that IS in `sequences`
               (valid synonym)
         x_c = a Hamming-1 neighbor of x_a that is NOT in `sequences`
               (candidate hallucination)

  3. Builds the 2D Gram-Schmidt plane in the REAL one-hot embedding space
     R^{seq_len * vocab_size} (build_plane() is verbatim from the toy
     script -- it never depended on the grammar being toy).

  4. At a REAL diffusion timestep `t` (an integer index into your actual
     alpha_bars schedule -- NOT an ad hoc sigma, see the "which timestep"
     discussion this replaces), evaluates the EXACT BP score on a grid over
     that plane via compute_bp_soft_targets + Tweedie's formula for the
     VP-SDE forward process your pipeline actually uses:

         x0_hat(x_t) = compute_bp_soft_targets(x_t, t, ..., factorized_layers=f)
                       # per-leaf category marginals = E[onehot(x0) | x_t]
                       # exactly, since bp_torch does exact sum-product over
                       # the tree with the per-symbol log-likelihood
                       # (`field`) computed from x_t already baked in.
         s(x_t) = (sqrt(alpha_bar_t) * x0_hat(x_t) - x_t) / (1 - alpha_bar_t)

     Derivation: p(x_t | x_0) = N(sqrt(ab_t) x_0, (1-ab_t) I) is the
     standard VP-SDE forward kernel; Tweedie's formula for a Gaussian
     channel y = a*x + noise(0, sigma^2) is
         score_y log p(y) = (a * E[x|y] - y) / sigma^2,
     giving exactly the expression above with a = sqrt(ab_t),
     sigma^2 = 1 - ab_t. This is the SAME field_intensity =
     sqrt(ab_t)/(1-ab_t) already inside compute_bp_soft_targets (that's
     a/sigma^2, the natural parameter of the Gaussian evidence fed into
     BP) -- so this script assumes no new convention, it just inverts the
     one compute_bp_soft_targets already commits to.

     f=0 is the TRUE oracle (exact posterior, full grammar/tree). A larger
     f is your existing partial-tree-factorization approximation, used
     here as the "model proxy" column IN PLACE OF an actual trained
     network (no checkpoint was provided). If you have a real trained
     model, swap in score_field_on_grid_model() below instead -- same
     Tweedie formula, x0_hat = softmax(model(x_t, t)) rather than BP
     marginals.

  5. Confirms x_c is a genuine hallucination via YOUR OWN classifier
     (build_parent_array / classification_stats from sequential_learning),
     not a hand-rolled grammar parser -- I don't have rho's internal
     production schema, and this project already has an established
     local/global/valid/memorized depth classification, so x_c is run
     through THAT rather than reinventing violation_depth() for real data.

CAVEATS / things to check before trusting the figure:
  - Grid evaluation is batched (score_field_on_grid_real's `batch_size`)
    but every single grid point still costs one BP forward pass over your
    real tree -- for a large k / vocab_size this can be slow. Start with a
    small --grid-n (e.g. 15-20) to sanity check runtime before scaling up.
  - find_companions_nn() searches `sequences` for a Hamming-1 valid/invalid
    pair by brute-force set membership. If `sequences` is very large,
    restrict the anchor search with --max-anchor-candidates rather than
    scanning everything.
  - classification_stats()'s exact return shape wasn't available to me
    (only its call signature, from train_and_eval.py's run_eval()) -- the
    printed result should be checked against what you know that function
    returns; adjust the two print lines near the bottom of main() if the
    keys differ.

Usage (edit paths/args for your setup)::

    python plot_rhm_attractor_field_real.py \
        --data-path ../data/labeled_data_restrictedfixed_6_4_1.0_4_0.npy \
        --reduced-length 4096 --pick-i-for-training 0 \
        --t-final 500 --t 250 --factorized-layers-proxy 2 \
        --grid-n 20 --out attractor_real.png

FIXED companions (skip the automatic search) + a checkpoint-DIRECTORY
sweep + the x_d = x_c + x_b - x_a parallelogram-completion analysis::

    python plot_rhm_attractor_field_real.py \
        --data-path ../data/labeled_data_restrictedfixed_6_4_1.0_4_0.npy \
        --reduced-length 4096 --pick-i-for-training 0 \
        --x-a 3,1,2,5,4,5,3,2,3,3,1,0,4,0,1,0 \
        --x-b 3,1,2,5,3,5,3,2,3,3,1,0,4,0,1,0 \
        --x-c 3,1,1,5,4,5,3,2,3,3,1,0,4,0,1,0 \
        --model-dir ../results_transformer_pick_0_for_training/models_restricted_continuous_0/test_model_script_12500_4096_cross_entropy_loss \
        --t 250 --grid-n 20 --out attractor_plots_fixed_xd

x_d's decoded sequence, its Hamming distances to x_a/x_b/x_c, and its
classification_stats() verdict (memorized/valid/local/'global') are
printed to stdout regardless of whether --model-dir/--model-path is
given at all -- this part runs even with a bare oracle-only invocation.
"""
import json
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


# ---------------------------------------------------------------------
# 0. Checkpoint loading -- mirrors train_and_eval.py's model construction
#    (create_model(model_args, vocab_size, seq_len, device)) exactly, then
#    loads a state dict saved by --save-final-model
#    (torch.save({"model_state_dict": ..., "global_step": ...}, path)).
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

    # torch.compile wraps the model in an OptimizedModule and prefixes every
    # saved key with "_orig_mod." -- the freshly-constructed model here is
    # NOT compiled, so the keys won't line up unless that prefix is
    # stripped first. Detect and strip it rather than requiring the caller
    # to know their training script used torch.compile.
    if any(key.startswith("_orig_mod.") for key in state_dict):
        n_prefixed = sum(key.startswith("_orig_mod.") for key in state_dict)
        print(f"stripping torch.compile's '_orig_mod.' prefix from {n_prefixed} "
              f"state_dict keys (checkpoint was saved from a compiled model)")
        state_dict = {(key[len("_orig_mod."):] if key.startswith("_orig_mod.") else key): val
                      for key, val in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    step = ckpt.get("global_step", "unknown") if isinstance(ckpt, dict) else "unknown"
    print(f"loaded checkpoint {model_path} (global_step={step}); "
          f"actual_hidden={actual_hidden}")
    model.eval()
    return model, step


# ---------------------------------------------------------------------
# 1. Companion selection -- IDENTICAL logic to the toy script. Nothing
#    here depends on the grammar's internals, only on set membership in
#    `sequences`, so it carries over to real data unchanged.
# ---------------------------------------------------------------------

def hamming1_neighbors(seq, vocab_size):
    seq = tuple(int(s) for s in seq)
    for pos in range(len(seq)):
        for sym in range(vocab_size):
            if sym == seq[pos]:
                continue
            yield seq[:pos] + (sym,) + seq[pos + 1:]


def classify_candidate(seq, parent_arr, k, train_set, train_arr, local_max_depth):
    """Run ONE candidate sequence through your project's own
    memorized/valid/local/global classifier (classification_stats), as a
    batch of size 1, and return (cls, depth_hist) for it. cls's 'global'/
    'local'/'valid'/'memorized' entries are FRACTIONS over the batch, so
    for a batch of 1 each is either 0.0 or 1.0 -- i.e. exactly which
    single bucket this one sequence landed in."""
    gen_probe = np.array([seq], dtype=np.int64)
    cls, depth_hist = classification_stats(
        gen_probe, parent_arr, ell=k, train_set=train_set, train_arr=train_arr,
        local_max_depth=local_max_depth,
    )
    return cls, depth_hist


def find_companions_nn(anchor_candidates, full_support_sequences, vocab_size,
                       classifier_fn, max_anchor_candidates=None,
                       max_c_candidates_per_anchor=None):
    """Same construction as the toy script's find_companions_nn(), but with
    two changes for real data:

      1. The anchor pool and the validity pool are kept SEPARATE:
           anchor_candidates      = sequences x_a is allowed to be drawn
                                     from (the training block, so x_a is
                                     genuinely "memorized" by that block)
           full_support_sequences = the pool used to test whether a
                                     Hamming-1 neighbor counts as "valid"
                                     -- this MUST be the full enumerated
                                     grammar, not just the training block.
                                     A real neighbor of x_a can be
                                     perfectly grammatical and simply not
                                     have been sampled into this one
                                     contiguous block; checking membership
                                     against the block alone makes almost
                                     everything look "invalid" and starves
                                     the search for x_b.

      2. x_c must be a GLOBALLY wrong sample, not merely off-grammar. An
         off-grammar Hamming-1 neighbor is not automatically a
         hallucination in this project's vocabulary -- it could instead be
         classified 'local' (a shallower, more surface-level break) by
         classifier_fn. Only a neighbor that classifier_fn's returned cls
         actually marks 'global' is accepted as x_c; every off-grammar
         neighbor of an anchor is tried (up to max_c_candidates_per_anchor)
         before giving up on that anchor and moving to the next one.

    classifier_fn: seq -> (cls, depth_hist), e.g. classify_candidate()
    bound to your parent_arr/k/train_set/train_arr/local_max_depth via
    functools.partial or a lambda -- kept as an injected callable rather
    than importing classification_stats directly in here, so this
    function doesn't need to know about rho/parent_arr at all.
    """
    support_set = set(full_support_sequences)
    candidates = (anchor_candidates[:max_anchor_candidates] if max_anchor_candidates
                 else anchor_candidates)
    for x_a in candidates:
        x_b = next((n for n in hamming1_neighbors(x_a, vocab_size) if n in support_set), None)
        if x_b is None:
            continue
        x_c, x_c_cls, x_c_depth_hist = None, None, None
        n_tried = 0
        for cand in hamming1_neighbors(x_a, vocab_size):
            if cand in support_set:
                continue  # grammatical -- not a hallucination candidate at all
            n_tried += 1
            if max_c_candidates_per_anchor and n_tried > max_c_candidates_per_anchor:
                break
            cls, depth_hist = classifier_fn(cand)
            if float(cls.get("global", 0.0)) > 0.5:
                x_c, x_c_cls, x_c_depth_hist = cand, cls, depth_hist
                break
        if x_c is None:
            continue  # every off-grammar neighbor here was 'local', not 'global' -- try next anchor
        return x_a, x_b, x_c, x_c_cls, x_c_depth_hist
    raise RuntimeError(
        "no anchor among the searched candidates has both (a) a Hamming-1 "
        "valid neighbor in the FULL grammar, and (b) a Hamming-1 off-grammar "
        "neighbor that classification_stats classifies as 'global' "
        "(globally wrong), rather than merely 'local'. Try "
        "--max-anchor-candidates None (search everything), a smaller "
        "--local-max-depth (a stricter local/global split makes 'global' "
        "easier to reach with a single edit), or a different "
        "--pick-i-for-training block."
    )


# ---------------------------------------------------------------------
# 2. Embedding + plane -- identical to the toy script, generalized to
#    real (seq_len, vocab_size) instead of (N_LEAVES, ALPHABET).
# ---------------------------------------------------------------------

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


def project_to_discrete(emb, seq_len, vocab_size):
    """Per-leaf argmax decoding of a (possibly non-one-hot) embedding-space
    point back to a discrete sequence -- ported from the toy script.
    emb is a flat (seq_len*vocab_size,) vector; reshape to (seq_len,
    vocab_size) and take argmax along the vocab axis at every position."""
    grid = np.asarray(emb, dtype=np.float64).reshape(seq_len, vocab_size)
    return tuple(int(v) for v in np.argmax(grid, axis=1))


# ---------------------------------------------------------------------
# 1b. Checkpoint-directory discovery -- for sweeping every checkpoint
#    file found under a directory of an ALREADY-TRAINED run (as opposed
#    to --model-path for a single file). Pairs each .pt with the
#    full_params_reg*.json that sits alongside it (same directory first,
#    then its parent), since train_and_eval.py writes that json next to
#    the checkpoints it produces.
# ---------------------------------------------------------------------

def find_checkpoints_in_dir(model_dir):
    model_dir = Path(model_dir)
    ckpts = sorted(model_dir.rglob("*.pt"))
    if not ckpts:
        raise RuntimeError(f"no .pt files found under {model_dir}")
    pairs = []
    for ckpt in ckpts:
        params_json = None
        for d in (ckpt.parent, ckpt.parent.parent):
            candidates = sorted(d.glob("full_params_reg*.json"))
            if candidates:
                params_json = candidates[0]
                break
        pairs.append((ckpt, params_json))
        if params_json is None:
            print(f"WARNING: no full_params_reg*.json found near {ckpt} -- "
                  f"will fall back to CLI/default architecture flags for it")
    return pairs


# ---------------------------------------------------------------------
# 3. Exact BP score field on a grid -- real Tweedie/BP relationship,
#    batched (each grid point is one real BP forward pass, so batching
#    matters far more here than in the toy script's brute-force version).
# ---------------------------------------------------------------------

@torch.no_grad()
def score_field_on_grid_real(ea, v1, v2, t, alpha_bars, rho, k, vocab_size, seq_len,
                             factorized_layers, device, grid_n=20,
                             alpha_extent=(-0.5, 1.5), beta_extent=(-0.5, 1.5),
                             batch_size=256):
    alphas = np.linspace(*alpha_extent, grid_n)
    betas = np.linspace(*beta_extent, grid_n)
    AA, BB = np.meshgrid(alphas, betas)                       # (grid_n, grid_n) each
    flat_a, flat_b = AA.reshape(-1), BB.reshape(-1)            # (grid_n^2,)
    pts = ea[None, :] + flat_a[:, None] * v1[None, :] + flat_b[:, None] * v2[None, :]  # (N, D)
    N = pts.shape[0]

    ab_t = alpha_bars[t - 1].item()
    x0_hats = np.zeros_like(pts)
    for i in range(0, N, batch_size):
        sl = slice(i, min(i + batch_size, N))
        xt = torch.as_tensor(pts[sl], dtype=torch.float32, device=device).reshape(-1, seq_len, vocab_size)
        t_tensor = torch.full((xt.shape[0],), int(t), device=device, dtype=torch.long)
        bp = compute_bp_soft_targets(xt, t_tensor, alpha_bars, rho, k, vocab_size,
                                     factorized_layers=factorized_layers)   # (B, L, V)
        x0_hats[sl] = bp.reshape(xt.shape[0], -1).double().cpu().numpy()

    scores = (np.sqrt(ab_t) * x0_hats - pts) / (1.0 - ab_t)     # (N, D)
    U = (scores @ v1).reshape(grid_n, grid_n)
    V = (scores @ v2).reshape(grid_n, grid_n)
    return alphas, betas, U, V


@torch.no_grad()
def score_field_on_grid_model(ea, v1, v2, t, alpha_bars, model, seq_len, vocab_size,
                              device, grid_n=20, alpha_extent=(-0.5, 1.5),
                              beta_extent=(-0.5, 1.5), batch_size=256):
    """Score field from an ACTUAL trained checkpoint instead of a
    factorized-BP proxy -- same Tweedie formula, x0_hat =
    softmax(model(x_t, t)) instead of BP marginals. Called by main() when
    --model-path is given (see load_checkpoint_model())."""
    alphas = np.linspace(*alpha_extent, grid_n)
    betas = np.linspace(*beta_extent, grid_n)
    AA, BB = np.meshgrid(alphas, betas)
    flat_a, flat_b = AA.reshape(-1), BB.reshape(-1)
    pts = ea[None, :] + flat_a[:, None] * v1[None, :] + flat_b[:, None] * v2[None, :]
    N = pts.shape[0]

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
    U = (scores @ v1).reshape(grid_n, grid_n)
    V = (scores @ v2).reshape(grid_n, grid_n)
    return alphas, betas, U, V


# ---------------------------------------------------------------------
# 4. Plotting -- verbatim from the toy script (label placement etc. is
#    grammar-agnostic; only the data feeding it changed).
# ---------------------------------------------------------------------

def plot_panel_col(fig, gs_col, alphas, betas, U, V, title, coord_ab):
    mag = np.sqrt(U ** 2 + V ** 2)
    ax_top = fig.add_subplot(gs_col[0])
    im = ax_top.pcolormesh(alphas, betas, mag, shading="auto", cmap="viridis")
    step = max(1, len(alphas) // 16)
    ax_top.quiver(alphas[::step], betas[::step],
                  U[::step, ::step], V[::step, ::step],
                  color="white", scale=None, width=0.004)
    ax_top.set_title(title, fontsize=12, color="black")
    fig.colorbar(im, ax=ax_top, fraction=0.046, pad=0.04, label=r"$\|s_\theta\|$")

    ax_bot = fig.add_subplot(gs_col[1])
    vmax = np.abs(U).max()
    im2 = ax_bot.pcolormesh(alphas, betas, U, shading="auto", cmap="coolwarm",
                            vmin=-vmax, vmax=vmax)
    fig.colorbar(im2, ax=ax_bot, fraction=0.046, pad=0.04,
                label=r"signed attraction along $v_{ab}$")

    for ax in (ax_top, ax_bot):
        a_lo, a_hi = alphas[0], alphas[-1]
        b_lo, b_hi = betas[0], betas[-1]
        edge_frac = 0.15
        near_left = lambda a: (a - a_lo) < edge_frac * (a_hi - a_lo)
        near_right = lambda a: (a_hi - a) < edge_frac * (a_hi - a_lo)
        edge_frac_b = 0.22
        near_bottom = lambda b: (b - b_lo) < edge_frac_b * (b_hi - b_lo)
        near_top = lambda b: (b_hi - b) < edge_frac_b * (b_hi - b_lo)
        points = list(coord_ab.items())
        for i, (name, (a, b)) in enumerate(points):
            ax.plot(a, b, "o", ms=9, mfc="white", mec="black", mew=1.5, zorder=5)
            ha, dx = "left", 10
            if near_right(a):
                ha, dx = "right", -10
            elif near_left(a):
                ha, dx = "left", 10
            else:
                nearby = [(a2, b2) for j, (_, (a2, b2)) in enumerate(points)
                         if j != i and abs(a - a2) < 0.35 * (a_hi - a_lo)
                         and abs(b - b2) < 0.35 * (b_hi - b_lo)]
                if nearby:
                    other_a = min((a2 for a2, _ in nearby), key=lambda a2: abs(a - a2))
                    if a < other_a:
                        ha, dx = "right", -10
            dy = 14 if b >= 0.5 * (b_lo + b_hi) else -18
            if near_bottom(b):
                dy = 14
            elif near_top(b):
                dy = -18
            ax.annotate(name, (a, b), textcoords="offset points", xytext=(dx, dy),
                       ha=ha, fontsize=10, color="black")
        ax.set_xlabel(r"$\alpha$ (toward $x_b$)", fontsize=10)
        ax.set_ylabel(r"$\beta$ (toward $x_c$)", fontsize=10)


# Marker shape per companion point -- used by plot_vector_only_panel() INSTEAD
# of text labels, so different points are still distinguishable at a glance
# without cluttering the plot with annotation text.
# x_c and x_d are BOTH confirmed 'global' hallucinations -- share the same
# marker shape so both read as "hallucinated" at a glance, distinct from
# x_a (memorized) and x_b (valid).
POINT_MARKERS = {
    r"$x_a$ (memorized)": ("o", "x_a"),
    r"$x_b$ (valid synonym)": ("s", "x_b"),
    r"$x_c$ (hallucinated)": ("^", "x_c"),
    r"$x_d = x_c+x_b-x_a$": ("^", "x_d (hallucinated)"),
}


# Short in-plot text label per point (x_a/x_b/x_c/x_d) -- separate from
# POINT_MARKERS's longer legend text, kept terse since it's drawn right
# next to the marker on a possibly-small subplot.
POINT_SHORT_LABEL = {
    r"$x_a$ (memorized)": r"$x_a$",
    r"$x_b$ (valid synonym)": r"$x_b$",
    r"$x_c$ (hallucinated)": r"$x_c$",
    r"$x_d = x_c+x_b-x_a$": r"$x_d$",
}


def plot_vector_only_panel(fig, ax, alphas, betas, U, V, title, coord_ab, quiver_step=None):
    """Quiver + magnitude-heatmap background, no signed-attraction sub-
    panel. Each companion point gets a distinct marker SHAPE (see
    POINT_MARKERS) AND a short text label (x_a/x_b/x_c/x_d) right next to
    it. The figure-level legend mapping shapes to full names is added
    ONCE by the caller (see add_point_legend()), not per-panel, so it
    doesn't repeat for every checkpoint/step in a side-by-side row. The
    background gradient is the field magnitude ||s_theta|| =
    sqrt(U^2+V^2) (same quantity as the top sub-panel in
    plot_panel_col()), now rendered with a 'hot' heatmap colormap."""
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
    single legend entry rather than duplicated."""
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


import re

_STEP_RE = re.compile(r"step_(\d+)\.pt$")


def find_checkpoint_by_step(model_dir, step):
    """Locate the .pt checkpoint (and its full_params_reg*.json, if any)
    for one specific global step under model_dir -- matches filenames like
    test_model_script_step_{step}.pt (the pattern train_and_eval.py's
    --save-final-model writes).

    Uses an EXACT regex match on the trailing "step_<digits>.pt", not a
    substring test -- a substring check like "_1053." in name is fragile
    (can't fully rule out collisions with other numbers embedding the same
    digits) and gives no visibility into what actually matched. This
    walks every .pt under model_dir once, extracts its integer step via
    regex, and requires an EXACT integer equality against `step`."""
    model_dir = Path(model_dir)
    all_pt = sorted(model_dir.rglob("*.pt"))
    exact = []
    unparsed = []
    for p in all_pt:
        m = _STEP_RE.search(p.name)
        if m is None:
            unparsed.append(p)
            continue
        if int(m.group(1)) == int(step):
            exact.append(p)
    if unparsed:
        print(f"  (note: {len(unparsed)} .pt file(s) under {model_dir} didn't match "
              f"the 'step_<digits>.pt' filename pattern at all -- ignored; if your "
              f"checkpoints use a different naming scheme, this matcher needs "
              f"updating, e.g. {unparsed[0].name!r})")
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


# ---------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------

def main(args):
    device = setup_device()

    vocab_size, sequences, seq_len, k, rho, _ = load_data(args.data_path, device)
    if not isinstance(rho, torch.Tensor):
        rho = torch.as_tensor(np.asarray(rho))
    rho = rho.to(device).float()

    alpha_bars = alpha_bars_schedule(args.t_final, device=device, s=args.s, schedule="linear")
    alphas_sched, _ = get_alpha_beta_from_alpha_bar(alpha_bars)

    L = args.reduced_length
    tr_lo = args.pick_i_for_training * L
    tr_hi = tr_lo + L
    if tr_hi > len(sequences):
        raise ValueError(f"train block [{tr_lo}:{tr_hi}] out of bounds ({len(sequences)} seqs)")
    train_block = sequences[tr_lo:tr_hi].detach().cpu().numpy().astype(np.int64)
    train_block_tuples = [tuple(int(v) for v in row) for row in train_block]

    # Validity pool for the Hamming-1 neighbor search MUST be the full
    # enumerated grammar (all of `sequences`), not just the training block
    # -- see find_companions_nn()'s docstring for why restricting it to the
    # block starves the search almost immediately.
    full_sequences_np = sequences.detach().cpu().numpy().astype(np.int64)
    if args.max_full_support_sequences is not None:
        full_sequences_np = full_sequences_np[:args.max_full_support_sequences]
    full_support_tuples = [tuple(int(v) for v in row) for row in full_sequences_np]

    print(f"vocab_size={vocab_size}, seq_len={seq_len}, k={k}, "
          f"train block size={len(train_block_tuples)}, "
          f"full support pool size={len(full_support_tuples)}")

    # x_c must be classified 'global' by your OWN classifier -- not just
    # off-grammar. build_parent_array/classification_stats is your
    # project's established memorized/valid/local/global split, so x_c is
    # verified against THAT rather than a hand-rolled grammar parser (I
    # don't have rho's internal production schema to write one). See
    # classify_candidate()/find_companions_nn()'s docstrings for why
    # "off-grammar" alone isn't enough -- a 'local' violation is a real
    # break but NOT the locally-coherent/globally-wrong pattern this
    # project calls a hallucination.
    parent_arr = build_parent_array(rho)
    train_set = set(train_block_tuples)
    classifier_fn = lambda seq: classify_candidate(
        seq, parent_arr, k, train_set, train_block, args.local_max_depth)

    if args.x_a is not None and args.x_b is not None and args.x_c is not None:
        # Fixed companions supplied directly on the CLI -- bypass
        # find_companions_nn() entirely and use exactly what was given.
        x_a = tuple(int(v) for v in args.x_a.split(","))
        x_b = tuple(int(v) for v in args.x_b.split(","))
        x_c = tuple(int(v) for v in args.x_c.split(","))
        if not (len(x_a) == len(x_b) == len(x_c) == seq_len):
            raise ValueError(f"--x-a/--x-b/--x-c must each have seq_len={seq_len} "
                             f"entries; got lengths {len(x_a)}, {len(x_b)}, {len(x_c)}")
        x_c_cls, x_c_depth_hist = classifier_fn(x_c)
        print(f"x_a (FIXED, from CLI) = {x_a}")
        print(f"x_b (FIXED, from CLI) = {x_b}")
        print(f"x_c (FIXED, from CLI) = {x_c}")
        print(f"  Hamming(x_a,x_b) = {sum(p != q for p, q in zip(x_a, x_b))}, "
              f"Hamming(x_a,x_c) = {sum(p != q for p, q in zip(x_a, x_c))}")
        print(f"x_c classification (classification_stats): {x_c_cls}")
        print(f"x_c depth histogram: {x_c_depth_hist}")
    else:
        x_a, x_b, x_c, x_c_cls, x_c_depth_hist = find_companions_nn(
            train_block_tuples, full_support_tuples, vocab_size, classifier_fn,
            max_anchor_candidates=args.max_anchor_candidates,
            max_c_candidates_per_anchor=args.max_c_candidates_per_anchor)
        print(f"x_a (memorized anchor)                       = {x_a}")
        print(f"x_b (valid synonym, Hamming-1)                = {x_b}")
        print(f"x_c (confirmed GLOBAL hallucination, Hamming-1) = {x_c}")
        print(f"  Hamming(x_a,x_b) = {sum(p != q for p, q in zip(x_a, x_b))}, "
              f"Hamming(x_a,x_c) = {sum(p != q for p, q in zip(x_a, x_c))}")
        print(f"x_c classification (classification_stats): {x_c_cls}")
        print(f"x_c depth histogram: {x_c_depth_hist}")
    # NOTE: verify these prints against classification_stats' actual return
    # shape in your codebase -- I only had its call signature to go on
    # (from train_and_eval.py's run_eval()), not its implementation. If its
    # keys aren't literally 'global'/'local'/'valid'/'memorized', update
    # classify_candidate()'s cls.get("global", ...) check accordingly.

    ea, v1, v2 = build_plane(x_a, x_b, x_c, seq_len, vocab_size)

    # ------------------------------------------------------------------
    # x_d = x_c + x_b - x_a in EMBEDDING space (parallelogram completion):
    # e_d = e_b + e_c - e_a, so e_d - e_a = (e_b-e_a) + (e_c-e_a). Its plane
    # coordinates are (e_d-e_a)@v1, (e_d-e_a)@v2 -- NOT (1,1) in general,
    # since v1/v2 are UNIT vectors (build_plane normalizes them), not the
    # raw (x_b-x_a)/(x_c-x_a) directions themselves. x_d's coords are the
    # SUM of x_b's and x_c's own plane coords (computed below as
    # x_d_alpha/x_d_beta) -- that's what actually closes the parallelogram
    # in the plot, e.g. (sqrt(2), sqrt(2)) rather than (1, 1) for two
    # Hamming-1 edits at different leaves.
    # It is generically NOT one-hot (some leaf may get a negative or
    # >1 coordinate wherever x_b and x_c disagree with x_a at the same
    # position), so it's decoded to a discrete sequence via per-leaf
    # argmax (project_to_discrete) before classification.
    # ------------------------------------------------------------------
    eb_full = onehot_embed(x_b, seq_len, vocab_size)
    ec_full = onehot_embed(x_c, seq_len, vocab_size)
    ea_full = onehot_embed(x_a, seq_len, vocab_size)
    e_d = eb_full + ec_full - ea_full
    x_d = project_to_discrete(e_d, seq_len, vocab_size)
    x_d_cls, x_d_depth_hist = classifier_fn(x_d)
    # Plane coords of x_d, computed the same way as x_b/x_c below (NOT
    # hardcoded (1,1) -- v1/v2 are UNIT vectors, so x_b/x_c generally don't
    # sit at (1,0)/(0,1) either; they sit at (||eb-ea||,0) and
    # (0,||ec-ea||)-ish, e.g. (sqrt(2),0) and (0,sqrt(2)) for Hamming-1
    # edits at different leaves. x_d's true coords are the SUM of x_b's and
    # x_c's coords: ((eb-ea)+(ec-ea))@v1, ((eb-ea)+(ec-ea))@v2 -- this is
    # what actually closes the parallelogram/rectangle in the plot.)
    x_d_alpha = e_d_minus_ea = (eb_full + ec_full - 2 * ea_full) @ v1
    x_d_beta = (eb_full + ec_full - 2 * ea_full) @ v2
    print()
    print(f"x_d = x_c + x_b - x_a (embedding-space parallelogram completion):")
    print(f"  plane coords: (alpha, beta) = ({x_d_alpha:.4g}, {x_d_beta:.4g})")
    print(f"  decoded x_d (per-leaf argmax) = {x_d}")
    print(f"  x_d == x_a? {x_d == x_a}   x_d == x_b? {x_d == x_b}   x_d == x_c? {x_d == x_c}")
    print(f"  Hamming(x_a,x_d) = {sum(p != q for p, q in zip(x_a, x_d))}, "
          f"Hamming(x_b,x_d) = {sum(p != q for p, q in zip(x_b, x_d))}, "
          f"Hamming(x_c,x_d) = {sum(p != q for p, q in zip(x_c, x_d))}")
    print(f"  x_d classification (classification_stats): {x_d_cls}")
    print(f"  x_d depth histogram: {x_d_depth_hist}")
    is_global_halluc = float(x_d_cls.get("global", 0.0)) > 0.5
    print(f"  x_d is a genuine (GLOBAL) hallucination: {is_global_halluc}")

    coord_ab = {
        r"$x_a$ (memorized)": (0.0, 0.0),
        r"$x_b$ (valid synonym)": ((onehot_embed(x_b, seq_len, vocab_size) - ea) @ v1,
                                   (onehot_embed(x_b, seq_len, vocab_size) - ea) @ v2),
        r"$x_c$ (hallucinated)": ((onehot_embed(x_c, seq_len, vocab_size) - ea) @ v1,
                                  (onehot_embed(x_c, seq_len, vocab_size) - ea) @ v2),
        r"$x_d = x_c+x_b-x_a$": (x_d_alpha, x_d_beta),
    }

    all_alphas = [0.0] + [coord_ab[key][0] for key in coord_ab]
    all_betas = [0.0] + [coord_ab[key][1] for key in coord_ab]
    pad_frac = 0.25
    a_lo, a_hi = min(all_alphas), max(all_alphas)
    b_lo, b_hi = min(all_betas), max(all_betas)
    alpha_extent = (a_lo - pad_frac * max(a_hi - a_lo, 1e-6), a_hi + pad_frac * max(a_hi - a_lo, 1e-6))
    beta_extent = (b_lo - pad_frac * max(b_hi - b_lo, 1e-6), b_hi + pad_frac * max(b_hi - b_lo, 1e-6))
    print(f"grid extent: alpha={alpha_extent}, beta={beta_extent}")

    ab_t = alpha_bars[args.t - 1].item()
    print(f"t={args.t} (of t_final={args.t_final})  ->  alpha_bar_t={ab_t:.6g}, "
          f"1-alpha_bar_t={1 - ab_t:.6g}")

    # Oracle field never depends on a checkpoint -- compute it ONCE and
    # reuse it across every checkpoint in the sweep (mirrors
    # train_and_eval.py's oracle-cached-once-per-t pattern).
    alphas_o, betas_o, U_o, V_o = score_field_on_grid_real(
        ea, v1, v2, args.t, alpha_bars, rho, k, vocab_size, seq_len,
        factorized_layers=0, device=device, grid_n=args.grid_n,
        alpha_extent=alpha_extent, beta_extent=beta_extent, batch_size=args.batch_size)

    def make_figure(alphas_m, betas_m, U_m, V_m, proxy_title, out_path):
        fig = plt.figure(figsize=(13, 7.5))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
        plot_panel_col(fig, [gs[0, 0], gs[1, 0]], alphas_o, betas_o, U_o, V_o,
                      "oracle (exact BP, factorized_layers=0)", coord_ab)
        plot_panel_col(fig, [gs[0, 1], gs[1, 1]], alphas_m, betas_m, U_m, V_m,
                      proxy_title, coord_ab)
        fig.suptitle(rf"Attractor field, $t={args.t}$ ($\bar\alpha_t={ab_t:.3g}$) -- "
                    f"$x_a$ memorized, $x_b$ valid synonym, $x_c$ confirmed GLOBAL "
                    f"hallucination, $x_d=x_c+x_b-x_a$  "
                    f"(real RHM: seq_len={seq_len}, vocab_size={vocab_size}, k={k})",
                    fontsize=11.5, color="black")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")

    if args.checkpoint_steps is not None:
        # ONE figure, one quiver-only panel per requested step, side by
        # side -- no heatmap, no oracle column, no text labels (see
        # plot_vector_only_panel()/POINT_MARKERS). --out is a single file.
        if args.model_dir is None:
            raise ValueError("--checkpoint-steps requires --model-dir")
        steps = [int(s) for s in args.checkpoint_steps.split(",")]
        fig, axes = plt.subplots(1, len(steps), figsize=(5.5 * len(steps), 5.2))
        if len(steps) == 1:
            axes = [axes]
        param_checksums = []
        for ax, step in zip(axes, steps):
            ckpt_path, params_json = find_checkpoint_by_step(args.model_dir, step)
            print(f"--- requested step {step}: checkpoint {ckpt_path} (params_json={params_json}) ---")
            model, ckpt_step = load_checkpoint_model(
                str(ckpt_path), str(params_json) if params_json else args.model_params_json,
                vocab_size, seq_len, device,
                args.hidden_size_encoder, args.nhead, args.num_layers, args.model_channels)
            # DIAGNOSTIC: a simple checksum of the loaded weights. If this
            # number is IDENTICAL across all requested steps, the same
            # checkpoint file is being loaded every time (a real bug to
            # chase down in find_checkpoint_by_step / the directory
            # listing) rather than the model genuinely not having moved.
            checksum = sum(p.detach().float().sum().item() for p in model.parameters())
            param_checksums.append((step, ckpt_step, checksum))
            print(f"  weight checksum (sum of all params) = {checksum:.6f}")
            alphas_m, betas_m, U_m, V_m = score_field_on_grid_model(
                ea, v1, v2, args.t, alpha_bars, model, seq_len, vocab_size,
                device=device, grid_n=args.grid_n,
                alpha_extent=alpha_extent, beta_extent=beta_extent, batch_size=args.batch_size)
            # DIAGNOSTIC: quick numeric summary of the field itself, so a
            # "looks identical" visual impression can be checked against
            # actual numbers rather than pixel-eyeballing a saved PNG.
            print(f"  field summary: |U| mean/max = {np.abs(U_m).mean():.6g}/{np.abs(U_m).max():.6g}, "
                  f"|V| mean/max = {np.abs(V_m).mean():.6g}/{np.abs(V_m).max():.6g}")
            plot_vector_only_panel(fig, ax, alphas_m, betas_m, U_m, V_m,
                                   f"step {ckpt_step}", coord_ab)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print()
        print("weight-checksum summary across requested steps (identical values here "
              "mean the SAME checkpoint got loaded for multiple steps -- a real bug -- "
              "not that training didn't change anything):")
        for step, ckpt_step, checksum in param_checksums:
            print(f"  requested_step={step}  loaded_global_step={ckpt_step}  checksum={checksum:.6f}")
        fig.suptitle(rf"Trained model score field without regularization, $t={args.t}$ "
                    rf"($\bar\alpha_t={ab_t:.3g}$)",
                    fontsize=12, color="black")
        add_point_legend(fig)
        fig.tight_layout(rect=[0, 0, 0.93, 1])  # leave a bit of room on the right for the legend
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")
        return

    if args.model_dir is not None:
        # Sweep EVERY checkpoint found under the directory, reusing the
        # same fixed plane/oracle field for all of them. One output file
        # per checkpoint, named after the checkpoint's stem, under
        # --out (treated as an output DIRECTORY in this mode).
        pairs = find_checkpoints_in_dir(args.model_dir)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"found {len(pairs)} checkpoint(s) under {args.model_dir}")
        for ckpt_path, params_json in pairs:
            print(f"--- checkpoint {ckpt_path} (params_json={params_json}) ---")
            model, ckpt_step = load_checkpoint_model(
                str(ckpt_path), str(params_json) if params_json else args.model_params_json,
                vocab_size, seq_len, device,
                args.hidden_size_encoder, args.nhead, args.num_layers, args.model_channels)
            alphas_m, betas_m, U_m, V_m = score_field_on_grid_model(
                ea, v1, v2, args.t, alpha_bars, model, seq_len, vocab_size,
                device=device, grid_n=args.grid_n,
                alpha_extent=alpha_extent, beta_extent=beta_extent, batch_size=args.batch_size)
            proxy_title = f"trained model ({ckpt_path.name}, step {ckpt_step})"
            out_path = out_dir / f"{ckpt_path.stem}_t{args.t}.png"
            make_figure(alphas_m, betas_m, U_m, V_m, proxy_title, out_path)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    elif args.model_path is not None:
        model, ckpt_step = load_checkpoint_model(
            args.model_path, args.model_params_json, vocab_size, seq_len, device,
            args.hidden_size_encoder, args.nhead, args.num_layers, args.model_channels)
        alphas_m, betas_m, U_m, V_m = score_field_on_grid_model(
            ea, v1, v2, args.t, alpha_bars, model, seq_len, vocab_size,
            device=device, grid_n=args.grid_n,
            alpha_extent=alpha_extent, beta_extent=beta_extent, batch_size=args.batch_size)
        proxy_title = f"trained model (checkpoint, step {ckpt_step})"
        make_figure(alphas_m, betas_m, U_m, V_m, proxy_title, args.out)
    else:
        alphas_m, betas_m, U_m, V_m = score_field_on_grid_real(
            ea, v1, v2, args.t, alpha_bars, rho, k, vocab_size, seq_len,
            factorized_layers=args.factorized_layers_proxy, device=device, grid_n=args.grid_n,
            alpha_extent=alpha_extent, beta_extent=beta_extent, batch_size=args.batch_size)
        proxy_title = f"partial-tree proxy (factorized_layers={args.factorized_layers_proxy})"
        print("no --model-path/--model-dir given -- second column is the "
              "factorized_layers BP proxy, not an actual trained network.")
        make_figure(alphas_m, betas_m, U_m, V_m, proxy_title, args.out)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--data-path", type=str,
                   default="../data/labeled_data_restrictedfixed_6_4_1.0_4_0.npy")
    p.add_argument("--reduced-length", type=int, default=4096)
    p.add_argument("--pick-i-for-training", type=int, default=0)
    p.add_argument("--max-anchor-candidates", type=int, default=None,
                   help="Cap how many training-block sequences find_companions_nn() "
                        "scans looking for a valid anchor. Default: search the whole "
                        "block (None) -- with the validity pool now correctly set to "
                        "the FULL grammar (see --max-full-support-sequences), a match "
                        "is usually found quickly, so there's less need to cap this.")
    p.add_argument("--max-full-support-sequences", type=int, default=None,
                   help="Cap how much of the full `sequences` pool is used to test "
                        "Hamming-1 validity (NOT the anchor search -- see "
                        "find_companions_nn()'s docstring). Default: use all of it. "
                        "Set this only if `sequences` is large enough that building "
                        "the full tuple set is slow.")
    p.add_argument("--max-c-candidates-per-anchor", type=int, default=None,
                   help="Cap how many off-grammar Hamming-1 neighbors of a given "
                        "anchor are run through classification_stats looking for a "
                        "'global' one before giving up on that anchor. Default: try "
                        "all of them (there are at most seq_len*(vocab_size-1), so "
                        "this is cheap unless classification_stats itself is slow "
                        "per call).")

    p.add_argument("--x-a", type=str, default=None,
                   help="Comma-separated seq_len-length sequence to use as a FIXED "
                        "x_a (e.g. '3,1,2,5,4,5,3,2,3,3,1,0,4,0,1,0'). If --x-a, "
                        "--x-b and --x-c are ALL given, find_companions_nn() is "
                        "skipped entirely and these exact sequences are used.")
    p.add_argument("--x-b", type=str, default=None,
                   help="Comma-separated FIXED x_b -- see --x-a.")
    p.add_argument("--x-c", type=str, default=None,
                   help="Comma-separated FIXED x_c -- see --x-a.")

    p.add_argument("--t-final", type=int, default=500)
    p.add_argument("--s", type=float, default=1e-4)
    p.add_argument("--t", type=int, default=250,
                   help="The ACTUAL diffusion timestep the score field is evaluated "
                        "at -- an index into the real alpha_bars schedule, replacing "
                        "the toy script's ad hoc --sigma.")

    p.add_argument("--factorized-layers-proxy", type=int, default=2,
                   help="factorized_layers value used for the second ('model proxy') "
                        "column WHEN --model-path is not given. Ignored once a real "
                        "checkpoint is loaded.")
    p.add_argument("--local-max-depth", type=int, default=2)

    # checkpoint loading -- second column becomes an actual trained model
    # once --model-path is set (see load_checkpoint_model()).
    p.add_argument("--model-path", type=str, default=None,
                   help="Path to a .pt checkpoint from train_and_eval.py's "
                        "--save-final-model (dict with 'model_state_dict'). "
                        "If given, the second column plots this model's REAL "
                        "Tweedie score instead of the factorized_layers proxy.")
    p.add_argument("--model-params-json", type=str, default=None,
                   help="Path to the full_params_reg{tag}.json written alongside "
                        "the checkpoint -- reads hidden_size_encoder/nhead/"
                        "num_layers/model_channels from there so the reconstructed "
                        "architecture is guaranteed to match. Strongly recommended "
                        "over the --hidden-size-encoder/etc CLI flags below. Used as "
                        "a FALLBACK in --model-dir mode, for any checkpoint where no "
                        "full_params_reg*.json is auto-discovered nearby.")
    p.add_argument("--model-dir", type=str, default=None,
                   help="Directory of an ALREADY-TRAINED run to sweep every .pt "
                        "checkpoint under (recursively), reusing the SAME fixed "
                        "plane/companions/oracle field for each one. Each "
                        "checkpoint's full_params_reg*.json is auto-discovered next "
                        "to it (see find_checkpoints_in_dir()). When set, --out is "
                        "treated as an OUTPUT DIRECTORY (one PNG per checkpoint, "
                        "named after its stem) rather than a single file. Takes "
                        "priority over --model-path if both are given.")
    p.add_argument("--checkpoint-steps", type=str, default=None,
                   help="Comma-separated list of specific global steps to plot from "
                        "--model-dir (e.g. '1053,9490,53366'), matched against "
                        "filenames like test_model_script_step_{step}.pt. When given, "
                        "output is a SINGLE figure with one quiver-ONLY panel per "
                        "step, side by side -- no heatmap, no oracle column, no text "
                        "labels (companion points get distinct marker shapes instead, "
                        "see POINT_MARKERS). Requires --model-dir. --out is a single "
                        "file path in this mode (not a directory).")
    p.add_argument("--hidden-size-encoder", type=int, default=512,
                   help="Fallback architecture arg if --model-params-json is not "
                        "given -- MUST match how the checkpoint was actually "
                        "trained (train_and_eval.py's own default shown here).")
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--model-channels", type=int, default=32)

    p.add_argument("--grid-n", type=int, default=20,
                   help="Grid resolution. Every point is a real BP forward pass -- "
                        "start small and scale up once you've checked runtime.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out", type=str, default="attractor_real.png",
                   help="Output PNG path in the default/--model-path modes. In "
                        "--model-dir mode this is instead an OUTPUT DIRECTORY -- "
                        "one PNG per discovered checkpoint is written there.")
    main(p.parse_args())