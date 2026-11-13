"""
sequential_learning.py

Reduced version of the reference `sequential_learning.py`, restricted to the
functionality of the single-model staircase experiment:

  - D_KL(BP_k || model) for k = 0..ell   (one curve per rung of the ladder)
  - evaluated on held-out test sequences at a fixed critical diffusion time
  - tracked across all available training checkpoints

Everything belonging to the two-model sample-split analysis has been removed:
there is one model, one panel, one family of curves.

Usage (run from scripts/ directory)::

    python sequential_learning_reduced.py \
        --models-root /abs/path/to/repo \
        --subpath results_transformer_pick_0_for_training/.../8_512 \
        --output-root ../plots/
"""
import logging
from argparse import ArgumentParser
from pathlib import Path
import math
import tempfile
import os
import json
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Use insert(0) to ensure we load from this directory, overriding any PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.noise_schedules import alpha_bars_schedule, get_alpha_beta_from_alpha_bar
from modules.diffusion import forward_process, backward_process, backward_process_gt
from modules.bp_torch import run_BP_diffusion_torch
from utils import (
    load_params,
    load_data,
    setup_device,
    create_model,
    load_checkpoint,
    kl_divergence_batch,
)
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
 
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.frameon": False,
        "grid.alpha": 0.3,
    }
)
 
 
def mean_and_se(x: np.ndarray):
    x = np.asarray(x)
    n = x.size
    if n <= 1:
        return float(x.mean()), 0.0
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(n))
 
 
# ------------------------ SAMPLE CLASSIFICATION ------------------------
#
# Four mutually exclusive buckets over UNCONDITIONALLY GENERATED samples --
# a fresh batch drawn from pure noise at each checkpoint, independent of the
# noised test batch the BP_k divergences are computed on.
#
# `memorized` is checked first and takes precedence: a memorized sample is by
# construction also a valid parse, so it must be excluded from `valid` rather
# than counted twice. The four rates therefore sum to 1.
 
RATE_KEYS = ["memorized", "valid", "local", "global"]
 
# Order the stackplot bands are laid down, bottom first. Kept separate from
# RATE_KEYS so the log line, npz keys and json stay in their existing order.
PLOT_ORDER = ["memorized", "valid", "global", "local"]
 
 
def build_parent_array(M) -> np.ndarray:
    """
    parent[b, c] = a  if the production a -> bc exists, else -1.
 
    Unambiguity of the grammar (M[a,b,c] * M[a',b,c] = 0 for a != a')
    guarantees at most one parent per child pair. `M` is the transition
    tensor that load_data returns as `rho`.
    """
    if isinstance(M, torch.Tensor):
        M = M.detach().cpu().numpy()
    M = np.asarray(M)
    q = M.shape[0]
    parent = np.full((q, q), -1, dtype=np.int64)
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if M[a, b, c] > 0:
                    parent[b, c] = a
    return parent
 
 
def violation_depth_batch(seqs: np.ndarray, parent_arr: np.ndarray, ell: int) -> np.ndarray:
    """
    Bottom-up parse of a whole batch at once. Returns an (n,) array giving the
    first merge depth at which the parse fails, or -1 for a valid sequence.
 
    Depth counts merge steps, so d is in 1..ell: d=1 means adjacent leaves form
    no legal production (most local), d=ell means the two halves are each
    internally grammatical but do not combine at the root (most global).
    """
    cur = np.asarray(seqs, dtype=np.int64).copy()
    fail = np.full(cur.shape[0], -1, dtype=np.int64)
    for d in range(1, ell + 1):
        b, c = cur[:, 0::2], cur[:, 1::2]
        nxt = parent_arr[b, c]
        bad = (nxt == -1).any(axis=1) & (fail == -1)
        fail[bad] = d
        # rows that already failed keep parsing on a dummy symbol; their
        # fail depth is pinned above and never overwritten
        cur = np.where(nxt == -1, 0, nxt)
    return fail
 
 
@torch.inference_mode()
def generate_samples(model, n_gen, seq_len, vocab_size, t_final, sched, device,
                     temperature=1.0, clamp=False, deterministic=False):
    """
    Ancestral sampling through the reverse process; returns (n, L) ints.
 
    Deliberately NOT routed through modules.diffusion.backward_process, which
    differs in two ways that matter over a 500-step trajectory:
      - it passes a length-1 time tensor rather than one entry per batch
        element, relying on broadcasting inside the denoiser;
      - it wraps the model call in torch.amp.autocast('cuda'), so the whole
        trajectory runs in fp16.
    This mirrors the reference sampler exactly instead.
 
    clamp: project x0_hat onto the nearest one-hot each step.
    deterministic: posterior mean only, no injected noise.
    """
    model.eval()
    ab = sched["alpha_bars"]
    alpha, beta = get_alpha_beta_from_alpha_bar(ab)
 
    x = torch.randn(n_gen, seq_len, vocab_size, device=device)
    for step in range(int(t_final), 0, -1):
        t = torch.full((n_gen,), step, device=device, dtype=torch.long)
        logits = model(x, t)
        if clamp:
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
 
 
def classification_stats(gen, parent_arr, ell, train_set, train_arr, local_max_depth=2):
    """
    Partition generated samples into memorized / valid / local / global, plus
    the mean Hamming distance to the nearest training sequence.
 
    local_max_depth splits the depth axis into "local" and "global" failures.
    With ell=4 the default of 2 gives depths {1,2} local and {3,4} global.
    """
    n = len(gen)
    mem_flags = np.array([tuple(s) in train_set for s in gen])
    depths = violation_depth_batch(gen, parent_arr, ell)
    novel = ~mem_flags
 
    n_valid = int((novel & (depths == -1)).sum())
    n_local = int((novel & (depths >= 1) & (depths <= local_max_depth)).sum())
    n_global = int((novel & (depths > local_max_depth)).sum())
 
    ham = float((gen[:, None, :] != train_arr[None, :, :]).sum(-1).min(axis=1).mean())
 
    # Degeneracy guard. A collapsed model emits near-constant sequences, and
    # since several diagonal pairs (s, s) are legal productions, those climb
    # a few merge levels by coincidence and score as "global" rather than
    # "local". n_distinct and the depth histogram make that visible instead
    # of letting it masquerade as global-only incoherence.
    n_distinct = len({tuple(s) for s in gen})
    depth_hist = {int(d): int((depths == d).sum()) for d in range(-1, ell + 1)
                  if int((depths == d).sum()) > 0}
 
    return {
        "memorized": float(mem_flags.mean()),
        "valid": n_valid / n,
        "local": n_local / n,
        "global": n_global / n,
        "nn_hamming": ham,
        "n_distinct": float(n_distinct),
    }, depth_hist
 
 
def safe_block_indices(
    *, num_seqs: int, reduced_length: int, train_block_i: int, test_block_i: int = 15, device=None
):
    """
    Train block is train_block_i, test block is test_block_i.
    Falls back deterministically if the test block is invalid or collides
    with the train block.
    """
    max_blocks = num_seqs // reduced_length
    if max_blocks <= 1:
        raise ValueError(f"Not enough sequences ({num_seqs}) for reduced_length={reduced_length}.")
 
    def block_to_idx(bi: int) -> torch.Tensor:
        left = bi * reduced_length
        right = left + reduced_length
        if left < 0 or right > num_seqs:
            raise ValueError(
                f"Block {bi} out of bounds for num_seqs={num_seqs}, reduced_length={reduced_length}"
            )
        return torch.arange(left, right, device=device)
 
    train_block_i = int(train_block_i)
    test_block_i = int(test_block_i)
 
    train_idx = block_to_idx(train_block_i)
 
    if (0 <= test_block_i < max_blocks) and (test_block_i != train_block_i):
        return train_idx, block_to_idx(test_block_i), test_block_i
 
    fb = 0 if train_block_i != 0 else 1
    if fb >= max_blocks:
        raise ValueError("Could not find a fallback test block.")
    return train_idx, block_to_idx(fb), fb
 
 
def get_schedule(schedule_cache: dict, *, t_test: int, s: float, device):
    """Cache the diffusion schedule (alpha_bars/alphas); it is reused heavily."""
    key = (int(t_test), float(s), device.type)
    if key in schedule_cache:
        return schedule_cache[key]
 
    alpha_bars = alpha_bars_schedule(int(t_test), device=device, s=float(s), schedule="linear")
    alphas, _ = get_alpha_beta_from_alpha_bar(alpha_bars)
 
    out = {
        "alpha_bars": alpha_bars,
        "alphas": alphas,
        "alpha_bars_np": alpha_bars.detach().cpu().numpy(),
        "alphas_np": alphas.detach().cpu().numpy(),
    }
    schedule_cache[key] = out
    return out
 
 
def build_experiment(path: Path, device, do_compile: bool):
    """Load params + data + model for the single experiment directory."""
    path = Path(path)
    params = load_params(path / "full_params.json")
 
    vocab_size, sequences, seq_len, k, rho, _ = load_data(params["data_path"], device)
    model, _, _ = create_model(params, vocab_size, seq_len, device)
 
    if do_compile:
        model = torch.compile(model)
 
    model.eval()
 
    return dict(
        path=path,
        params=params,
        vocab_size=vocab_size,
        sequences=sequences,
        seq_len=seq_len,
        k=k,
        rho=rho,
        model=model,
    )


def average_pairwise_hamming(sequences):
    """
    Average nearest-neighbor Hamming distance between distinct
    training sequences.

    sequences: [N, L]
    """
    N, L = sequences.shape

    # Compute pairwise distances in chunks to avoid constructing
    # the full [N, N, L] comparison tensor.
    chunk_size = 1024
    nn_distances = torch.full(
        (N,),
        float("inf"),
        device=sequences.device
    )

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)

        # [chunk, 1, L] vs [1, N, L]
        chunk_sequences = sequences[start:end]

        distances = (
            chunk_sequences[:, None, :] != sequences[None, :, :]
        ).sum(dim=-1)

        # Exclude self-comparison
        row_indices = torch.arange(
            start,
            end,
            device=sequences.device
        )

        col_indices = torch.arange(
            end - start,
            device=sequences.device
        )

        distances[col_indices, row_indices] = L + 1

        # Nearest neighbor for each sequence in this chunk
        nn_distances[start:end] = distances.min(dim=1).values

    return nn_distances.float().mean()
    
 
def load_epoch_weights(exp: dict, ckpt_path: Path, device):
    if (not ckpt_path.exists()) or (
        not load_checkpoint(exp["model"], str(ckpt_path), device=device, training=False)
    ):
        raise RuntimeError(f"Checkpoint load failed: {ckpt_path}")
    exp["model"].eval()
 
 
# ----------------------------- BP (FACTORISED) -----------------------------
 
 
def _compute_bp_factorized_np(
    x_noised_np: np.ndarray,
    *,
    t_noise: int,
    bp_params: tuple,
    sched: dict,
    factorized_layers: int,
) -> np.ndarray:
    """BP marginals for a given factorized_layers, numpy (CPU) path."""
    rho, k, vocab_size = bp_params
    _, bp_x0 = backward_process_gt(
        x_noised_np,
        int(t_noise),
        (rho, k, vocab_size),
        device=None,
        alpha_bars=sched["alpha_bars_np"],
        alphas=sched["alphas_np"],
        fix_noise=False,
        factorized_layers=int(factorized_layers),
    )
    if isinstance(bp_x0, torch.Tensor):
        bp_x0 = bp_x0.detach().cpu().numpy()
    return bp_x0
 
 
def _compute_bp_factorized_torch(
    x_noised: torch.Tensor,
    *,
    t_noise: int,
    bp_params: tuple,
    sched: dict,
    factorized_layers: int,
    device,
) -> np.ndarray:
    """BP marginals for a given factorized_layers, PyTorch (GPU) path."""
    M, l, q = bp_params
 
    if isinstance(M, np.ndarray):
        M = torch.from_numpy(M).float().to(device)
    else:
        M = M.to(device).float()
 
    alpha_bars = sched["alpha_bars"]
    if isinstance(alpha_bars, np.ndarray):
        field_intensity = np.sqrt(alpha_bars[t_noise - 1]) / (1 - alpha_bars[t_noise - 1])
        field_intensity = torch.tensor(field_intensity, device=device, dtype=torch.float32)
    else:
        field_intensity = torch.sqrt(alpha_bars[t_noise - 1]) / (1 - alpha_bars[t_noise - 1])
        field_intensity = field_intensity.to(device)
 
    field = x_noised * field_intensity
    bp_marginals = run_BP_diffusion_torch(M, l, q, field, factorized_layers=int(factorized_layers))
    return bp_marginals.detach().cpu().numpy()
 
 
def get_cached_bp_factorized(pack: dict, factorized_layers: int, backend: str = "numpy") -> np.ndarray:
    """Cache BP(f) inside the pack, keyed by backend and f."""
    f = int(factorized_layers)
    cache_key = f"bp_preds_factorized_{backend}"
    cache = pack.setdefault(cache_key, {})
    if f in cache:
        return cache[f]
 
    if backend == "torch":
        bp = _compute_bp_factorized_torch(
            pack["x_noised"],
            t_noise=int(pack["t_noise"]),
            bp_params=pack["bp_params"],
            sched=pack["schedule"],
            factorized_layers=f,
            device=pack["x_noised"].device,
        )
    else:
        if pack.get("x_noised_np") is None:
            pack["x_noised_np"] = pack["x_noised"].detach().cpu().numpy()
        bp = _compute_bp_factorized_np(
            pack["x_noised_np"],
            t_noise=int(pack["t_noise"]),
            bp_params=pack["bp_params"],
            sched=pack["schedule"],
            factorized_layers=f,
        )
 
    cache[f] = bp
    return bp
 
 
# ----------------------------- BP PACK (TEST ONLY) -----------------------------
 
 
def get_bp_test_pack(
    *,
    exp: dict,
    device,
    schedule_cache: dict,
    t_final_test_override: Optional[int],
    t_denoising_ratio: float,
    N_test: int,
    reps: int,
    test_block_i: int,
):
    """
    Noise a held-out test block at the critical time; BP(f) is computed lazily
    on top of this and cached in the returned pack.
    """
    params = exp["params"]
    sequences = exp["sequences"]
    vocab_size = exp["vocab_size"]
    rho, k = exp["rho"], exp["k"]
 
    reduced_length = int(params["reduced_length"])
    pick_i = int(params.get("pick_i_for_training", 0) or 0)
 
    t_final = int(params["t_final"])
    t_test = int(t_final_test_override) if t_final_test_override is not None else t_final
    t_noise = int(t_denoising_ratio * t_final)
 
    sched = get_schedule(schedule_cache, t_test=t_test, s=float(params["s"]), device=device)
 
    _, test_idx, used_test_block = safe_block_indices(
        num_seqs=sequences.shape[0],
        reduced_length=reduced_length,
        train_block_i=pick_i,
        test_block_i=test_block_i,
        device=sequences.device,
    )
    if int(used_test_block) != int(test_block_i):
        logger.warning(
            "Requested test_block_i=%d invalid/overlapping; using fallback test_block=%d",
            int(test_block_i),
            int(used_test_block),
        )
 
    test_seqs = sequences[test_idx]
    test_oh = torch.nn.functional.one_hot(test_seqs.long(), num_classes=vocab_size).float().to(device)
    x_test_noised = forward_process(
        test_oh[:N_test].repeat(reps, 1, 1),
        torch.tensor(t_noise, device=device, dtype=torch.long),
        alpha_bars=sched["alpha_bars"],
    )
 
    return dict(
        schedule=sched,
        t_final=t_final,
        t_test=t_test,
        t_noise=t_noise,
        test_block_i=int(test_block_i),
        used_test_block=int(used_test_block),
        x_noised=x_test_noised,
        x_noised_np=None,
        bp_params=(rho, k, vocab_size),
        vocab_size=int(vocab_size),
        seq_len=int(exp["seq_len"]),
        tree_depth=int(k),
    )
 
 
# ------------------------- MODEL PREDICTION -------------------------
 
 
def denoise(
    *,
    model,
    x_noised: torch.Tensor,
    t_noise: int,
    t_final: int,
    sched: dict,
    device,
) -> np.ndarray:
    """Model posterior mean x0_hat at the fixed critical time."""
    with torch.inference_mode():
        _, x0_hat = backward_process(
            x_noised,
            int(t_noise),
            int(t_final),
            model,
            device,
            alpha_bars=sched["alpha_bars"],
            alphas=sched["alphas"],
            clamp=False,
            temperature=None,
            fix_noise=False,
        )
    return x0_hat.detach().cpu().numpy()
 
 
# ------------------------------ ATOMIC SAVES ------------------------------
 
 
def _atomic_savefig(fig, out_path: Path, **savefig_kwargs):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(out_path.parent), suffix=out_path.suffix)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        fig.savefig(tmp_path, **savefig_kwargs)
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
 
 
def _atomic_savez(out_path: Path, **npz_kwargs):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(out_path.parent), suffix=out_path.suffix)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez(tmp_path, **npz_kwargs)
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
 
 
def save_summary_json(out_path: Path, data: dict):
    def to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
 
    def convert_structure(d):
        if isinstance(d, dict):
            return {k: convert_structure(v) for k, v in d.items()}
        if isinstance(d, list):
            return [convert_structure(i) for i in d]
        return to_list(d)
 
    with open(out_path, "w") as f:
        json.dump(convert_structure(data), f, indent=2)
 
 
# ------------------------------ PLOTTING ------------------------------
 
 
def plot_and_save(
    *,
    args_obj,
    plots_dir: Path,
    epochs_plot,
    title_time: int,
    t_denoising_ratio: float,
    reduced_length: int,
    reps: int,
    N_test: int,
    kl_bp_f: Dict[int, List[float]],
    kl_bp_f_se: Dict[int, List[float]],
    rates: Optional[Dict[str, List[float]]] = None,
    tag: str = "latest",
    job_id: str = None,
    ckpt_tag: str = "step",
):
    """
    Left panel:  D_KL(BP_k || model) vs epoch, one curve per rung.
    Right panel: sample classification rates vs epoch (only if `rates` given).
    """
    colors_f = ["#d62728", "#1f77b4", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2"]
 
    has_rates = bool(rates) and len(rates.get("valid", [])) > 0
    if has_rates:
        fig, (ax, axr) = plt.subplots(1, 2, figsize=(14.5, 6))
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        axr = None
    epochs = np.asarray(epochs_plot, dtype=float)
 
    for f in sorted(kl_bp_f.keys()):
        col = colors_f[f % len(colors_f)]
        vals = np.asarray(kl_bp_f[f], dtype=float)
        errs = np.asarray(kl_bp_f_se[f], dtype=float)
        label = r"exact score  $\mathrm{BP}_0$" if f == 0 else f"BP(f={f})"
        ax.plot(epochs, vals, "-o", ms=3, lw=2.0, color=col, label=label, alpha=0.85)
        ax.fill_between(epochs, vals - errs, vals + errs, alpha=0.12, color=col, lw=0)
 
    # mark the model-oracle minimum (closest approach to the exact score)
    if 0 in kl_bp_f and len(kl_bp_f[0]) > 0:
        i_exact = int(np.argmin(np.asarray(kl_bp_f[0], dtype=float)))
        ax.axvline(epochs[i_exact], color="#d62728", ls=":", lw=1.4)
    
    x_axis_label = "Global step" if ckpt_tag == "step" else "Epoch"
    ax.set_xlabel(x_axis_label, fontsize=13)
    ax.set_ylabel(r"$D_{\mathrm{KL}}(\mathrm{BP}_k \,\|\, \mathrm{model})$", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("position on the ladder (test)", fontsize=14)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(fontsize=9)
 
    if has_rates:
        rate_cols = {
            "memorized": "#7f7f7f",
            "valid": "#2ca02c",
            "local": "#d62728",
            "global": "#ff7f0e",
        }
        axr.stackplot(
            epochs,
            *[np.asarray(rates[k], dtype=float) for k in PLOT_ORDER],
            labels=PLOT_ORDER,
            colors=[rate_cols[k] for k in PLOT_ORDER],
            alpha=0.85,
        )
        axr.set_xscale("log")
        if epochs.size > 1:
            axr.set_xlim(epochs.min(), epochs.max())
        axr.set_ylim(0, 1)
        axr.set_xlabel(x_axis_label, fontsize=13)
        axr.set_ylabel("fraction of generated samples", fontsize=12)
        axr.set_title("sample classification", fontsize=14)
        axr.grid(True, linestyle=":", alpha=0.4)
        axr.legend(fontsize=9, loc="upper left", framealpha=0.8)
 
    fig.suptitle(
        f"KL with Factorized BP at time {title_time} (ratio={t_denoising_ratio})",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
 
    base = (
        f"KL_FACT_train_size_{reduced_length}_reps_{reps}_Ntest_{N_test}_"
        f"ratio_{t_denoising_ratio}_{tag}"
    )
    if job_id:
        base = f"{base}_{job_id}"
 
    out_png = plots_dir / f"{base}.png"
    out_npz = plots_dir / f"{base}.npz"
    out_json = plots_dir / f"{base}.json"
 
    _atomic_savefig(fig, out_png, bbox_inches="tight")
    plt.close(fig)
 
    save = {
        "args": {k: str(v) for k, v in vars(args_obj).items()},
        "epochs": np.asarray(epochs_plot),
    }
    for f, vals in kl_bp_f.items():
        save[f"kl_bp_f{f}"] = np.asarray(vals)
        save[f"kl_bp_f{f}_se"] = np.asarray(kl_bp_f_se[f])
 
    if rates:
        for key, vals in rates.items():
            save[key] = np.asarray(vals)
 
    _atomic_savez(out_npz, **save)
    save_summary_json(out_json, save)
 
    logger.info("Saved plot: %s", out_png)
    logger.info("Saved data: %s", out_npz)
    logger.info("Saved json: %s", out_json)
 
 
# ------------------------------ MAIN ------------------------------
 
 
def main(args):
    device = setup_device()
 
    try:
        f_list = [int(x) for x in args.bp_factorized_layers.split(",")]
    except ValueError:
        raise ValueError(f"Invalid format for --bp-factorized-layers: {args.bp_factorized_layers}")
 
    model_dir = Path(args.models_root) / args.subpath
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model directory: {model_dir}")
    logger.info("Model: %s", model_dir)
 
    exp = build_experiment(model_dir, device, do_compile=(not args.no_compile))
    reduced_length = int(exp["params"]["reduced_length"])
    logger.info("reduced_length (from full_params.json): %d", reduced_length)
 
    _plots_base = Path(args.output_root) if args.output_root else Path("..") / "plots"
    plots_dir = (
        _plots_base
        / "exp6_factorized"
        / f"N_train_{reduced_length}"
        / f"N_test_{args.N_test}"
        / f"ratio_{args.t_denoising_ratio}"
        / f"backend_{args.bp_backend}"
    )
    plots_dir.mkdir(parents=True, exist_ok=True)
 
    schedule_cache: Dict[Tuple[Any, ...], dict] = {}
 
    pack = get_bp_test_pack(
        exp=exp,
        device=device,
        schedule_cache=schedule_cache,
        t_final_test_override=args.t_final_test,
        t_denoising_ratio=args.t_denoising_ratio,
        N_test=args.N_test,
        reps=args.reps,
        test_block_i=args.test_block_i,
    )
    title_time = int(pack["t_noise"])
    tree_depth = int(pack["tree_depth"])
 
    for f in f_list:
        if not (0 <= f <= tree_depth):
            raise ValueError(f"factorized_layers={f} invalid for tree_depth={tree_depth}")
 
    kl_bp_f = {f: [] for f in f_list}
    kl_bp_f_se = {f: [] for f in f_list}
 
    # --- setup for sample classification (skipped when --n-gen 0) ---
    rates: Optional[Dict[str, List[float]]] = None
    parent_arr = train_set = train_arr = None
    if args.n_gen > 0:
        parent_arr = build_parent_array(exp["rho"])
        train_idx, _, _ = safe_block_indices(
            num_seqs=exp["sequences"].shape[0],
            reduced_length=reduced_length,
            train_block_i=int(exp["params"].get("pick_i_for_training", 0) or 0),
            test_block_i=args.test_block_i,
            device=exp["sequences"].device,
        )
        train_arr = exp["sequences"][train_idx].detach().cpu().numpy().astype(np.int64)
        train_set = set(map(tuple, train_arr))
        rates = {k: [] for k in RATE_KEYS}
        rates["nn_hamming"] = []
        rates["n_distinct"] = []
        n_allowed = int((parent_arr >= 0).sum())
        diag_legal = [s for s in range(parent_arr.shape[0]) if parent_arr[s, s] >= 0]
        logger.info(
            "Classification enabled: n_gen=%d unconditional samples per checkpoint; "
            "train block has %d sequences (%d distinct); grammar admits %d of %d "
            "child pairs (legal diagonal pairs (s,s): %s -- constant sequences can "
            "climb several levels by coincidence); local/global split at depth %d of ell=%d",
            args.n_gen,
            len(train_arr),
            len(train_set),
            n_allowed,
            parent_arr.size,
            diag_legal,
            args.local_max_depth,
            tree_depth,
        )
 
    logger.info("Scanning for checkpoints...")
    if args.ckpt_tag == "epoch":
            ckpts = list(model_dir.glob("test_model_script_epoch_*.pt"))
    elif args.ckpt_tag == "step":
            ckpts = list(model_dir.glob("test_model_script_step_*.pt"))
    
    def _parse_ckpt(c: Path):
        stem = c.stem  # e.g. "test_model_script_epoch_1134" or "..._step_15000"
        if "_epoch_" in stem:
            tag, num_str = "epoch", stem.split("_epoch_")[1]
        elif "_step_" in stem:
            tag, num_str = "step", stem.split("_step_")[1]
        else:
            return None
        if not num_str.isdigit():
            return None
        return tag, int(num_str), c
    
    parsed = [r for r in (_parse_ckpt(c) for c in ckpts) if r is not None]
    if not parsed:
        raise RuntimeError(f"No checkpoints found in: {model_dir}")
    
    tags_present = {tag for tag, _, _ in parsed}
    if len(tags_present) > 1:
        raise RuntimeError(
            f"Mixed epoch- and step-named checkpoints in {model_dir}: {tags_present}. "
            "Evaluate one naming convention at a time."
        )
    ckpt_tag = tags_present.pop()
    
    # number -> path, deduplicated (in case of stray duplicates)
    ckpt_by_num = {num: path for _, num, path in parsed}
    epochs_avail = sorted(ckpt_by_num.keys())
    
    logger.info(
        "Found %d checkpoints (%s-based). Range: %d .. %d",
        len(epochs_avail), ckpt_tag, epochs_avail[0], epochs_avail[-1],
    )
 
    for epoch in epochs_avail:
        logger.info("################ Processing epoch %d ################", epoch)
        load_epoch_weights(exp, ckpt_by_num[epoch], device)
 
        pred = denoise(
            model=exp["model"],
            x_noised=pack["x_noised"],
            t_noise=pack["t_noise"],
            t_final=pack["t_final"],
            sched=pack["schedule"],
            device=device,
        )
 
        for f in f_list:
            bp_preds = get_cached_bp_factorized(pack, f, backend=args.bp_backend)
            m, se = mean_and_se(kl_divergence_batch(bp_preds, pred))
            kl_bp_f[f].append(m)
            kl_bp_f_se[f].append(se)
 
        if args.n_gen > 0:
            gen = generate_samples(
                exp["model"],
                args.n_gen,
                int(exp["seq_len"]),
                int(exp["vocab_size"]),
                pack["t_final"],
                pack["schedule"],
                device,
            )
            cls, depth_hist = classification_stats(
                gen, parent_arr, tree_depth, train_set, train_arr,
                local_max_depth=args.local_max_depth,
            )
            for key, val in cls.items():
                rates[key].append(val)
 
        log_parts = [" ".join(f"bp{f} {kl_bp_f[f][-1]:.4f}" for f in f_list)]
        if args.n_gen > 0:
            log_parts.append(
                f"mem {cls['memorized']:.3f}  valid {cls['valid']:.3f}  "
                f"local {cls['local']:.3f}  global {cls['global']:.3f}  "
                f"ham {cls['nn_hamming']:.2f}  "
                f"distinct {int(cls['n_distinct'])}/{args.n_gen}  "
                f"depths {depth_hist}"
            )
        logger.info("epoch %d | %s", epoch, "  |  ".join(log_parts))
 
        plot_and_save(
            args_obj=args,
            plots_dir=plots_dir,
            epochs_plot=epochs_avail[: len(kl_bp_f[f_list[0]])],
            title_time=title_time,
            t_denoising_ratio=args.t_denoising_ratio,
            reduced_length=reduced_length,
            reps=args.reps,
            N_test=args.N_test,
            kl_bp_f=kl_bp_f,
            kl_bp_f_se=kl_bp_f_se,
            rates=rates,
            tag="latest",
            job_id=args.job_id,
            ckpt_tag=args.ckpt_tag,   # NEW
        )
 
 
if __name__ == "__main__":
    p = ArgumentParser()
 
    p.add_argument("--models-root", type=str, required=True,
                   help="Absolute root directory containing the model dir.")
    p.add_argument("--subpath", type=str, required=True,
                   help="Subpath to the model dir (relative to --models-root).")
 
    p.add_argument("--N-test", dest="N_test", type=int, default=1000)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--t-final-test", dest="t_final_test", type=int, default=None)
    p.add_argument("--t-denoising-ratio", type=float, default=0.15)
    p.add_argument("--test-block-i", type=int, default=0, help="Preferred test block index.")
 
    p.add_argument("--n-gen", type=int, default=256,
                   help="Unconditional samples generated per checkpoint for classification. "
                        "0 disables classification.")
    p.add_argument("--local-max-depth", type=int, default=2,
                   help="Parse depths <= this count as 'local' violations, above as 'global'.")
 
    p.add_argument("--no-compile", action="store_true", help="Disable torch.compile.")
    p.add_argument("--bp-factorized-layers", type=str, default="0,1,2,3,4",
                   help="Comma-separated factorized_layers to sweep. 0 means exact.")
    p.add_argument("--bp-backend", type=str, choices=["torch", "numpy"], default="torch",
                   help="Backend for BP computation: 'torch' (GPU) or 'numpy' (CPU).")
    p.add_argument("--output-root", type=str, default=None,
                   help="Root dir for output plots (default: ../plots)")
    p.add_argument("--job-id", type=str, default=None,
                   help="SLURM job ID to append to output files.")
    p.add_argument("--ckpt-tag", type=str, choices=["epoch", "step"], default="step")
 
    args = p.parse_args()
    main(args)