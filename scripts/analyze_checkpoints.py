"""
analyze_checkpoints.py

Post-hoc ladder diagnostic: build ONE LadderEvaluator (frozen (x_t,t) draws +
oracle ladder, computed once) and sweep it over every saved checkpoint of a
trained model. Decoupled from training entirely -- run any time after the
fact, against any eval block you choose.

Usage (run from scripts/ directory, same layout as train_transformer.py)::

    python analyze_checkpoints.py \
        --model-dir /abs/path/to/results_transformer_.../8_512 \
        --ladder-batch-size 512

By default this evaluates on the SAME block the model was trained on
(pick_i_for_training from full_params.json) -- a train-set diagnostic, same
scope as the --ladder-live path in train_transformer.py. Pass
--eval-block-i to point at a different (held-out) block instead; see the
NOTE in build_eval_set() for what "held out" means here.
"""
import os
import re
import sys
import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SCRIPT_DIR = Path(__file__).resolve().parent   # scripts/ -- data_path in full_params.json
                                                # was written relative to this directory
                                                # (train_transformer.py's own default,
                                                # e.g. "../data/...npy"), NOT the launch cwd.

from modules.noise_schedules import alpha_bars_schedule, get_alpha_beta_from_alpha_bar
from modules.ladder_diagnostics import LadderEvaluator, level_excess
from utils import load_data, create_model, setup_device, load_checkpoint, load_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# --------------------------- checkpoint discovery ---------------------------

_CKPT_RE = re.compile(r"test_model_script_(epoch|step)_(\d+)\.pt$")


def find_checkpoints(model_dir: Path):
    """
    Returns (tag, sorted list of (num, path)) where tag is 'epoch' or 'step'.
    Mirrors sequential_learning_reduced.py's _parse_ckpt: refuses to mix
    epoch- and step-named checkpoints in the same directory.
    """
    parsed = []
    for p in model_dir.glob("test_model_script_*.pt"):
        m = _CKPT_RE.match(p.name)
        if m:
            parsed.append((m.group(1), int(m.group(2)), p))
    if not parsed:
        raise RuntimeError(f"No checkpoints found in: {model_dir}")

    tags_present = {tag for tag, _, _ in parsed}
    if len(tags_present) > 1:
        raise RuntimeError(
            f"Mixed epoch- and step-named checkpoints in {model_dir}: {tags_present}. "
            "Evaluate one naming convention at a time."
        )
    tag = tags_present.pop()
    by_num = {num: path for _, num, path in parsed}   # de-dup stray repeats
    nums = sorted(by_num.keys())
    return tag, [(n, by_num[n]) for n in nums]


def strip_compile_prefix(state_dict):
    """torch.compile wraps the model, prefixing saved keys with '_orig_mod.'
    Strip it so the state dict loads into a plain (uncompiled) model."""
    if not any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return state_dict
    return {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
            for k, v in state_dict.items()}


# --------------------------- eval-set construction ---------------------------

def build_eval_set(params, device, eval_block_i):
    """
    Loads the same data file the model trained on and slices out one block
    of `reduced_length` sequences, one-hot encoded, ready to hand to
    LadderEvaluator as x0_oh_eval.

    NOTE on "held out": blocks are carved from the same underlying dataset
    by contiguous index ranges, same convention as safe_block_indices() in
    sequential_learning_reduced.py. Passing --eval-block-i different from
    the training block's pick_i_for_training gives you sequences the model
    never trained on, PROVIDED the dataset itself has no duplicate/near-
    duplicate sequences across blocks -- not verified here.
    """
    # Seed BEFORE load_data, same order train_transformer.py uses -- if
    # load_data falls back to auto-generating data (file not found at
    # data_path), this ensures it reproduces the exact dataset training
    # saw, rather than a different draw from whatever RNG state happened
    # to be active here.
    seed = int(params["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # data_path in full_params.json is whatever string was passed to
    # --data-path at training time -- typically relative, e.g.
    # "../data/labeled_data_....npy", meant to be resolved against
    # scripts/ (where train_transformer.py / this script live), NOT
    # against whatever directory the job happens to be launched from.
    # An already-absolute data_path passes through unchanged.
    data_path = Path(params["data_path"])
    if not data_path.is_absolute():
        data_path = (SCRIPT_DIR / data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(
            f"data_path resolved to {data_path}, which does not exist. "
            f"full_params.json stored data_path={params['data_path']!r}, "
            f"resolved relative to {SCRIPT_DIR}. If the data file has moved, "
            f"update full_params.json or pass the correct file's location."
        )
    vocab_size, sequences, seq_len, k, rho, _ = load_data(str(data_path), device)
    reduced_length = int(params["reduced_length"])
    train_block_i = int(params.get("pick_i_for_training", 0) or 0)

    block_i = eval_block_i if eval_block_i is not None else train_block_i
    if eval_block_i is None:
        logger.warning(
            "No --eval-block-i given; evaluating on the TRAINING block "
            "(pick_i_for_training=%d). This is a train-set diagnostic, not "
            "a generalization measure. Pass --eval-block-i to use a "
            "different block.", train_block_i,
        )
    elif block_i == train_block_i:
        logger.warning(
            "--eval-block-i=%d matches pick_i_for_training=%d -- "
            "evaluating on the training block.", block_i, train_block_i,
        )

    left = block_i * reduced_length
    right = left + reduced_length
    if left < 0 or right > len(sequences):
        raise ValueError(
            f"Requested eval block [{left}:{right}] out of bounds for "
            f"sequence length {len(sequences)} (reduced_length={reduced_length})"
        )
    eval_sequences = sequences[left:right]
    x0_oh_eval = torch.nn.functional.one_hot(eval_sequences.long(), vocab_size).float()

    logger.info(
        "Eval set: block %d, %d sequences, shape %s",
        block_i, len(eval_sequences), tuple(x0_oh_eval.shape),
    )
    return dict(
        vocab_size=vocab_size, seq_len=seq_len, k=k, rho=rho,
        x0_oh_eval=x0_oh_eval, eval_block_i=block_i,
    )


# --------------------------------- main ---------------------------------

def main(args):
    device = setup_device()
    torch.set_float32_matmul_precision("high")

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model directory: {model_dir}")

    params_path = model_dir / "full_params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"full_params.json not found in: {model_dir}")
    params = load_params(str(params_path))
    logger.info("Loaded params from %s", params_path)

    data = build_eval_set(params, device, args.eval_block_i)
    vocab_size, seq_len, k, rho = data["vocab_size"], data["seq_len"], data["k"], data["rho"]
    x0_oh_eval = data["x0_oh_eval"]

    alpha_bars = alpha_bars_schedule(
        int(params["t_final"]), device=device, s=float(params["s"]), schedule="linear"
    )
    alphas, _ = get_alpha_beta_from_alpha_bar(alpha_bars)

    f_list = (
        [int(x) for x in args.bp_factorized_layers.split(",")]
        if args.bp_factorized_layers else list(range(k + 1))
    )

    # Built ONCE: this is where the frozen (x_t, t) draws and the oracle
    # ladder get computed. Every checkpoint below is scored against the
    # SAME frozen pairs, so checkpoints are directly comparable.
    evaluator = LadderEvaluator(
        x0_oh_eval=x0_oh_eval,
        t_final=int(params["t_final"]),
        alpha_bars=alpha_bars,
        alphas=alphas,
        rho=rho,
        k=k,
        vocab_size=vocab_size,
        factorized_layers_list=f_list,
        device=device,
        batch_size=args.ladder_batch_size,
        reweight=bool(params.get("reweighting", False)),
        rescale_loss=bool(params.get("rescale_loss", False)),
        seed=int(params["seed"]),   # match training's seed for comparability
        compute_oracle=True,
    )
    logger.info(
        "LadderEvaluator built: ell=%d f_list=%s reweight=%s rescale_loss=%s seed=%d N=%d",
        k, f_list, evaluator.rescale_loss, evaluator.rescale_loss,
        int(params["seed"]), len(x0_oh_eval),
    )

    model_args = {
        "hidden_size_encoder": params["hidden_size_encoder"],
        "nhead": params["nhead"],
        "num_layers": params["num_layers"],
        "model_channels": params["model_channels"],
    }
    # Build the model UNCOMPILED -- torch.compile is not needed for eval-only
    # forward passes here, and skipping it avoids the '_orig_mod.' key
    # mismatch entirely (strip_compile_prefix below is a fallback in case
    # checkpoints were saved some other way).
    model, _, _ = create_model(model_args, vocab_size, seq_len, device)

    ckpt_tag, ckpts = find_checkpoints(model_dir)
    logger.info(
        "Found %d checkpoints (%s-based). Range: %d .. %d",
        len(ckpts), ckpt_tag, ckpts[0][0], ckpts[-1][0],
    )

    rows = []
    out_json_path = model_dir / "ladder_posthoc.json"
    for num, ckpt_path in ckpts:
        state = torch.load(ckpt_path, map_location=device)
        state_dict = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
        model.load_state_dict(strip_compile_prefix(state_dict))
        model.to(device)

        out = evaluator.evaluate(model)
        out[ckpt_tag] = num
        rows.append(out)

        logger.info(
            "%s %d | %s | loss_total %.4f",
            ckpt_tag, num,
            "  ".join(f"bp{f} {out[f'bp_{f}']:.4f}" for f in f_list),
            out["loss_total"],
        )

        # Write after every checkpoint so a killed job still leaves usable
        # partial output -- same spirit as ladder_live.json during training.
        with open(out_json_path, "w") as fp:
            json.dump(rows, fp)

    logger.info("Wrote per-checkpoint rows: %s", out_json_path)

    # ---- aggregate R_k / delta_k across all checkpoints ----
    # NOTE on naming: level_excess() in this codebase now returns its array
    # reversed (R = R[::-1]) so row k lines up with rung k of bp_ladder,
    # rather than refinement-step d. R[k] = g_k - g_{k+1} for k < ell, and
    # R[ell] = g_ell (the one row that isn't a plain gap -- see the ladder
    # module docstring). We use 'k' in all keys/variable names below to
    # match that, NOT 'd' -- evaluate()'s own out[f'R_{d}']/out[f'delta_{d}']
    # keys are stale (still d-labeled) and hold these same k-indexed values;
    # if you rely on ladder_posthoc.json's R_*/delta_* keys elsewhere,
    # rename them there too so nothing downstream misreads rung 0 as rung ell.
    if evaluator._full_ladder:
        f_list_full = list(range(k + 1))
        ladder_dict = {f: [r[f"bp_{f}"] for r in rows] for f in f_list_full}
        R_raw = level_excess(ladder_dict, ell=k)
        R_cor = level_excess(ladder_dict, ell=k, oracle_ladder=evaluator.bp_oracle)

        npz_path = model_dir / "ladder_posthoc.npz"
        np.savez(
            npz_path,
            **{ckpt_tag: np.array([r[ckpt_tag] for r in rows])},
            R=R_raw, delta=R_cor,
            **{f"bp_{f}": np.array(ladder_dict[f]) for f in f_list_full},
        )
        logger.info(
            "Wrote aggregate: %s (R %s, delta %s, indexed by rung k ascending)",
            npz_path, R_raw.shape, R_cor.shape,
        )
    else:
        logger.info(
            "f_list=%s is not the full contiguous 0..%d ladder -- "
            "skipping R_k/delta_k aggregation (bp_f/g_f are still in %s).",
            f_list, k, out_json_path,
        )


if __name__ == "__main__":
    p = ArgumentParser(description="Post-hoc ladder evaluation across saved checkpoints.")
    p.add_argument("--model-dir", type=str, required=True,
                   help="Directory containing full_params.json and test_model_script_*.pt checkpoints.")
    p.add_argument("--eval-block-i", type=int, default=None,
                   help="Data block index to evaluate on. Default: the training block "
                        "(pick_i_for_training from full_params.json) -- a train-set diagnostic. "
                        "Pass a different block index for a held-out eval set.")
    p.add_argument("--bp-factorized-layers", type=str, default=None,
                   help="Comma-separated k values; default is the full ladder 0..ell.")
    p.add_argument("--ladder-batch-size", type=int, default=512,
                   help="Batch size for ladder eval (no backward pass -> can exceed training batch size).")
    args = p.parse_args()
    main(args)