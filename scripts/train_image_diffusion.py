"""
Train an x0-prediction diffusion U-Net on the hierarchical shapes dataset,
logging the band-resolved risk vector at log-spaced checkpoints.

Usage
-----
    python scripts/train_image_diffusion.py \
        --n-train 8192 --n-iter 200 --base-ch 32 \
        --levels 4 --results-dir ../results/shapes_n8192

N-sweep (the sample-complexity axis):
    for N in 512 2048 8192 32768; do
        python scripts/train_image_diffusion.py --n-train $N \
            --results-dir ../results/shapes_n$N
    done

Width sweep (the spectral-bias control -- narrow nets retain spectral
ordering, wide nets wash it out, so if your band ordering is IDENTICAL
across widths it is not ordinary spectral bias):
    for CH in 8 16 32 64; do
        python scripts/train_image_diffusion.py --base-ch $CH \
            --results-dir ../results/shapes_ch$CH
    done
"""
import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.band_diagnostics import BandEvaluator, grouped_bands
from modules.gaussian_baseline import GaussianBandBaseline
from modules.image_diffusion import alpha_bars_schedule, q_sample, sample_timesteps
from modules.shapes_data import ShapesConfig, make_train_test
from modules.unet_small import UNetSmall

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "full_params.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    cfg = ShapesConfig(image_size=args.image_size, n_slots=args.n_slots)
    (x_tr, lab_tr), (x_te, lab_te) = make_train_test(args.n_train, args.n_test, cfg, seed=args.seed)
    logger.info("train %s  test %s", tuple(x_tr.shape), tuple(x_te.shape))
    torch.save({"labels_train": lab_tr, "labels_test": lab_te}, out_dir / "labels.pt")

    alpha_bars = alpha_bars_schedule(args.t_final, schedule=args.schedule, device=device)

    model = UNetSmall(in_ch=1, base_ch=args.base_ch).to(device)
    logger.info("model params: %d", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- baseline: fitted ONCE on training data, no model involved ---
    logger.info("fitting Gaussian band baseline (diagonal=%s)...", args.diagonal_baseline)
    baseline = GaussianBandBaseline(
        levels=args.levels, shape=tuple(x_tr.shape[1:]),
        diagonal=args.diagonal_baseline, shrinkage=args.shrinkage, device=device,
    ).fit(x_tr, batch_size=args.batch_size)

    ev_tr = BandEvaluator(x_tr[: args.n_eval], alpha_bars, args.levels, baseline,
                          device, args.batch_size, args.n_t_samples, seed=args.seed)
    ev_te = BandEvaluator(x_te[: args.n_eval], alpha_bars, args.levels, baseline,
                          device, args.batch_size, args.n_t_samples, seed=args.seed)

    steps_per_epoch = math.ceil(args.n_train / args.batch_size)
    total_steps = args.n_iter * steps_per_epoch
    ckpt_steps = sorted(set(
        torch.logspace(math.log10(10), math.log10(total_steps), args.n_checkpoints).long().tolist()
    ))
    logger.info("total_steps=%d  checkpoints at %s", total_steps, ckpt_steps)

    x_tr_dev = x_tr.to(device)
    rows, gstep = [], 0

    def log_row(step):
        r_tr, r_te = ev_tr.evaluate(model), ev_te.evaluate(model)
        row = {"step": step}
        row.update({f"train_{k}": v for k, v in r_tr.items()})
        row.update({f"test_{k}": v for k, v in r_te.items()})
        # emp/pop gap per band -- the finite-N floor, same role as Omega_d on RHM
        for n in ev_tr.names:
            row[f"gap_{n}"] = r_te[f"Delta_{n}"] - r_tr[f"Delta_{n}"]
        for pre, r in (("train", r_tr), ("test", r_te)):
            for k, v in grouped_bands(r, args.levels, args.group_at).items():
                row[f"{pre}_{k}"] = v
        rows.append(row)
        with open(out_dir / "band_ladder.json", "w") as fh:
            json.dump(rows, fh, indent=2)
        logger.info("step %d | train loss %.5f | D_local %.5f  D_global %.5f | parseval %.2e",
                    step, r_tr["loss_total"], row["train_D_local"], row["train_D_global"],
                    r_tr["parseval_residual"])

    for epoch in range(args.n_iter):
        perm = torch.randperm(x_tr_dev.shape[0], device=device)
        for i in range(0, x_tr_dev.shape[0], args.batch_size):
            gstep += 1
            x0 = x_tr_dev[perm[i:i + args.batch_size]]
            t = sample_timesteps(x0.shape[0], args.t_final, device)
            xt, _ = q_sample(x0, t, alpha_bars)
            loss = (x0 - model(xt, t)).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if gstep in ckpt_steps:
                torch.save({"model_state_dict": model.state_dict(), "step": gstep},
                           out_dir / f"ckpt_step_{gstep}.pt")
                log_row(gstep)

    log_row(gstep)
    logger.info("done -> %s", out_dir / "band_ladder.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    # data
    p.add_argument("--n-train", type=int, default=8192)
    p.add_argument("--n-test", type=int, default=2048)
    p.add_argument("--n-eval", type=int, default=1024, help="eval subset size; see BandEvaluator note on N-sweeps")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--n-slots", type=int, default=4)
    # model
    p.add_argument("--base-ch", type=int, default=32, help="width -- sweep this as the spectral-bias control")
    # diffusion
    p.add_argument("--t-final", type=int, default=500)
    p.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear"])
    # training
    p.add_argument("--n-iter", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--n-checkpoints", type=int, default=45)
    # diagnostics
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--group-at", type=int, default=2)
    p.add_argument("--n-t-samples", type=int, default=4)
    p.add_argument("--diagonal-baseline", action="store_true",
                   help="per-coefficient variances only: cheaper, but a strictly weaker baseline")
    p.add_argument("--shrinkage", type=float, default=1e-3)
    main(p.parse_args())
