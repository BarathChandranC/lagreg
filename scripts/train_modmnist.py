"""
train_modmnist.py

Train an eps-prediction U-Net on mod-K MNIST composites and, at log-spaced
steps, evaluate IN-PROCESS (no checkpoints written):

  * exact per-t decomposition of the denoising risk on a held-out population
    set ("pop") and on the training set ("emp"):
        loss = C_t + R_local + R_global,   n_local, n_global, align_global
    (see modules/decomposition.py; the identity is exact on pop only), plus
        sw_*   sample-wise split, exact on any set: on emp it IS the split
               of the training loss (loss - C = sw_D_base + sw_D_global)
        lin_*  linear / alignment n's, the RHM-ladder analogue
  * composition of freshly generated samples:
        memorized / valid / local (locally wrong) / global (globally wrong)

Output: <out-root>/K{K}_tile{s}/seed{seed}_N{N}/
          full_params.json
          train_eval_log.json   one row per eval step, keys:
              step, lr, train_loss (running mean since last eval),
              emp_t{t}_{...}, pop_t{t}_{...} for every t in --eval-t-fracs,
              memorized, valid, local, global, tile_recognised, ...
          final_model.pt        only with --save-final-model

Optional hierarchical-lag regularizer (--reg-lambda > 0):
    L = eps-MSE + lambda * ReLU(n_global - sg[n_local] - margin)^2
with the evaluator's own n_global / n_local (projection-based, t-averaged),
estimated per training batch from conditional completions
(modules/regularizer.py). Per-eval means of the batch values are logged as
batch_n_global / batch_n_local / batch_lag / batch_reg.

Usage (from modmnist/scripts):
    python train_modmnist.py --K 5 --n-train 4096 --seed 0 --total-steps 200000
"""
import json
import logging
import math
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.data import (load_mnist, preprocess_tiles, ResiduePools,
                          sample_composites)
from modules.schedule import CosineSchedule
from modules.oracle import ModKOracle
from modules.unet import UNet
from modules.decomposition import DecompositionEvaluator
from modules.sampling import ddim_sample, classify_samples
from modules.fid import FIDEvaluator
from modules.regularizer import LagRegularizer
from modules.tile_classifier import (train_tile_classifier, load_tile_classifier,
                                     save_tile_classifier)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("modmnist")


def build_classifier(args, x_tr, y_tr, x_te, y_te, device):
    path = Path(args.classifier_path or f"../checkpoints/tile_cnn_tile{args.tile_size}.pt")
    if path.exists():
        clf, rep = load_tile_classifier(path, args.tile_size, device)
        te = rep.get("test", {})
        log.info("loaded tile classifier %s (test digit acc %.4f, negatives accepted %.4f, T=%.3f)",
                 path, te.get("digit_acc", float("nan")), te.get("neg_accepted", float("nan")),
                 rep.get("temperature", float("nan")))
        if abs(rep.get("conf_thresh", args.conf_thresh) - args.conf_thresh) > 1e-9:
            log.warning("classifier was validated at threshold %s but --conf-thresh is %s",
                        rep.get("conf_thresh"), args.conf_thresh)
        return clf
    log.info("no classifier checkpoint -- training one (run train_tile_classifier.py "
             "beforehand to avoid this inside a sweep)")
    clf, rep = train_tile_classifier(x_tr, y_tr, x_te, y_te, args.tile_size, device,
                                     conf_thresh=args.conf_thresh, seed=1234, log=log.info)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_tile_classifier(clf, rep, path)
    log.info("saved classifier -> %s", path)
    return clf


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---------------------------------------------------------------- data
    xtr_u8, ytr, xte_u8, yte = load_mnist(args.mnist_root, args.mnist_npz)
    s = args.tile_size
    x_tr = preprocess_tiles(xtr_u8, s)
    x_te = preprocess_tiles(xte_u8, s)
    # The pool (and therefore the population and the oracle) is fixed by
    # --pool-seed, NOT the run seed, so every (N, seed) run in a sweep shares
    # one population and one oracle.
    pools = ResiduePools(x_tr, ytr, args.K, per_digit_cap=args.pool_per_digit,
                         seed=args.pool_seed, device=device)
    log.info("pool: K=%d tile=%dx%d sizes per residue %s (total %d)",
             args.K, s, s, pools.sizes.tolist(), pools.flat.shape[0])

    g_train = torch.Generator(device=device).manual_seed(10_000 + args.seed)
    g_pop = torch.Generator(device=device).manual_seed(20_000 + args.pool_seed)
    x_train, c_train, _ = sample_composites(pools, args.n_train, generator=g_train)
    x_pop, _, _ = sample_composites(pools, args.n_eval, generator=g_pop)
    assert ((c_train.sum(1) % args.K) == 0).all()
    log.info("train set: %d composites %s | pop eval set: %d", args.n_train,
             tuple(x_train.shape[1:]), args.n_eval)

    sched = CosineSchedule(args.T, device=device)
    oracle = ModKOracle(pools, dtype=getattr(torch, args.oracle_dtype), chunk=args.oracle_chunk)
    t_list = sorted({sched.t_from_frac(float(f)) for f in args.eval_t_fracs.split(",")})
    log.info("eval t grid: %s (T=%d)", t_list, args.T)

    t0 = time.time()
    ev_pop = DecompositionEvaluator(oracle, sched, x_pop, t_list, args.n_completions,
                                    seed=30_000 + args.pool_seed,
                                    batch_size=args.eval_batch_size, label="pop")
    n_emp = min(args.n_eval, args.n_train)
    ev_emp = DecompositionEvaluator(oracle, sched, x_train[:n_emp], t_list, args.n_completions,
                                    seed=40_000 + args.seed,
                                    batch_size=args.eval_batch_size, label="emp")
    log.info("evaluators built in %.1fs", time.time() - t0)
    for it in ev_pop.items:
        log.info("  t=%4d a=%.3f s=%.3f  E|m_loc|^2=%.4g  E|m_glob|^2=%.4g (per pixel)",
                 it["t"], it["a"], it["s"],
                 float((it["m_loc"] ** 2).sum((1, 2)).mean() / ev_pop.D),
                 float((it["m_glob"] ** 2).sum((1, 2)).mean() / ev_pop.D))

    clf = build_classifier(args, x_tr, ytr, x_te, yte, device) if args.n_gen > 0 else None
    fid_ev = None
    if clf is not None and args.fid:
        t0 = time.time()
        fid_ev = FIDEvaluator(clf, pools, args.n_gen, n_ref=args.fid_n_ref,
                              seed=args.pool_seed, log=log.info)
        log.info("FID reference built in %.1fs", time.time() - t0)

    # ---------------------------------------------------------------- model
    img = 2 * s
    ch_mult = tuple(int(c) for c in args.ch_mult.split(","))
    model = UNet(img_size=img, base_ch=args.base_ch, ch_mult=ch_mult, n_res=args.n_res,
                 attn_res=tuple(int(r) for r in args.attn_res.split(",")),
                 dropout=args.dropout).to(device)
    log.info("U-Net params: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    regz = None
    if args.reg_lambda > 0 or args.reg_monitor:
        regz = LagRegularizer(oracle, sched, pools, margin=args.reg_margin,
                              n_completions=args.reg_completions,
                              n_ref=args.reg_n_ref, seed=args.pool_seed, log=log.info)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: min(1.0, (st + 1) / max(1, args.warmup_steps)))

    # ---------------------------------------------------------------- output
    out_dir = Path(args.out_root) / f"K{args.K}_tile{s}" / f"seed{args.seed}_N{args.n_train}"
    if args.add_flag:
        out_dir = out_dir.with_name(out_dir.name + f"_{args.add_flag}")
    if args.reg_lambda > 0:
        out_dir = out_dir.with_name(out_dir.name + "_reg" + str(args.reg_lambda).replace(".", "p"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "full_params.json", "w") as f:
        json.dump({**vars(args), "t_list": t_list, "pool_sizes": pools.sizes.tolist()}, f, indent=2)
    # (t-average weights are logged at startup and recomputable from t_list and T)
    log_path = out_dir / "train_eval_log.json"

    first = max(1, args.first_eval_step)
    eval_steps = sorted({0} | set(np.unique(np.round(
        np.logspace(math.log10(first), math.log10(args.total_steps), args.n_evals)
    ).astype(int)).tolist()))
    log.info("%d eval steps: %s", len(eval_steps), eval_steps)
    eval_set = set(eval_steps)

    # ---- averaging over t, tied to the test loss ----
    # Training draws t ~ Uniform{1..T} with plain eps-MSE, so the test loss is
    # the uniform average over t of loss(t) = C_t + R_local(t) + R_global(t).
    # Each grid point stands for the t's closer to it than to its neighbours
    # (edges at midpoints, ends at 1 and T), so weights sum to 1. Accuracy of
    # these averages depends on the grid covering [1, T]; a uniform grid
    # (e.g. --eval-t-fracs 0.05,0.1,...,0.95) makes them a midpoint rule.
    ts_np = np.array(t_list, dtype=float)
    edges = np.concatenate([[0.5], (ts_np[1:] + ts_np[:-1]) / 2, [args.T + 0.5]])
    t_w = np.diff(edges) / args.T
    log.info("t-average weights: %s", {t: round(float(w), 4) for t, w in zip(t_list, t_w)})

    def add_t_averages(row, pref):
        A = lambda k: float(sum(w * row[f"{pref}_t{t}_{k}"] for w, t in zip(t_w, t_list)))
        for k in ("loss", "C", "excess", "R_local", "R_global", "R_global_sub",
                  "cond_var", "loc_energy", "glob_energy"):
            row[f"{pref}_avg_{k}"] = A(k)
        El, Eg = row[f"{pref}_avg_loc_energy"], row[f"{pref}_avg_glob_energy"]
        align_num = sum(w * row[f"{pref}_t{t}_align_global"] * row[f"{pref}_t{t}_glob_energy"]
                        for w, t in zip(t_w, t_list))
        row[f"{pref}_avg_n_local"] = row[f"{pref}_avg_R_local"] / El
        row[f"{pref}_avg_n_global"] = row[f"{pref}_avg_R_global"] / Eg
        row[f"{pref}_avg_align_global"] = float(align_num) / Eg          # energy-weighted, RHM-comparable
        row[f"{pref}_avg_leakage"] = row[f"{pref}_avg_cond_var"] / Eg    # n_global = 1 - 2 align + leakage
        row[f"{pref}_avg_identity_resid"] = (row[f"{pref}_avg_loss"] - row[f"{pref}_avg_C"]
                                             - row[f"{pref}_avg_R_local"] - row[f"{pref}_avg_R_global"])
        # sample-wise split: avg(loss) - avg(C) = avg sw_D_base + avg sw_D_global exactly
        for k in ("cross_loc", "sw_D_base", "sw_D_global", "ip_f_glob", "ip_f_loc", "ip_loc_glob"):
            row[f"{pref}_avg_{k}"] = A(k)
        row[f"{pref}_avg_sw_n_local"] = row[f"{pref}_avg_sw_D_base"] / El
        row[f"{pref}_avg_sw_n_global"] = row[f"{pref}_avg_sw_D_global"] / Eg
        # linear / alignment n's (RHM-ladder analogue), from t-averaged inner products
        ilg = row[f"{pref}_avg_ip_loc_glob"]
        row[f"{pref}_avg_lin_n_global"] = 1 - (row[f"{pref}_avg_ip_f_glob"] - ilg) / Eg
        row[f"{pref}_avg_lin_n_local"] = 1 - (row[f"{pref}_avg_ip_f_loc"] - ilg) / El

    def run_eval(step, train_loss, reg_stats=None):
        t1 = time.time()
        row = {"step": int(step), "lr": opt.param_groups[0]["lr"], "train_loss": train_loss}
        if reg_stats:                         # means over training batches since the last eval
            row.update({f"batch_{k}": v for k, v in reg_stats.items()})
        for pref, ev in (("pop", ev_pop), ("emp", ev_emp)):
            for k, v in ev.evaluate(model).items():
                row[f"{pref}_{k}"] = v
            add_t_averages(row, pref)
        if clf is not None:
            gen = ddim_sample(model, sched, args.n_gen, img, device, n_steps=args.sample_steps,
                              eta=args.sample_eta, batch_size=args.eval_batch_size,
                              generator=torch.Generator(device=device).manual_seed(50_000 + args.seed))
            row.update(classify_samples(gen, clf, args.K, x_train, conf_thresh=args.conf_thresh,
                                        mem_rule=args.mem_rule, mem_ratio=args.mem_ratio,
                                        mem_rms=args.mem_rms))
            if fid_ev is not None:
                row.update(fid_ev(gen))
            if args.save_samples:
                torch.save(gen[:64].cpu(), out_dir / f"samples_step{step}.pt")
        row["eval_seconds"] = time.time() - t1
        rows.append(row)
        with open(log_path, "w") as f:
            json.dump(rows, f, indent=1)
        tm = t_list[len(t_list) // 2]
        msg = (f"step {step:>7d} | train {train_loss if train_loss is not None else float('nan'):.4f} "
               f"| t={tm}: pop n_loc {row[f'pop_t{tm}_n_local']:.3f} n_glob {row[f'pop_t{tm}_n_global']:.3f} "
               f"align {row[f'pop_t{tm}_align_global']:.3f} pyth {row[f'pop_t{tm}_pyth_resid']:+.2e} "
               f"| emp(sw) n_loc {row['emp_avg_sw_n_local']:+.3f} n_glob {row['emp_avg_sw_n_global']:+.3f} "
               f"| lin pop n_glob {row['pop_avg_lin_n_global']:+.3f} "
               f"| avg-t pop n_loc {row['pop_avg_n_local']:.3f} n_glob {row['pop_avg_n_global']:.3f} "
               f"align {row['pop_avg_align_global']:+.3f} leak {row['pop_avg_leakage']:.3f}")
        if clf is not None:
            msg += (f" | mem {row['memorized']:.3f} valid {row['valid']:.3f} "
                    f"local {row['local']:.3f} global {row['global']:.3f} "
                    f"tile_rej {row['tile_rejected']:.3f} "
                    f"v|rec {row['valid_given_recognised']:.3f} (1/K={1 / args.K:.3f})")
        if reg_stats:
            msg += (f" | batch n_glob {reg_stats['n_global']:.3f} n_loc {reg_stats['n_local']:.3f} "
                    f"lag {reg_stats['lag']:+.3f} reg {reg_stats['reg']:.3g}")
        if fid_ev is not None:
            msg += (f" | FID {row['fid_composite']:.3g} (floor {row['fid_composite_real_floor']:.3g}) "
                    f"KID {row['kid_composite']:.2e}")
        log.info("%s  [%.0fs]", msg, row["eval_seconds"])

    # ---------------------------------------------------------------- train
    rows = []
    run_eval(0, None)                      # zero-init output layer -> f = 0 -> n_local = n_global = 1
    model.train()
    acc_loss, acc_n = 0.0, 0
    reg_acc, reg_n = {}, 0
    amp = (args.amp == "bf16" and device.type == "cuda")
    for step in range(1, args.total_steps + 1):
        idx = torch.randint(0, args.n_train, (args.batch_size,), device=device, generator=g_train)
        x0 = x_train[idx]
        t = torch.randint(1, args.T + 1, (args.batch_size,), device=device, generator=g_train)
        eps = torch.randn(x0.shape, device=device, generator=g_train)
        zt = sched.q_sample(x0, t, eps)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            pred = model(zt, t)
        main = F.mse_loss(pred.float(), eps)
        loss = main
        if regz is not None and step % args.reg_every == 0:
            apply = args.reg_lambda > 0 and step > args.reg_warmup_steps
            rb = min(args.reg_batch, args.batch_size)
            with torch.set_grad_enabled(apply):            # monitor-only: no graph
                reg, info = regz(model, pred[:rb], zt[:rb], t[:rb],
                                 torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp))
            if apply:
                loss = main + args.reg_lambda * reg
            for k, v in info.items():
                reg_acc[k] = reg_acc.get(k, 0.0) + v
            reg_n += 1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        sch.step()
        acc_loss += float(main.detach()); acc_n += 1          # main loss only (reg logged separately)
        if step in eval_set:
            run_eval(step, acc_loss / acc_n,
                     {k: v / reg_n for k, v in reg_acc.items()} if reg_n else None)
            acc_loss, acc_n = 0.0, 0
            reg_acc, reg_n = {}, 0
            model.train()

    if args.save_final_model:
        torch.save({"model_state_dict": model.state_dict(), "args": vars(args)},
                   out_dir / "final_model.pt")
    log.info("done -> %s", log_path)


if __name__ == "__main__":
    p = ArgumentParser(description="mod-K MNIST diffusion: train + exact local/global risk decomposition")
    # data
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--tile-size", type=int, default=14, choices=[7, 14, 28])
    p.add_argument("--pool-per-digit", type=int, default=None,
                   help="Cap tiles per digit in the pool (defines the population). "
                        "None = all ~6000/digit. Oracle cost is linear in pool size.")
    p.add_argument("--pool-seed", type=int, default=0,
                   help="Fixes pool subsampling AND the pop eval set; keep constant across a sweep.")
    p.add_argument("--mnist-root", type=str, default="../data")
    p.add_argument("--mnist-npz", type=str, default=None)
    p.add_argument("--n-train", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    # model / diffusion
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--base-ch", type=int, default=64)
    p.add_argument("--ch-mult", type=str, default="1,2,2",
                   help="'1,2,2' for 28x28 composites (tile 14); '1,2,2,2' for 56x56 (tile 28).")
    p.add_argument("--n-res", type=int, default=2)
    p.add_argument("--attn-res", type=str, default="7,14")
    p.add_argument("--dropout", type=float, default=0.0)
    # optimisation
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", type=str, default="none", choices=["none", "bf16"],
                   help="Autocast for the TRAINING forward only; evaluation is always fp32.")
    # evaluation
    p.add_argument("--n-evals", type=int, default=30)
    p.add_argument("--first-eval-step", type=int, default=50)
    p.add_argument("--eval-t-fracs", type=str, default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.8",
                   help="t/T values at which the decomposition is reported (never aggregated: "
                        "m_glob energy and a/s vary by orders of magnitude across t).")
    p.add_argument("--n-eval", type=int, default=256, help="Composites per eval set (pop and emp).")
    p.add_argument("--n-completions", type=int, default=16, help="M conditional completions per (composite, region).")
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--oracle-dtype", type=str, default="float64", choices=["float64", "float32"])
    p.add_argument("--oracle-chunk", type=int, default=256)
    # samples
    p.add_argument("--n-gen", type=int, default=1000, help="0 disables sample classification.")
    p.add_argument("--sample-steps", type=int, default=250)
    p.add_argument("--sample-eta", type=float, default=1.0)
    p.add_argument("--conf-thresh", type=float, default=0.9)
    p.add_argument("--mem-rule", type=str, default="ratio", choices=["ratio", "rms"])
    p.add_argument("--mem-ratio", type=float, default=1 / 3)
    p.add_argument("--mem-rms", type=float, default=0.1)
    p.add_argument("--classifier-path", type=str, default=None,
                   help="Default ../checkpoints/tile_cnn_tile{tile_size}.pt; trained and saved there if missing.")
    p.add_argument("--save-samples", action="store_true")
    p.add_argument("--fid", action="store_true", default=True,
                   help="FID/KID of the n_gen samples in tile-classifier feature space "
                        "(local quality only -- blind to the mod-K rule, see modules/fid.py).")
    p.add_argument("--no-fid", dest="fid", action="store_false")
    p.add_argument("--fid-n-ref", type=int, default=10000,
                   help="Real population composites used as the FID/KID reference.")
    # hierarchical-lag regularizer
    p.add_argument("--reg-lambda", type=float, default=0.0,
                   help="Weight of ReLU(n_global - sg[n_local] - margin)^2 added to the eps-MSE. "
                        "0 disables. Runs with lambda > 0 get a _reg<lambda> folder suffix.")
    p.add_argument("--reg-margin", type=float, default=0.0,
                   help="Dead zone on the lag before the penalty starts (0 = paper Eq. 3).")
    p.add_argument("--reg-warmup-steps", type=int, default=0,
                   help="No penalty before this step (n_global/n_local are still logged).")
    p.add_argument("--reg-every", type=int, default=1,
                   help="Compute the per-batch n's (and the penalty) every k steps.")
    p.add_argument("--reg-batch", type=int, default=32,
                   help="Composites of each training batch used for the penalty. Each needs "
                        "4 x --reg-completions extra forward+backward passes.")
    p.add_argument("--reg-completions", type=int, default=2,
                   help="Conditional completions M per (composite, region); >= 2.")
    p.add_argument("--reg-n-ref", type=int, default=2048,
                   help="Reference composites used once to fix the normalisers G_loc, G_glob.")
    p.add_argument("--reg-monitor", action="store_true",
                   help="Compute and log the per-batch n_global/n_local/lag even when "
                        "--reg-lambda 0 (the matched control for a regularized run).")
    # output
    p.add_argument("--out-root", type=str, default="../results_modmnist")
    p.add_argument("--add-flag", type=str, default=None)
    p.add_argument("--save-final-model", action="store_true")
    main(p.parse_args())