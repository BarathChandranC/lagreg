"""
rates_at_step.py

For every discovered per-N run (same discovery logic as combine_runs.py),
convert epochs -> gradient steps (steps = epoch * ceil(N / batch_size)) and
report the 'valid' and 'global' percentages AT a target gradient-step count
(e.g. 5000) via linear interpolation between the two bracketing checkpoints.
If the target falls outside the logged range for a given N, the nearest
endpoint is used and the row is flagged as extrapolated.

Usage (run from scripts/, same conventions as combine_runs.py)::

    python rates_at_step.py --plots-root ../plots --batch-size 512 --target-step 5000
    python rates_at_step.py --plots-root ../plots --batch-size 512 --target-step 5000 \
        --n-test 1000 --ratio 0.15 --backend torch
"""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from combine_runs import discover  # reuse exact same file-discovery logic


def nearest_index(steps: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(steps - target)))


def interp_at(steps: np.ndarray, values: np.ndarray, target: float):
    """
    Linearly interpolate `values` at `target` gradient step.

    Returns (value, lo_step, hi_step, extrapolated: bool). If target falls
    outside [steps.min(), steps.max()], clamps to the nearest endpoint and
    flags extrapolated=True rather than silently extrapolating past logged
    checkpoints (there is no data to interpolate against there).
    """
    steps = np.asarray(steps, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(steps)
    steps, values = steps[order], values[order]

    if target <= steps[0]:
        return float(values[0]), float(steps[0]), float(steps[0]), True
    if target >= steps[-1]:
        return float(values[-1]), float(steps[-1]), float(steps[-1]), True

    hi = int(np.searchsorted(steps, target))
    lo = hi - 1
    s_lo, s_hi = steps[lo], steps[hi]
    v_lo, v_hi = values[lo], values[hi]

    frac = (target - s_lo) / (s_hi - s_lo) if s_hi > s_lo else 0.0
    val = v_lo + frac * (v_hi - v_lo)
    return float(val), float(s_lo), float(s_hi), False


def main(args):
    runs = discover(Path(args.plots_root), n_test=args.n_test,
                    ratio=args.ratio, backend=args.backend)
    if not runs:
        raise SystemExit(
            f"No npz found under {args.plots_root} matching "
            f"N_test={args.n_test} ratio={args.ratio} backend={args.backend}"
        )
    print(f"found {len(runs)} runs: {sorted(runs)}\n")

    header = (f"{'N':>8}  {'steps/epoch':>11}  {'bracket [lo, hi]':>19}  "
              f"{'valid':>8}  {'global':>8}  {'local':>8}  {'memorized':>10}")
    print(header)
    print("-" * len(header))

    rows = []
    for N, npz in runs.items():
        d = np.load(npz)
        if "epochs" not in d or "valid" not in d or "global" not in d:
            print(f"{N:>8}  -- missing epochs/valid/global in {npz.name}, skipping "
                  f"(was it run with --n-gen > 0?)")
            continue

        steps_per_epoch = int(np.ceil(N / args.batch_size))
        steps = np.asarray(d["epochs"], dtype=float) * steps_per_epoch

        valid, s_lo, s_hi, extrap_v = interp_at(steps, d["valid"], args.target_step)
        glob, _, _, extrap_g = interp_at(steps, d["global"], args.target_step)
        local, _, _, _ = interp_at(steps, d["local"], args.target_step) if "local" in d else (float("nan"), 0, 0, False)
        mem, _, _, _ = interp_at(steps, d["memorized"], args.target_step) if "memorized" in d else (float("nan"), 0, 0, False)

        bracket = f"[{s_lo:.0f}, {s_hi:.0f}]"
        flag = "  (extrapolated - target outside logged range)" if (extrap_v or extrap_g) else ""
        print(f"{N:>8}  {steps_per_epoch:>11}  {bracket:>19}  "
              f"{valid:>8.4f}  {glob:>8.4f}  {local:>8.4f}  {mem:>10.4f}{flag}")
        rows.append((N, args.target_step, valid, glob, local, mem))

    if not rows:
        return

    print(f"\n--- trend across N at step~={args.target_step} ---")
    rows.sort(key=lambda r: r[0])
    Ns = [r[0] for r in rows]
    valids = [r[2] for r in rows]
    globs = [r[3] for r in rows]
    print(f"N sweep: {Ns}")
    print(f"valid:   {[f'{v:.3f}' for v in valids]}")
    print(f"global:  {[f'{g:.3f}' for g in globs]}")
    if len(rows) > 1:
        d_valid = valids[-1] - valids[0]
        d_glob = globs[-1] - globs[0]
        print(f"\nvalid change from smallest to largest N: {d_valid:+.4f}")
        print(f"global change from smallest to largest N: {d_glob:+.4f}")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--plots-root", type=str, default="../plots")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--target-step", type=float, default=5000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--ratio", type=str, default=None)
    p.add_argument("--backend", type=str, default=None)
    main(p.parse_args())