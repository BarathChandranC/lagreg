"""
Post-hoc analysis of band_ladder.json: emergence ordering, the directional
spectral-bias test, and plots.

    python scripts/band_loss_decomposition.py --run-dir ../results/shapes_n8192 --plots

Cross-run (the actual controls):
    python scripts/band_loss_decomposition.py \
        --run-dir ../results/shapes_ch8 ../results/shapes_ch64 --compare --plots

THE DIRECTIONAL TEST, which is the load-bearing analysis here
-------------------------------------------------------------
There is no oracle on images, so you cannot verify the vector against ground
truth. What you CAN do is check the direction of the emergence ordering
against the competing hypothesis:

    spectral bias  -> COARSE bands resolve FIRST (high-variance, low-frequency
                      directions are learned fastest)
    hierarchy      -> FINE/local structure resolves first, global coherence last

These predict OPPOSITE orderings, so observing one is evidence against the
other. If the ordering also FLIPS with channel width (narrow nets retain
spectral ordering, wide nets lose it) that is the strongest available
separation, since it changes the mechanism while holding the data fixed.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.band_diagnostics import emergence_steps
from modules.wavelet import band_names


def load(run_dir):
    with open(Path(run_dir) / "band_ladder.json") as f:
        rows = json.load(f)
    with open(Path(run_dir) / "full_params.json") as f:
        params = json.load(f)
    return rows, params


def report(rows, params, split="train"):
    levels = params["levels"]
    names = band_names(levels)
    print(f"\n{'='*78}\nrun: n_train={params['n_train']}  base_ch={params['base_ch']}  "
          f"levels={levels}  split={split}\n{'='*78}")

    pr = max(r[f"{split}_parseval_residual"] for r in rows)
    print(f"max Parseval residual: {pr:.2e}" + ("   <-- BANDS INVALID" if pr > 1e-4 else "   (ok)"))

    print(f"\n{'step':>8}" + "".join(f"{n.replace('detail_','d').replace('approx_','a'):>11}" for n in names))
    print(f"{'':>8}" + "".join(f"{'Delta':>11}" for _ in names))
    for r in rows[:: max(1, len(rows) // 12)]:
        print(f"{r['step']:>8}" + "".join(f"{r[f'{split}_Delta_{n}']:>11.5f}" for n in names))

    print("\nemergence step (50% of own decay range):")
    tau = {}
    for n in names:
        t = emergence_steps(rows, f"{split}_Delta_{n}")
        tau[n] = t
        print(f"   {n:>10}: {t}")

    finite = {k: v for k, v in tau.items() if v is not None}
    if len(finite) >= 2:
        order = sorted(finite, key=finite.get)
        print(f"\n   resolved first -> last: {' -> '.join(order)}")
        fine_first = order[0].startswith("detail_1") or order[0].startswith("detail_2")
        print("   DIRECTION: " + ("FINE-first (hierarchy-consistent, anti-spectral-bias)"
                                  if fine_first else
                                  "COARSE-first (spectral-bias-consistent)"))
        print("   NOTE: one run, one seed. Confirm with a width sweep before claiming either.")

    print(f"\n{'step':>8}{'D_local':>12}{'D_global':>12}{'ratio':>10}")
    for r in rows[:: max(1, len(rows) // 12)]:
        dl, dg = r[f"{split}_D_local"], r[f"{split}_D_global"]
        print(f"{r['step']:>8}{dl:>12.5f}{dg:>12.5f}{(dg/dl if abs(dl)>1e-12 else float('nan')):>10.3f}")

    if split == "train":
        print(f"\nemp/pop gap per band (test Delta - train Delta), final checkpoint:")
        for n in names:
            print(f"   {n:>10}: {rows[-1][f'gap_{n}']:+.6f}")


def plot(rows, params, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = params["levels"]
    names = band_names(levels)
    steps = np.array([r["step"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    for n in names:
        ax.plot(steps, [r[f"train_Delta_{n}"] for r in rows], marker="o", ms=3, label=n)
    ax.set(xscale="log", xlabel="step", ylabel=r"$\Delta_j$ (train)",
           title=r"Band excess over Gaussian baseline")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.axhline(0, color="gray", ls=":", lw=1); ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = axes[0, 1]
    for n in names:
        ax.plot(steps, [r[f"train_L_{n}"] for r in rows], marker="o", ms=3, label=f"model {n}")
        ax.plot(steps, [r[f"train_Lbase_{n}"] for r in rows], ls="--", lw=1, alpha=.6)
    ax.set(xscale="log", yscale="log", xlabel="step", ylabel="per-coeff MSE",
           title="raw band loss (solid) vs Gaussian baseline (dashed)")
    ax.legend(fontsize=7); ax.grid(alpha=.3)

    ax = axes[1, 0]
    ax.plot(steps, [r["train_D_local"] for r in rows], marker="o", ms=3, label="D_local (fine)")
    ax.plot(steps, [r["train_D_global"] for r in rows], marker="s", ms=3, label="D_global (coarse)")
    ax.set(xscale="log", xlabel="step", ylabel=r"$\Delta$", title="local vs global")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.axhline(0, color="gray", ls=":", lw=1); ax.legend(); ax.grid(alpha=.3)

    ax = axes[1, 1]
    for n in names:
        ax.plot(steps, [r[f"gap_{n}"] for r in rows], marker="^", ms=3, label=n)
    ax.set(xscale="log", xlabel="step", ylabel="test - train", title="emp/pop gap (finite-N floor)")
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.axhline(0, color="gray", ls=":", lw=1); ax.legend(fontsize=8); ax.grid(alpha=.3)

    fig.suptitle(f"Band ladder | n_train={params['n_train']} base_ch={params['base_ch']}")
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved plot -> {out_path}")


def compare(run_dirs):
    print(f"\n{'='*78}\nCROSS-RUN EMERGENCE ORDERING\n{'='*78}")
    print("If the ordering is IDENTICAL across widths, ordinary spectral bias is")
    print("unlikely to be the driver (it is width-modulated). If it FLIPS, it is.\n")
    for rd in run_dirs:
        rows, params = load(rd)
        names = band_names(params["levels"])
        tau = {n: emergence_steps(rows, f"train_Delta_{n}") for n in names}
        finite = {k: v for k, v in tau.items() if v is not None}
        order = " -> ".join(sorted(finite, key=finite.get)) if finite else "(none resolved)"
        print(f"  n={params['n_train']:>6} ch={params['base_ch']:>3}: {order}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, nargs="+", required=True)
    p.add_argument("--split", type=str, default="train", choices=["train", "test"])
    p.add_argument("--plots", action="store_true")
    p.add_argument("--compare", action="store_true")
    a = p.parse_args()

    for rd in a.run_dir:
        rows, params = load(rd)
        report(rows, params, a.split)
        if a.plots:
            plot(rows, params, Path(rd) / "band_ladder.png")
    if a.compare and len(a.run_dir) > 1:
        compare(a.run_dir)
