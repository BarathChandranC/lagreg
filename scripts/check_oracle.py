"""
Self-test for the oracle + decomposition evaluator. Run this BEFORE training:

    python check_oracle.py                 # synthetic tiny pool, CPU/GPU, ~1 min
    python check_oracle.py --mnist         # also a real-MNIST smoke test of the evaluator

Checks
  1. m*, m^loc from ModKOracle == brute-force enumeration over all
     (x_0..x_3) in pool^4 (tiny synthetic pool, K=3).
  2. Evaluator on oracle 'models':
       f = m*    -> R_local ~ 0, R_global ~ 0, excess = 0
       f = m^loc -> R_local ~ 0, n_global ~ 1, align_global ~ 0
       f = 0     -> n_local = 1, n_global = 1, align_global = 0
     (R_local / n_global are Monte Carlo: compared within a few SE.)
  3. Pythagoras for an arbitrary (random-init) U-Net on the pop set:
       R_global (subtraction) ~ R_global (direct), within SE.
"""
import itertools
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.data import ResiduePools, sample_composites, to_tiles, from_tiles
from modules.schedule import CosineSchedule
from modules.oracle import ModKOracle, OracleDenoiser
from modules.decomposition import DecompositionEvaluator
from modules.unet import UNet

torch.set_default_dtype(torch.float32)


def brute_force(pools, z_tiles, a, s):
    """E[X | z] and E[X_i | z_i] by explicit enumeration of pool^4 (float64)."""
    X = pools.flat.double()                                     # (P,d)
    P = X.shape[0]
    res = torch.repeat_interleave(torch.arange(pools.K), pools.sizes.cpu())
    inv_size = 1.0 / pools.sizes.double().cpu()[res]            # (P,)
    out_full, out_loc = [], []
    for zb, ab, sb in zip(z_tiles.double(), a.double(), s.double()):
        logl = -((zb[:, None, :] - ab * X[None]) ** 2).sum(-1) / (2 * sb ** 2)   # (4,P)
        logl = logl - logl.max(1, keepdim=True).values
        l = logl.exp() * inv_size[None]                          # (4,P) incl. prior 1/|D_c|
        # joint: sum over pool^4 with sum residues == 0
        idx = torch.cartesian_prod(*[torch.arange(P)] * 4)       # (P^4,4)
        ok = (res[idx].sum(1) % pools.K) == 0
        idx = idx[ok]
        w = l[0, idx[:, 0]] * l[1, idx[:, 1]] * l[2, idx[:, 2]] * l[3, idx[:, 3]]
        w = w / w.sum()
        Ex = torch.stack([(w[:, None] * X[idx[:, i]]).sum(0) for i in range(4)])
        out_full.append(Ex)
        # local: X_i marginal = (1/K) Unif(D_c): weight l_i(x) (already has 1/|D_c|)
        wl = l / l.sum(1, keepdim=True)
        out_loc.append(wl @ X)
    return torch.stack(out_full), torch.stack(out_loc)


def check_bruteforce(device):
    g = torch.Generator().manual_seed(0)
    K, ts = 3, 2
    tiles = torch.rand(20, ts, ts, generator=g) * 2 - 1
    labels = torch.arange(10).repeat(2).numpy()
    pools = ResiduePools(tiles, labels, K, device="cpu")
    oracle = ModKOracle(pools)
    sched = CosineSchedule(1000)
    worst = 0.0
    for t in (50, 300, 700):
        x, _, _ = sample_composites(pools, 4, generator=torch.Generator().manual_seed(t))
        tt = torch.full((4,), t)
        z = sched.q_sample(x, tt, torch.randn(x.shape, generator=g))
        a, s = sched.a_s(tt)
        zt = to_tiles(z)
        tg = oracle.targets(zt, a, s)
        Ex_full, Ex_loc = brute_force(pools, zt, a, s)
        e1 = (tg["Ex_full"] - Ex_full).abs().max().item()
        e2 = (tg["Ex_loc"] - Ex_loc).abs().max().item()
        worst = max(worst, e1, e2)
        print(f"  t={t:4d}: max|E[X|z] err| {e1:.2e}   max|E[X_i|z_i] err| {e2:.2e}   "
              f"A range [{tg['A'].min():.3f}, {tg['A'].max():.3f}]")
    assert worst < 1e-8, f"oracle disagrees with brute force (max err {worst:g})"
    print("  [ok] oracle == brute force")


def check_evaluator(pools, device, t_fracs=(0.1, 0.3, 0.5), n=128, M=16, img_ch=None):
    sched = CosineSchedule(1000, device=device)
    oracle = ModKOracle(pools)
    x, _, _ = sample_composites(pools, n, generator=torch.Generator(device=device).manual_seed(1))
    t_list = [sched.t_from_frac(f) for f in t_fracs]
    ev = DecompositionEvaluator(oracle, sched, x, t_list, n_completions=M, seed=3,
                                batch_size=512, label="pop")
    ok = True

    def show(name, out):
        for t in t_list:
            k = lambda q: out[f"t{t}_{q}"]
            print(f"  {name:5s} t={t:4d}: R_loc {k('R_local'):+.3e}±{k('R_local_se'):.1e}  "
                  f"R_glob {k('R_global'):+.3e}±{k('R_global_se'):.1e}  R_glob_sub {k('R_global_sub'):+.3e}  "
                  f"n_loc {k('n_local'):+.3f} n_glob {k('n_global'):+.3f} align {k('align_global'):+.3f}  "
                  f"E|mg|^2 {k('glob_energy'):.2e}")

    for mode in ("star", "loc", "zero"):
        out = ev.evaluate(OracleDenoiser(oracle, sched, mode).to(device))
        show(mode, out)
        for t in t_list:
            k = lambda q: out[f"t{t}_{q}"]
            tol_l = 5 * k("R_local_se") + 1e-6
            if mode == "star":
                ok &= abs(k("excess")) < 1e-6 and abs(k("R_local")) < tol_l
            elif mode == "loc":
                # f = m^loc is exactly local: completions all agree -> U = 0, G = |m_glob|^2
                ok &= abs(k("R_local")) < 1e-9 and abs(k("n_global") - 1) < 1e-5
                # E<m_loc, m_glob> = 0 only in expectation: loose statistical bound
                ok &= abs(k("align_global")) < 0.25
            else:
                ok &= abs(k("n_local") - 1) < 1e-5 and abs(k("n_global") - 1) < 1e-5
    print("  [ok] oracle models behave as predicted" if ok else "  [FAIL] oracle-model checks")

    torch.manual_seed(0)
    net = UNet(img_size=2 * pools.tile_size, base_ch=32, ch_mult=(1, 2) if pools.tile_size == 7 else (1, 2, 2),
               attn_res=(7, 14), n_res=1).to(device)
    torch.nn.init.normal_(net.out[-1].weight, std=0.05)          # non-zero output
    out = ev.evaluate(net)
    show("unet", out)
    for t in t_list:
        k = lambda q: out[f"t{t}_{q}"]
        z = k("pyth_resid") / (k("R_global_se") + k("excess_se") + 1e-12)
        print(f"  unet t={t}: pyth_resid {k('pyth_resid'):+.3e} = {z:+.2f} SE")
        ok &= abs(z) < 4
    print("  [ok] Pythagoras holds within MC error" if ok else "  [FAIL] see above")
    return ok


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--mnist", action="store_true")
    p.add_argument("--mnist-root", default="../data")
    p.add_argument("--mnist-npz", default=None)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("1) brute force (synthetic pool, K=3, 2x2 tiles)")
    check_bruteforce(device)

    print("2-3) evaluator on synthetic 7x7 pool, K=5")
    g = torch.Generator().manual_seed(5)
    # class-dependent blobs so the class posterior is informative
    proto = torch.rand(10, 7, 7, generator=g) * 2 - 1
    tiles = (proto.repeat(30, 1, 1) + 0.3 * torch.randn(300, 7, 7, generator=g)).clamp(-1, 1)
    labels = torch.arange(10).repeat(30).numpy()
    ok = check_evaluator(ResiduePools(tiles, labels, 5, device=device), device)

    if args.mnist:
        from modules.data import load_mnist, preprocess_tiles
        xtr, ytr, _, _ = load_mnist(args.mnist_root, args.mnist_npz)
        pools = ResiduePools(preprocess_tiles(xtr, 14), ytr, 5, per_digit_cap=500, device=device)
        print("4) real MNIST (tile 14, 500/digit), K=5")
        ok &= check_evaluator(pools, device, t_fracs=(0.2, 0.4, 0.6), n=64, M=8)
    sys.exit(0 if ok else 1)
