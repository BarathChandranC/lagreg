"""
plot_modmnist.py -- one summary figure for a mod-K MNIST sweep.

Reads   <root>/seed<S>_N<N>[_<flag>]/train_eval_log.json  (+ full_params.json)
Writes  <out-dir>/modmnist_summary.png   (default out-dir: <root>/plots)

Rows = training-set size N (largest first), seeds averaged (band = +- s.e.m.),
x = gradient steps (log). Columns:
  1  held-out ("pop"), t-averaged: projection n_local / n_global (solid) and the
     linear / alignment n's, RHM-ladder analogue (dashed)
  2  training set ("emp"), t-averaged: the SAMPLE-WISE split sw_n_local /
     sw_n_global (solid), which sums exactly to the training loss minus C and
     can go negative under memorization; linear n's dashed; the regularizer's
     per-batch values (batch_n_*) dotted. Older logs without sw_ keys fall back
     to the projection emp n's, labelled "pop-referenced" -- those are NOT a
     split of the training loss.
  3  generated-sample composition in %: memorized / valid / global (hallucinated)
     / local (bad), with valid|recognised and the 1/K chance line
  4  composite FID, with the real-data floor
  5  memorized samples, %
  6  test loss (t-averaged, held-out), with the Bayes floor C
Columns 1-2 each get their own y-range per N. Columns 4-6 share the time axis; they are separate panels because the three
quantities live on unrelated scales (one shared y-axis would flatten two of them).

Older logs without t-averaged keys are handled: the averages are recomputed
from the per-t keys with the same weights train_modmnist.py uses.

Usage (from modmnist/scripts):
    python plot_modmnist.py --root ../results_modmnist/K5_tile14 --flag pool100
"""
import json
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_RE = re.compile(r"^seed(\d+)_N(\d+)(?:_(.+))?$")

# categorical slots (fixed order) -- local / global / valid / memorized keep the
# same meaning in every figure: local = orange, global = blue, valid = aqua, mem = grey
C_LOCAL, C_GLOBAL, C_VALID, C_MEM = "#eb6834", "#2a78d6", "#1baf7a", "#9a9a95"
INK, INK2, GRID = "#1f1f1e", "#6b6b66", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 180, "font.size": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2.0,
})


# ------------------------------------------------------------------ loading

def discover(root, flag, n_vals):
    runs, flags_seen = {}, set()
    for d in sorted(Path(root).iterdir()):
        m = RUN_RE.match(d.name)
        if not m or not (d / "train_eval_log.json").exists():
            continue
        f = m.group(3) or ""
        flags_seen.add(f)
        if flag is not None and f != flag:
            continue
        N = int(m.group(2))
        if n_vals and N not in n_vals:
            continue
        runs.setdefault(N, []).append((int(m.group(1)), d))
    if flag is None and len(flags_seen) > 1:
        raise SystemExit(f"runs with different suffixes found {sorted(flags_seen)}; "
                         f"pick one with --flag (use --flag \"\" for no suffix)")
    if not runs:
        raise SystemExit(f"no seed*_N*/train_eval_log.json under {root} (flag={flag!r})")
    return dict(sorted(runs.items(), reverse=True))


def t_list_from(rows, params):
    if params.get("t_list"):
        return [int(t) for t in params["t_list"]]
    return sorted({int(k.split("_")[1][1:]) for k in rows[0]
                   if re.match(r"^pop_t\d+_R_local$", k)})


def ensure_averages(row, t_list, T):
    """Fill <split>_avg_* from per-t keys when an older log lacks them."""
    ts = np.array(t_list, float)
    edges = np.concatenate([[0.5], (ts[1:] + ts[:-1]) / 2, [T + 0.5]])
    w = np.diff(edges) / T
    for p in ("pop", "emp"):
        if f"{p}_avg_n_local" in row or f"{p}_t{t_list[0]}_R_local" not in row:
            continue
        A = lambda k: float(sum(wi * row[f"{p}_t{t}_{k}"] for wi, t in zip(w, t_list)))
        for k in ("loss", "C", "excess", "R_local", "R_global", "cond_var",
                  "loc_energy", "glob_energy"):
            row[f"{p}_avg_{k}"] = A(k)
        El, Eg = row[f"{p}_avg_loc_energy"], row[f"{p}_avg_glob_energy"]
        an = sum(wi * row[f"{p}_t{t}_align_global"] * row[f"{p}_t{t}_glob_energy"]
                 for wi, t in zip(w, t_list))
        row[f"{p}_avg_n_local"] = row[f"{p}_avg_R_local"] / El
        row[f"{p}_avg_n_global"] = row[f"{p}_avg_R_global"] / Eg
        row[f"{p}_avg_align_global"] = an / Eg
        row[f"{p}_avg_leakage"] = row[f"{p}_avg_cond_var"] / Eg
        row[f"{p}_avg_identity_resid"] = (row[f"{p}_avg_loss"] - row[f"{p}_avg_C"]
                                          - row[f"{p}_avg_R_local"] - row[f"{p}_avg_R_global"])
        if f"{p}_t{t_list[0]}_sw_D_base" in row:
            for k in ("sw_D_base", "sw_D_global", "ip_f_glob", "ip_f_loc", "ip_loc_glob"):
                row[f"{p}_avg_{k}"] = A(k)
            row[f"{p}_avg_sw_n_local"] = row[f"{p}_avg_sw_D_base"] / El
            row[f"{p}_avg_sw_n_global"] = row[f"{p}_avg_sw_D_global"] / Eg
            ilg = row[f"{p}_avg_ip_loc_glob"]
            row[f"{p}_avg_lin_n_global"] = 1 - (row[f"{p}_avg_ip_f_glob"] - ilg) / Eg
            row[f"{p}_avg_lin_n_local"] = 1 - (row[f"{p}_avg_ip_f_loc"] - ilg) / El
    # older logs: linear n_global from the alignment alone. Drops the oracle-only
    # term E<m_loc, m_glob>/E||m_glob||^2 (zero in expectation on pop; not exactly
    # on the training subset). lin n_local needs E<f, m_loc>, never logged before.
    for p in ("pop", "emp"):
        if f"{p}_avg_lin_n_global" not in row and f"{p}_avg_align_global" in row:
            row[f"{p}_avg_lin_n_global_approx"] = 1 - row[f"{p}_avg_align_global"]


def load_run(d):
    rows = json.load(open(d / "train_eval_log.json"))
    pp = d / "full_params.json"
    params = json.load(open(pp)) if pp.exists() else {}
    tl = t_list_from(rows, params)
    for r in rows:
        ensure_averages(r, tl, int(params.get("T", 1000)))
    return rows, params, tl


class Group:
    """All seeds of one N, aligned on their common eval steps."""

    def __init__(self, N, seed_dirs):
        self.N = N
        loaded = [load_run(d) for _, d in sorted(seed_dirs)]
        self.seeds = [s for s, _ in sorted(seed_dirs)]
        self.params = loaded[0][1]
        self.t_list = loaded[0][2]
        self.K = int(self.params.get("K", 5))
        steps = sorted(set.intersection(*[{r["step"] for r in rows} for rows, _, _ in loaded]))
        self.steps = np.array(steps, float)
        by = [{r["step"]: r for r in rows} for rows, _, _ in loaded]
        self.rows = [[b[s] for s in steps] for b in by]           # [seed][step] -> row

    def has(self, key):
        return any(key in r for rows in self.rows for r in rows)

    def mat(self, key):
        """(n_seeds, n_steps) float array, NaN where missing / None."""
        def val(r):
            v = r.get(key)
            return float(v) if isinstance(v, (int, float)) and v is not None else np.nan
        return np.array([[val(r) for r in rows] for rows in self.rows])

    def mean_sem(self, key):
        m = self.mat(key)
        with np.errstate(all="ignore"):
            mu = np.nanmean(m, 0)
            n = np.sum(~np.isnan(m), 0)
            se = np.where(n > 1, np.nanstd(m, 0, ddof=1) / np.sqrt(np.maximum(n, 1)), 0.0)
        return mu, se

    def last_hist(self, key):
        hs = [np.asarray(rows[-1].get(key), float) for rows in self.rows if rows[-1].get(key)]
        if not hs:
            return None
        hs = [h / h.sum() for h in hs]
        return np.mean(hs, 0)


# ---------------------------------------------------------------- helpers

def pos(g):
    return g.steps > 0                                             # log x drops step 0


def line(ax, g, key, color, ls="-", label=None, shade=True):
    if not g.has(key):
        return False
    mu, se = g.mean_sem(key)
    m = pos(g) & np.isfinite(mu)
    ax.plot(g.steps[m], mu[m], ls, color=color, label=label)
    if shade and len(g.seeds) > 1:
        ax.fill_between(g.steps[m], (mu - se)[m], (mu + se)[m], color=color, alpha=0.15, lw=0)
    return True


def lin_lines(ax, g, pref):
    """Linear / alignment n's, dashed. Exact keys if logged, else 1 - align (approx)."""
    if g.has(f"{pref}_avg_lin_n_global"):
        line(ax, g, f"{pref}_avg_lin_n_local", C_LOCAL, "--", "n_local (linear)", shade=False)
        line(ax, g, f"{pref}_avg_lin_n_global", C_GLOBAL, "--", "n_global (linear)", shade=False)
    else:
        line(ax, g, f"{pref}_avg_lin_n_global_approx", C_GLOBAL, "--",
             "n_global (linear, = 1 - align)", shade=False)


def missing(ax, what):
    ax.text(0.5, 0.5, f"{what}\nnot logged", ha="center", va="center",
            transform=ax.transAxes, color=INK2, fontsize=8)


def finish_rows(axes, groups, xlabel="gradient steps"):
    for r, g in enumerate(groups):
        axes[r, 0].set_ylabel(f"N = {g.N:g}" + (f"\n({len(g.seeds)} seeds)" if len(g.seeds) > 1 else ""),
                              fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    for ax in axes.flat:
        ax.set_xscale("log")


def shared_ylim(axes_col, lo=None, hi=None):
    live = [ax for ax in axes_col if ax.has_data()]
    if not live:
        return
    a = min(ax.get_ylim()[0] for ax in live) if lo is None else lo
    b = max(ax.get_ylim()[1] for ax in live) if hi is None else hi
    for ax in live:
        ax.set_ylim(a, b)


def legends_first_row(axes):
    """One legend per column, on the first row that has labelled data."""
    for c in range(axes.shape[1]):
        for ax in axes[:, c]:
            h, l = ax.get_legend_handles_labels()
            if h:
                ax.legend(h, l, fontsize=7, loc="best", frameon=True, framealpha=0.85, edgecolor="none")
                break



# --------------------------------------------------------------- figure

def figure(groups, out):
    nr = len(groups)
    fig, axes = plt.subplots(nr, 6, figsize=(24, 2.6 * nr + 0.6), sharex=True, squeeze=False,
                             gridspec_kw={"width_ratios": [1.2, 1.2, 1.2, 1, 1, 1]})
    for r, g in enumerate(groups):
        a = axes[r]

        ok = line(a[0], g, "pop_avg_n_local", C_LOCAL, "-", "n_local (projection)")
        line(a[0], g, "pop_avg_n_global", C_GLOBAL, "-", "n_global (projection)")
        lin_lines(a[0], g, "pop")
        a[0].axhline(1, color=INK2, lw=0.8, ls=":")
        if not ok:
            missing(a[0], "t-averaged n")

        if g.has("emp_avg_sw_n_local"):
            ok = line(a[1], g, "emp_avg_sw_n_local", C_LOCAL, "-", "n_local (split of train loss)")
            line(a[1], g, "emp_avg_sw_n_global", C_GLOBAL, "-", "n_global (split of train loss)")
        else:
            ok = line(a[1], g, "emp_avg_n_local", C_LOCAL, "-", "n_local (pop-referenced, old log)")
            line(a[1], g, "emp_avg_n_global", C_GLOBAL, "-", "n_global (pop-referenced, old log)")
        lin_lines(a[1], g, "emp")
        line(a[1], g, "batch_n_local", C_LOCAL, ":", "n_local (regularizer batches)", shade=False)
        line(a[1], g, "batch_n_global", C_GLOBAL, ":", "n_global (regularizer batches)", shade=False)
        a[1].axhline(1, color=INK2, lw=0.8, ls=":")
        a[1].axhline(0, color=INK2, lw=0.8, ls="-")
        if not ok:
            missing(a[1], "t-averaged n")
        a = a[1:]                                  # remaining panels keep their old code

        if g.has("valid"):
            m = pos(g)
            ys = [100 * np.nan_to_num(g.mean_sem(k)[0][m])
                  for k in ("memorized", "valid", "global", "local")]
            a[1].stackplot(g.steps[m], *ys, colors=[C_MEM, C_VALID, C_GLOBAL, C_LOCAL],
                           labels=["memorized", "valid", "global (hallucinated)", "local (bad)"],
                           alpha=0.85, edgecolor="white", linewidth=0.5)
            if g.has("valid_given_recognised"):
                mu, _ = g.mean_sem("valid_given_recognised")
                a[1].plot(g.steps[m], 100 * mu[m], color=INK, label="valid | recognised")
            a[1].axhline(100 / g.K, color=INK, lw=1, ls=":", label="1/K (chance)")
            a[1].set_ylim(0, 100)
        else:
            missing(a[1], "samples")

        if g.has("fid_composite"):
            line(a[2], g, "fid_composite", INK, "-", "composite FID")
            if g.has("fid_composite_real_floor"):
                a[2].axhline(np.nanmean(g.mat("fid_composite_real_floor")), color=INK2,
                             lw=1, ls=":", label="real-data floor")
            a[2].set_yscale("log")
        else:
            missing(a[2], "FID")

        if g.has("memorized"):
            mu, se = g.mean_sem("memorized")
            m = pos(g)
            a[3].plot(g.steps[m], 100 * mu[m], color=C_MEM, label="memorized")
            if len(g.seeds) > 1:
                a[3].fill_between(g.steps[m], 100 * (mu - se)[m], 100 * (mu + se)[m],
                                  color=C_MEM, alpha=0.2, lw=0)
        else:
            missing(a[3], "memorization")

        if not line(a[4], g, "pop_avg_loss", INK, "-", "test loss (held-out)"):
            missing(a[4], "test loss")
        else:
            line(a[4], g, "pop_avg_C", INK2, ":", "Bayes floor C", shade=False)
            a[4].set_yscale("log")

    titles = ["n's, held-out (t-averaged)",
              "n's, train (t-averaged)", "generated samples (%)",
              "composite FID", "memorized samples (%)", "test loss (t-averaged)"]
    for c, t in enumerate(titles):
        axes[0, c].set_title(t, fontsize=10, color=INK)
    for ax in list(axes[:, 0]) + list(axes[:, 1]):  # n panels: own y-range per N
        if ax.has_data():
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(0.0, lo), hi)
    shared_ylim(axes[:, 3])
    shared_ylim(axes[:, 4], lo=0)
    shared_ylim(axes[:, 5])
    finish_rows(axes, groups)
    legends_first_row(axes)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- main

def main(a):
    n_vals = {int(x) for x in a.n_vals.split(",")} if a.n_vals else None
    runs = discover(a.root, a.flag, n_vals)
    groups = [Group(N, sd) for N, sd in runs.items()]
    out = Path(a.out_dir or Path(a.root) / "plots")
    out.mkdir(parents=True, exist_ok=True)
    for g in groups:
        print(f"N={g.N:>7}  seeds {g.seeds}  {len(g.steps)} common eval steps (last {g.steps[-1]:.0f})")
    path = out / "modmnist_summary.png"
    figure(groups, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    p = ArgumentParser(description="Summary figure for train_modmnist.py runs")
    p.add_argument("--root", required=True, help=".../results_modmnist/K5_tile14")
    p.add_argument("--flag", default=None,
                   help="Run-folder suffix to select (e.g. pool100); \"\" = no suffix. "
                        "Required when several suffixes exist.")
    p.add_argument("--n-vals", default=None, help="Comma-separated N to include.")
    p.add_argument("--out-dir", default=None)
    main(p.parse_args())