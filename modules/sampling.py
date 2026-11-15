"""
Sampling + sample classification for mod-K MNIST.

Tiles are read by an 11-way CNN (digits + REJECT). A tile counts as a
recognised digit only if the argmax is a digit AND its prob >= conf_thresh.

Categories (mutually exclusive, sum to 1), in precedence order:
  memorized  nearest-training-composite test (see `memorized_mask`)
  local      novel, and at least one tile is not a confidently recognised
             digit                                   -> locally wrong
  global     novel, all four tiles recognised, sum of residues != 0 mod K
                                                     -> globally wrong (hallucinated)
  valid      novel, all four recognised, constraint satisfied

Reference point: a sampler that gets every tile right but ignores the
coupling has valid / (valid + global) = 1/K.
"""
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import to_tiles


# ------------------------------------------------------------------ sampler

@torch.no_grad()
def ddim_sample(model, sched, n, img_size, device, n_steps=250, eta=1.0,
                batch_size=1024, generator=None, amp=None):
    """
    DDIM with stochasticity eta (eta=1 ~ ancestral DDPM on the strided grid,
    eta=0 deterministic). x0-hat is clipped to [-1,1] before the update.

    amp = torch.float16 / torch.bfloat16 runs the DENOISER forward in that dtype
    (the DDIM update arithmetic stays fp32). Roughly 2x. Safe here because the
    samples are only classified / FID'd. V100 has fp16 tensor cores but no bf16.
    """
    model.eval()
    T = sched.T
    ts = torch.linspace(T, 1, n_steps, device=device).round().long().unique_consecutive()
    out = []
    for lo in range(0, n, batch_size):
        m = min(batch_size, n - lo)
        x = torch.randn(m, 1, img_size, img_size, device=device, generator=generator)
        for k, t in enumerate(ts):
            tb = torch.full((m,), int(t), dtype=torch.long, device=device)
            ab = sched.abar[t - 1].float()
            ab_prev = (sched.abar[ts[k + 1] - 1].float() if k + 1 < len(ts)
                       else torch.tensor(1.0, device=device))
            with (torch.autocast("cuda", dtype=amp)
                  if amp is not None and device.type == "cuda"
                  else contextlib.nullcontext()):
                e = model(x, tb)
            e = e.float()
            x0 = ((x - (1 - ab).sqrt() * e) / ab.sqrt()).clamp(-1, 1)
            e = (x - ab.sqrt() * x0) / (1 - ab).sqrt()                  # consistent with clipped x0
            sig = eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).clamp_min(0).sqrt()
            dir_ = (1 - ab_prev - sig ** 2).clamp_min(0).sqrt() * e
            x = ab_prev.sqrt() * x0 + dir_
            if k + 1 < len(ts):
                x = x + sig * torch.randn(x.shape, device=device, generator=generator)
        out.append(x.clamp(-1, 1))
    return torch.cat(out)


# --------------------------------------------------------- tile classifier
# (lives in tile_classifier.py; re-exported here for existing imports)
from .tile_classifier import (TileCNN, REJECT, train_tile_classifier,  # noqa: F401,E402
                              load_tile_classifier, save_tile_classifier)


# ----------------------------------------------------------- memorization

@torch.no_grad()
def nn_distances(gen, train, chunk=512):
    """
    gen (n,1,H,W), train (N,1,H,W) -> (d1, d2): L2 distance to the nearest
    and second-nearest training composite. Squared-norm expansion, chunked.
    """
    G = gen.flatten(1).float()
    Tr = train.flatten(1).float()
    tn = (Tr * Tr).sum(1)
    d1s, d2s = [], []
    for lo in range(0, G.shape[0], chunk):
        g = G[lo:lo + chunk]
        d2 = (g * g).sum(1, keepdim=True) - 2 * g @ Tr.T + tn[None]
        k = min(2, Tr.shape[0])
        v = d2.clamp_min(0).topk(k, dim=1, largest=False).values.sqrt()
        d1s.append(v[:, 0])
        d2s.append(v[:, 1] if k > 1 else torch.full_like(v[:, 0], float("inf")))
    return torch.cat(d1s), torch.cat(d2s)


def memorized_mask(d1, d2, D, rule="ratio", ratio=1 / 3, rms=0.1):
    """
    rule='ratio': d1 < ratio * d2 (nearest training image much closer than
                  the runner-up; Carlini et al. 2023-style extraction test).
    rule='rms'  : d1 / sqrt(D) < rms (absolute per-pixel RMS distance, [-1,1] scale).
    """
    if rule == "ratio":
        return d1 < ratio * d2
    if rule == "rms":
        return d1 / D ** 0.5 < rms
    raise ValueError(rule)


@torch.no_grad()
def classify_samples(gen, clf, rule, train_imgs, conf_thresh=0.9, mem_rule="ratio",
                     mem_ratio=1 / 3, mem_rms=0.1, return_per_sample=False):
    n = gen.shape[0]
    s = gen.shape[-1] // 2
    tiles = to_tiles(gen).view(n * 4, 1, s, s)
    nc = getattr(clf, "n_classes", 10)
    probs = torch.softmax(clf(tiles).float(), -1).view(n, 4, nc)
    conf, cls = probs.max(-1)                                         # (n,4)
    is_digit = cls < 10                                               # not REJECT
    digit = probs[..., :10].argmax(-1)
    residue = digit % rule.C
    tile_ok = is_digit & (conf >= conf_thresh)
    recognised = tile_ok.all(1)
    sat = rule.satisfies(residue)

    d1, d2 = nn_distances(gen, train_imgs)
    mem = memorized_mask(d1, d2, gen[0].numel(), mem_rule, mem_ratio, mem_rms)
    novel = ~mem
    valid = novel & recognised & sat
    glob = novel & recognised & ~sat
    loc = novel & ~recognised
    n_rec = int((novel & recognised).sum())
    out = {
        "memorized": float(mem.float().mean()),
        "valid": float(valid.float().mean()),
        "local": float(loc.float().mean()),
        "global": float(glob.float().mean()),
        "tile_recognised": float(tile_ok.float().mean()),
        "tile_rejected": float((~is_digit).float().mean()),
        "valid_given_recognised": (float(valid.sum()) / n_rec) if n_rec else float("nan"),
        "nn_dist_mean": float(d1.mean()),
        "nn_ratio_mean": float((d1 / d2).mean()),
        "residue_hist": torch.bincount(residue.flatten(), minlength=rule.C).tolist(),
        "sum_hist": torch.bincount(residue.sum(1), minlength=1).tolist(),
    }
    if not return_per_sample:
        return out
    per = {"valid": valid, "global": glob, "local": loc, "memorized": mem,
           "residue": residue, "conf": conf, "tile_ok": tile_ok, "sat": sat,
           "recognised": recognised, "d1": d1, "d2": d2}
    return out, per


# ------------------------------------------------------- viewable sample dumps

CATEGORIES = ("valid", "global", "local", "memorized")


def _show(ax, img):
    ax.imshow((img.float().cpu() + 1) / 2, cmap="gray", vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def _readout(per, i, rule):
    """'3 4 | 2 5  sum 7=7  OK' -- the classifier's read-out of one sample,
    slots row-major (TL TR / BL BR). '?' marks a tile below conf_thresh."""
    ok = per["tile_ok"][i]
    r = per["residue"][i].tolist()
    s = [str(v) if bool(ok[k]) else "?" for k, v in enumerate(r)]
    head = f"{s[0]} {s[1]} / {s[2]} {s[3]}"
    if not bool(ok.all()):
        return head + "  (tile unread)"
    tag = "rule OK" if bool(per["sat"][i]) else "rule BROKEN"
    if rule.name.startswith("sumpair"):
        head += f"   {r[0]}+{r[1]}={r[0] + r[1]} vs {r[2]}+{r[3]}={r[2] + r[3]}"
    elif rule.name.startswith(("modk", "checksum")):
        head += f"   sum={sum(r) % rule.C} mod {rule.C}"
    return head + f"\n{tag}"


def save_sample_png(path, gen, per, rule, n_per_cat=1, dpi=110):
    """
    One actual image per sample category (valid / global / local / memorized),
    titled with what the tile classifier read off it. This is the sanity check
    the scalar rates cannot give: whether a 'valid' sample really is four clean
    digits obeying the rule, or whether the classifier is being fooled.

    Returns {category: count in this batch}.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = max(1, int(n_per_cat))
    counts = {c: int(per[c].sum()) for c in CATEGORIES}
    fig, axes = plt.subplots(len(CATEGORIES), cols, squeeze=False,
                             figsize=(2.3 * cols, 2.7 * len(CATEGORIES)))
    for r, cat in enumerate(CATEGORIES):
        idx = per[cat].nonzero(as_tuple=True)[0][:cols]
        for j in range(cols):
            ax = axes[r][j]
            if j >= len(idx):
                ax.axis("off")
                if j == 0:
                    ax.text(0.5, 0.5, f"{cat}\n(none in this batch)", ha="center",
                            va="center", fontsize=9, color="0.4")
                continue
            i = int(idx[j])
            _show(ax, gen[i, 0])
            ttl = _readout(per, i, rule)
            if j == 0:
                ttl = f"[{cat}]  {counts[cat]}/{gen.shape[0]}\n" + ttl
            ax.set_title(ttl, fontsize=8, fontfamily="monospace")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return counts


def save_sample_grid(path, gen, n=8, dpi=110):
    """Plain n x n grid of the first n*n samples -- overall image quality at a glance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = max(1, min(int(n), int(gen.shape[0] ** 0.5)))
    fig, axes = plt.subplots(n, n, squeeze=False, figsize=(1.1 * n, 1.1 * n))
    for k in range(n * n):
        ax = axes[k // n][k % n]
        _show(ax, gen[k, 0])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)