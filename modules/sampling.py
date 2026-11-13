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
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import to_tiles


# ------------------------------------------------------------------ sampler

@torch.no_grad()
def ddim_sample(model, sched, n, img_size, device, n_steps=250, eta=1.0,
                batch_size=1024, generator=None):
    """
    DDIM with stochasticity eta (eta=1 ~ ancestral DDPM on the strided grid,
    eta=0 deterministic). x0-hat is clipped to [-1,1] before the update.
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
            e = model(x, tb).float()
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
def classify_samples(gen, clf, K, train_imgs, conf_thresh=0.9, mem_rule="ratio",
                     mem_ratio=1 / 3, mem_rms=0.1):
    n = gen.shape[0]
    s = gen.shape[-1] // 2
    tiles = to_tiles(gen).view(n * 4, 1, s, s)
    nc = getattr(clf, "n_classes", 10)
    probs = torch.softmax(clf(tiles).float(), -1).view(n, 4, nc)
    conf, cls = probs.max(-1)                                         # (n,4)
    is_digit = cls < 10                                               # not REJECT
    digit = probs[..., :10].argmax(-1)
    residue = digit % K
    tile_ok = is_digit & (conf >= conf_thresh)
    recognised = tile_ok.all(1)
    sat = (residue.sum(1) % K) == 0

    d1, d2 = nn_distances(gen, train_imgs)
    mem = memorized_mask(d1, d2, gen[0].numel(), mem_rule, mem_ratio, mem_rms)
    novel = ~mem
    valid = novel & recognised & sat
    glob = novel & recognised & ~sat
    loc = novel & ~recognised
    n_rec = int((novel & recognised).sum())
    return {
        "memorized": float(mem.float().mean()),
        "valid": float(valid.float().mean()),
        "local": float(loc.float().mean()),
        "global": float(glob.float().mean()),
        "tile_recognised": float(tile_ok.float().mean()),
        "tile_rejected": float((~is_digit).float().mean()),
        "valid_given_recognised": (float(valid.sum()) / n_rec) if n_rec else float("nan"),
        "nn_dist_mean": float(d1.mean()),
        "nn_ratio_mean": float((d1 / d2).mean()),
        "residue_hist": torch.bincount(residue.flatten(), minlength=K).tolist(),
        "sum_mod_hist": torch.bincount(residue.sum(1) % K, minlength=K).tolist(),
    }