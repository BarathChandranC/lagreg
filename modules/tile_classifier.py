"""
Tile classifier used to sort generated samples: 11 classes = digits 0-9 + REJECT.

Why each piece is here
----------------------
* REJECT class. A clean-digit-only CNN is confidently wrong on noise: an
  untrained diffusion model's samples were read as four identical digits with
  p > 0.9, which scored pure noise as 'global' (hallucinated). The reject
  class is trained on synthetic non-digits (make_negatives) so garbage lands
  in 'local' instead.
* Augmentation (augment): small rotation / scale / sub-pixel translation
  with background-preserving padding (no wrap-around), plus mild Gaussian
  noise (sigma <= 0.3). It is applied LABEL-INDEPENDENTLY -- to a random
  subset of every batch, digits and negatives alike -- so that neither
  resampling blur nor mild noise nor their absence can separate the classes.
  (An earlier version augmented only the digits; the network then learned
  "sharp image = not a digit" and rejected clean test digits.) The
  noised-digit negatives use much heavier noise (sigma >= 0.87) than the
  mild augmentation noise, so the two never overlap.
* Validation split held out from MNIST-train; best epoch chosen by validation
  NLL on digits + negatives jointly.
* Temperature scaling fitted on validation (one scalar T, stored as a buffer
  so forward() returns calibrated logits). This makes "prob >= 0.9" a
  calibrated statement rather than an artefact of training length.
* Reports, on MNIST-test and on negatives built from test digits:
    digit accuracy, false-reject rate, accepted@thr (correct / wrong),
    negatives rejected, negatives accepted@thr overall and per negative type.
  'negatives accepted@thr' is the number that matters most: it is the rate at
  which garbage would be counted as a recognised digit.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

REJECT = 10
NEG_TYPES = ("gauss_noise", "salt_pepper", "blobs", "overlay",
             "noised_digit", "near_blank", "scribble")


# ------------------------------------------------------------------- model

class TileCNN(nn.Module):
    def __init__(self, tile_size=14, width=48, n_classes=11):
        super().__init__()
        w = width
        self.f = nn.Sequential(
            nn.Conv2d(1, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(),
            nn.Conv2d(w, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(w, 2 * w, 3, padding=1), nn.BatchNorm2d(2 * w), nn.ReLU(),
            nn.Conv2d(2 * w, 2 * w, 3, padding=1), nn.BatchNorm2d(2 * w), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.2), nn.Linear(2 * w, n_classes),
        )
        self.tile_size, self.width, self.n_classes = tile_size, width, n_classes
        self.register_buffer("temperature", torch.ones(()))

    def forward(self, x):
        """Calibrated logits (raw logits / temperature)."""
        return self.f(x) / self.temperature


# ------------------------------------------------------------ augmentation

def augment(x, p_geom=0.6, max_rot_deg=12.0, scale=(0.9, 1.1), max_shift_px=1.5,
            p_noise=0.3, noise_max=0.3):
    """
    x: (n,1,s,s) in [-1,1]. Each image independently gets the random affine
    with prob p_geom and mild noise with prob p_noise; the rest pass through
    unchanged. Must be applied to digits and negatives alike (see module doc).
    Background (-1) is preserved at the borders.
    """
    n, _, s, _ = x.shape
    dev = x.device
    ang = (torch.rand(n, device=dev) * 2 - 1) * math.radians(max_rot_deg)
    sc = torch.empty(n, device=dev).uniform_(*scale)
    tx = (torch.rand(n, device=dev) * 2 - 1) * max_shift_px * 2 / s
    ty = (torch.rand(n, device=dev) * 2 - 1) * max_shift_px * 2 / s
    c, si = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([c, -si, tx], 1),
                         torch.stack([si, c, ty], 1)], 1)                 # (n,2,3)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    # shift to [0,2] so zero padding == background, then back
    warped = F.grid_sample(x + 1, grid, mode="bilinear", padding_mode="zeros",
                           align_corners=False) - 1
    geo = (torch.rand(n, 1, 1, 1, device=dev) < p_geom)
    y = torch.where(geo, warped, x)
    noisy = torch.rand(n, 1, 1, 1, device=dev) < p_noise
    sig = torch.rand(n, 1, 1, 1, device=dev) * noise_max
    y = torch.where(noisy, y + sig * torch.randn_like(y), y)
    return y.clamp(-1, 1)


augment_digits = augment          # backwards-compatible name


# --------------------------------------------------------------- negatives

def _scribbles(n, s, dev, k=7):
    """Tangled random-walk strokes: k connected segments, soft stroke width."""
    start = torch.rand(n, 1, 2, device=dev) * 0.6 * s + 0.2 * s
    steps = torch.randn(n, k, 2, device=dev) * 0.3 * s
    pts = torch.cat([start, start + steps.cumsum(1)], 1).clamp(0, s - 1)  # (n,k+1,2)
    a, b = pts[:, :-1], pts[:, 1:]                                        # (n,k,2)
    yy, xx = torch.meshgrid(torch.arange(s, device=dev, dtype=torch.float32),
                            torch.arange(s, device=dev, dtype=torch.float32),
                            indexing="ij")
    P = torch.stack([xx, yy], -1).view(1, 1, s * s, 2)
    ab = (b - a).unsqueeze(2)                                             # (n,k,1,2)
    ap = P - a.unsqueeze(2)                                               # (n,k,ss,2)
    tpar = ((ap * ab).sum(-1) / (ab.pow(2).sum(-1) + 1e-6)).clamp(0, 1)
    d = (ap - tpar.unsqueeze(-1) * ab).norm(dim=-1).min(1).values         # (n,ss)
    w = 0.5 + 0.6 * torch.rand(n, 1, device=dev)
    return (2 * torch.exp(-(d / w) ** 2) - 1).view(n, 1, s, s)


def make_negatives(n, X, Y, device):
    """
    n synthetic non-digit tiles (n,1,s,s) in [-1,1] and their type index
    (into NEG_TYPES). X (P,1,s,s) / Y (P,) supply real digits for the
    'overlay' and 'noised_digit' types -- pass digits from the SAME split
    (train digits for training, val/test digits for evaluation).
    """
    s = X.shape[-1]
    u = lambda *shape: torch.rand(*shape, device=device)
    cands = [
        torch.randn(n, 1, s, s, device=device) * (0.3 + 1.2 * u(n, 1, 1, 1))
        + (-1 + 1.3 * u(n, 1, 1, 1)),                                     # gauss_noise
        torch.where(u(n, 1, s, s) < 0.05 + 0.45 * u(n, 1, 1, 1),
                    torch.ones((), device=device), -torch.ones((), device=device)),  # salt_pepper
        F.interpolate(torch.randn(n, 1, 4, 4, device=device) * 1.5 - 0.5,
                      size=(s, s), mode="bilinear", align_corners=False),  # blobs
    ]
    ia = torch.randint(0, X.shape[0], (n,), device=device)
    ib = torch.randint(0, X.shape[0], (n,), device=device)
    same = Y[ia] == Y[ib]
    ib[same] = (ib[same] + 1) % X.shape[0]
    cands.append(torch.maximum(X[ia], X[ib]))                             # overlay
    a = 0.1 + 0.4 * u(n, 1, 1, 1)
    cands.append(a * X[ia] + (1 - a ** 2).sqrt()
                 * torch.randn(n, 1, s, s, device=device))                # noised_digit
    cands.append(-1 + 0.3 * torch.randn(n, 1, s, s, device=device).abs())  # near_blank
    cands.append(_scribbles(n, s, device))                                # scribble
    kind = torch.randint(0, len(cands), (n,), device=device)
    out = torch.stack(cands, 0)[kind, torch.arange(n, device=device)]
    return out.clamp(-1, 1), kind


# -------------------------------------------------------------- evaluation

@torch.no_grad()
def _logits(clf, x, batch=4096):
    clf.eval()
    return torch.cat([clf(x[i:i + batch]).float() for i in range(0, x.shape[0], batch)])


@torch.no_grad()
def evaluate(clf, Xd, Yd, Xn, Kn, conf_thresh=0.9, logits=None):
    """Metrics on real digits (Xd,Yd) and negatives (Xn, types Kn)."""
    ld, ln = logits if logits is not None else (_logits(clf, Xd), _logits(clf, Xn))
    rej = torch.full((ln.shape[0],), REJECT, device=ln.device, dtype=torch.long)
    nll = (F.cross_entropy(ld, Yd, reduction="sum")
           + F.cross_entropy(ln, rej, reduction="sum")) / (ld.shape[0] + ln.shape[0])
    cd, pd = torch.softmax(ld, -1).max(-1)
    cn, pn = torch.softmax(ln, -1).max(-1)
    acc_d = (pd < 10) & (cd >= conf_thresh)
    acc_n = (pn < 10) & (cn >= conf_thresh)
    m = lambda v: float(v.float().mean())
    out = {
        "nll": float(nll),
        "digit_acc": m(pd == Yd),
        "false_reject": m(pd == REJECT),
        "accepted_correct": m(acc_d & (pd == Yd)),
        "accepted_wrong": m(acc_d & (pd != Yd)),
        "neg_rejected": m(pn == REJECT),
        "neg_accepted": m(acc_n),
    }
    out["neg_accepted_by_type"] = {
        NEG_TYPES[k]: m(acc_n[Kn == k]) for k in range(len(NEG_TYPES)) if (Kn == k).any()}
    return out


@torch.no_grad()
def fit_temperature(clf, Xd, Yd, Xn):
    """Single scalar T minimising NLL on (digits + negatives); grid search."""
    clf.temperature.fill_(1.0)
    ld, ln = _logits(clf, Xd), _logits(clf, Xn)
    L = torch.cat([ld, ln])
    T_lab = torch.cat([Yd, torch.full((ln.shape[0],), REJECT, device=ln.device, dtype=torch.long)])
    Ts = torch.logspace(math.log10(0.3), math.log10(10.0), 200, device=L.device)
    nll = torch.stack([F.cross_entropy(L / T, T_lab) for T in Ts])
    T = float(Ts[nll.argmin()])
    clf.temperature.fill_(T)
    return T


# ---------------------------------------------------------------- training

def train_tile_classifier(tr_x, tr_y, te_x, te_y, tile_size, device, epochs=15,
                          batch_size=512, lr=1e-3, weight_decay=1e-4, neg_frac=0.33,
                          val_size=5000, conf_thresh=0.9, width=48, seed=0, log=print):
    """
    tr_x / te_x: (n,s,s) float in [-1,1]; tr_y / te_y: digit labels.
    Returns (clf, report). clf carries its fitted temperature; report holds
    validation-selected epoch, temperature and all test metrics.
    """
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    X = torch.as_tensor(tr_x).float().unsqueeze(1).to(device)
    Y = torch.as_tensor(tr_y).long().to(device)
    perm = torch.randperm(X.shape[0], generator=g).to(device)
    Xv, Yv = X[perm[:val_size]], Y[perm[:val_size]]
    Xt, Yt = X[perm[val_size:]], Y[perm[val_size:]]
    Xn_v, Kn_v = make_negatives(val_size, Xv, Yv, device)          # fixed val negatives

    clf = TileCNN(tile_size, width=width).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_ep = -(-Xt.shape[0] // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_ep,
                                                pct_start=0.15)
    n_neg = max(1, int(neg_frac * batch_size))
    best = (math.inf, None, -1)
    for ep in range(epochs):
        clf.train()
        order = torch.randperm(Xt.shape[0], device=device)
        for lo in range(0, Xt.shape[0], batch_size):
            idx = order[lo:lo + batch_size]
            xn, _ = make_negatives(n_neg, Xt, Yt, device)
            xb = augment(torch.cat([Xt[idx], xn]))        # label-independent augmentation
            yb = torch.cat([Yt[idx], torch.full((n_neg,), REJECT, device=device, dtype=torch.long)])
            loss = F.cross_entropy(clf(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        v = evaluate(clf, Xv, Yv, Xn_v, Kn_v, conf_thresh)
        log(f"epoch {ep + 1:2d}/{epochs}  val: nll {v['nll']:.4f}  digit acc {v['digit_acc']:.4f}  "
            f"false reject {v['false_reject']:.4f}  neg rejected {v['neg_rejected']:.4f}  "
            f"neg accepted@{conf_thresh} {v['neg_accepted']:.4f}")
        if v["nll"] < best[0]:
            best = (v["nll"], copy.deepcopy(clf.state_dict()), ep + 1)

    clf.load_state_dict(best[1])
    T = fit_temperature(clf, Xv, Yv, Xn_v)
    v = evaluate(clf, Xv, Yv, Xn_v, Kn_v, conf_thresh)
    log(f"best epoch {best[2]}, temperature {T:.3f}; calibrated val: digit acc {v['digit_acc']:.4f}  "
        f"neg accepted@{conf_thresh} {v['neg_accepted']:.4f}")

    Xte = torch.as_tensor(te_x).float().unsqueeze(1).to(device)
    Yte = torch.as_tensor(te_y).long().to(device)
    torch.manual_seed(seed + 12345)
    Xn_te, Kn_te = make_negatives(Xte.shape[0], Xte, Yte, device)
    te = evaluate(clf, Xte, Yte, Xn_te, Kn_te, conf_thresh)
    log(f"TEST (MNIST-test + negatives from test digits), threshold {conf_thresh}:")
    log(f"  digit acc {te['digit_acc']:.4f} | false reject {te['false_reject']:.4f} | "
        f"accepted & correct {te['accepted_correct']:.4f} | accepted & wrong {te['accepted_wrong']:.4f}")
    log(f"  negatives rejected {te['neg_rejected']:.4f} | negatives accepted {te['neg_accepted']:.4f}")
    log("  negatives accepted by type: " + "  ".join(
        f"{k} {val:.4f}" for k, val in te["neg_accepted_by_type"].items()))
    clf.eval()
    report = {"best_epoch": best[2], "temperature": T, "conf_thresh": conf_thresh,
              "val": v, "test": te, "test_acc": te["digit_acc"]}
    return clf, report


def load_tile_classifier(path, tile_size, device):
    """Load a checkpoint written by save_tile_classifier; refuses obsolete ones."""
    st = torch.load(path, map_location=device, weights_only=True)
    if st.get("tile_size") != tile_size:
        raise ValueError(f"classifier at {path} is for tile_size={st.get('tile_size')}, need {tile_size}")
    if st.get("n_classes", 10) != 11 or "temperature" not in st["state_dict"]:
        raise ValueError(
            f"classifier at {path} is an old version (no reject class or no calibration). "
            f"Retrain: python train_tile_classifier.py --tile-size {tile_size}")
    clf = TileCNN(tile_size, width=st.get("width", 48), n_classes=11).to(device)
    clf.load_state_dict(st["state_dict"])
    clf.eval()
    return clf, st.get("report", {})


def save_tile_classifier(clf, report, path):
    torch.save({"state_dict": clf.state_dict(), "tile_size": clf.tile_size,
                "width": clf.width, "n_classes": clf.n_classes, "report": report}, path)