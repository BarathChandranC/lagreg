"""
Train the tile classifier used to sort generated samples (digits 0-9 + REJECT).
Run ONCE per tile size before a sweep, on a GPU:

    srun --gpus=1 --time=00:30:00 --mem=16G python -u train_tile_classifier.py

Writes ../checkpoints/tile_cnn_tile{s}.pt (weights + fitted temperature +
full validation/test report) and, unless --no-preview, a PNG next to it
showing every negative type and some augmented digits with the classifier's
call on each -- look at it once to check the reject class makes sense.

What to look for in the printed TEST block:
  digit acc            ~0.99
  false reject         < 0.01   (real digits sent to 'not a digit')
  negatives accepted   < 0.01   (garbage counted as a confident digit) -- the key one
  per-type accepted    all small; a large value names the kind of garbage that leaks
"""
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.data import TILE_SIZE, load_mnist, preprocess_tiles
from modules.tile_classifier import (train_tile_classifier, save_tile_classifier,
                                     make_negatives, augment, NEG_TYPES, REJECT)


def preview(clf, X, Y, device, out_png, conf_thresh, n_per_row=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    torch.manual_seed(0)
    rows = []
    for k, name in enumerate(NEG_TYPES):          # one row per negative type
        xs, kinds = [], []
        while sum(len(x) for x in xs) < n_per_row:
            xn, kn = make_negatives(200, X, Y, device)
            xs.append(xn[kn == k])
        rows.append((name, torch.cat(xs)[:n_per_row]))
    idx = torch.randint(0, X.shape[0], (n_per_row,), device=device)
    rows.append(("digit (augmented)", augment(X[idx], p_geom=1.0, p_noise=0.5), Y[idx]))
    rows.append(("digit (clean)", X[idx], Y[idx]))
    rows = [r if len(r) == 3 else (r[0], r[1], None) for r in rows]

    fig, axes = plt.subplots(len(rows), n_per_row, figsize=(1.1 * n_per_row, 1.25 * len(rows)))
    with torch.no_grad():
        for r, (name, xs, ys) in enumerate(rows):
            conf, pred = torch.softmax(clf(xs), -1).max(-1)
            accepted = (pred < 10) & (conf >= conf_thresh)
            for c in range(n_per_row):
                ax = axes[r, c]
                ax.imshow(xs[c, 0].cpu(), cmap="gray", vmin=-1, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                lab = "rej" if int(pred[c]) == REJECT else str(int(pred[c]))
                if ys is None:                       # negative: wrong if accepted as a digit
                    wrong = bool(accepted[c])
                else:                                # digit: wrong unless accepted as its label
                    wrong = not (bool(accepted[c]) and int(pred[c]) == int(ys[c]))
                ax.set_title(f"{lab} {float(conf[c]):.2f}", fontsize=7,
                             color="red" if wrong else "black")
                if c == 0:
                    ax.set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle(f"classifier calls (red = would be miscounted at threshold {conf_thresh})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--neg-frac", type=float, default=0.33,
                   help="Negatives per batch as a fraction of the digit batch size.")
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--conf-thresh", type=float, default=0.9,
                   help="Threshold the metrics are reported at; use the same value as "
                        "train_modmnist.py --conf-thresh.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--mnist-root", type=str, default="../data")
    p.add_argument("--mnist-npz", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--no-preview", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    xtr, ytr, xte, yte = load_mnist(args.mnist_root, args.mnist_npz)
    s = TILE_SIZE
    clf, report = train_tile_classifier(
        preprocess_tiles(xtr), ytr, preprocess_tiles(xte), yte, s, device,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, neg_frac=args.neg_frac,
        val_size=args.val_size, conf_thresh=args.conf_thresh, width=args.width, seed=args.seed)

    out = Path(args.out or f"../checkpoints/tile_cnn_tile{s}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_tile_classifier(clf, report, out)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump({**report, "args": vars(args)}, f, indent=2)
    print(f"saved {out} and {out.with_suffix('.json')}")

    if not args.no_preview:
        try:
            Xte = preprocess_tiles(xte).unsqueeze(1).to(device)
            Yte = torch.as_tensor(yte).long().to(device)
            png = out.with_name(out.stem + "_preview.png")
            preview(clf, Xte, Yte, device, png, args.conf_thresh)
            print(f"saved {png}")
        except Exception as e:                      # preview is a convenience only
            print(f"preview skipped: {e}")