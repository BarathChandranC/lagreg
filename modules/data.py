"""
Mod-K MNIST ("double MNIST") data.

A composite image is a 2x2 grid of MNIST tiles (regions i = 0..3, row-major:
0 top-left, 1 top-right, 2 bottom-left, 3 bottom-right). Each tile carries a
residue class c_i = digit_i mod K. Composites are drawn from

    (c_0, c_1, c_2) ~ Unif(Z_K^3),   c_3 = -(c_0 + c_1 + c_2) mod K,
    X_i | c_i       ~ Unif(pool[c_i])  independently,

so every proper subset of the four (c_i) -- and of the four tiles -- is
independent, and the only dependence is the 4-way constraint.

The POPULATION is this distribution over the finite tile pool (MNIST-train
tiles, optionally capped per digit). The oracle in oracle.py is exact for
that population. A training set of N composites is N i.i.d. draws from it;
the held-out ("pop") eval set is a fresh draw with a different seed.

Pixels are mapped to [-1, 1].
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------- MNIST I/O

def load_mnist(root="../data", npz_path=None):
    """
    Returns (x_train, y_train, x_test, y_test) as uint8 (n,28,28) / int64 (n,).

    npz_path: a keras-style mnist.npz (keys x_train, y_train, x_test, y_test).
    Otherwise torchvision.datasets.MNIST(root, download=True) is used.
    """
    if npz_path is not None:
        d = np.load(npz_path)
        return (d["x_train"], d["y_train"].astype(np.int64),
                d["x_test"], d["y_test"].astype(np.int64))
    from torchvision.datasets import MNIST
    tr = MNIST(root, train=True, download=True)
    te = MNIST(root, train=False, download=True)
    return (tr.data.numpy(), tr.targets.numpy().astype(np.int64),
            te.data.numpy(), te.targets.numpy().astype(np.int64))


def preprocess_tiles(x_uint8, tile_size):
    """uint8 (n,28,28) -> float32 (n, tile_size, tile_size) in [-1, 1]."""
    x = torch.as_tensor(x_uint8).float().div(255.0).unsqueeze(1)    # (n,1,28,28)
    if tile_size != 28:
        if 28 % tile_size != 0:
            raise ValueError("tile_size must divide 28 (28, 14, 7)")
        x = F.avg_pool2d(x, 28 // tile_size)
    return (x * 2.0 - 1.0).squeeze(1).contiguous()


# --------------------------------------------------------- tile <-> image

def to_tiles(img):
    """(B,1,2s,2s) -> (B,4,s*s), regions row-major."""
    B, C, H, W = img.shape
    s = H // 2
    t = img.view(B, C, 2, s, 2, s).permute(0, 2, 4, 1, 3, 5)        # (B,2,2,C,s,s)
    return t.reshape(B, 4, C * s * s)


def from_tiles(tiles, s):
    """(B,4,s*s) -> (B,1,2s,2s)."""
    B = tiles.shape[0]
    t = tiles.view(B, 2, 2, 1, s, s).permute(0, 3, 1, 4, 2, 5)      # (B,1,2,s,2,s)
    return t.reshape(B, 1, 2 * s, 2 * s)


# ------------------------------------------------------------- tile pools

class ResiduePools:
    """
    Flat pool of tiles grouped by residue class.

    flat     (P, d)  tiles, residue blocks contiguous
    offsets  (K,)    start index of each residue block
    sizes    (K,)    block sizes
    digits   (P,)    original digit label of each tile (for diagnostics)
    """

    def __init__(self, tiles, labels, K, per_digit_cap=None, seed=0, device="cpu"):
        rng = np.random.default_rng(seed)
        labels = np.asarray(labels)
        keep = []
        for dgt in range(10):
            idx = np.flatnonzero(labels == dgt)
            if per_digit_cap is not None and len(idx) > per_digit_cap:
                idx = rng.choice(idx, per_digit_cap, replace=False)
            keep.append(np.sort(idx))
        keep = np.concatenate(keep)
        res = labels[keep] % K
        order = np.argsort(res, kind="stable")
        keep, res = keep[order], res[order]

        self.K = int(K)
        self.tile_size = int(tiles.shape[-1])
        self.d = self.tile_size ** 2
        self.flat = tiles[torch.as_tensor(keep)].reshape(len(keep), -1).to(device)
        self.digits = torch.as_tensor(labels[keep]).to(device)
        sizes = np.bincount(res, minlength=K)
        if (sizes == 0).any():
            raise ValueError(f"empty residue pool(s): sizes={sizes.tolist()}")
        self.sizes = torch.as_tensor(sizes, dtype=torch.long, device=device)
        self.offsets = torch.as_tensor(np.concatenate([[0], np.cumsum(sizes)[:-1]]),
                                       dtype=torch.long, device=device)
        self.device = torch.device(device)

    def sample_idx(self, classes, generator=None):
        """Uniform tile index within each residue pool. classes: long tensor, any shape."""
        u = torch.rand(classes.shape, device=self.device, generator=generator)
        within = (u * self.sizes[classes].double()).long()
        within = torch.minimum(within, self.sizes[classes] - 1)
        return self.offsets[classes] + within

    def tiles(self, idx):
        return self.flat[idx]                                          # (..., d)


def sample_classes(n, K, generator=None, device="cpu"):
    """(n,4) residues: three uniform, the 4th fixed by sum == 0 mod K."""
    c = torch.randint(0, K, (n, 4), device=device, generator=generator)
    c[:, 3] = (-(c[:, 0] + c[:, 1] + c[:, 2])) % K
    return c


def sample_composites(pools, n, generator=None):
    """Returns (images (n,1,2s,2s), classes (n,4), tile_idx (n,4))."""
    c = sample_classes(n, pools.K, generator=generator, device=pools.device)
    idx = pools.sample_idx(c, generator=generator)
    img = from_tiles(pools.tiles(idx), pools.tile_size)
    return img, c, idx
