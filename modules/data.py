"""
Composite MNIST ("double MNIST") data.

A composite image is 64x64: a 2x2 grid of 32x32 tiles (regions i = 0..3,
row-major: 0 top-left, 1 top-right, 2 bottom-left, 3 bottom-right), each
holding one native 28x28 MNIST digit centred with a 2px border. Each tile carries a
class c_i = digit_i mod K -- so K = 10 means the class IS the digit. Composites
are drawn from

    (c_0, c_1, c_2, c_3) ~ Unif(V),        V = the rule's valid set
    X_i | c_i            ~ Unif(pool[c_i]) independently,

where V comes from a TupleRule (modules/rules.py) handed to ResiduePools; the
default is sum_i c_i = 0 (mod K). The rule is carried on the pool so that
everything downstream -- oracle, evaluator, regularizer, samplers -- gets it
without a separate argument.

WHICH V YOU PICK CHANGES THE LEARNING PROBLEM, not just the labels. Under
sum = 0 (mod K) the valid set is a subgroup, so every proper subset of the
(c_i) is exactly independent and ALL of the constraint lives in the 4-way
joint: there is no lower-order correlation for gradient descent to follow, and
FID (a 1st/2nd-order statistic) is provably blind to the rule. Under
c_0 + c_1 = c_2 + c_3 the box truncation breaks the group structure, pairwise
correlations appear (corr = 1/3), and both of those statements fail. See
modules/rules.py.

The POPULATION is this distribution over the finite tile pool (MNIST-train
tiles, optionally capped per digit). The oracle in oracle.py is exact for that
population. A training set of N composites is N i.i.d. draws from it; the
held-out ("pop") eval set is a fresh draw with a different seed.

Pixels are mapped to [-1, 1].
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


TILE_SIZE = 32                 # composite edge is 2 * TILE_SIZE = 64
PAD = (TILE_SIZE - 28) // 2    # MNIST is 28x28


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


def preprocess_tiles(x_uint8):
    """
    uint8 (n,28,28) -> float32 (n,32,32) in [-1, 1].

    The native 28x28 digit is centred on a blank 32x32 canvas (2px border on
    each side). No resampling, so the pixel statistics stay MNIST's own and
    each digit sits in its quadrant with a margin -- four digits in the corners
    of a 64x64 composite, i.e. Lu et al.'s Quarter-MNIST geometry. Padding is
    applied before the [-1,1] map, so the border is 0 -> -1, exactly the MNIST
    background value.
    """
    x = torch.as_tensor(x_uint8).float().div(255.0).unsqueeze(1)    # (n,1,28,28)
    x = F.pad(x, (PAD, PAD, PAD, PAD), value=0.0)                   # (n,1,32,32)
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
    Flat pool of tiles grouped by class (digit mod K), plus the composition
    rule the classes must satisfy.

    flat     (P, d)  tiles, class blocks contiguous
    offsets  (K,)    start index of each class block
    sizes    (K,)    block sizes
    digits   (P,)    original digit label of each tile (for diagnostics)
    rule             TupleRule; defaults to sum = 0 (mod K)
    """

    def __init__(self, tiles, labels, K, per_digit_cap=None, seed=0, device="cpu",
                 rule=None):
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
        # Composition rule on the four tile classes (default: sum = 0 mod K).
        # Carried on the pool so that everything downstream which already has
        # `pools` -- oracle, evaluator, regularizer, samplers -- gets the rule.
        from .rules import modk
        self.rule = (rule if rule is not None else modk(int(K))).to(device)
        if self.rule.C != int(K):
            raise ValueError(f"rule has C={self.rule.C} classes but pools were built with K={K}")
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


def sample_classes(n, rule, generator=None, device="cpu"):
    """(n,4) tile classes drawn uniformly from the rule's valid set."""
    return rule.sample(n, generator=generator, device=device)


def sample_composites(pools, n, generator=None):
    """Returns (images (n,1,2s,2s), classes (n,4), tile_idx (n,4))."""
    c = sample_classes(n, pools.rule, generator=generator, device=pools.device)
    idx = pools.sample_idx(c, generator=generator)
    img = from_tiles(pools.tiles(idx), pools.tile_size)
    return img, c, idx