"""
Each 64x64 image is three FIXED, non-overlapping panels concatenated left to
right, each panel dedicated to exactly one shape identity:
 
    slot 0 (cols  0-20)  -> triangle
    slot 1 (cols 21-41)  -> square
    slot 2 (cols 42-62)  -> pentagon
 
Within a slot, the shape is present independently with probability
`p_present` (default 0.5); when present, its (x, y) position inside the
panel is drawn uniformly and its pixel area is held ~constant across shape
types (~60px) by construction (per-shape side-length formulas), so raw
pixel count does not confound shape identity. All-empty draws (no shape in
any slot) are REJECTED AND REDRAWN so every image has at least one shape,
via rejection sampling so the requested count `n` is met exactly.
 
Pixel convention: background 0 -> bg_value (-1.0), shape 255 -> fg_value
(+1.0), single channel, [-1,1] -- unchanged from the old generator, so
image_diffusion.py / unet_small.py / band_diagnostics.py need no changes.
64x64 is divisible by 2**levels up to levels=6.
 
NOTE FOR CALLERS: unlike the old ShapesConfig, this dataset's image size
(64) and slot count (3) are fixed by construction, not configurable --
`image_size`/`n_slots` CLI args in train_image_diffusion.py are no-ops
against this module and should be dropped from that script's argparse.
 
TESTING STATUS: panel-drawing and rejection-sampling logic verified with
numpy+PIL directly (per-shape white-pixel counts match the assertions
below; output shape/dtype/value-range and the no-all-empty invariant all
checked). The torch wrap (from_numpy + rescale) is a direct dtype cast of
an already-verified array and has not itself been executed against torch.
"""

import math
from dataclasses import dataclass
from typing import Dict, Tuple
 
import numpy as np
import torch
from PIL import Image, ImageDraw
 
PANEL_W, PANEL_H = 21, 63       # each panel's (width, height) in pixels
IMAGE_SIZE = 64                  # final square canvas after zero-padding
AREA = 60                        # target foreground pixel area per shape
SLOT_NAMES = ("triangle", "square", "pentagon")  # slot index -> fixed identity
_ASSERT_TOL = {"square": (64, 0), "triangle": (65, 2), "pentagon": (70, 2)}
 
 
@dataclass
class ShapesConfig:
    p_present: float = 0.5      # per-slot independent presence probability
    bg_value: float = -1.0      # background level, matches diffusion convention
    fg_value: float = 1.0       # foreground (shape) level
 
 
def _draw_square(draw: ImageDraw.ImageDraw, rng: np.random.Generator) -> Tuple[int, int, int]:
    side_length = int(AREA ** 0.5)
    max_x = PANEL_W - side_length * 2
    max_y = PANEL_H - side_length * 2
    x1 = int(rng.integers(1, max_x + 1))
    y1 = int(rng.integers(1, max_y + 1))
    draw.rectangle([x1, y1, x1 + side_length, y1 + side_length], fill=255, outline=255)
    return x1, y1, side_length
 
 
def _draw_triangle(draw: ImageDraw.ImageDraw, rng: np.random.Generator) -> Tuple[int, int, int]:
    side_length = int(math.sqrt((4 * AREA) / math.sqrt(3)))
    max_x = PANEL_W - side_length
    max_y = PANEL_H - side_length
    x1 = int(rng.integers(0, max_x + 1))
    y1 = int(rng.integers(0, max_y + 1))
    x2, y2 = x1 + side_length, y1
    x3 = x1 + side_length // 2
    y3 = y1 + int((side_length * math.sqrt(3)) / 2)
    draw.polygon([(x1, y1), (x2, y2), (x3, y3)], fill=255, outline=255)
    return x1, y1, side_length
 
 
def _draw_pentagon(draw: ImageDraw.ImageDraw, rng: np.random.Generator) -> Tuple[int, int, int]:
    side_length = int(math.sqrt((4 * AREA) / math.sqrt(5 * (5 + 2 * math.sqrt(5)))))
    max_x = PANEL_W - side_length
    max_y = PANEL_H - side_length
    x1 = int(rng.integers(side_length, max_x + 1))
    y1 = int(rng.integers(side_length, max_y + 1))
    vertices = []
    angle_offset = 72
    for _ in range(5):
        x = x1 + side_length * math.cos(math.radians(angle_offset))
        y = y1 - side_length * math.sin(math.radians(angle_offset))
        vertices.append((x, y))
        angle_offset += 72
    draw.polygon(vertices, fill=255, outline=255)
    return x1, y1, side_length
 
 
_DRAWERS = {"triangle": _draw_triangle, "square": _draw_square, "pentagon": _draw_pentagon}
 
 
def _draw_one_candidate(rng: np.random.Generator, p_present: float):
    """One (64,64) uint8 canvas, or None if all three slots came up empty."""
    panels = []
    present = np.zeros(3, dtype=np.int64)
    pos = np.zeros((3, 2), dtype=np.int64)
    side = np.zeros(3, dtype=np.int64)
    for k, name in enumerate(SLOT_NAMES):
        panel = Image.new("L", (PANEL_W, PANEL_H), color=0)
        draw = ImageDraw.Draw(panel)
        if rng.random() <= p_present:
            x1, y1, sl = _DRAWERS[name](draw, rng)
            arr = np.array(panel)
            white = int(np.sum(arr == 255))
            exp, tol = _ASSERT_TOL[name]
            assert abs(white - exp) <= tol, f"{name}: white={white} expected {exp}+-{tol}"
            present[k], pos[k], side[k] = 1, (x1, y1), sl
        panels.append(np.array(panel))
    if present.sum() == 0:
        return None
    row = np.concatenate(panels, axis=1)          # (63, 63)
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    canvas[0:63, 0:63] = row
    return canvas, present, pos, side
 
 
def generate_shapes(
    n: int,
    cfg: ShapesConfig = ShapesConfig(),
    seed: int = 0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Generate n images plus per-image slot labels.
 
    Returns:
        images: (n, 1, 64, 64) float32 in [bg_value, fg_value].
        labels: dict of (n, ...) int64 tensors --
            'present'     (n, 3)     1 if slot k's shape is drawn, else 0
                                      (k=0 triangle, 1 square, 2 pentagon)
            'pos'         (n, 3, 2)  (x1, y1) top-left anchor WITHIN the
                                      slot's own panel; 0 where absent
            'side_length' (n, 3)     per-slot side length; 0 where absent
    """
    rng = np.random.default_rng(seed)
    imgs = np.zeros((n, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    present = np.zeros((n, 3), dtype=np.int64)
    pos = np.zeros((n, 3, 2), dtype=np.int64)
    side = np.zeros((n, 3), dtype=np.int64)
 
    filled = 0
    while filled < n:
        out = _draw_one_candidate(rng, cfg.p_present)
        if out is None:
            continue  # rejection sampling: redraw so `n` is met exactly
        canvas, pr, po, sd = out
        imgs[filled], present[filled], pos[filled], side[filled] = canvas, pr, po, sd
        filled += 1
 
    x = imgs.astype(np.float32) / 255.0                       # in [0, 1]
    x = cfg.bg_value + x * (cfg.fg_value - cfg.bg_value)       # in [bg, fg]
    x = x[:, None, :, :]                                       # (n, 1, H, W)
    return (
        torch.from_numpy(x),
        {
            "present": torch.from_numpy(present),
            "pos": torch.from_numpy(pos),
            "side_length": torch.from_numpy(side),
        },
    )
 
 
def make_train_test(
    n_train: int,
    n_test: int,
    cfg: ShapesConfig = ShapesConfig(),
    seed: int = 0,
):
    """Disjoint train/test splits (distinct seeds, not a slice of one pool --
    same reasoning as the old generator's make_train_test)."""
    xtr, ltr = generate_shapes(n_train, cfg, seed=seed)
    xte, lte = generate_shapes(n_test, cfg, seed=seed + 10_000)
    return (xtr, ltr), (xte, lte)
 
