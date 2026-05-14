"""Synthetic plant-leaf generator for AgroLeaf.

Produces a green leaf-shaped silhouette with brown/yellow disease lesions
scattered on it, on a contrasting (paper or soil) background. Designed so the
RGB->HSI segmentation pipeline has clear bands to threshold against.
"""

from __future__ import annotations

import cv2
import numpy as np


def _leaf_outline(w: int, h: int) -> np.ndarray:
    """Generate a soft, vaguely teardrop-shaped leaf silhouette mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    # Big ellipse, slightly rotated, makes the body
    cv2.ellipse(mask, (cx, cy), (int(w * 0.40), int(h * 0.30)), 20,
                0, 360, 255, -1)
    # A small "tip" extension
    pts = np.array([[cx + 80, cy - 100], [cx + 130, cy - 160], [cx + 60, cy - 50]])
    cv2.fillPoly(mask, [pts], 255)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask = (mask > 64).astype(np.uint8) * 255
    return mask


def make_diseased_leaf(width: int = 640, height: int = 480,
                       n_lesions: int = 6, seed: int | None = 1) -> np.ndarray:
    """Green leaf with brown/yellow lesions on a tan background. Returns BGR."""
    rng = np.random.default_rng(seed)

    # Tan/soil background
    bg = np.full((height, width, 3), 0, dtype=np.uint8)
    bg[:] = (110, 145, 170)  # BGR tan

    # Leaf body — slight gradient from dark to light green
    lm = _leaf_outline(width, height)
    leaf_color = np.zeros_like(bg)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    grad = 0.55 + 0.45 * (yy / height)
    leaf_color[..., 0] = (40 * grad).astype(np.uint8)            # B
    leaf_color[..., 1] = (160 * grad).astype(np.uint8)           # G
    leaf_color[..., 2] = (50 * grad).astype(np.uint8)            # R
    out = np.where(lm[..., None] > 0, leaf_color, bg)

    # Lesions: brown circles with yellow halo. Place inside the leaf mask.
    ys, xs = np.where(lm > 0)
    for _ in range(n_lesions):
        i = rng.integers(0, len(xs))
        cx, cy = int(xs[i]), int(ys[i])
        r_inner = int(rng.integers(8, 22))
        r_outer = r_inner + int(rng.integers(4, 10))
        # Yellow halo
        cv2.circle(out, (cx, cy), r_outer, (35, 200, 230), -1)
        # Brown core
        cv2.circle(out, (cx, cy), r_inner, (30, 60, 110), -1)

    # Sensor noise
    out = np.clip(out.astype(np.float32) + rng.normal(0, 4, out.shape), 0, 255).astype(np.uint8)
    return out


def make_healthy_leaf(width: int = 640, height: int = 480, seed: int | None = 2) -> np.ndarray:
    """Healthy leaf, no lesions — sanity check (severity should be ~0%)."""
    return make_diseased_leaf(width, height, n_lesions=0, seed=seed)


if __name__ == "__main__":
    cv2.imwrite("samples/diseased.png", make_diseased_leaf())
    cv2.imwrite("samples/healthy.png", make_healthy_leaf())
    print("Wrote samples/diseased.png, samples/healthy.png")
