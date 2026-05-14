"""Synthetic satellite / aerial scene generator for OrbitRestore.

Produces a textured aerial-photo-ish image: agricultural plots in a grid +
straight roads + a darker river curve. Sharp edges make the deblur quality
visually obvious.
"""

from __future__ import annotations

import cv2
import numpy as np


def make_aerial(width: int = 512, height: int = 384, seed: int | None = 1) -> np.ndarray:
    """Synthetic aerial RGB. Returns BGR uint8."""
    rng = np.random.default_rng(seed)

    img = np.full((height, width, 3), 0, dtype=np.uint8)
    # Base = brownish soil tone
    img[:] = (50, 80, 120)

    # Agricultural plots — grid of differently-coloured rectangles
    palette = [(40, 150, 60), (90, 130, 60), (40, 110, 110), (70, 90, 130),
               (30, 90, 30), (160, 200, 90), (50, 60, 110)]
    rows = 6; cols = 8
    cell_h, cell_w = height // rows, width // cols
    for r in range(rows):
        for c in range(cols):
            color = palette[(r * cols + c) % len(palette)]
            jitter = rng.integers(-5, 5)
            cv2.rectangle(img,
                          (c * cell_w + 2 + jitter, r * cell_h + 2),
                          ((c + 1) * cell_w - 2, (r + 1) * cell_h - 2),
                          color, -1)

    # Roads
    cv2.line(img, (0, height // 3), (width, height // 3 + 40), (200, 200, 200), 4)
    cv2.line(img, (width // 4, 0), (width // 4 + 80, height), (200, 200, 200), 3)

    # River — winding dark blue curve
    pts = []
    for x in range(0, width, 8):
        y = int(height * 0.7 + 25 * np.sin(x / 35.0))
        pts.append([x, y])
    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(img, [pts], False, (150, 70, 30), 6)

    # Soft texture
    img = np.clip(img.astype(np.float32) + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    return img


def make_hazy_aerial(width: int = 512, height: int = 384,
                     haze: float = 0.55, seed: int | None = 2) -> np.ndarray:
    """Aerial scene with atmospheric haze (simulated)."""
    img = make_aerial(width, height, seed=seed).astype(np.float32) / 255.0
    A = np.array([0.93, 0.92, 0.95], dtype=np.float32)  # atmospheric light
    # Transmission falls off with image y (further away = more haze)
    yy = np.linspace(0.4, 0.95, height, dtype=np.float32)
    t = 1.0 - haze * yy[:, None]
    t = np.clip(t, 0.15, 1.0)
    out = img * t[..., None] + A * (1.0 - t[..., None])
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    cv2.imwrite("samples/aerial.png", make_aerial())
    cv2.imwrite("samples/aerial_hazy.png", make_hazy_aerial())
    print("Wrote samples/aerial.png, samples/aerial_hazy.png")
