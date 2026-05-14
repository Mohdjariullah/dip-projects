"""Synthetic night-sky generator for AstroVision.

Produces a dark frame with N stars (Gaussian PSFs of varied brightness +
FWHM), one extended fuzzy "galaxy" or "nebula", and optionally a long
satellite trail crossing the frame. Adds Poisson + read noise.
"""

from __future__ import annotations

import cv2
import numpy as np


def _gaussian_blob(canvas: np.ndarray, cx: float, cy: float,
                   peak: float, sigma: float) -> None:
    """Add a Gaussian PSF in-place at (cx, cy)."""
    h, w = canvas.shape
    r = int(4 * sigma) + 1
    x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r + 1)
    y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    g = peak * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], g)


def make_star_field(width: int = 640, height: int = 480, n_stars: int = 120,
                    extended: bool = True, trail: bool = False,
                    seed: int | None = 1) -> np.ndarray:
    """Dark frame + Gaussian stars + (optional) extended object + trail.

    Returns a uint8 grayscale image. Pre-asinh-stretch — very dark!
    """
    rng = np.random.default_rng(seed)

    sky = 6.0  # background DN
    canvas = np.full((height, width), sky, dtype=np.float32)

    # Stars: brightness distribution is heavy-tailed
    for _ in range(n_stars):
        cx = rng.uniform(8, width - 8)
        cy = rng.uniform(8, height - 8)
        sigma = rng.choice([1.1, 1.3, 1.5, 1.8])
        peak = rng.choice([30, 50, 80, 120, 200, 240])
        _gaussian_blob(canvas, cx, cy, float(peak), float(sigma))

    # Extended object — broad faint Gaussian to mimic a galaxy / nebula
    if extended:
        gx, gy = width * 0.65, height * 0.55
        _gaussian_blob(canvas, gx, gy, 35.0, 28.0)
        _gaussian_blob(canvas, gx + 5, gy + 3, 25.0, 14.0)

    # Satellite trail
    if trail:
        x1, y1 = 10, int(height * 0.2)
        x2, y2 = width - 10, int(height * 0.7)
        cv2.line(canvas, (x1, y1), (x2, y2), 80, 2, cv2.LINE_AA)

    # Poisson-style + read noise
    canvas = canvas + rng.normal(0, 1.5, canvas.shape)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def make_satellite_streaked_frame(width: int = 640, height: int = 480,
                                  seed: int | None = 2) -> np.ndarray:
    return make_star_field(width, height, n_stars=140, extended=False,
                           trail=True, seed=seed)


if __name__ == "__main__":
    cv2.imwrite("samples/starfield.png", make_star_field())
    cv2.imwrite("samples/satellite_streak.png", make_satellite_streaked_frame())
    print("Wrote samples/starfield.png, samples/satellite_streak.png")
