"""Synthetic star-field pair generator.

Produces two grayscale images of the "same" star field, with:
  - one star (the "asteroid") shifted by a known offset between A and B,
  - a tiny global rotation/translation simulating telescope drift,
  - Gaussian sensor noise.

This lets the asteroid-detection demo work reliably without real telescope
data, and gives a known ground truth (the asteroid position).
"""

from __future__ import annotations

import numpy as np
import cv2


def _draw_star(canvas: np.ndarray, x: float, y: float, brightness: float, sigma: float = 1.4) -> None:
    """Add a 2D Gaussian "star" PSF centered at (x, y) onto canvas in-place."""
    h, w = canvas.shape
    # Bounding box of influence (truncate at 4*sigma)
    r = int(np.ceil(4 * sigma))
    x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
    y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    g = brightness * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma))
    canvas[y0:y1, x0:x1] = np.clip(canvas[y0:y1, x0:x1] + g, 0, 255)


def make_sky_pair(
    width: int = 640,
    height: int = 480,
    num_stars: int = 80,
    asteroid_offset: tuple = (15, 8),
    telescope_drift_px: float = 1.5,
    telescope_drift_deg: float = 0.3,
    noise: float = 0.015,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Generate two sky images of the same field with one moving asteroid.

    Returns (image_a, image_b, asteroid_position_in_a).
    image_a and image_b are uint8 grayscale 2D arrays.
    """
    rng = np.random.default_rng(seed)

    # Scatter stars randomly. Brighter stars are rarer (power-law-ish).
    star_x = rng.uniform(20, width - 20, num_stars)
    star_y = rng.uniform(20, height - 20, num_stars)
    star_b = rng.uniform(60, 250, num_stars) * rng.uniform(0.4, 1.0, num_stars)
    star_sigma = rng.uniform(1.1, 1.8, num_stars)

    # Pick one star to act as the asteroid.
    ast_idx = int(rng.integers(0, num_stars))
    ast_x_a, ast_y_a = float(star_x[ast_idx]), float(star_y[ast_idx])
    ast_b = float(star_b[ast_idx])
    ast_sigma = float(star_sigma[ast_idx])

    # ----- Image A -----
    a = np.zeros((height, width), dtype=np.float32)
    for i in range(num_stars):
        _draw_star(a, star_x[i], star_y[i], star_b[i], star_sigma[i])

    # ----- Image B (same stars + asteroid moved + telescope drift) -----
    # We render B in its "true" frame (asteroid moved, other stars fixed),
    # then apply a small affine warp to simulate telescope drift.
    b = np.zeros((height, width), dtype=np.float32)
    for i in range(num_stars):
        if i == ast_idx:
            new_x = star_x[i] + asteroid_offset[0]
            new_y = star_y[i] + asteroid_offset[1]
            _draw_star(b, new_x, new_y, ast_b, ast_sigma)
        else:
            _draw_star(b, star_x[i], star_y[i], star_b[i], star_sigma[i])

    # Apply small rigid warp to image B (drift).
    if telescope_drift_px != 0 or telescope_drift_deg != 0:
        cx, cy = width / 2, height / 2
        M = cv2.getRotationMatrix2D((cx, cy), telescope_drift_deg, 1.0)
        M[0, 2] += telescope_drift_px
        M[1, 2] += telescope_drift_px * 0.5
        b = cv2.warpAffine(b, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Add Gaussian sensor noise to both.
    if noise > 0:
        a = a + rng.normal(0, noise * 255, a.shape).astype(np.float32)
        b = b + rng.normal(0, noise * 255, b.shape).astype(np.float32)

    a = np.clip(a, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)

    return a, b, (ast_x_a, ast_y_a)


if __name__ == "__main__":
    a, b, pos = make_sky_pair()
    cv2.imwrite("samples/sky_a.png", a)
    cv2.imwrite("samples/sky_b.png", b)
    print(f"Wrote samples/sky_a.png and samples/sky_b.png. Asteroid at A: ({pos[0]:.1f}, {pos[1]:.1f})")
