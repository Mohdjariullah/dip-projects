"""Synthetic clean reference image for NoiseLab.

A simple geometric scene (sky gradient + sun + landscape blocks) — enough
texture and edges to make noise visible without needing a real photograph.
"""

from __future__ import annotations

import cv2
import numpy as np


def make_clean_scene(width: int = 512, height: int = 384,
                     seed: int | None = 1) -> np.ndarray:
    """A blocky synthetic landscape — grayscale uint8."""
    rng = np.random.default_rng(seed)

    # Sky gradient (light at top, slightly darker near horizon)
    sky = np.tile(np.linspace(200, 170, height // 2, dtype=np.float32)[:, None],
                  (1, width))
    # Ground (uniform mid-tone)
    ground = np.full((height - height // 2, width), 90, dtype=np.float32)
    img = np.vstack([sky, ground])

    # Sun
    cv2.circle(img, (int(width * 0.78), int(height * 0.25)), 28, 240, -1)

    # Distant mountain outline
    pts = np.array([[0, height // 2],
                    [width * 0.25, int(height * 0.32)],
                    [width * 0.50, int(height * 0.45)],
                    [width * 0.78, int(height * 0.28)],
                    [width, int(height * 0.40)],
                    [width, height // 2]], dtype=np.int32)
    cv2.fillPoly(img, [pts], 120)

    # A few "houses" in the foreground (sharp edges → useful for filter tests)
    for x in (60, 200, 360):
        cv2.rectangle(img, (x, int(height * 0.6)), (x + 80, int(height * 0.85)), 160, -1)
        cv2.rectangle(img, (x + 10, int(height * 0.7)), (x + 30, int(height * 0.85)), 60, -1)

    # Subtle texture so geometric/harmonic means have something to do
    img += rng.normal(0, 1.5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    cv2.imwrite("samples/clean.png", make_clean_scene())
    print("Wrote samples/clean.png")
