"""Synthetic X-ray-like image generator for RadioLens.

A real X-ray is a transmission image: brighter where the X-ray was absorbed
more (bone), darker where it passed through (lung air). We fake that with a
soft radial gradient + a few elliptical bright "bones" + low contrast + noise.
"""

from __future__ import annotations

import cv2
import numpy as np


def _radial_bg(h: int, w: int, brightness: int = 90, falloff: float = 0.6) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2, w / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (0.5 * max(h, w))
    field = brightness * (1.0 - falloff * np.clip(r, 0, 1))
    return field


def _add_ellipse(img: np.ndarray, cx: int, cy: int, a: int, b: int,
                 angle: float, intensity: int, soft: int = 25) -> None:
    canvas = np.zeros_like(img)
    cv2.ellipse(canvas, (cx, cy), (a, b), angle, 0, 360, intensity, -1)
    canvas = cv2.GaussianBlur(canvas, (soft | 1, soft | 1), 0)
    np.maximum(img, canvas, out=img)


def make_chest_xray(width: int = 512, height: int = 512,
                    contrast: float = 0.55, noise: float = 6.0,
                    seed: int | None = 1) -> np.ndarray:
    """Low-contrast chest-xray-ish synthetic image.

    Designed so the user sees obvious improvement after applying gamma<1 or
    contrast stretching. Returns a single-channel uint8.
    """
    rng = np.random.default_rng(seed)

    img = _radial_bg(height, width, brightness=80).astype(np.float32)

    # Spinal column (vertical bright stripe)
    cv2.rectangle(img, (width // 2 - 12, 20), (width // 2 + 12, height - 20),
                  200, -1)

    # Rib pairs (curved ellipses)
    for i, frac in enumerate(np.linspace(0.18, 0.78, 7)):
        y = int(height * frac)
        _add_ellipse(img, width // 2 - 90, y, 80, 12, -15, 180, soft=15)
        _add_ellipse(img, width // 2 + 90, y, 80, 12, 15, 180, soft=15)

    # Heart shadow (darker region)
    _add_ellipse(img, width // 2 - 20, int(height * 0.55), 110, 80, 0, 30, soft=35)

    # Soft tissue noise
    img = img + rng.normal(0, noise, img.shape)

    # Compress dynamic range to simulate low-contrast capture
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    img = (contrast * img + (1 - contrast) * 0.4) * 255

    return np.clip(img, 0, 255).astype(np.uint8)


def make_hand_xray(width: int = 512, height: int = 512, seed: int | None = 2) -> np.ndarray:
    """Hand-with-fracture-style synthetic (5 finger bones + palm)."""
    rng = np.random.default_rng(seed)
    img = _radial_bg(height, width, brightness=60).astype(np.float32)

    # Palm region
    _add_ellipse(img, width // 2, int(height * 0.7), 110, 80, 0, 170, soft=25)

    # Five fingers
    for i, x in enumerate(np.linspace(width * 0.25, width * 0.75, 5)):
        x = int(x)
        # Three phalanges per finger
        for j, y_frac in enumerate([0.20, 0.34, 0.48]):
            y = int(height * y_frac)
            _add_ellipse(img, x, y, 14, 28, 0, 200, soft=11)

    # Simulated fracture: small dark line on middle finger
    cv2.line(img, (width // 2, int(height * 0.30)),
             (width // 2 + 3, int(height * 0.35)), 50, 2)

    img = img + rng.normal(0, 5, img.shape)
    img = (img - img.min()) / (img.max() - img.min() + 1e-9) * 200 + 20
    return np.clip(img, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    cv2.imwrite("samples/chest.png", make_chest_xray())
    cv2.imwrite("samples/hand.png", make_hand_xray())
    print("Wrote samples/chest.png, samples/hand.png")
