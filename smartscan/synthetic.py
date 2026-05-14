"""Synthetic mock-ID generator for SmartScan.

Generates a "license" card with named fields: a photo placeholder, name,
date of birth, ID number, address, and a signature line — on a tilted/rotated
background to demonstrate the perspective-rectification step.
"""

from __future__ import annotations

import cv2
import numpy as np


def _put(canvas: np.ndarray, txt: str, pos: tuple[int, int],
         scale: float = 0.7, color=(20, 20, 20), thick: int = 2) -> None:
    cv2.putText(canvas, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _make_card(width: int = 540, height: int = 340) -> np.ndarray:
    card = np.full((height, width, 3), 245, dtype=np.uint8)
    # Tinted header band
    cv2.rectangle(card, (0, 0), (width, 50), (180, 200, 230), -1)
    _put(card, "STATE ID CARD", (16, 34), 0.9, (40, 40, 90), 2)

    # Photo placeholder
    cv2.rectangle(card, (20, 70), (140, 230), (60, 60, 60), -1)
    cv2.putText(card, "PHOTO", (45, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # Text fields — each line is its own field for the projection-profile detector
    _put(card, "Name:    Asif Mohdjariullah", (160, 95), 0.7)
    _put(card, "DOB:     2003-08-21",           (160, 130), 0.7)
    _put(card, "ID No:   K6 472 195 A",          (160, 165), 0.7)
    _put(card, "Address: 12 Park Lane, City",   (160, 200), 0.7)

    # Signature line
    cv2.line(card, (160, 270), (500, 270), (40, 40, 40), 2)
    _put(card, "Signature", (160, 295), 0.6, (90, 90, 90), 1)

    # Outer border
    cv2.rectangle(card, (0, 0), (width - 1, height - 1), (120, 120, 120), 2)
    return card


def make_tilted_id(width: int = 800, height: int = 600,
                   angle: float = 18.0, seed: int | None = 1) -> np.ndarray:
    """Place a generated card onto a textured background at an angle."""
    rng = np.random.default_rng(seed)
    bg = np.full((height, width, 3), 90, dtype=np.uint8)
    # Slight texture
    bg = bg + rng.normal(0, 8, bg.shape).astype(np.int32)
    bg = np.clip(bg, 0, 255).astype(np.uint8)

    card = _make_card()
    ch, cw = card.shape[:2]
    # Centred on the background, rotated `angle` degrees
    canvas = bg.copy()
    M = cv2.getRotationMatrix2D((cw / 2, ch / 2), angle, 1.0)
    rotated = cv2.warpAffine(card, M, (cw + 60, ch + 60),
                             borderValue=(90, 90, 90))
    rh, rw = rotated.shape[:2]
    x = (width - rw) // 2; y = (height - rh) // 2
    canvas[y:y + rh, x:x + rw] = np.where(rotated.any(axis=-1, keepdims=True),
                                           rotated, canvas[y:y + rh, x:x + rw])
    return canvas


def make_straight_id() -> np.ndarray:
    """Already-rectified card on a uniform background — for the field detector test."""
    return make_tilted_id(angle=0.0)


if __name__ == "__main__":
    cv2.imwrite("samples/id_tilted.png", make_tilted_id())
    cv2.imwrite("samples/id_straight.png", make_straight_id())
    print("Wrote samples/id_tilted.png, id_straight.png")
