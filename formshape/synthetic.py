"""Synthetic industrial-part silhouette generator for FormShape.

Produces simple silhouettes of common manufactured parts (hex screw, gear,
bottle outline) at controllable rotation / scale, plus optional "defects"
(notches, chips) for the test-vs-reference comparison.
"""

from __future__ import annotations

import cv2
import numpy as np


def _blank(w: int = 400, h: int = 400) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def hex_screw(width: int = 400, height: int = 400, radius: int = 120,
              rotation: float = 0.0, defect: bool = False) -> np.ndarray:
    """Hex screw-head silhouette. Optional small chip on one edge."""
    canvas = _blank(width, height)
    cy, cx = height // 2, width // 2
    pts = []
    for k in range(6):
        a = np.deg2rad(60 * k + rotation)
        pts.append([cx + radius * np.cos(a), cy + radius * np.sin(a)])
    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(canvas, [pts], 255)
    if defect:
        # Carve a small triangular chip on one edge
        a = np.deg2rad(30 + rotation)
        chip_center = (int(cx + (radius - 10) * np.cos(a)),
                       int(cy + (radius - 10) * np.sin(a)))
        cv2.circle(canvas, chip_center, 18, 0, -1)
    return canvas


def gear(width: int = 400, height: int = 400, outer: int = 130, inner: int = 90,
         teeth: int = 12, rotation: float = 0.0, defect: bool = False) -> np.ndarray:
    """Gear silhouette with N teeth. Optional missing tooth."""
    canvas = _blank(width, height)
    cy, cx = height // 2, width // 2
    pts = []
    n = teeth * 4
    for k in range(n):
        # Alternate outer / inner / outer / inner around each tooth
        phase = k % 4
        r = outer if phase < 2 else inner
        a = np.deg2rad((360 * k / n) + rotation)
        if defect and (k // 4) == 3:
            r = inner  # missing tooth
        pts.append([cx + r * np.cos(a), cy + r * np.sin(a)])
    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(canvas, [pts], 255)
    return canvas


def bottle(width: int = 320, height: int = 480, rotation: float = 0.0,
           defect: bool = False) -> np.ndarray:
    """Simple bottle outline (neck + body). Defect = dent in the side."""
    canvas = _blank(width, height)
    body_pts = np.array([
        [width // 2 - 30, 40],
        [width // 2 + 30, 40],
        [width // 2 + 30, 110],
        [width // 2 + 90, 150],
        [width // 2 + 90, height - 40],
        [width // 2 - 90, height - 40],
        [width // 2 - 90, 150],
        [width // 2 - 30, 110],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [body_pts], 255)
    if defect:
        cv2.circle(canvas, (width // 2 + 90, height // 2), 28, 0, -1)
    if abs(rotation) > 1e-3:
        M = cv2.getRotationMatrix2D((width / 2, height / 2), rotation, 1.0)
        canvas = cv2.warpAffine(canvas, M, (width, height))
    return canvas


PARTS = {
    "Hex screw": hex_screw,
    "Gear":      gear,
    "Bottle":    bottle,
}


if __name__ == "__main__":
    cv2.imwrite("samples/screw_ref.png",    hex_screw())
    cv2.imwrite("samples/screw_defect.png", hex_screw(defect=True))
    cv2.imwrite("samples/gear_ref.png",     gear())
    cv2.imwrite("samples/gear_defect.png",  gear(defect=True))
    cv2.imwrite("samples/bottle_ref.png",   bottle())
    cv2.imwrite("samples/bottle_defect.png",bottle(defect=True))
    print("Wrote 6 part silhouettes to samples/")
