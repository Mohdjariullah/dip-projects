"""Traffic Signal Detection — pure OpenCV pipeline (no Streamlit).

Pipeline:
    BGR image
      -> HSV color-space conversion
      -> per-color HSV inRange thresholding (Red/Yellow/Green)
      -> morphological opening + closing
      -> contour finding + circularity/area filter
      -> annotated image + list of detections

Each function does one thing so it can be explained on its own in a viva.
"""

from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


# Default HSV ranges (OpenCV uses H: 0-179, S: 0-255, V: 0-255).
# Red wraps around 0/180, so it needs TWO ranges combined.
DEFAULT_HSV_RANGES = {
    "red": [
        ((0, 120, 120), (10, 255, 255)),
        ((170, 120, 120), (179, 255, 255)),
    ],
    "yellow": [((18, 120, 120), (35, 255, 255))],
    "green": [((40, 80, 80), (90, 255, 255))],
}

DISPLAY_BGR = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 200, 0),
}


@dataclass
class Detection:
    state: str            # "red" | "yellow" | "green"
    bbox: tuple           # (x, y, w, h)
    area: float           # contour area in px
    circularity: float    # 4*pi*A / P^2 (1.0 = perfect circle)


def to_hsv(bgr: np.ndarray) -> np.ndarray:
    """Step 1 — BGR -> HSV conversion."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def make_color_masks(hsv: np.ndarray, ranges: dict = None) -> dict:
    """Step 2 — HSV inRange thresholding for each color.

    Returns a dict {color_name: binary_mask}. Red mask is the OR of two
    sub-ranges (because red hue wraps around 0/180).
    """
    if ranges is None:
        ranges = DEFAULT_HSV_RANGES

    masks = {}
    for color, sub_ranges in ranges.items():
        m = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in sub_ranges:
            m = cv2.bitwise_or(m, cv2.inRange(hsv, np.array(lo), np.array(hi)))
        masks[color] = m
    return masks


def clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Step 3 — Morphological opening (erode->dilate) then closing.

    Opening removes salt-and-pepper noise (small bright specks in the mask).
    Closing fills small holes inside the lit signal disc.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k)
    return closed


def find_blobs(
    mask: np.ndarray,
    min_area: float = 80,
    max_area: float = 50000,
    min_circularity: float = 0.55,
) -> list:
    """Step 4 — Contour finding + circularity/area filter.

    A traffic light disc is roughly circular: 4*pi*A / P^2 should be near 1.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        perim = cv2.arcLength(c, closed=True)
        if perim <= 0:
            continue
        circ = 4 * np.pi * area / (perim * perim)
        if circ < min_circularity:
            continue
        x, y, w, h = cv2.boundingRect(c)
        out.append({"bbox": (x, y, w, h), "area": float(area), "circularity": float(circ)})
    return out


def detect(
    bgr: np.ndarray,
    ranges: dict = None,
    kernel_size: int = 5,
    min_area: float = 80,
    max_area: float = 50000,
    min_circularity: float = 0.55,
) -> tuple:
    """Step 5 — Full pipeline. Returns (annotated_bgr, masks_dict, detections).

    detections is a list of Detection dataclass instances.
    """
    hsv = to_hsv(bgr)
    raw_masks = make_color_masks(hsv, ranges)

    cleaned = {color: clean_mask(m, kernel_size) for color, m in raw_masks.items()}

    detections: list[Detection] = []
    for color, mask in cleaned.items():
        for blob in find_blobs(mask, min_area, max_area, min_circularity):
            detections.append(
                Detection(
                    state=color,
                    bbox=blob["bbox"],
                    area=blob["area"],
                    circularity=blob["circularity"],
                )
            )

    annotated = annotate(bgr, detections)
    return annotated, cleaned, detections


def annotate(bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw labelled bounding boxes on a copy of the image."""
    out = bgr.copy()
    for i, d in enumerate(detections, start=1):
        x, y, w, h = d.bbox
        color = DISPLAY_BGR[d.state]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = f"{i}:{d.state.upper()}"
        # Text background for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(
            out, label, (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )
    return out


def colored_mask_overlay(masks: dict, shape: tuple) -> np.ndarray:
    """Combine R/Y/G binary masks into a single BGR visualization image."""
    out = np.zeros(shape, dtype=np.uint8)
    for color, m in masks.items():
        bgr_color = np.array(DISPLAY_BGR[color], dtype=np.uint8)
        layer = np.zeros(shape, dtype=np.uint8)
        layer[m > 0] = bgr_color
        out = cv2.add(out, layer)
    return out


def count_by_state(detections: list[Detection]) -> dict:
    counts = {"red": 0, "yellow": 0, "green": 0}
    for d in detections:
        counts[d.state] += 1
    return counts
