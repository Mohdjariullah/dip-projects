"""RadioLens — pure-NumPy intensity transforms for X-ray contrast enhancement.

Every transform is a 1-D lookup table (LUT) built once and applied via cv2.LUT,
so a slider in the Streamlit UI re-renders the image in well under a frame.

Module 2 syllabus coverage:
    - power-law (gamma)
    - log transform
    - contrast stretching (piecewise linear with 2 control points)
    - piecewise-linear with N control points
    - bit-plane slicing
    - intensity-level slicing
    - global thresholding (used for the bone-isolation lens)
"""

from __future__ import annotations

import cv2
import numpy as np


# ----------------------------- LUT helpers ----------------------------------

def _apply_lut(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a 256-entry LUT to a uint8 image. Handles 3-channel by broadcasting."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut.astype(np.uint8))


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# ----------------------------- Transforms -----------------------------------

def gamma_lut(gamma: float) -> np.ndarray:
    """Power-law: s = 255 * (r/255)^gamma.

    gamma < 1 brightens (reveals dark detail — lung texture);
    gamma > 1 darkens (reveals bright detail — bone interior).
    """
    r = np.arange(256, dtype=np.float64) / 255.0
    return np.clip(255.0 * np.power(r, gamma), 0, 255)


def log_lut(c: float | None = None) -> np.ndarray:
    """Log transform: s = c * log(1 + r). Auto-scales c so output spans [0,255]."""
    r = np.arange(256, dtype=np.float64)
    if c is None:
        c = 255.0 / np.log1p(255.0)
    return np.clip(c * np.log1p(r), 0, 255)


def contrast_stretch_lut(r1: int, s1: int, r2: int, s2: int) -> np.ndarray:
    """Classic two-knee piecewise-linear contrast stretch.

    (r1,s1) and (r2,s2) are the two control points; the rest is linearly
    extrapolated. Setting (r1,s1)=(p_low,0), (r2,s2)=(p_high,255) gives
    full-range stretching between percentiles.
    """
    r = np.arange(256, dtype=np.float64)
    s = np.zeros_like(r)
    # Segment 1: 0..r1
    if r1 > 0:
        s[:r1] = (s1 / r1) * r[:r1]
    # Segment 2: r1..r2
    if r2 > r1:
        s[r1:r2] = s1 + (s2 - s1) * (r[r1:r2] - r1) / (r2 - r1)
    # Segment 3: r2..255
    if r2 < 255:
        s[r2:] = s2 + (255 - s2) * (r[r2:] - r2) / (255 - r2)
    return np.clip(s, 0, 255)


def piecewise_lut(control_points: list[tuple[int, int]]) -> np.ndarray:
    """N-point piecewise-linear LUT from a list of (r, s) control points.

    Endpoints (0, .) and (255, .) are added automatically if missing.
    """
    pts = sorted(control_points)
    if pts[0][0] != 0:
        pts = [(0, pts[0][1])] + pts
    if pts[-1][0] != 255:
        pts = pts + [(255, pts[-1][1])]
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    r = np.arange(256, dtype=np.float64)
    return np.clip(np.interp(r, xs, ys), 0, 255)


def intensity_slice(img: np.ndarray, lo: int, hi: int, preserve_bg: bool = True) -> np.ndarray:
    """Highlight intensities in [lo, hi].

    preserve_bg=True keeps non-selected pixels at their original value;
    preserve_bg=False produces a binary highlight (white inside, black outside).
    """
    g = _to_gray(img)
    mask = (g >= lo) & (g <= hi)
    if preserve_bg:
        out = g.copy()
        out[mask] = 255
        return out
    return (mask.astype(np.uint8) * 255)


# ----------------------------- Bit-plane stack -------------------------------

def bit_plane(img: np.ndarray, k: int) -> np.ndarray:
    """Return the k-th bit-plane (0..7) as a uint8 image (0 or 255)."""
    g = _to_gray(img)
    return (((g >> k) & 1) * 255).astype(np.uint8)


def all_bit_planes(img: np.ndarray) -> list[np.ndarray]:
    """Return all 8 bit-planes, LSB (0) first."""
    return [bit_plane(img, k) for k in range(8)]


def reconstruct_from_planes(img: np.ndarray, selected: list[int]) -> np.ndarray:
    """Reconstruct an image using only the given bit-plane indices.

    Demonstrates that MSB planes (5..7) carry coarse anatomy while LSB
    planes (0..2) are mostly sensor noise.
    """
    g = _to_gray(img).astype(np.int32)
    out = np.zeros_like(g)
    for k in selected:
        out += ((g >> k) & 1) << k
    return np.clip(out, 0, 255).astype(np.uint8)


# ----------------------------- Bone isolation --------------------------------

def isolate_bone(img: np.ndarray, threshold: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Threshold (Otsu by default) + small closing to highlight bright bone-like regions.

    Returns (binary_mask, red_overlay_bgr).
    """
    g = _to_gray(img)
    if threshold is None:
        _, mask = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(g, threshold, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    bgr[mask > 0] = (0.4 * bgr[mask > 0] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    return mask, bgr


# ----------------------------- Convenience -----------------------------------

def apply(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a LUT to a (possibly 3-channel) image. Returns gray uint8."""
    return _apply_lut(_to_gray(img), lut)


def transformation_curve_points(lut: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For plotting: (input intensity, output intensity)."""
    return np.arange(256), lut.astype(np.int32)
