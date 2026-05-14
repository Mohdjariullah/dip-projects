"""DocuClean — adaptive document binarization with multiple algorithms.

Module 2 syllabus coverage:
    - global thresholding (Otsu)
    - adaptive (mean / Gaussian) thresholding
    - Niblack local thresholding
    - Sauvola local thresholding
    - intensity transform (gamma / contrast stretch) as pre-stage
    - morphology (opening, closing, despeckle) as post-stage

Niblack and Sauvola are written from scratch using cv2.boxFilter +
cv2.sqrBoxFilter for O(1)-per-pixel local mean/std via integral-image trick.
"""

from __future__ import annotations

import cv2
import numpy as np


# ----------------------------- helpers --------------------------------------

def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _odd(n: int) -> int:
    n = int(max(3, n))
    return n if n % 2 == 1 else n + 1


def _local_mean_std(gray: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """O(1)-per-pixel local mean and std via box filter on integral images."""
    g = gray.astype(np.float32)
    ksize = (window, window)
    mean = cv2.boxFilter(g, ddepth=cv2.CV_32F, ksize=ksize, normalize=True)
    mean_sq = cv2.sqrBoxFilter(g, ddepth=cv2.CV_32F, ksize=ksize, normalize=True)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(var)
    return mean, std


# ----------------------------- intensity pre-stage --------------------------

def gamma_correct(img: np.ndarray, gamma: float) -> np.ndarray:
    lut = np.clip(255.0 * (np.arange(256) / 255.0) ** gamma, 0, 255).astype(np.uint8)
    return cv2.LUT(to_gray(img), lut)


def remove_shading(gray: np.ndarray, kernel: int = 51) -> np.ndarray:
    """Remove uneven background illumination by subtracting a large-kernel opening."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(kernel), _odd(kernel)))
    bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, k)
    # Subtract background and rescale to full range
    diff = cv2.subtract(gray, bg)
    out = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return out.astype(np.uint8)


# ----------------------------- thresholding ---------------------------------

def thresh_otsu(gray: np.ndarray) -> np.ndarray:
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def thresh_adaptive_gaussian(gray: np.ndarray, window: int = 25, C: int = 10) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        _odd(window), C,
    )


def thresh_niblack(gray: np.ndarray, window: int = 25, k: float = -0.2) -> np.ndarray:
    """Niblack: T(x,y) = mean(x,y) + k * std(x,y).

    Pixel is foreground (black, 0) where gray < T. Returns standard binary
    image (255 = paper, 0 = ink) to match other thresholders.
    """
    mean, std = _local_mean_std(to_gray(gray), _odd(window))
    T = mean + k * std
    bw = (gray.astype(np.float32) > T).astype(np.uint8) * 255
    return bw


def thresh_sauvola(gray: np.ndarray, window: int = 25, k: float = 0.2, R: float = 128.0) -> np.ndarray:
    """Sauvola: T(x,y) = mean * (1 + k * (std/R - 1)).

    Designed to handle uneven illumination by dampening the threshold when
    local contrast is low (background regions get clean, not speckled).
    """
    mean, std = _local_mean_std(to_gray(gray), _odd(window))
    T = mean * (1.0 + k * (std / R - 1.0))
    bw = (gray.astype(np.float32) > T).astype(np.uint8) * 255
    return bw


def thresh_wolf(gray: np.ndarray, window: int = 25, k: float = 0.5) -> np.ndarray:
    """Wolf-Jolion: adapts Sauvola to the image-wide minimum.

    T = (1 - k) * mean + k * I_min + k * (std / max_std) * (mean - I_min)
    """
    g = to_gray(gray)
    mean, std = _local_mean_std(g, _odd(window))
    i_min = float(g.min())
    max_std = float(std.max() if std.max() > 0 else 1.0)
    T = (1 - k) * mean + k * i_min + k * (std / max_std) * (mean - i_min)
    bw = (g.astype(np.float32) > T).astype(np.uint8) * 255
    return bw


THRESHOLD_METHODS = {
    "Otsu": thresh_otsu,
    "Adaptive Gaussian": thresh_adaptive_gaussian,
    "Niblack": thresh_niblack,
    "Sauvola": thresh_sauvola,
    "Wolf-Jolion": thresh_wolf,
}


# ----------------------------- post-process ---------------------------------

def despeckle(bw: np.ndarray, min_area: int = 6) -> np.ndarray:
    """Remove connected components (BLACK ink blobs) smaller than min_area."""
    inv = cv2.bitwise_not(bw)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    keep = np.zeros_like(inv)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return cv2.bitwise_not(keep)


def morph_clean(bw: np.ndarray, kernel: int = 1) -> np.ndarray:
    if kernel <= 1:
        return bw
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (_odd(kernel), _odd(kernel)))
    return cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)


# ----------------------------- metrics --------------------------------------

def foreground_ratio(bw: np.ndarray) -> float:
    """Fraction of pixels classified as ink (dark)."""
    return float((bw == 0).sum()) / bw.size


def component_count(bw: np.ndarray) -> int:
    """Number of ink connected components — proxy for OCR-friendliness."""
    inv = cv2.bitwise_not(bw)
    n, _, _, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    return max(0, n - 1)


def diff_overlay(bw_a: np.ndarray, bw_b: np.ndarray) -> np.ndarray:
    """Visualise pixels where the two binarizations disagree.

    Red = only A says ink. Blue = only B says ink.
    """
    a_ink = bw_a == 0
    b_ink = bw_b == 0
    only_a = a_ink & ~b_ink
    only_b = b_ink & ~a_ink
    canvas = cv2.cvtColor(bw_a, cv2.COLOR_GRAY2BGR)
    canvas[only_a] = (0, 0, 220)   # red in BGR
    canvas[only_b] = (220, 0, 0)   # blue in BGR
    return canvas


# ----------------------------- end-to-end pipeline --------------------------

def run_pipeline(img: np.ndarray, *,
                 gamma: float = 1.0,
                 shade_kernel: int = 0,
                 method: str = "Sauvola",
                 window: int = 25,
                 k_or_C: float = 0.2,
                 R: float = 128.0,
                 despeckle_min: int = 0,
                 close_kernel: int = 1) -> np.ndarray:
    """Full pipeline: intensity → shading removal → threshold → post-process."""
    g = to_gray(img)
    if abs(gamma - 1.0) > 1e-3:
        g = gamma_correct(g, gamma)
    if shade_kernel >= 3:
        g = remove_shading(g, shade_kernel)

    fn = THRESHOLD_METHODS[method]
    if method == "Otsu":
        bw = fn(g)
    elif method == "Adaptive Gaussian":
        bw = fn(g, window=window, C=int(k_or_C))
    elif method == "Niblack":
        bw = fn(g, window=window, k=k_or_C)
    elif method == "Sauvola":
        bw = fn(g, window=window, k=k_or_C, R=R)
    else:
        bw = fn(g, window=window, k=k_or_C)

    if despeckle_min > 0:
        bw = despeckle(bw, despeckle_min)
    bw = morph_clean(bw, close_kernel)
    return bw
