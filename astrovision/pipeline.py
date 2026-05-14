"""AstroVision — star, galaxy & satellite-trail segmentation for astrophoto.

Module 4 syllabus coverage:
    - Point detection: Laplacian-of-Gaussian (LoG)
    - Local maxima → star coordinates
    - Sub-pixel centroiding via 2-D Gaussian / parabolic fit
    - Line detection: probabilistic Hough on a bright-pixel mask
    - Adaptive thresholding: sigma-clipped sky background
    - Region growing: BFS for galaxy / nebula extent

Plus a Module 2 nod — `asinh_stretch` makes the data visible at all.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


# ============================================================================
# 1. Display stretch
# ============================================================================

def asinh_stretch(img: np.ndarray, soft: float = 0.05) -> np.ndarray:
    """asinh stretch: log-like for bright pixels, linear for faint ones.

    Standard astrophotography display — reveals nebulosity without saturating stars.
    """
    g = to_gray(img).astype(np.float32) / 255.0
    out = np.arcsinh(g / soft) / np.arcsinh(1.0 / soft)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def percentile_stretch(img: np.ndarray, low: float = 1.0, high: float = 99.5) -> np.ndarray:
    """Linear stretch between two percentile clip points."""
    g = to_gray(img).astype(np.float32)
    lo, hi = np.percentile(g, [low, high])
    if hi - lo < 1e-9:
        return g.astype(np.uint8)
    return np.clip((g - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


# ============================================================================
# 2. Sigma-clipped sky background
# ============================================================================

def sigma_clipped_stats(arr: np.ndarray, sigma: float = 3.0,
                        max_iter: int = 5) -> tuple[float, float]:
    """Iteratively reject outliers >sigma·std from the median.

    Returns (sky_median, sky_std) — robust to bright stars / hot pixels.
    """
    data = arr.astype(np.float64).ravel()
    for _ in range(max_iter):
        m = np.median(data)
        s = np.std(data)
        keep = np.abs(data - m) < sigma * s
        if keep.sum() == len(data) or keep.sum() < 100:
            break
        data = data[keep]
    return float(np.median(data)), float(np.std(data))


# ============================================================================
# 3. Point detection (LoG / multi-scale)
# ============================================================================

@dataclass
class Star:
    id: int
    x: float; y: float
    peak: float
    flux: float         # aperture-minus-sky integrated flux
    fwhm: float
    ellipticity: float


def log_response(gray: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Laplacian-of-Gaussian response.

    LoG = Laplacian( Gaussian(I, sigma) ). Negative response near star centres
    (because LoG of a bright blob is negative on top, positive on the rim) — we
    flip the sign so peaks indicate point sources.
    """
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    lap = cv2.Laplacian(g, ddepth=cv2.CV_32F, ksize=3)
    return -lap  # so peaks correspond to bright point sources


def find_peaks(response: np.ndarray, threshold: float,
               min_distance: int = 3) -> np.ndarray:
    """Local maxima above `threshold`, non-max suppressed at radius `min_distance`."""
    kernel_size = 2 * min_distance + 1
    max_local = cv2.dilate(response, np.ones((kernel_size, kernel_size), np.float32))
    peaks = (response == max_local) & (response > threshold)
    ys, xs = np.where(peaks)
    return np.stack([xs, ys], axis=-1)


def _subpixel_centroid(window: np.ndarray) -> tuple[float, float]:
    """2-D parabolic centroid refinement on a small intensity window.

    Equivalent to a Gaussian fit in log-space for symmetric peaks.
    """
    if window.size == 0:
        return 0.0, 0.0
    w = window.astype(np.float64)
    total = w.sum()
    if total <= 0:
        return window.shape[1] / 2.0, window.shape[0] / 2.0
    yy, xx = np.mgrid[0:window.shape[0], 0:window.shape[1]]
    cx = (xx * w).sum() / total
    cy = (yy * w).sum() / total
    return float(cx), float(cy)


def detect_stars(gray: np.ndarray, sigmas: list[float] = (1.2, 2.0, 3.5),
                 sigma_clip: float = 3.0, k_thresh: float = 6.0,
                 min_distance: int = 4,
                 aperture: int = 6) -> list[Star]:
    """Multi-scale point detection + sub-pixel centroid + aperture photometry."""
    g = gray.astype(np.float32)
    sky_med, sky_std = sigma_clipped_stats(g, sigma=sigma_clip)
    threshold = k_thresh * sky_std

    # Collect peaks across scales, then deduplicate by minimum-distance
    all_peaks = []
    for s in sigmas:
        r = log_response(g, s)
        # Multiply by sigma^2 for scale normalisation (so different sigmas are comparable)
        pks = find_peaks(r * (s ** 2), threshold=threshold, min_distance=min_distance)
        for x, y in pks:
            all_peaks.append((x, y, s))

    # Deduplicate: keep one per min_distance neighbourhood (priority = response)
    keep = []
    used = np.zeros_like(g, dtype=bool)
    for x, y, s in sorted(all_peaks, key=lambda p: -g[p[1], p[0]]):
        y0, y1 = max(0, y - min_distance), min(g.shape[0], y + min_distance + 1)
        x0, x1 = max(0, x - min_distance), min(g.shape[1], x + min_distance + 1)
        if not used[y0:y1, x0:x1].any():
            keep.append((x, y, s))
            used[y0:y1, x0:x1] = True

    # Refine + photometry
    out = []
    annulus = aperture + 4
    for i, (x, y, s) in enumerate(keep, start=1):
        # Sub-pixel centroid in a small window
        ws = max(3, int(2 * s + 1))
        y0, y1 = max(0, y - ws), min(g.shape[0], y + ws + 1)
        x0, x1 = max(0, x - ws), min(g.shape[1], x + ws + 1)
        win = g[y0:y1, x0:x1] - sky_med
        win = np.maximum(win, 0)
        cx, cy = _subpixel_centroid(win)
        fx, fy = x0 + cx, y0 + cy

        # Aperture flux
        ay0, ay1 = max(0, int(fy) - aperture), min(g.shape[0], int(fy) + aperture + 1)
        ax0, ax1 = max(0, int(fx) - aperture), min(g.shape[1], int(fx) + aperture + 1)
        flux = float((g[ay0:ay1, ax0:ax1] - sky_med).clip(min=0).sum())

        # FWHM ≈ 2.355 · σ (assume detection scale)
        fwhm = float(2.355 * s)

        # Ellipticity from second-moment matrix of the patch
        patch = win + 1e-3
        total = patch.sum()
        my = (np.mgrid[0:patch.shape[0], 0:patch.shape[1]][0] - cy)
        mx = (np.mgrid[0:patch.shape[0], 0:patch.shape[1]][1] - cx)
        mu20 = (mx ** 2 * patch).sum() / total
        mu02 = (my ** 2 * patch).sum() / total
        mu11 = (mx * my * patch).sum() / total
        a = mu20 + mu02
        b = np.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2)
        lam1 = (a + b) / 2.0; lam2 = (a - b) / 2.0
        ecc = float(np.sqrt(max(0.0, 1.0 - lam2 / max(lam1, 1e-9))))

        out.append(Star(id=i, x=fx, y=fy, peak=float(g[int(fy), int(fx)]),
                        flux=flux, fwhm=fwhm, ellipticity=ecc))
    return out


def classify_extended(star: Star, stellar_fwhm: float = 4.5,
                      ecc_cutoff: float = 0.55) -> str:
    """Star / galaxy classifier from FWHM and ellipticity (pure DIP, no ML)."""
    if star.fwhm > 1.6 * stellar_fwhm or star.ellipticity > ecc_cutoff:
        return "galaxy/nebula"
    return "star"


# ============================================================================
# 4. Region growing for extended objects
# ============================================================================

def region_grow_extended(gray: np.ndarray, seed: tuple[int, int],
                         k_sigma: float = 1.5,
                         max_pixels: int = 200_000) -> np.ndarray:
    """BFS region growing using sigma-clipped sky stats.

    Pixel accepted if `I > sky_median + k_sigma * sky_std`.
    """
    h, w = gray.shape
    sx, sy = seed
    if not (0 <= sx < w and 0 <= sy < h):
        return np.zeros((h, w), dtype=np.uint8)
    sky_med, sky_std = sigma_clipped_stats(gray)
    cutoff = sky_med + k_sigma * sky_std

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[sy, sx] = 255
    queue = deque([(sx, sy)])
    grown = 1
    while queue and grown < max_pixels:
        x, y = queue.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not mask[ny, nx]:
                if gray[ny, nx] > cutoff:
                    mask[ny, nx] = 255
                    queue.append((nx, ny))
                    grown += 1
    return mask


# ============================================================================
# 5. Satellite-trail detection (Hough) + inpainting
# ============================================================================

def detect_trails(gray: np.ndarray, k_sigma: float = 4.0,
                  min_length: int = 80) -> list[tuple[int, int, int, int]]:
    """Threshold bright pixels at sky + k_sigma·std; probabilistic Hough."""
    sky_med, sky_std = sigma_clipped_stats(gray)
    bright = (gray > sky_med + k_sigma * sky_std).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(bright, 1, np.pi / 180, threshold=60,
                            minLineLength=min_length, maxLineGap=10)
    if lines is None:
        return []
    return [tuple(int(v) for v in l[0]) for l in lines]


def inpaint_trails(gray: np.ndarray, trails: list[tuple[int, int, int, int]],
                   thickness: int = 5) -> np.ndarray:
    """Cover the detected trails with a dilated mask and inpaint (Telea)."""
    mask = np.zeros_like(gray)
    for x1, y1, x2, y2 in trails:
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
    if mask.sum() == 0:
        return gray.copy()
    if gray.ndim == 2:
        src = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        out = cv2.inpaint(src, mask, 3, cv2.INPAINT_TELEA)
        return cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)


# ============================================================================
# 6. Annotation
# ============================================================================

def annotate_stars(img: np.ndarray, stars: list[Star]) -> np.ndarray:
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    for s in stars:
        kind = classify_extended(s)
        color = (0, 200, 255) if kind == "star" else (255, 80, 180)
        r = max(4, int(s.fwhm))
        cv2.circle(out, (int(round(s.x)), int(round(s.y))), r, color, 1)
    return out
