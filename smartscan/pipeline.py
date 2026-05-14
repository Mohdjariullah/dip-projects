"""SmartScan ID — projection profiles + region growing + marker watershed.

Module 4 syllabus coverage:
    - Hough line detection (perspective rectification, table-line detection)
    - Otsu thresholding (pre-stage)
    - Projection-profile field detection (horizontal / vertical sums)
    - Marker-controlled watershed (segments touching fields)
    - Region growing from a seed (BFS, 4/8-connectivity, intensity tolerance)
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


# ----------------------------- Perspective rectification --------------------

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Return TL, TR, BR, BL ordering for a 4-point quad."""
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],   # TL
        pts[np.argmin(diff)], # TR
        pts[np.argmax(s)],   # BR
        pts[np.argmax(diff)], # BL
    ], dtype=np.float32)


def rectify_document(bgr: np.ndarray) -> np.ndarray:
    """Find the document edges (Canny + largest 4-point contour) and warp it
    flat. Falls back to the original image if no quad is found.
    """
    g = to_gray(bgr)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    edges = cv2.Canny(g, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bgr
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    img_area = bgr.shape[0] * bgr.shape[1]
    for c in contours:
        peri = cv2.arcLength(c, True)
        # Try progressively looser polygon approximations until we get 4 vertices
        for eps in (0.02, 0.03, 0.05, 0.08):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4:
                break
        if len(approx) == 4 and cv2.contourArea(approx) > 0.10 * img_area:
            pts = _order_points(approx)
            tl, tr, br, bl = pts
            wA = np.linalg.norm(br - bl); wB = np.linalg.norm(tr - tl)
            hA = np.linalg.norm(tr - br); hB = np.linalg.norm(tl - bl)
            W, H = int(max(wA, wB)), int(max(hA, hB))
            dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]],
                           dtype=np.float32)
            M = cv2.getPerspectiveTransform(pts, dst)
            return cv2.warpPerspective(bgr, M, (W, H))
    return bgr


# ----------------------------- Projection profiles --------------------------

def projection_profiles(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal and vertical sums of ink (low intensity = ink).

    We invert and threshold first so 'ink' becomes 1 and background 0;
    summing rows gives a 1-D profile that peaks at text-line bands.
    """
    _, bw = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h_prof = bw.sum(axis=1)  # one entry per row
    v_prof = bw.sum(axis=0)  # one entry per column
    return h_prof, v_prof


def detect_text_lines(gray: np.ndarray, min_height: int = 8) -> list[tuple[int, int]]:
    """Return (y0, y1) bands using the horizontal projection profile."""
    h_prof, _ = projection_profiles(gray)
    smooth = cv2.GaussianBlur(h_prof.astype(np.float32), (1, 11), 0).ravel()
    threshold = max(2.0, smooth.mean() + 0.5 * smooth.std())
    in_band = smooth > threshold
    bands = []
    start = None
    for i, on in enumerate(in_band):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= min_height:
                bands.append((start, i))
            start = None
    if start is not None and len(in_band) - start >= min_height:
        bands.append((start, len(in_band)))
    return bands


# ----------------------------- Line detection (Hough) -----------------------

def detect_lines(gray: np.ndarray, min_length: int = 60) -> list[tuple[int, int, int, int]]:
    """Probabilistic Hough — straight separators (table lines / underlines)."""
    edges = cv2.Canny(gray, 80, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=min_length, maxLineGap=10)
    if lines is None:
        return []
    return [tuple(int(v) for v in l[0]) for l in lines]


# ----------------------------- Region growing (BFS) -------------------------

def region_grow(gray: np.ndarray, seed: tuple[int, int],
                tolerance: int = 15, connectivity: int = 4) -> np.ndarray:
    """Flood-fill from `seed` while pixels remain within `tolerance` of the
    seed AND within tolerance of the running region mean.

    Returns a binary mask of grown pixels.
    """
    h, w = gray.shape[:2]
    sx, sy = seed
    if not (0 <= sx < w and 0 <= sy < h):
        return np.zeros((h, w), dtype=np.uint8)
    seed_val = int(gray[sy, sx])
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[sy, sx] = 255

    if connectivity == 8:
        nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    queue = deque([(sx, sy)])
    sum_ = float(seed_val); count = 1
    while queue:
        x, y = queue.popleft()
        mean = sum_ / count
        for dx, dy in nbrs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not mask[ny, nx]:
                v = int(gray[ny, nx])
                if abs(v - seed_val) <= tolerance and abs(v - mean) <= tolerance:
                    mask[ny, nx] = 255
                    queue.append((nx, ny))
                    sum_ += v; count += 1
    return mask


# ----------------------------- Watershed (marker-controlled) ---------------

def watershed_segment(bgr: np.ndarray, dist_threshold: float = 0.45) -> np.ndarray:
    """Marker-controlled watershed.

    Workflow:
      1. Threshold → foreground mask
      2. Distance transform on foreground
      3. Local maxima (peaks above dist_threshold * max_dist) = sure-foreground markers
      4. Dilation of foreground = sure-background
      5. Unknown = sure_bg - sure_fg → labelled by cv2.watershed
    """
    g = to_gray(bgr)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opened, kernel, iterations=3)
    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, dist_threshold * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)

    n, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    color = bgr if bgr.ndim == 3 else cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers)
    return markers


def watershed_overlay(bgr: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """Colour the watershed regions and draw boundaries in red."""
    h, w = markers.shape
    rng = np.random.default_rng(0)
    out = bgr.copy() if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    n = int(markers.max()) + 1
    colors = (rng.integers(60, 220, size=(n + 1, 3))).astype(np.uint8)
    for label in range(2, n + 1):
        out[markers == label] = (0.5 * out[markers == label]
                                 + 0.5 * colors[label]).astype(np.uint8)
    out[markers == -1] = (0, 0, 255)  # boundary
    return out


# ----------------------------- Field detection from profiles ----------------

@dataclass
class Field:
    id: int
    bbox: tuple[int, int, int, int]  # x, y, w, h
    height: int


def extract_fields(bgr: np.ndarray) -> tuple[list[Field], np.ndarray]:
    """Extract each text-band as a Field. Also returns an annotated image."""
    rectified = rectify_document(bgr)
    g = to_gray(rectified)
    bands = detect_text_lines(g, min_height=10)

    fields = []
    out = rectified.copy() if rectified.ndim == 3 else cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for i, (y0, y1) in enumerate(bands, start=1):
        x, w = 0, out.shape[1]
        # Trim horizontally: find left/right extent of ink in this band
        strip = g[y0:y1]
        _, sb = cv2.threshold(strip, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        vp = sb.sum(axis=0)
        nz = np.where(vp > 0)[0]
        if len(nz) > 0:
            x = int(max(0, nz[0] - 4))
            w = int(min(out.shape[1], nz[-1] + 4) - x)
        fields.append(Field(id=i, bbox=(x, y0, w, y1 - y0), height=y1 - y0))
        cv2.rectangle(out, (x, y0), (x + w, y1), (0, 200, 0), 2)
        cv2.putText(out, f"#{i}", (x + 4, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    return fields, out
