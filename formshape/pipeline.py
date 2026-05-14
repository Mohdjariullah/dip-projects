"""FormShape Inspector — boundary descriptors for defect detection.

Module 5 syllabus coverage (boundary descriptors):
    - Moore-neighbour boundary tracing (from scratch)
    - Freeman chain codes (4-conn, 8-conn)
    - Chain-code rotation invariance via first difference
    - Fourier descriptors (FFT of boundary as complex sequence)
    - FD translation / scale / rotation / start-point invariance
    - Shape signature (radial distance vs. boundary index)

The Moore-tracing implementation is the marks-bearing artefact.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


# ============================================================================
# 1. Silhouette extraction
# ============================================================================

def silhouette(img: np.ndarray) -> np.ndarray:
    """Otsu binarisation → largest connected component → uint8 mask."""
    g = to_gray(img)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # The foreground is whichever class is smaller (the object).
    if (bw == 255).sum() > (bw == 0).sum():
        bw = cv2.bitwise_not(bw)
    # Keep only the largest blob
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 1:
        return bw
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((lbl == biggest).astype(np.uint8) * 255)


# ============================================================================
# 2. Moore-neighbour boundary tracing (from scratch)
# ============================================================================

# 8-neighbour offsets in clockwise order starting from "east"
_MOORE_OFFSETS = [(0, 1), (1, 1), (1, 0), (1, -1),
                  (0, -1), (-1, -1), (-1, 0), (-1, 1)]


def moore_boundary(mask: np.ndarray, max_steps: int | None = None) -> np.ndarray:
    """Return ordered boundary pixels of `mask` (uint8, 0 / 255) as Nx2 (y, x).

    Algorithm:
      1. Find the topmost-leftmost foreground pixel s. Boundary starts there.
      2. Walk the 8-neighbourhood clockwise from the direction `back` we came
         from; the first foreground neighbour is the next boundary pixel.
      3. Stop when we revisit s entering from the same direction.
    """
    h, w = mask.shape
    # Find starting pixel (top-left foreground)
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    start_y, start_x = int(ys[0]), int(xs[xs.argmin()] if False else 0)  # placeholder

    # Re-find correctly: top-most row first, then left-most column on that row
    top = int(ys.min())
    cols_top = xs[ys == top]
    start_x = int(cols_top.min())
    start_y = top
    start = (start_y, start_x)
    boundary = [start]

    # Initial "back" direction: came from west of the start pixel
    back_dir = 6  # offset index (-1, 0) — north... actually we want WEST = (0,-1) which is index 4
    back_dir = 4  # came from west

    if max_steps is None:
        max_steps = 8 * (h + w)

    current = start
    prev_back = back_dir
    for _ in range(max_steps):
        cy, cx = current
        # Start checking from the cell clockwise next to `back`
        found_next = None
        for k in range(1, 9):
            idx = (prev_back + k) % 8
            dy, dx = _MOORE_OFFSETS[idx]
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] > 0:
                found_next = (idx, (ny, nx))
                break
        if found_next is None:
            break
        idx, nxt = found_next
        # The new "back" direction is opposite of the step we just took
        prev_back = (idx + 4) % 8
        if nxt == start and len(boundary) > 1:
            break
        boundary.append(nxt)
        current = nxt
    return np.array(boundary, dtype=np.int32)


# ============================================================================
# 3. Freeman chain code
# ============================================================================

# 8-direction Freeman codes (E=0, NE=1, N=2, NW=3, W=4, SW=5, S=6, SE=7)
# Note: y axis points down in image coordinates.
_DIR8 = {
    (0, 1): 0, (-1, 1): 1, (-1, 0): 2, (-1, -1): 3,
    (0, -1): 4, (1, -1): 5, (1, 0): 6, (1, 1): 7,
}
# 4-direction (E=0, N=1, W=2, S=3)
_DIR4 = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}


def chain_code_8(boundary: np.ndarray) -> np.ndarray:
    """8-connectivity Freeman code for an ordered boundary array."""
    if len(boundary) < 2:
        return np.zeros((0,), dtype=np.int8)
    diffs = np.diff(boundary, axis=0)
    # Wrap around (close the loop)
    diffs = np.vstack([diffs, boundary[0] - boundary[-1]])
    codes = []
    for dy, dx in diffs:
        # Clamp to {-1, 0, 1} (Moore tracing only produces these steps)
        dy = int(np.sign(dy)); dx = int(np.sign(dx))
        if (dy, dx) in _DIR8:
            codes.append(_DIR8[(dy, dx)])
    return np.array(codes, dtype=np.int8)


def chain_code_4(boundary: np.ndarray) -> np.ndarray:
    """4-connectivity Freeman code (diagonal steps are skipped)."""
    if len(boundary) < 2:
        return np.zeros((0,), dtype=np.int8)
    codes = []
    for i in range(len(boundary)):
        dy, dx = boundary[(i + 1) % len(boundary)] - boundary[i]
        dy = int(np.sign(dy)); dx = int(np.sign(dx))
        if (dy, dx) in _DIR4:
            codes.append(_DIR4[(dy, dx)])
    return np.array(codes, dtype=np.int8)


def chain_first_difference(code: np.ndarray, base: int = 8) -> np.ndarray:
    """First-difference of a chain code → rotation-invariant under 45° / 90°."""
    if len(code) == 0:
        return code
    diff = np.diff(code, append=code[0])
    return (diff % base).astype(np.int8)


# ============================================================================
# 4. Fourier descriptors
# ============================================================================

def fourier_descriptors(boundary: np.ndarray, n_coef: int = 32,
                        n_resample: int = 256) -> np.ndarray:
    """Boundary → complex sequence z[n] = x + j·y → FFT → normalise.

    Normalisations applied:
      - Translation:  drop F[0]    (centroid)
      - Scale:        divide by |F[1]|
      - Rotation:     take magnitudes only
      - Start-point:  resample boundary to fixed length first
    """
    if len(boundary) < 4:
        return np.zeros(n_coef, dtype=np.float64)
    # Resample to fixed length for start-point + length invariance
    bn = _resample(boundary, n_resample)
    z = bn[:, 1].astype(np.float64) + 1j * bn[:, 0].astype(np.float64)
    F = np.fft.fft(z)
    # Drop F[0] (translation)
    F = F[1:]
    # Normalise scale by |F[0]| of the shifted array (which is the old F[1])
    if abs(F[0]) > 1e-9:
        F = F / abs(F[0])
    # Magnitudes for rotation invariance, truncate to n_coef
    return np.abs(F)[:n_coef]


def _resample(boundary: np.ndarray, n: int) -> np.ndarray:
    """Linearly resample a closed boundary to exactly n points."""
    # Cumulative arc length
    diffs = np.diff(boundary, axis=0, append=boundary[:1])
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    cum = np.concatenate([[0], np.cumsum(dists)])
    total = cum[-1]
    if total < 1e-9:
        return np.tile(boundary[0], (n, 1)).astype(np.int32)
    targets = np.linspace(0, total, n, endpoint=False)
    ys = np.interp(targets, cum, np.append(boundary[:, 0], boundary[0, 0]))
    xs = np.interp(targets, cum, np.append(boundary[:, 1], boundary[0, 1]))
    return np.stack([ys, xs], axis=-1).astype(np.int32)


# ============================================================================
# 5. Shape signature (radial distance from centroid)
# ============================================================================

def shape_signature(boundary: np.ndarray, n: int = 360) -> np.ndarray:
    """Radial distance r(θ) sampled at n angles around the centroid.

    Periodic, scale-invariant after normalisation by the mean radius.
    """
    if len(boundary) < 3:
        return np.zeros(n)
    cy = boundary[:, 0].mean()
    cx = boundary[:, 1].mean()
    angles = np.arctan2(boundary[:, 0] - cy, boundary[:, 1] - cx) % (2 * np.pi)
    radii = np.sqrt((boundary[:, 0] - cy) ** 2 + (boundary[:, 1] - cx) ** 2)
    order = np.argsort(angles)
    a = angles[order]; r = radii[order]
    targets = np.linspace(0, 2 * np.pi, n, endpoint=False)
    sig = np.interp(targets, a, r, period=2 * np.pi)
    return sig / (sig.mean() + 1e-9)


# ============================================================================
# 6. Distance + defect score
# ============================================================================

def fd_distance(fd_a: np.ndarray, fd_b: np.ndarray) -> tuple[float, np.ndarray]:
    """Euclidean distance between FD vectors + per-coefficient contributions.

    Returns (total_distance, per_coef_squared_errors).
    """
    n = min(len(fd_a), len(fd_b))
    diff = (fd_a[:n] - fd_b[:n]) ** 2
    return float(np.sqrt(diff.sum())), diff


# ============================================================================
# 7. Render helpers
# ============================================================================

def render_boundary(shape: tuple[int, int], boundary: np.ndarray,
                    color: tuple[int, int, int] = (0, 200, 255)) -> np.ndarray:
    """Return a 3-channel image with the boundary drawn on a black background."""
    h, w = shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if len(boundary) >= 2:
        pts = boundary[:, [1, 0]].reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
    return canvas
