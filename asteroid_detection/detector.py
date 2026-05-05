"""Asteroid Detection — pure OpenCV pipeline (no Streamlit).

Pipeline (two sky images of the same field, taken at different times):
    grayscale + Gaussian blur (denoise)
      -> ORB feature detection + BFMatcher + RANSAC homography (alignment)
      -> absdiff (stars cancel; movers leave a residue)
      -> threshold (Otsu or manual)
      -> morphological opening (kill 1-pixel noise)
      -> connected-components (each blob = candidate)
      -> area filter
      -> annotate the original image
"""

from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Candidate:
    idx: int
    centroid: tuple        # (x, y)
    bbox: tuple            # (x, y, w, h)
    area: int
    mean_brightness: float


# ---------- Step 1: preprocess ----------
def preprocess(img: np.ndarray, blur_k: int = 3) -> np.ndarray:
    """Grayscale + Gaussian blur. Reduces sensor noise that would survive differencing."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    if blur_k > 1:
        # ensure odd kernel
        if blur_k % 2 == 0:
            blur_k += 1
        gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    return gray


# ---------- Step 2: align ----------
def align(
    img_ref: np.ndarray,
    img_to_warp: np.ndarray,
    motion: str = "euclidean",
    iterations: int = 200,
    eps: float = 1e-6,
):
    """Align img_to_warp onto img_ref using ECC (Enhanced Correlation Coefficient).

    ECC is intensity-based and works well on sparse-feature images like
    star fields where ORB struggles (no distinctive corners — every star
    looks the same). It iteratively maximizes the correlation coefficient
    between the warped image and the reference.

    motion: "translation" | "euclidean" (default; rot+trans) | "affine" | "homography"

    Returns (warped, warp_matrix, ecc_score). If ECC fails to converge, falls
    back to ORB feature matching, then to the unchanged image.
    """
    motion_map = {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean":   cv2.MOTION_EUCLIDEAN,
        "affine":      cv2.MOTION_AFFINE,
        "homography":  cv2.MOTION_HOMOGRAPHY,
    }
    motion_type = motion_map.get(motion, cv2.MOTION_EUCLIDEAN)

    if motion_type == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, eps)

    try:
        score, warp_matrix = cv2.findTransformECC(
            img_ref, img_to_warp, warp_matrix, motion_type, criteria, None, 5
        )
    except cv2.error:
        # ECC didn't converge — try ORB fallback
        return _align_orb_fallback(img_ref, img_to_warp)

    h, w = img_ref.shape[:2]
    if motion_type == cv2.MOTION_HOMOGRAPHY:
        warped = cv2.warpPerspective(
            img_to_warp, warp_matrix, (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )
    else:
        warped = cv2.warpAffine(
            img_to_warp, warp_matrix, (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )
    return warped, warp_matrix, float(score)


def _align_orb_fallback(img_ref: np.ndarray, img_to_warp: np.ndarray, max_features: int = 1000):
    """ORB-based alignment used only when ECC fails to converge."""
    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(img_ref, None)
    kp2, des2 = orb.detectAndCompute(img_to_warp, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return img_to_warp.copy(), None, 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)
    matches = matches[: max(4, int(len(matches) * 0.8))]
    if len(matches) < 4:
        return img_to_warp.copy(), None, 0.0
    src = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return img_to_warp.copy(), None, 0.0
    h, w = img_ref.shape[:2]
    warped = cv2.warpPerspective(img_to_warp, H, (w, h))
    return warped, H, float(mask.sum()) if mask is not None else 0.0


# ---------- Step 3: difference ----------
def difference(a: np.ndarray, b_aligned: np.ndarray) -> np.ndarray:
    """Absolute pixel difference. Stars (well aligned) cancel, movers stand out."""
    return cv2.absdiff(a, b_aligned)


# ---------- Step 4: threshold ----------
def threshold(diff: np.ndarray, mode: str = "otsu", manual_value: int = 25) -> np.ndarray:
    """Binarize the difference image. Mode: 'otsu' or 'manual'."""
    if mode == "otsu":
        _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(diff, manual_value, 255, cv2.THRESH_BINARY)
    return binary


# ---------- Step 5: morphology ----------
def morphological_clean(binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Opening removes single-pixel noise (cosmic rays, sensor hot pixels)."""
    if kernel_size < 1:
        return binary
    if kernel_size % 2 == 0:
        kernel_size += 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)


# ---------- Step 6+7: candidates ----------
def find_candidates(
    binary: np.ndarray,
    diff: np.ndarray,
    min_area: int = 4,
    max_area: int = 500,
) -> list[Candidate]:
    """Connected components on the binary mask, filtered by area.

    Mean brightness comes from the difference image (how strong the change was).
    """
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out: list[Candidate] = []
    next_idx = 1
    for i in range(1, n):  # skip background (label 0)
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_area or a > max_area:
            continue
        mask = (labels == i)
        mean_b = float(diff[mask].mean()) if mask.any() else 0.0
        cx, cy = centroids[i]
        out.append(Candidate(
            idx=next_idx,
            centroid=(float(cx), float(cy)),
            bbox=(x, y, w, h),
            area=a,
            mean_brightness=mean_b,
        ))
        next_idx += 1
    return out


# ---------- Step 8: annotate ----------
def annotate(img: np.ndarray, candidates: list[Candidate]) -> np.ndarray:
    """Circle and number each candidate on a copy of img.

    Accepts grayscale (2D) or BGR (3D); always returns BGR for display.
    """
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out = img.copy()
    for c in candidates:
        cx, cy = int(round(c.centroid[0])), int(round(c.centroid[1]))
        radius = max(8, int(np.sqrt(c.area) * 2))
        cv2.circle(out, (cx, cy), radius, (0, 255, 255), 2)
        cv2.putText(
            out, str(c.idx), (cx + radius + 2, cy + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
        )
    return out


def diff_heatmap(diff: np.ndarray) -> np.ndarray:
    """Convert a grayscale difference image to a HOT colormap for display."""
    if diff.dtype != np.uint8:
        diff = np.clip(diff, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(diff, cv2.COLORMAP_HOT)


# ---------- One-call pipeline ----------
def detect_pair(
    img_a: np.ndarray,
    img_b: np.ndarray,
    blur_k: int = 3,
    threshold_mode: str = "otsu",
    manual_threshold: int = 25,
    morph_kernel: int = 3,
    min_area: int = 4,
    max_area: int = 500,
) -> dict:
    """Run the full pipeline. Returns a dict with every intermediate stage,
    so the Streamlit UI can display them all."""
    a_gray = preprocess(img_a, blur_k)
    b_gray = preprocess(img_b, blur_k)
    b_aligned, warp, score = align(a_gray, b_gray)
    diff = difference(a_gray, b_aligned)
    binary = threshold(diff, threshold_mode, manual_threshold)
    cleaned = morphological_clean(binary, morph_kernel)
    candidates = find_candidates(cleaned, diff, min_area, max_area)
    annotated = annotate(img_a, candidates)
    return {
        "a_gray": a_gray,
        "b_gray": b_gray,
        "b_aligned": b_aligned,
        "diff": diff,
        "diff_heatmap": diff_heatmap(diff),
        "binary": binary,
        "cleaned": cleaned,
        "candidates": candidates,
        "annotated": annotated,
        "warp_matrix": warp,
        "alignment_score": score,
    }
