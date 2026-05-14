"""AgroLeaf — HSI color-space plant disease analyzer with region descriptors.

Module 5 syllabus coverage:
    - RGB -> HSI conversion from scratch (textbook Gonzalez & Woods formulas)
    - HSI vs HSV comparison
    - region descriptors: area, perimeter, compactness, eccentricity, solidity,
      Euler number
    - thresholding in HSI space for disease segmentation

The RGB -> HSI implementation is the marks-bearing piece. We do NOT use
cv2.cvtColor for HSI — only for HSV, and only for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


# ----------------------------- RGB -> HSI ----------------------------------

def rgb_to_hsi(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 image to HSI float32, H in [0, 1), S in [0, 1], I in [0, 1].

    Formulas (Gonzalez & Woods, Eq. 6.2-2):

        I = (R + G + B) / 3
        S = 1 - (3 / (R + G + B)) * min(R, G, B)
        theta = arccos( 0.5 * ((R - G) + (R - B)) /
                        sqrt((R - G)^2 + (R - B)(G - B)) )
        H = theta        if B <= G
        H = 2*pi - theta otherwise
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    I = (R + G + B) / 3.0

    # Saturation
    rgb_sum = R + G + B
    rgb_min = np.minimum(np.minimum(R, G), B)
    # avoid division-by-zero on pure black pixels
    safe_sum = np.where(rgb_sum > 1e-9, rgb_sum, 1.0)
    S = 1.0 - 3.0 * rgb_min / safe_sum
    S = np.where(rgb_sum > 1e-9, S, 0.0)

    # Hue
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
    # cos(theta) clamped to [-1, 1] for arccos
    cos_theta = np.where(den > 1e-9, num / den, 0.0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    H = np.where(B <= G, theta, 2 * np.pi - theta)
    # Normalise H to [0, 1)
    H = H / (2 * np.pi)

    out = np.stack([H, S, I], axis=-1).astype(np.float32)
    return out


def hsi_to_display(hsi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return three uint8 single-channel images: H (as hue wheel-ish), S, I."""
    H = (hsi[..., 0] * 255).astype(np.uint8)
    S = (hsi[..., 1] * 255).astype(np.uint8)
    I = (hsi[..., 2] * 255).astype(np.uint8)
    return H, S, I


# ----------------------------- Leaf segmentation ---------------------------

def leaf_mask(bgr: np.ndarray) -> np.ndarray:
    """Bounding mask separating the leaf (any colour, healthy or diseased)
    from the background.

    Strategy: leaves are darker than typical paper/soil/tan backgrounds, so
    Otsu on the inverted L (lightness) channel of Lab catches both green
    tissue AND brown/yellow disease lesions — which is critical, because the
    disease mask is computed as `leaf ∩ ¬healthy` and we MUST keep diseased
    pixels in the leaf bounding mask.

    Uses morphological close+open to fill small intra-leaf holes (e.g. yellow
    halo around a lesion) and reject background specks, then keeps only the
    single largest connected component.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    _, m = cv2.threshold(255 - L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    n, lab_img, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab_img == biggest).astype(np.uint8) * 255
    # Fill any remaining internal holes (lesions create dark holes that the
    # leaf bounding mask must include)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        filled = np.zeros_like(m)
        cv2.drawContours(filled, contours, -1, 255, thickness=-1)
        m = filled
    return m


# ----------------------------- Disease segmentation ------------------------

def healthy_mask(hsi: np.ndarray, h_lo: float = 0.20, h_hi: float = 0.42,
                 s_min: float = 0.15) -> np.ndarray:
    """Pixels with green hue and meaningful saturation = healthy tissue.

    Hue is in [0, 1); 0.20..0.42 spans green. Saturation must exceed a small
    threshold to reject washed-out / grayish pixels.
    """
    H = hsi[..., 0]
    S = hsi[..., 1]
    healthy = ((H >= h_lo) & (H <= h_hi)) & (S >= s_min)
    return (healthy.astype(np.uint8) * 255)


def disease_mask(bgr: np.ndarray, h_lo: float = 0.20, h_hi: float = 0.42,
                 s_min: float = 0.15, min_lesion_area: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Disease mask = leaf-bounded AND NOT-healthy.

    Returns (disease_mask, leaf_mask), both uint8 0/255.
    """
    leaf = leaf_mask(bgr)
    hsi = rgb_to_hsi(bgr)
    healthy = healthy_mask(hsi, h_lo, h_hi, s_min)
    diseased = cv2.bitwise_and(leaf, cv2.bitwise_not(healthy))
    # Remove tiny specks
    n, lab, stats, _ = cv2.connectedComponentsWithStats(diseased, connectivity=8)
    keep = np.zeros_like(diseased)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_lesion_area:
            keep[lab == i] = 255
    return keep, leaf


# ----------------------------- Region descriptors --------------------------

@dataclass
class Lesion:
    id: int
    area: int           # px
    perimeter: float    # px
    compactness: float  # P^2 / (4*pi*A) — 1.0 = perfect circle, >1 elongated
    eccentricity: float # from second moments, 0 = circle, ~1 = line
    solidity: float     # area / convex-hull-area
    holes: int          # number of holes inside the lesion
    euler: int          # 1 - holes (single component)
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]  # x, y, w, h


def _eccentricity_from_moments(m: dict) -> float:
    """Eccentricity from central moments: see Gonzalez & Woods 11.3-6."""
    mu20 = m["mu20"]; mu02 = m["mu02"]; mu11 = m["mu11"]
    a = mu20 + mu02
    b = np.sqrt(4 * mu11 * mu11 + (mu20 - mu02) ** 2)
    lam1 = (a + b) / 2.0
    lam2 = (a - b) / 2.0
    if lam1 <= 1e-9:
        return 0.0
    return float(np.sqrt(max(0.0, 1.0 - lam2 / lam1)))


def lesion_descriptors(disease_mask_img: np.ndarray) -> list[Lesion]:
    """Per-component region descriptors for every lesion."""
    out: list[Lesion] = []
    # External contours give us the lesion boundaries
    contours, hier = cv2.findContours(disease_mask_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return out
    hier = hier[0]  # shape (N, 4)

    next_id = 1
    for i, c in enumerate(contours):
        # Skip if this contour is a hole (parent != -1 means inner contour)
        if hier[i][3] != -1:
            continue
        area = float(cv2.contourArea(c))
        if area < 1:
            continue
        perim = float(cv2.arcLength(c, closed=True))
        compactness = (perim * perim) / (4 * np.pi * area) if area > 0 else 0.0

        moms = cv2.moments(c)
        ecc = _eccentricity_from_moments(moms)

        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull)) if hull is not None else area
        solidity = area / hull_area if hull_area > 0 else 1.0

        # Holes = number of inner contours whose parent is this one
        holes = sum(1 for j in range(len(contours)) if hier[j][3] == i)
        euler = 1 - holes

        x, y, w, h = cv2.boundingRect(c)
        cx = moms["m10"] / moms["m00"] if moms["m00"] > 0 else x + w / 2
        cy = moms["m01"] / moms["m00"] if moms["m00"] > 0 else y + h / 2

        out.append(Lesion(
            id=next_id, area=int(area), perimeter=perim,
            compactness=float(compactness), eccentricity=float(ecc),
            solidity=float(solidity), holes=holes, euler=euler,
            centroid=(float(cx), float(cy)), bbox=(x, y, w, h),
        ))
        next_id += 1
    return out


def severity_score(leaf_mask_img: np.ndarray, disease_mask_img: np.ndarray,
                   lesions: list[Lesion]) -> dict:
    """Leaf-level severity: % area + lesion count + simple risk class."""
    leaf_area = int((leaf_mask_img > 0).sum())
    disease_area = int((disease_mask_img > 0).sum())
    pct = (100.0 * disease_area / leaf_area) if leaf_area > 0 else 0.0

    if pct < 5:
        label = "Healthy"
    elif pct < 15:
        label = "Mild"
    elif pct < 35:
        label = "Moderate"
    else:
        label = "Severe"

    return {
        "leaf_area_px": leaf_area,
        "disease_area_px": disease_area,
        "percent_affected": round(pct, 2),
        "lesion_count": len(lesions),
        "risk": label,
    }


# ----------------------------- HSI vs HSV ----------------------------------

def hsv_disease_mask(bgr: np.ndarray, h_lo: int = 36, h_hi: int = 86,
                     s_min: int = 40) -> np.ndarray:
    """Same idea but using OpenCV's HSV — for the comparison panel.

    OpenCV HSV: H in [0, 179], S/V in [0, 255]. Green ≈ 36..86.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv[..., 0]; S = hsv[..., 1]
    healthy = ((H >= h_lo) & (H <= h_hi)) & (S >= s_min)
    leaf = leaf_mask(bgr)
    return cv2.bitwise_and(leaf, cv2.bitwise_not(healthy.astype(np.uint8) * 255))


# ----------------------------- Overlay -------------------------------------

def render_overlay(bgr: np.ndarray, disease_mask_img: np.ndarray,
                   lesions: list[Lesion]) -> np.ndarray:
    """Red disease overlay + numbered lesion labels."""
    out = bgr.copy()
    red = np.zeros_like(out); red[..., 2] = 255
    mask3 = cv2.cvtColor(disease_mask_img, cv2.COLOR_GRAY2BGR) > 0
    out[mask3] = (0.5 * out[mask3] + 0.5 * red[mask3]).astype(np.uint8)
    for l in lesions:
        cx, cy = int(l.centroid[0]), int(l.centroid[1])
        cv2.putText(out, str(l.id), (cx - 8, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out
