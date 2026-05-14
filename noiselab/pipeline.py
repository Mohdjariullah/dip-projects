"""NoiseLab — noise models + spatial-domain restoration filters.

Module 3 syllabus coverage:
    Noise models:
        Gaussian, Rayleigh, Erlang/Gamma, Exponential, Uniform,
        Salt-and-pepper, Speckle (multiplicative Gaussian)
    Restoration in spatial domain:
        arithmetic mean, geometric mean, harmonic mean,
        contraharmonic mean (Q-parameterised), median,
        adaptive median (level-A / level-B from Gonzalez & Woods),
        midpoint, alpha-trimmed mean
    Metrics:
        PSNR, SSIM (Wang 2004), MSE, MAE
"""

from __future__ import annotations

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# ============================================================================
# Noise models
# ============================================================================

def add_gaussian(img: np.ndarray, sigma: float = 20.0, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def add_rayleigh(img: np.ndarray, scale: float = 25.0, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = rng.rayleigh(scale, img.shape) - scale * np.sqrt(np.pi / 2)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def add_gamma(img: np.ndarray, shape: float = 2.0, scale: float = 15.0,
              seed: int | None = None) -> np.ndarray:
    """Erlang/Gamma noise (shape=int → Erlang)."""
    rng = np.random.default_rng(seed)
    n = rng.gamma(shape, scale, img.shape) - shape * scale
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def add_exponential(img: np.ndarray, scale: float = 20.0,
                    seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = rng.exponential(scale, img.shape) - scale
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def add_uniform(img: np.ndarray, low: float = -25, high: float = 25,
                seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = rng.uniform(low, high, img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def add_salt_pepper(img: np.ndarray, density: float = 0.10,
                    seed: int | None = None) -> np.ndarray:
    """Replace `density` of pixels with random salt (255) or pepper (0)."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    n_total = img.size if img.ndim == 2 else img.shape[0] * img.shape[1]
    n_corrupt = int(density * n_total)
    if n_corrupt == 0:
        return out
    coords = rng.integers(0, [img.shape[0], img.shape[1]], size=(n_corrupt, 2))
    half = n_corrupt // 2
    salt_y, salt_x = coords[:half, 0], coords[:half, 1]
    pep_y, pep_x = coords[half:, 0], coords[half:, 1]
    if out.ndim == 2:
        out[salt_y, salt_x] = 255
        out[pep_y, pep_x] = 0
    else:
        out[salt_y, salt_x] = 255
        out[pep_y, pep_x] = 0
    return out


def add_speckle(img: np.ndarray, sigma: float = 0.1,
                seed: int | None = None) -> np.ndarray:
    """Multiplicative Gaussian: I' = I + I * N(0, sigma)."""
    rng = np.random.default_rng(seed)
    n = rng.normal(0, sigma, img.shape)
    out = img.astype(np.float32) * (1 + n)
    return np.clip(out, 0, 255).astype(np.uint8)


NOISE_MODELS = {
    "Gaussian": add_gaussian,
    "Rayleigh": add_rayleigh,
    "Gamma (Erlang)": add_gamma,
    "Exponential": add_exponential,
    "Uniform": add_uniform,
    "Salt & Pepper": add_salt_pepper,
    "Speckle": add_speckle,
}


# ============================================================================
# Restoration filters (spatial domain)
# ============================================================================

def _pad_reflect(img: np.ndarray, w: int) -> np.ndarray:
    return cv2.copyMakeBorder(img, w, w, w, w, cv2.BORDER_REFLECT_101)


def arithmetic_mean(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.blur(img, (ksize, ksize))


def geometric_mean(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """exp( mean(log(I + 1)) ) — log-space arithmetic mean."""
    f = img.astype(np.float32) + 1.0
    logf = np.log(f)
    logmean = cv2.blur(logf, (ksize, ksize))
    out = np.exp(logmean) - 1.0
    return np.clip(out, 0, 255).astype(np.uint8)


def harmonic_mean(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """mn / sum(1/I).  Good for salt noise (kills bright outliers)."""
    eps = 1e-6
    f = img.astype(np.float32) + eps
    inv = 1.0 / f
    mean_inv = cv2.blur(inv, (ksize, ksize))
    out = 1.0 / (mean_inv + eps)
    return np.clip(out, 0, 255).astype(np.uint8)


def contraharmonic_mean(img: np.ndarray, ksize: int = 3, Q: float = 0.0) -> np.ndarray:
    """sum(I^(Q+1)) / sum(I^Q).

    Q > 0 removes pepper (dark outliers); Q < 0 removes salt; Q = 0 = arithmetic mean.
    """
    eps = 1e-6
    f = img.astype(np.float32) + eps
    num = cv2.blur(f ** (Q + 1), (ksize, ksize))
    den = cv2.blur(f ** Q, (ksize, ksize))
    out = num / (den + eps)
    return np.clip(out, 0, 255).astype(np.uint8)


def median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.medianBlur(img, ksize)


def midpoint_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """(max + min) / 2 — works well on uniform / Gaussian noise."""
    f = img.astype(np.float32)
    mx = cv2.dilate(f, np.ones((ksize, ksize), np.uint8))
    mn = cv2.erode(f, np.ones((ksize, ksize), np.uint8))
    return np.clip((mx + mn) / 2.0, 0, 255).astype(np.uint8)


def alpha_trimmed_mean(img: np.ndarray, ksize: int = 3, d: int = 2) -> np.ndarray:
    """Delete d/2 smallest and d/2 largest per window; mean the rest.

    Bridges median (d = ksize²-1) and arithmetic mean (d = 0).
    Pure-Python loop for clarity; fine for demo images.
    """
    g = to_gray(img).astype(np.float32)
    h, w = g.shape
    pad = ksize // 2
    P = _pad_reflect(g.astype(np.uint8), pad).astype(np.float32)
    out = np.zeros_like(g)
    keep = ksize * ksize - d
    for y in range(h):
        for x in range(w):
            win = P[y:y + ksize, x:x + ksize].ravel()
            win.sort()
            half = d // 2
            out[y, x] = win[half:ksize * ksize - (d - half)].mean()
    return np.clip(out, 0, 255).astype(np.uint8)


def adaptive_median(img: np.ndarray, smax: int = 7) -> np.ndarray:
    """Gonzalez & Woods adaptive median filter.

    Algorithm (per pixel):
      Level A — compute z_min, z_med, z_max in window of size S.
                if z_min < z_med < z_max, go to B.
                else expand S by 2 and repeat (until S > smax → output z_med).
      Level B — if z_min < z_xy < z_max, output z_xy (original).
                else output z_med.

    Pepper/salt impulse noise: median works up to ~20%; this works to ~80%.
    """
    g = to_gray(img).astype(np.uint8)
    h, w = g.shape
    pad = smax // 2
    P = _pad_reflect(g, pad)
    out = g.copy()

    for y in range(h):
        for x in range(w):
            s = 3
            while True:
                r = s // 2
                yy, xx = y + pad, x + pad
                win = P[yy - r:yy + r + 1, xx - r:xx + r + 1]
                zmin, zmed, zmax = int(win.min()), int(np.median(win)), int(win.max())
                if zmin < zmed < zmax:
                    # Level B
                    zxy = int(g[y, x])
                    out[y, x] = zxy if (zmin < zxy < zmax) else zmed
                    break
                else:
                    s += 2
                    if s > smax:
                        out[y, x] = zmed
                        break
    return out


FILTERS = {
    "Arithmetic mean": arithmetic_mean,
    "Geometric mean":  geometric_mean,
    "Harmonic mean":   harmonic_mean,
    "Contraharmonic":  contraharmonic_mean,
    "Median":          median_filter,
    "Adaptive median": adaptive_median,
    "Midpoint":        midpoint_filter,
    "Alpha-trimmed":   alpha_trimmed_mean,
}


# ============================================================================
# Metrics
# ============================================================================

def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = mse(a, b)
    if m < 1e-12:
        return 99.0
    return float(10.0 * np.log10((255.0 ** 2) / m))


def ssim(a: np.ndarray, b: np.ndarray, window: int = 7) -> float:
    """Simplified single-scale SSIM (Wang 2004), grayscale only."""
    a = to_gray(a).astype(np.float64)
    b = to_gray(b).astype(np.float64)
    K1, K2, L = 0.01, 0.03, 255.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu_a = cv2.GaussianBlur(a, (window, window), 1.5)
    mu_b = cv2.GaussianBlur(b, (window, window), 1.5)
    var_a = cv2.GaussianBlur(a * a, (window, window), 1.5) - mu_a ** 2
    var_b = cv2.GaussianBlur(b * b, (window, window), 1.5) - mu_b ** 2
    cov   = cv2.GaussianBlur(a * b, (window, window), 1.5) - mu_a * mu_b
    num = (2 * mu_a * mu_b + C1) * (2 * cov + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (var_a + var_b + C2)
    return float((num / den).mean())


# ============================================================================
# Noise-type rule-based classifier (no ML)
# ============================================================================

def classify_noise(img: np.ndarray) -> dict:
    """Return relative "confidence" for each noise type from histogram stats.

    Pure histogram-feature rules — useful as a viva talking-point on how
    a non-ML estimator picks signal from skewness/kurtosis/impulse spikes.
    """
    g = to_gray(img).astype(np.float64)
    h = cv2.calcHist([to_gray(img)], [0], None, [256], [0, 256]).ravel()
    total = max(1.0, h.sum())
    spike_0 = h[0] / total
    spike_255 = h[255] / total
    # Skewness / kurtosis (Fisher) of the histogram
    mu = g.mean()
    sd = g.std() + 1e-9
    skew = float(((g - mu) ** 3).mean() / (sd ** 3))
    kurt = float(((g - mu) ** 4).mean() / (sd ** 4) - 3.0)

    # Salt-and-pepper is the only noise that creates significant probability
    # mass exactly at 0 and 255. If we see >0.5% at either extreme, it
    # dominates — override all other scores.
    sp_evidence = max(spike_0, spike_255)
    sp_score = 100.0 * sp_evidence  # >> all others when impulses present

    scores = {
        "Salt & Pepper": sp_score,
        "Gaussian":      max(0.0, 1.0 - abs(skew) - 0.05 * abs(kurt)),
        "Rayleigh":      max(0.0, min(1.0, 0.5 + skew)),       # right-skewed
        "Exponential":   max(0.0, min(1.0, 0.4 + 0.5 * skew + 0.05 * kurt)),
        "Uniform":       max(0.0, 1.0 - abs(kurt) / 2.0 - abs(skew)),
        "Gamma":         max(0.0, min(1.0, 0.3 + 0.4 * skew)),
        "Speckle":       max(0.0, 0.5 - abs(skew) + 0.1 * kurt),
    }
    s = sum(scores.values()) or 1.0
    probs = {k: round(v / s, 3) for k, v in scores.items()}
    best = max(probs, key=probs.get)
    return {"prediction": best, "scores": probs, "skew": round(skew, 3),
            "kurt": round(kurt, 3), "spike0": round(spike_0, 4),
            "spike255": round(spike_255, 4)}


# ============================================================================
# Recommendation
# ============================================================================

RECOMMENDED_FILTER = {
    "Salt & Pepper":  "Adaptive median",
    "Gaussian":       "Alpha-trimmed",
    "Rayleigh":       "Geometric mean",
    "Exponential":    "Geometric mean",
    "Uniform":        "Arithmetic mean",
    "Gamma":          "Alpha-trimmed",
    "Speckle":        "Median",
}
