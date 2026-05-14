"""OrbitRestore — frequency-domain restoration for satellite & aerial images.

Module 3 syllabus coverage:
    - degradation model g = h * f + n
    - direct inverse filtering with regularisation
    - Wiener filter (frequency domain)
    - Lucy-Richardson iterative deconvolution
    - line-noise (stripe) detection & subtraction
    - dark-channel prior dehaze (He et al. 2009 — classical, no DL)
"""

from __future__ import annotations

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


# ============================================================================
# PSF kernels (degradation models)
# ============================================================================

def motion_psf(length: int = 15, angle: float = 0.0, size: int = 31) -> np.ndarray:
    """Linear-motion blur kernel of given pixel `length` and `angle` (deg)."""
    k = np.zeros((size, size), dtype=np.float32)
    cx = size // 2
    cy = size // 2
    th = np.deg2rad(angle)
    dx, dy = np.cos(th), np.sin(th)
    half = length / 2
    pts = [(cx + dx * t, cy + dy * t) for t in np.linspace(-half, half, length * 4)]
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < size and 0 <= yi < size:
            k[yi, xi] = 1.0
    s = k.sum()
    return k / s if s > 0 else k


def defocus_psf(radius: int = 5, size: int = 31) -> np.ndarray:
    """Uniform disc kernel of `radius` pixels."""
    k = np.zeros((size, size), dtype=np.float32)
    cv2.circle(k, (size // 2, size // 2), radius, 1.0, -1)
    s = k.sum()
    return k / s if s > 0 else k


def gaussian_psf(sigma: float = 2.5, size: int = 31) -> np.ndarray:
    """Gaussian PSF for atmospheric turbulence."""
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


PSFS = {
    "Motion blur":   motion_psf,
    "Defocus disc":  defocus_psf,
    "Gaussian (turbulence)": gaussian_psf,
}


# ============================================================================
# Degradation
# ============================================================================

def degrade(img: np.ndarray, psf: np.ndarray, noise_sigma: float = 0.0,
            seed: int | None = None) -> np.ndarray:
    """g = h * f + n.  Uses BORDER_REFLECT to suppress edge artefacts."""
    blurred = cv2.filter2D(img.astype(np.float32), -1, psf,
                           borderType=cv2.BORDER_REFLECT_101)
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        blurred = blurred + rng.normal(0, noise_sigma, blurred.shape)
    return np.clip(blurred, 0, 255).astype(np.uint8)


# ============================================================================
# Restoration
# ============================================================================

def _fft_kernel(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Zero-pad PSF to image shape and centre-shift, then FFT."""
    pad = np.zeros(shape, dtype=np.float32)
    kh, kw = psf.shape
    pad[:kh, :kw] = psf
    # Centre the kernel so the phase doesn't shift the image
    pad = np.roll(pad, -kh // 2, axis=0)
    pad = np.roll(pad, -kw // 2, axis=1)
    return np.fft.fft2(pad)


def inverse_filter(g: np.ndarray, psf: np.ndarray,
                   epsilon: float = 1e-2) -> np.ndarray:
    """Regularised inverse: F̂ = G / (H + ε·sign(H)).

    Without ε, this blows up wherever H ≈ 0.
    """
    G = np.fft.fft2(g.astype(np.float32))
    H = _fft_kernel(psf, g.shape)
    F = G / (H + epsilon * np.exp(1j * np.angle(H)))
    out = np.real(np.fft.ifft2(F))
    return np.clip(out, 0, 255).astype(np.uint8)


def wiener_filter(g: np.ndarray, psf: np.ndarray, K: float = 0.01) -> np.ndarray:
    """Wiener: F̂ = (H*/(|H|² + K)) · G."""
    G = np.fft.fft2(g.astype(np.float32))
    H = _fft_kernel(psf, g.shape)
    Hc = np.conj(H)
    F = (Hc / (np.abs(H) ** 2 + K)) * G
    out = np.real(np.fft.ifft2(F))
    return np.clip(out, 0, 255).astype(np.uint8)


def lucy_richardson(g: np.ndarray, psf: np.ndarray,
                    iterations: int = 10) -> np.ndarray:
    """Iterative maximum-likelihood deconvolution (Richardson 1972, Lucy 1974).

    f_{n+1} = f_n · ( h^T * ( g / (h * f_n) ) )
    """
    f = g.astype(np.float32) / 255.0
    g_norm = g.astype(np.float32) / 255.0
    psf_flip = np.flipud(np.fliplr(psf))
    eps = 1e-6
    for _ in range(iterations):
        denom = cv2.filter2D(f, -1, psf, borderType=cv2.BORDER_REFLECT_101) + eps
        ratio = g_norm / denom
        update = cv2.filter2D(ratio, -1, psf_flip, borderType=cv2.BORDER_REFLECT_101)
        f = f * update
    return np.clip(f * 255.0, 0, 255).astype(np.uint8)


# ============================================================================
# Destripe (line-noise removal)
# ============================================================================

def add_stripes(img: np.ndarray, n_stripes: int = 8, magnitude: int = 30,
                seed: int | None = 1) -> np.ndarray:
    """Inject horizontal scan-line noise (defective CCD rows)."""
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32).copy()
    rows = rng.integers(0, img.shape[0], n_stripes)
    sign = rng.choice([-1, 1], size=n_stripes)
    for r, s in zip(rows, sign):
        out[r:r + 2, :] += s * magnitude
    return np.clip(out, 0, 255).astype(np.uint8)


def _median_1d(profile: np.ndarray, kernel: int) -> np.ndarray:
    """1-D median filter via a strided sliding window (NumPy only)."""
    k = kernel if kernel % 2 else kernel + 1
    pad = k // 2
    padded = np.pad(profile, pad, mode="reflect")
    # Build (N, k) view of windows
    windows = np.lib.stride_tricks.sliding_window_view(padded, k)
    return np.median(windows, axis=1)


def destripe(img: np.ndarray, kernel: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Subtract per-row anomaly from a median-filtered row profile.

    Returns (cleaned_image, row_anomaly_profile).
    """
    g = to_gray(img).astype(np.float32)
    row_mean = g.mean(axis=1)
    row_mean_smooth = _median_1d(row_mean, kernel)
    anomaly = row_mean - row_mean_smooth
    out = g - anomaly[:, None]
    return np.clip(out, 0, 255).astype(np.uint8), anomaly


# ============================================================================
# Dark-channel-prior dehaze (He, Sun, Tang 2009 — classical, no DL)
# ============================================================================

def _dark_channel(img: np.ndarray, patch: int = 15) -> np.ndarray:
    """min over RGB then min over patch×patch."""
    if img.ndim == 2:
        mn = img
    else:
        mn = img.min(axis=2)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(mn, k)


def dehaze(bgr: np.ndarray, patch: int = 15, omega: float = 0.95,
           t_min: float = 0.1) -> np.ndarray:
    """Dehaze using the dark-channel prior.

    1. Estimate atmospheric light A from the brightest 0.1% pixels in the dark channel.
    2. Transmission t(x) = 1 - ω · darkChannel(I(x) / A).
    3. Recover scene radiance: J = (I - A) / max(t, t_min) + A.
    """
    f = bgr.astype(np.float32) / 255.0
    dc = _dark_channel((f * 255).astype(np.uint8), patch).astype(np.float32) / 255.0

    # Atmospheric light: top 0.1% brightest pixels of dark channel → take max of I there
    n_pixels = dc.size
    n_top = max(1, n_pixels // 1000)
    flat_dc = dc.ravel()
    idx = np.argpartition(flat_dc, -n_top)[-n_top:]
    flat_I = f.reshape(-1, f.shape[2] if f.ndim == 3 else 1)
    A = flat_I[idx].max(axis=0)

    norm = f / np.maximum(A, 1e-3)
    t = 1.0 - omega * _dark_channel((norm * 255).astype(np.uint8), patch).astype(np.float32) / 255.0
    t = np.clip(t, t_min, 1.0)

    if f.ndim == 3:
        J = (f - A) / t[..., None] + A
    else:
        J = (f - A) / t + A
    return np.clip(J * 255.0, 0, 255).astype(np.uint8)


# ============================================================================
# Metrics (re-use a minimal copy)
# ============================================================================

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if m < 1e-12:
        return 99.0
    return float(10.0 * np.log10(255.0 ** 2 / m))


def ssim(a: np.ndarray, b: np.ndarray, window: int = 7) -> float:
    a = to_gray(a).astype(np.float64); b = to_gray(b).astype(np.float64)
    K1, K2, L = 0.01, 0.03, 255.0
    C1 = (K1 * L) ** 2; C2 = (K2 * L) ** 2
    mu_a = cv2.GaussianBlur(a, (window, window), 1.5)
    mu_b = cv2.GaussianBlur(b, (window, window), 1.5)
    var_a = cv2.GaussianBlur(a * a, (window, window), 1.5) - mu_a ** 2
    var_b = cv2.GaussianBlur(b * b, (window, window), 1.5) - mu_b ** 2
    cov   = cv2.GaussianBlur(a * b, (window, window), 1.5) - mu_a * mu_b
    num = (2 * mu_a * mu_b + C1) * (2 * cov + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (var_a + var_b + C2)
    return float((num / den).mean())
