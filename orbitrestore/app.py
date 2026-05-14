"""Streamlit frontend for OrbitRestore — Satellite/Aerial Restoration.

Run: streamlit run app.py
"""

from __future__ import annotations

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
# UNCONDITIONALLY put this project'''s dir at the FRONT of sys.path. Any
# previously-visited project'''s dir has its own pipeline.py/synthetic.py
# of the same name, so it must NOT shadow ours. Remove any prior copy and
# re-insert at position 0.
sys.path[:] = [p for p in sys.path if p != _HERE]
sys.path.insert(0, _HERE)
# Drop cached "pipeline"/"synthetic"/"detector" entries so the next import
# loads THIS project'''s versions, not whichever was visited first.
for _stale in ("pipeline", "synthetic", "detector"):
    sys.modules.pop(_stale, None)

import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from pipeline import (
    to_gray, PSFS, degrade,
    inverse_filter, wiener_filter, lucy_richardson,
    add_stripes, destripe, dehaze,
    psnr, ssim,
)
from synthetic import make_aerial, make_hazy_aerial


try:
    st.set_page_config(page_title="OrbitRestore — Satellite Restoration", layout="wide")
except Exception:
    pass  # parent multipage app already set the page config
st.title("OrbitRestore")
st.caption(
    "Module 3 satellite image restoration. Wiener filter & Lucy-Richardson run "
    "in the **frequency domain via FFT**, dark-channel-prior dehaze, and CCD-"
    "stripe destriping — all classical, no DL."
)


tabs = st.tabs(["Deblur", "Destripe", "Dehaze"])


# ============================================================================
# Deblur
# ============================================================================
with tabs[0]:
    st.subheader("Deblur — simulate & restore")
    c1, c2, c3 = st.columns(3)
    psf_kind = c1.selectbox("Degradation", list(PSFS.keys()))
    # PSF params per type
    if psf_kind == "Motion blur":
        length = c2.slider("length (px)", 3, 31, 15, 2)
        angle = c3.slider("angle (deg)", 0, 180, 30, 5)
        psf = PSFS[psf_kind](length=length, angle=float(angle))
    elif psf_kind == "Defocus disc":
        radius = c2.slider("radius (px)", 1, 12, 5, 1)
        psf = PSFS[psf_kind](radius=radius)
    else:
        sigma = c2.slider("σ", 0.5, 5.0, 2.5, 0.1)
        psf = PSFS[psf_kind](sigma=sigma)

    sigma_n = st.slider("Additive Gaussian noise σ", 0.0, 20.0, 3.0, 0.5)

    clean = make_aerial()
    blurred = degrade(clean, psf, noise_sigma=sigma_n, seed=1)

    method = st.radio("Restoration method", ["Inverse (regularised)", "Wiener", "Lucy-Richardson"],
                      horizontal=True)
    if method == "Inverse (regularised)":
        eps = st.slider("ε regularisation", 1e-3, 1e-1, 1e-2, 1e-3, format="%.3f")
        restored = inverse_filter(to_gray(blurred), psf, epsilon=eps)
    elif method == "Wiener":
        K = st.slider("K (noise-to-signal ratio)", 1e-4, 1e-1, 1e-2, 1e-4, format="%.4f")
        restored = wiener_filter(to_gray(blurred), psf, K=K)
    else:
        n_iter = st.slider("iterations", 1, 60, 15, 1)
        restored = lucy_richardson(to_gray(blurred), psf, iterations=n_iter)

    # PSF preview
    c4, c5 = st.columns([1, 5])
    psf_vis = (psf / psf.max() * 255).astype(np.uint8)
    c4.image(psf_vis, caption=f"PSF h(x,y)\n{psf.shape}", width="stretch", clamp=True)
    with c5:
        c1, c2, c3 = st.columns(3)
        c1.image(clean, caption="Clean (ground truth)", width="stretch", clamp=True)
        c2.image(blurred, caption=f"Degraded ({psf_kind} + noise)", width="stretch", clamp=True)
        c3.image(restored, caption=f"Restored ({method})", width="stretch", clamp=True)

    cleg = to_gray(clean)
    c1, c2 = st.columns(2)
    c1.metric("PSNR restored", f"{psnr(cleg, restored):.2f} dB",
              delta=f"{psnr(cleg, restored) - psnr(cleg, to_gray(blurred)):+.2f} dB")
    c2.metric("SSIM restored", f"{ssim(cleg, restored):.3f}",
              delta=f"{ssim(cleg, restored) - ssim(cleg, to_gray(blurred)):+.3f}")


# ============================================================================
# Destripe
# ============================================================================
with tabs[1]:
    st.subheader("Destripe — defective-CCD-row removal")
    n_stripes = st.slider("Number of stripes", 1, 40, 10, 1)
    mag = st.slider("Stripe magnitude", 5, 80, 30, 5)
    kernel = st.slider("Median-profile kernel (odd)", 3, 51, 25, 2)

    clean = make_aerial()
    striped = add_stripes(clean, n_stripes=n_stripes, magnitude=mag, seed=1)
    cleaned, profile = destripe(striped, kernel=kernel)

    c1, c2, c3 = st.columns(3)
    c1.image(clean, caption="Clean", width="stretch", clamp=True)
    c2.image(striped, caption=f"With {n_stripes} stripes", width="stretch", clamp=True)
    c3.image(cleaned, caption="Destriped", width="stretch", clamp=True)

    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(profile, lw=1.0)
    ax.set_xlabel("row index"); ax.set_ylabel("row anomaly")
    ax.set_title("Per-row mean minus median-smoothed profile (spikes = stripes)")
    ax.grid(alpha=0.3)
    st.pyplot(fig)


# ============================================================================
# Dehaze
# ============================================================================
with tabs[2]:
    st.subheader("Dehaze — dark-channel prior (He et al. 2009)")
    patch = st.slider("Dark-channel patch size", 5, 31, 15, 2)
    omega = st.slider("ω (haze keep ratio)", 0.5, 0.99, 0.95, 0.01)
    t_min = st.slider("t_min (transmission floor)", 0.05, 0.3, 0.1, 0.01)

    src = st.radio("Source", ["Synthetic hazy aerial", "Upload"], horizontal=True)
    if src == "Synthetic hazy aerial":
        hazy = make_hazy_aerial()
    else:
        f = st.file_uploader("Upload hazy image", type=["png", "jpg", "jpeg"])
        if f is None:
            st.info("Upload a hazy image, or use the synthetic sample.")
            st.stop()
        pil = Image.open(f).convert("RGB")
        hazy = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    dehazed = dehaze(hazy, patch=patch, omega=omega, t_min=t_min)
    c1, c2 = st.columns(2)
    c1.image(cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB), caption="Hazy", width="stretch")
    c2.image(cv2.cvtColor(dehazed, cv2.COLOR_BGR2RGB), caption="Dehazed", width="stretch")


st.markdown("---")
st.caption("OrbitRestore · Module 3 · Owner: Sudeep")
