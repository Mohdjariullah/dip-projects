"""Streamlit frontend for NoiseLab — Noise Models + Restoration.

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
from PIL import Image

from pipeline import (
    to_gray, NOISE_MODELS, FILTERS,
    psnr, ssim, mse, mae,
    classify_noise, RECOMMENDED_FILTER,
    contraharmonic_mean, adaptive_median, alpha_trimmed_mean,
)
from synthetic import make_clean_scene


try:
    st.set_page_config(page_title="NoiseLab — Restoration Suite", layout="wide")
except Exception:
    pass  # parent multipage app already set the page config
st.title("NoiseLab Forensics")
st.caption(
    "Module 3 noise models + spatial-domain restoration. Adaptive median, "
    "contraharmonic mean, and a rule-based noise-type classifier — all from scratch."
)


# ---------------- Input ----------------
st.sidebar.header("Clean reference")
src = st.sidebar.radio("Source", ["Synthetic scene", "Upload"], index=0)
if src == "Synthetic scene":
    clean = make_clean_scene()
else:
    f = st.sidebar.file_uploader("Upload a clean image", type=["png", "jpg", "jpeg", "bmp"])
    if f is None:
        st.info("Upload an image, or use the synthetic landscape sample.")
        st.stop()
    pil = Image.open(f).convert("L")
    clean = np.array(pil)


# ---------------- Noise injection ----------------
st.sidebar.header("Noise injection")
noise_name = st.sidebar.selectbox("Noise model", list(NOISE_MODELS.keys()))
seed = st.sidebar.number_input("Seed", 0, 99, 1, 1)

# Per-model param
if noise_name == "Gaussian":
    sigma = st.sidebar.slider("σ", 1.0, 80.0, 20.0, 1.0)
    noisy = NOISE_MODELS[noise_name](clean, sigma=sigma, seed=int(seed))
elif noise_name == "Rayleigh":
    scale = st.sidebar.slider("scale", 1.0, 60.0, 25.0, 1.0)
    noisy = NOISE_MODELS[noise_name](clean, scale=scale, seed=int(seed))
elif noise_name == "Gamma (Erlang)":
    shape = st.sidebar.slider("shape (k)", 1.0, 10.0, 2.0, 0.5)
    scale = st.sidebar.slider("scale (θ)", 1.0, 40.0, 15.0, 1.0)
    noisy = NOISE_MODELS[noise_name](clean, shape=shape, scale=scale, seed=int(seed))
elif noise_name == "Exponential":
    scale = st.sidebar.slider("scale", 1.0, 60.0, 20.0, 1.0)
    noisy = NOISE_MODELS[noise_name](clean, scale=scale, seed=int(seed))
elif noise_name == "Uniform":
    bound = st.sidebar.slider("± bound", 1.0, 80.0, 25.0, 1.0)
    noisy = NOISE_MODELS[noise_name](clean, low=-bound, high=bound, seed=int(seed))
elif noise_name == "Salt & Pepper":
    density = st.sidebar.slider("density", 0.0, 0.5, 0.10, 0.01)
    noisy = NOISE_MODELS[noise_name](clean, density=density, seed=int(seed))
else:  # Speckle
    sp_sig = st.sidebar.slider("σ (mult)", 0.0, 0.5, 0.10, 0.01)
    noisy = NOISE_MODELS[noise_name](clean, sigma=sp_sig, seed=int(seed))


# ---------------- Tabs ----------------
tabs = st.tabs(["Inject", "Classify", "Restore", "All-filters grid"])


# ---- Inject ----
with tabs[0]:
    c1, c2 = st.columns(2)
    c1.image(clean, caption="Clean reference", width="stretch", clamp=True)
    c2.image(noisy, caption=f"With {noise_name} noise", width="stretch", clamp=True)
    c3, c4 = st.columns(2)
    c3.metric("PSNR (noisy vs clean)", f"{psnr(clean, noisy):.2f} dB")
    c4.metric("SSIM (noisy vs clean)", f"{ssim(clean, noisy):.3f}")


# ---- Classify ----
with tabs[1]:
    st.subheader("Histogram-feature noise classifier (rule-based, non-ML)")
    cls = classify_noise(noisy)
    st.metric("Predicted noise model", cls["prediction"])
    st.write({"skewness": cls["skew"], "kurtosis": cls["kurt"],
              "spike@0": cls["spike0"], "spike@255": cls["spike255"]})
    st.write("Confidences:")
    st.bar_chart(cls["scores"])

    rec = RECOMMENDED_FILTER.get(cls["prediction"], "Median")
    st.info(f"Recommended filter for {cls['prediction']}: **{rec}**")


# ---- Restore ----
with tabs[2]:
    st.subheader("Spatial-domain restoration")
    filt = st.selectbox("Filter", list(FILTERS.keys()), index=5)
    k = st.slider("Kernel size (odd)", 3, 11, 3, 2)

    extra_kwargs = {}
    if filt == "Contraharmonic":
        Q = st.slider("Q (positive removes pepper, negative removes salt)",
                      -3.0, 3.0, 0.0, 0.1)
        restored = contraharmonic_mean(to_gray(noisy), k, Q=Q)
    elif filt == "Adaptive median":
        smax = st.slider("Max window size", 3, 11, 7, 2)
        with st.spinner("Adaptive median is a per-pixel algorithm — running…"):
            restored = adaptive_median(to_gray(noisy), smax=smax)
    elif filt == "Alpha-trimmed":
        d = st.slider("d (trimmed count)", 0, k * k - 1, 2, 1)
        with st.spinner("Alpha-trimmed mean — running…"):
            restored = alpha_trimmed_mean(to_gray(noisy), k, d=d)
    else:
        restored = FILTERS[filt](to_gray(noisy), k)

    c1, c2, c3 = st.columns(3)
    c1.image(clean, caption="Clean", width="stretch", clamp=True)
    c2.image(noisy, caption="Noisy", width="stretch", clamp=True)
    c3.image(restored, caption=f"Restored ({filt})", width="stretch", clamp=True)

    c4, c5, c6, c7 = st.columns(4)
    c4.metric("PSNR restored", f"{psnr(clean, restored):.2f} dB",
              delta=f"{psnr(clean, restored) - psnr(clean, noisy):+.2f} dB")
    c5.metric("SSIM restored", f"{ssim(clean, restored):.3f}",
              delta=f"{ssim(clean, restored) - ssim(clean, noisy):+.3f}")
    c6.metric("MSE", f"{mse(clean, restored):.1f}")
    c7.metric("MAE", f"{mae(clean, restored):.1f}")


# ---- All-filters grid ----
with tabs[3]:
    st.subheader("Compare every filter at fixed kernel size")
    grid_k = st.slider("Kernel size for grid (odd)", 3, 7, 3, 2, key="grid_k")
    rows = []
    images = []
    for name, fn in FILTERS.items():
        if name == "Adaptive median":
            r = adaptive_median(to_gray(noisy), smax=grid_k if grid_k >= 3 else 3)
        elif name == "Alpha-trimmed":
            r = alpha_trimmed_mean(to_gray(noisy), grid_k, d=max(2, grid_k * grid_k // 4))
        elif name == "Contraharmonic":
            r = contraharmonic_mean(to_gray(noisy), grid_k, Q=0.0)
        else:
            r = fn(to_gray(noisy), grid_k)
        rows.append({"Filter": name, "PSNR (dB)": round(psnr(clean, r), 2),
                     "SSIM": round(ssim(clean, r), 3),
                     "MSE": round(mse(clean, r), 1)})
        images.append((name, r))
    st.dataframe(rows, width="stretch")
    cols = st.columns(4)
    for (name, im), col in zip(images, (cols * 2)[:len(images)]):
        col.image(im, caption=name, width="stretch", clamp=True)


st.markdown("---")
st.caption("NoiseLab · Module 3 · Owner: Asif")
