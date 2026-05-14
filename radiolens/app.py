"""Streamlit frontend for RadioLens — X-ray Contrast Enhancement Studio.

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
import matplotlib.pyplot as plt

from pipeline import (
    gamma_lut, log_lut, contrast_stretch_lut, piecewise_lut,
    intensity_slice, all_bit_planes, reconstruct_from_planes,
    isolate_bone, apply, transformation_curve_points,
)
from synthetic import make_chest_xray, make_hand_xray


try:
    st.set_page_config(page_title="RadioLens — X-Ray Contrast Studio", layout="wide")
except Exception:
    pass  # parent multipage app already set the page config
st.title("RadioLens")
st.caption(
    "Module 2 intensity-domain X-ray contrast studio. Six lenses — gamma, log, "
    "piecewise, bit-plane, intensity-slicing, and bone isolation — each a LUT."
)


# ---------------- Input ----------------
st.sidebar.header("Input")
src = st.sidebar.radio("Source", ["Synthetic: chest", "Synthetic: hand", "Upload"], index=0)

if src == "Synthetic: chest":
    img = make_chest_xray()
elif src == "Synthetic: hand":
    img = make_hand_xray()
else:
    f = st.sidebar.file_uploader("Upload X-ray image", type=["png", "jpg", "jpeg", "bmp", "tif"])
    if f is None:
        st.info("Upload an X-ray, or use the synthetic chest/hand samples in the sidebar.")
        st.stop()
    pil = Image.open(f).convert("L")
    img = np.array(pil)


def _curve_fig(lut: np.ndarray, title: str):
    xs, ys = transformation_curve_points(lut)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot(xs, ys, lw=2)
    ax.plot([0, 255], [0, 255], "k--", lw=0.6, alpha=0.5)
    ax.set_xlim(0, 255); ax.set_ylim(0, 255)
    ax.set_xlabel("input intensity r"); ax.set_ylabel("output s")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    return fig


def _hist_fig(arr: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.hist(arr.ravel(), bins=64, color="#3b6ea5")
    ax.set_xlim(0, 255)
    ax.set_title(title, fontsize=9)
    return fig


tabs = st.tabs(["Gamma", "Log", "Piecewise", "Bit-plane", "Intensity slicing", "Bone isolation"])


# ---- 1. Gamma ----
with tabs[0]:
    st.subheader("Power-law (gamma) transform")
    gamma = st.slider("gamma", 0.10, 3.00, 0.50, 0.05)
    lut = gamma_lut(gamma)
    out = apply(img, lut)
    c1, c2, c3 = st.columns([2, 2, 1.5])
    c1.image(img, caption="Original", width="stretch", clamp=True)
    c2.image(out, caption=f"gamma = {gamma:.2f}", width="stretch", clamp=True)
    c3.pyplot(_curve_fig(lut, "Mapping"))
    st.expander("Why this matters").markdown(
        "gamma < 1 expands dark tones → reveals **soft-tissue / lung texture**. "
        "gamma > 1 expands bright tones → reveals **bone interior** detail."
    )


# ---- 2. Log ----
with tabs[1]:
    st.subheader("Log transform")
    lut = log_lut()
    out = apply(img, lut)
    c1, c2, c3 = st.columns([2, 2, 1.5])
    c1.image(img, caption="Original", width="stretch", clamp=True)
    c2.image(out, caption="log(1 + r), auto-scaled", width="stretch", clamp=True)
    c3.pyplot(_curve_fig(lut, "Mapping"))
    st.expander("Why this matters").markdown(
        "Compresses bright values, expands dark values. Useful when the "
        "histogram is heavily skewed to bright pixels (overexposed X-ray)."
    )


# ---- 3. Piecewise / contrast stretching ----
with tabs[2]:
    st.subheader("Piecewise-linear contrast stretching")
    c_l, c_r = st.columns(2)
    r1 = c_l.slider("r1 (low knee in)", 0, 254, 70)
    s1 = c_r.slider("s1 (low knee out)", 0, 255, 0)
    r2 = c_l.slider("r2 (high knee in)", r1 + 1, 255, 180)
    s2 = c_r.slider("s2 (high knee out)", 0, 255, 255)
    lut = contrast_stretch_lut(r1, s1, r2, s2)
    out = apply(img, lut)
    c1, c2, c3 = st.columns([2, 2, 1.5])
    c1.image(img, caption="Original", width="stretch", clamp=True)
    c2.image(out, caption=f"stretch ({r1},{s1})→({r2},{s2})", width="stretch", clamp=True)
    c3.pyplot(_curve_fig(lut, "Mapping"))
    st.expander("Why this matters").markdown(
        "Two-knee linear map. Setting (r1,s1)=(p_low,0), (r2,s2)=(p_high,255) "
        "spreads any middle intensity band across the full 0–255 range — the "
        "textbook contrast-stretching demo."
    )


# ---- 4. Bit-plane ----
with tabs[3]:
    st.subheader("Bit-plane slicing")
    planes = all_bit_planes(img)
    cols = st.columns(8)
    for k, col in enumerate(cols):
        col.image(planes[k], caption=f"plane {k}{' (MSB)' if k == 7 else ' (LSB)' if k == 0 else ''}",
                  width="stretch", clamp=True)
    sel = st.multiselect("Reconstruct from these planes", list(range(8)),
                         default=[5, 6, 7])
    if sel:
        out = reconstruct_from_planes(img, sel)
        c1, c2 = st.columns(2)
        c1.image(img, caption="Original", width="stretch", clamp=True)
        c2.image(out, caption=f"Reconstructed from planes {sel}", width="stretch", clamp=True)
    st.expander("Why this matters").markdown(
        "Higher-order planes (5,6,7 = MSBs) carry coarse anatomy. Low planes "
        "(0,1,2 = LSBs) are mostly sensor noise. Reconstructing from only the "
        "top 3–4 planes typically preserves diagnostic content with ~50% less data."
    )


# ---- 5. Intensity slicing ----
with tabs[4]:
    st.subheader("Intensity-level slicing")
    lo, hi = st.slider("Highlight intensities in [lo, hi]", 0, 255, (180, 255))
    preserve = st.radio("Mode", ["Preserve background", "Binary highlight"], horizontal=True) \
        == "Preserve background"
    out = intensity_slice(img, lo, hi, preserve_bg=preserve)
    c1, c2 = st.columns(2)
    c1.image(img, caption="Original", width="stretch", clamp=True)
    c2.image(out, caption=f"sliced [{lo}, {hi}]", width="stretch", clamp=True)
    st.expander("Why this matters").markdown(
        "Isolates a specific intensity band — e.g., setting [180, 255] highlights "
        "bone-bright regions on a chest X-ray. Binary-highlight mode is the "
        "starting point for downstream segmentation."
    )


# ---- 6. Bone isolation ----
with tabs[5]:
    st.subheader("Bone isolation (Otsu + morphology)")
    use_manual = st.checkbox("Use manual threshold instead of Otsu")
    if use_manual:
        t = st.slider("Threshold", 0, 255, 160)
        mask, overlay = isolate_bone(img, threshold=t)
    else:
        mask, overlay = isolate_bone(img, threshold=None)
    c1, c2, c3 = st.columns(3)
    c1.image(img, caption="Original", width="stretch", clamp=True)
    c2.image(mask, caption="Binary mask", width="stretch", clamp=True)
    c3.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Red overlay", width="stretch")
    st.expander("Why this matters").markdown(
        "Otsu picks the threshold that maximises between-class variance — the "
        "standard global thresholding answer for bimodal histograms. The closing "
        "step fills small holes inside the segmented bone region."
    )


st.markdown("---")
st.caption("RadioLens · Module 2 · Owner: Sudeep")
