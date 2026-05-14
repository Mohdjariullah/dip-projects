"""Streamlit frontend for AgroLeaf — HSI plant disease analyzer.

Run: streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import cv2
import streamlit as st
from PIL import Image

from pipeline import (
    rgb_to_hsi, hsi_to_display, leaf_mask,
    disease_mask, lesion_descriptors, severity_score,
    hsv_disease_mask, render_overlay,
)
from synthetic import make_diseased_leaf, make_healthy_leaf


st.set_page_config(page_title="AgroLeaf — HSI Plant Disease Analyzer", layout="wide")
st.title("AgroLeaf")
st.caption(
    "Module 5 HSI-space leaf disease segmentation. RGB→HSI implemented from "
    "scratch using the Gonzalez & Woods textbook formulas — no `cv2.cvtColor` "
    "for HSI."
)


# ---------------- Input ----------------
st.sidebar.header("Input")
src = st.sidebar.radio("Source", ["Synthetic: diseased", "Synthetic: healthy", "Upload"], index=0)

if src == "Synthetic: diseased":
    bgr = make_diseased_leaf()
elif src == "Synthetic: healthy":
    bgr = make_healthy_leaf()
else:
    f = st.sidebar.file_uploader("Upload a leaf image", type=["png", "jpg", "jpeg", "bmp"])
    if f is None:
        st.info("Upload a leaf photo, or use the synthetic samples in the sidebar.")
        st.stop()
    pil = Image.open(f).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ---------------- HSI params ----------------
st.sidebar.header("Healthy-hue window")
h_lo = st.sidebar.slider("H low (×2π)", 0.0, 1.0, 0.20, 0.01)
h_hi = st.sidebar.slider("H high (×2π)", 0.0, 1.0, 0.42, 0.01)
s_min = st.sidebar.slider("S min", 0.0, 1.0, 0.15, 0.01)
min_lesion = st.sidebar.slider("Min lesion area (px)", 1, 500, 40, 1)


# ---------------- Tabs ----------------
tabs = st.tabs(["HSI Explorer", "Disease Map", "Descriptors", "HSI vs HSV"])


# ---- 1. HSI Explorer ----
with tabs[0]:
    st.subheader("RGB → HSI (from scratch)")
    hsi = rgb_to_hsi(bgr)
    H, S, I = hsi_to_display(hsi)
    c1, c2, c3, c4 = st.columns(4)
    c1.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Original (BGR)", width="stretch")
    c2.image(H, caption="H (Hue, [0, 1) normalised)", width="stretch", clamp=True)
    c3.image(S, caption="S (Saturation)", width="stretch", clamp=True)
    c4.image(I, caption="I (Intensity = (R+G+B)/3)", width="stretch", clamp=True)
    st.expander("The textbook formulas").markdown(r"""
    $$I = \tfrac{R+G+B}{3}, \quad S = 1 - \tfrac{3}{R+G+B}\min(R,G,B)$$

    $$\theta = \cos^{-1}\!\left(\tfrac{0.5\bigl[(R-G)+(R-B)\bigr]}
    {\sqrt{(R-G)^2 + (R-B)(G-B)}}\right)$$

    $$H = \begin{cases}\theta & B \le G\\ 2\pi-\theta & B > G\end{cases}$$
    """)


# ---- 2. Disease Map ----
with tabs[1]:
    dmask, lmask = disease_mask(bgr, h_lo, h_hi, s_min, min_lesion)
    lesions = lesion_descriptors(dmask)
    sev = severity_score(lmask, dmask, lesions)
    overlay = render_overlay(bgr, dmask, lesions)

    c1, c2 = st.columns(2)
    c1.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Original", width="stretch")
    c2.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
             caption=f"Disease overlay ({sev['lesion_count']} lesions)", width="stretch")

    c3, c4, c5 = st.columns(3)
    c3.metric("Affected area", f"{sev['percent_affected']}%")
    c4.metric("Lesion count", sev["lesion_count"])
    c5.metric("Risk", sev["risk"])

    c6, c7 = st.columns(2)
    c6.image(lmask, caption="Leaf bounding mask (Lab a* + Otsu)", width="stretch", clamp=True)
    c7.image(dmask, caption="Disease mask", width="stretch", clamp=True)


# ---- 3. Descriptors ----
with tabs[2]:
    dmask, _ = disease_mask(bgr, h_lo, h_hi, s_min, min_lesion)
    lesions = lesion_descriptors(dmask)
    if not lesions:
        st.info("No lesions found at the current settings.")
    else:
        rows = [{
            "ID": l.id, "Area": l.area, "Perimeter": round(l.perimeter, 1),
            "Compactness": round(l.compactness, 3),
            "Eccentricity": round(l.eccentricity, 3),
            "Solidity": round(l.solidity, 3),
            "Holes": l.holes, "Euler": l.euler,
            "BBox": str(l.bbox),
        } for l in lesions]
        st.dataframe(rows, width="stretch")
        st.caption(
            "Compactness = P²/(4πA): 1.0 = perfect disc, larger = more elongated/irregular. "
            "Eccentricity ∈ [0, 1): 0 = circle, near 1 = line. "
            "Euler number = 1 − holes (per Module 5 region descriptors)."
        )


# ---- 4. HSI vs HSV ----
with tabs[3]:
    st.subheader("Why HSI, not HSV?")
    dmask_hsi, _ = disease_mask(bgr, h_lo, h_hi, s_min, min_lesion)
    dmask_hsv = hsv_disease_mask(bgr)
    c1, c2 = st.columns(2)
    c1.image(dmask_hsi, caption="Disease mask via HSI (custom)", width="stretch", clamp=True)
    c2.image(dmask_hsv, caption="Disease mask via OpenCV HSV", width="stretch", clamp=True)
    st.markdown(
        "**HSI** uses `I = (R+G+B)/3` and a hue from inverse-cosine geometry. "
        "**HSV** uses `V = max(R,G,B)`. HSI is closer to perceptual brightness; "
        "HSV is faster but distorts intensity for highly-saturated colours."
    )


st.markdown("---")
st.caption("AgroLeaf · Module 5 · Owner: Asif")
