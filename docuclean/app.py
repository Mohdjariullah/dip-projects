"""Streamlit frontend for DocuClean — Adaptive Document Binarization Studio.

Run: streamlit run app.py
"""

from __future__ import annotations

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# When this file is loaded as a page inside the unified streamlit_app.py,
# Python caches "pipeline"/"synthetic"/"detector" from whichever project
# was visited first. Clear them so this page imports its OWN local copies.
for _stale in ("pipeline", "synthetic", "detector"):
    sys.modules.pop(_stale, None)

import numpy as np
import cv2
import streamlit as st
from PIL import Image

from pipeline import (
    to_gray, gamma_correct, remove_shading,
    thresh_otsu, thresh_adaptive_gaussian,
    thresh_niblack, thresh_sauvola, thresh_wolf,
    despeckle, morph_clean,
    foreground_ratio, component_count, diff_overlay,
)
from synthetic import make_faded_document, make_low_contrast_doc


try:
    st.set_page_config(page_title="DocuClean — Adaptive Document Binarizer", layout="wide")
except Exception:
    pass  # parent multipage app already set the page config
st.title("DocuClean Pro")
st.caption(
    "Module 2 adaptive thresholding studio. Sauvola and Niblack implemented "
    "from scratch via box-filter integral images (O(1) per pixel)."
)


# ---------------- Input ----------------
st.sidebar.header("Input")
src = st.sidebar.radio("Source", ["Synthetic: faded", "Synthetic: low-contrast", "Upload"], index=0)

if src == "Synthetic: faded":
    bgr = make_faded_document()
elif src == "Synthetic: low-contrast":
    bgr = make_low_contrast_doc()
else:
    f = st.sidebar.file_uploader("Upload a document", type=["png", "jpg", "jpeg", "bmp", "tif"])
    if f is None:
        st.info("Upload a document, or pick a synthetic sample from the sidebar.")
        st.stop()
    pil = Image.open(f).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

gray = to_gray(bgr)


# ---------------- Pre-stage ----------------
st.sidebar.header("Pre-stage")
gamma = st.sidebar.slider("Gamma correction", 0.3, 2.5, 1.0, 0.05)
shade = st.sidebar.slider("Shading-removal kernel (0 = off)", 0, 121, 51, 2)

g = gray
if abs(gamma - 1.0) > 1e-3:
    g = gamma_correct(g, gamma)
if shade >= 3:
    g = remove_shading(g, shade)


# ---------------- Per-method params ----------------
st.sidebar.header("Local window")
window = st.sidebar.slider("Window size", 5, 75, 25, 2)

st.sidebar.header("Niblack")
niblack_k = st.sidebar.slider("Niblack k", -1.0, 1.0, -0.2, 0.05)

st.sidebar.header("Sauvola")
sauvola_k = st.sidebar.slider("Sauvola k", 0.0, 0.8, 0.20, 0.02)
sauvola_R = st.sidebar.slider("Sauvola R", 32, 256, 128, 1)

st.sidebar.header("Adaptive Gaussian")
adaptive_C = st.sidebar.slider("Adaptive C (offset)", -30, 30, 10, 1)

st.sidebar.header("Wolf")
wolf_k = st.sidebar.slider("Wolf k", 0.0, 1.0, 0.5, 0.05)

st.sidebar.header("Post-process")
desp = st.sidebar.slider("Despeckle min component area", 0, 80, 6, 1)
close_k = st.sidebar.slider("Closing kernel (1 = off)", 1, 9, 1, 2)


# ---------------- Compute all 5 methods ----------------
def _post(bw: np.ndarray) -> np.ndarray:
    if desp > 0:
        bw = despeckle(bw, desp)
    return morph_clean(bw, close_k)

results = {
    "Otsu": _post(thresh_otsu(g)),
    "Adaptive Gaussian": _post(thresh_adaptive_gaussian(g, window, adaptive_C)),
    "Niblack": _post(thresh_niblack(g, window, niblack_k)),
    "Sauvola": _post(thresh_sauvola(g, window, sauvola_k, sauvola_R)),
    "Wolf-Jolion": _post(thresh_wolf(g, window, wolf_k)),
}


# ---------------- Display ----------------
tabs = st.tabs(["5-up grid", "Tuner", "Diff overlay", "Metrics"])

with tabs[0]:
    st.markdown("**Same image, five thresholding methods.** "
                "Otsu is global; the other four are local/adaptive.")
    c0, c1 = st.columns(2)
    c0.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Original (with degradation)", width="stretch")
    c1.image(g, caption="After pre-stage (gamma + shading removal)", width="stretch", clamp=True)

    cols = st.columns(5)
    for col, (name, bw) in zip(cols, results.items()):
        col.image(bw, caption=name, width="stretch", clamp=True)

with tabs[1]:
    method = st.selectbox("Tune this method", list(results.keys()), index=3)
    bw = results[method]
    c1, c2 = st.columns(2)
    c1.image(g, caption="Pre-stage output", width="stretch", clamp=True)
    c2.image(bw, caption=f"{method} → binarized", width="stretch", clamp=True)

with tabs[2]:
    c1, c2 = st.columns(2)
    name_a = c1.selectbox("Method A", list(results.keys()), index=0)
    name_b = c2.selectbox("Method B", list(results.keys()), index=3)
    overlay = diff_overlay(results[name_a], results[name_b])
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
             caption=f"Disagreement: red = ink only in {name_a}; blue = ink only in {name_b}",
             width="stretch")

with tabs[3]:
    rows = []
    for name, bw in results.items():
        rows.append({
            "Method": name,
            "Foreground (ink) %": round(100 * foreground_ratio(bw), 2),
            "Connected components": component_count(bw),
        })
    st.dataframe(rows, width="stretch")
    st.caption(
        "Foreground % near 0 or 100 = the method collapsed. "
        "Connected components is a rough OCR-friendliness proxy — too many = noise, "
        "too few = broken/merged characters."
    )


st.markdown("---")
st.caption("DocuClean · Module 2 · Owner: Asif")
