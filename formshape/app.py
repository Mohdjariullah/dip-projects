"""Streamlit frontend for FormShape Inspector — boundary descriptors.

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
import matplotlib.pyplot as plt
from PIL import Image

from pipeline import (
    silhouette, moore_boundary, chain_code_4, chain_code_8,
    chain_first_difference, fourier_descriptors, shape_signature,
    fd_distance, render_boundary,
)
from synthetic import PARTS, hex_screw, gear


try:
    st.set_page_config(page_title="FormShape Inspector", layout="wide")
except Exception:
    pass  # parent multipage app already set the page config
st.title("FormShape Inspector")
st.caption(
    "Module 5 boundary descriptors — Moore-neighbour tracing, Freeman chain "
    "codes, and Fourier descriptors implemented from scratch."
)


# ---------------- Input ----------------
st.sidebar.header("Reference part")
ref_kind = st.sidebar.selectbox("Reference shape", list(PARTS.keys()), index=0)
ref_rot = st.sidebar.slider("Reference rotation (deg)", -90, 90, 0, 1)

st.sidebar.header("Test part")
test_kind = st.sidebar.selectbox("Test shape", list(PARTS.keys()), index=0, key="test_kind")
test_rot = st.sidebar.slider("Test rotation (deg)", -90, 90, 0, 1, key="test_rot")
test_defect = st.sidebar.checkbox("Inject defect on test part", value=True)

st.sidebar.header("Descriptor")
n_coef = st.sidebar.slider("FD coefficients to keep", 4, 64, 24, 1)
threshold = st.sidebar.slider("Defect score threshold", 0.0, 2.0, 0.5, 0.05)


# Generate parts
ref_img = PARTS[ref_kind](rotation=float(ref_rot), defect=False)
test_img = PARTS[test_kind](rotation=float(test_rot), defect=bool(test_defect))


# Run pipeline on both
def _process(img):
    sil = silhouette(img)
    b = moore_boundary(sil)
    return sil, b


ref_sil, ref_b = _process(ref_img)
test_sil, test_b = _process(test_img)

ref_fd = fourier_descriptors(ref_b, n_coef=n_coef)
test_fd = fourier_descriptors(test_b, n_coef=n_coef)
dist, per_coef = fd_distance(ref_fd, test_fd)
verdict = "FAIL (defective)" if dist > threshold else "PASS"


# ---------------- Tabs ----------------
tabs = st.tabs(["Boundaries", "Chain code", "Shape signature", "Fourier descriptors", "Verdict"])


# ---- 1. Boundaries ----
with tabs[0]:
    c1, c2 = st.columns(2)
    c1.image(ref_sil, caption=f"Reference silhouette ({len(ref_b)} boundary pts)", width="stretch", clamp=True)
    c1.image(render_boundary(ref_sil.shape, ref_b, (0, 220, 0)),
             caption="Moore-traced boundary (green)", width="stretch")
    c2.image(test_sil, caption=f"Test silhouette ({len(test_b)} boundary pts)", width="stretch", clamp=True)
    c2.image(render_boundary(test_sil.shape, test_b, (0, 220, 255)),
             caption="Moore-traced boundary (cyan)", width="stretch")


# ---- 2. Chain code ----
with tabs[1]:
    cc8_ref = chain_code_8(ref_b)
    cc8_test = chain_code_8(test_b)
    cc4_ref = chain_code_4(ref_b)
    fd_ref = chain_first_difference(cc8_ref)

    c1, c2 = st.columns(2)
    c1.markdown("**8-conn Freeman code (reference)** — first 80 codes")
    c1.code(" ".join(map(str, cc8_ref[:80].tolist())))
    c1.markdown("**First-difference (rotation invariant under 45°)** — first 80")
    c1.code(" ".join(map(str, fd_ref[:80].tolist())))
    c2.markdown("**8-conn Freeman code (test)** — first 80 codes")
    c2.code(" ".join(map(str, cc8_test[:80].tolist())))
    c2.markdown("**4-conn Freeman code (reference)** — first 80 codes")
    c2.code(" ".join(map(str, cc4_ref[:80].tolist())))


# ---- 3. Shape signature ----
with tabs[2]:
    sig_ref = shape_signature(ref_b)
    sig_test = shape_signature(test_b)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(sig_ref, label="reference", lw=2)
    ax.plot(sig_test, label="test", lw=2, alpha=0.8)
    ax.set_xlabel("angle index (0..359)"); ax.set_ylabel("r(θ) / mean(r)")
    ax.set_title("Radial-distance shape signature")
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig)
    st.caption(
        "Periodic deviations indicate symmetry; sharp local dips on the test "
        "curve are usually defects (chips, notches)."
    )


# ---- 4. Fourier descriptors ----
with tabs[3]:
    fig, ax = plt.subplots(figsize=(8, 3))
    idx = np.arange(len(ref_fd))
    ax.bar(idx - 0.18, ref_fd, width=0.36, label="reference", color="#3b6ea5")
    ax.bar(idx + 0.18, test_fd, width=0.36, label="test", color="#e88a3c", alpha=0.9)
    ax.set_xlabel("FD coefficient k"); ax.set_ylabel("|F_k| (normalised)")
    ax.set_title("Fourier descriptor magnitudes")
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.bar(idx, per_coef, color="#a44")
    ax2.set_xlabel("FD coefficient k"); ax2.set_ylabel("(ref - test)²")
    ax2.set_title("Per-coefficient distance contribution")
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)
    st.caption(
        "FDs are translation/scale/rotation invariant after normalisation, so "
        "matching parts of any orientation should give near-zero distance. "
        "Defects show up as spikes at specific k."
    )


# ---- 5. Verdict ----
with tabs[4]:
    c1, c2, c3 = st.columns(3)
    c1.metric("FD distance", f"{dist:.4f}")
    c2.metric("Threshold", f"{threshold:.2f}")
    c3.metric("Verdict", verdict)
    st.progress(min(1.0, dist / max(threshold, 1e-3)))
    st.caption(
        "Verdict = `dist > threshold ⇒ FAIL`. The threshold is engineering-tuned "
        "on a batch of known-OK parts in production."
    )


st.markdown("---")
st.caption("FormShape Inspector · Module 5 · Owner: Sudeep")
