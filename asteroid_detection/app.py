"""Streamlit frontend for Asteroid Detection.

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

from detector import detect_pair
from synthetic import make_sky_pair


try:
    st.set_page_config(page_title="Asteroid Detection", layout="wide")
except Exception:
    pass  # parent multipage app already set the page config
st.title("Asteroid Detection")
st.caption("Two sky images -> ECC align -> absdiff -> threshold -> morphology -> connected components.")


# ---------------- Sidebar ----------------
st.sidebar.header("Input")
input_mode = st.sidebar.radio(
    "Source",
    ["Synthetic Pair", "Upload Pair", "Webcam (two captures)"],
    index=0,
)

st.sidebar.header("Pipeline parameters")
blur_k       = st.sidebar.slider("Pre-blur kernel", 1, 11, 3, step=2)
threshold_mode = st.sidebar.radio(
    "Threshold", ["manual", "otsu"], index=0, horizontal=True,
    help="Manual works best for sparse change (asteroid). Otsu works better when changes cover a large area.",
)
manual_thresh  = st.sidebar.slider("Manual threshold", 0, 255, 30, disabled=(threshold_mode == "otsu"))
morph_k      = st.sidebar.slider("Morph kernel", 1, 11, 3, step=2)
min_area     = st.sidebar.slider("Min blob area (px)", 1, 200, 4)
max_area     = st.sidebar.slider("Max blob area (px)", 50, 5000, 500)
show_overlay = st.sidebar.checkbox("Show alignment overlay (debug)", value=False)

st.sidebar.header("Synthetic options")
num_stars        = st.sidebar.slider("Stars", 10, 300, 80)
ast_off_x        = st.sidebar.slider("Asteroid offset X (px)", 1, 60, 15)
ast_off_y        = st.sidebar.slider("Asteroid offset Y (px)", -30, 30, 8)
drift_px         = st.sidebar.slider("Telescope drift (px)", 0.0, 8.0, 1.5, step=0.1)
drift_deg        = st.sidebar.slider("Telescope drift (deg)", 0.0, 3.0, 0.3, step=0.1)


# ---------------- Helpers ----------------
def _to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _render(img_a: np.ndarray, img_b: np.ndarray, ground_truth: tuple = None):
    result = detect_pair(
        img_a, img_b,
        blur_k=blur_k,
        threshold_mode=threshold_mode,
        manual_threshold=manual_thresh,
        morph_kernel=morph_k,
        min_area=min_area,
        max_area=max_area,
    )

    # 2x3 grid
    r1c1, r1c2, r1c3 = st.columns(3)
    r2c1, r2c2, r2c3 = st.columns(3)
    with r1c1:
        st.subheader("1. Image A")
        st.image(_to_rgb(img_a), width="stretch")
    with r1c2:
        st.subheader("2. Image B")
        st.image(_to_rgb(img_b), width="stretch")
    with r1c3:
        st.subheader("3. B aligned to A")
        st.image(_to_rgb(result["b_aligned"]), width="stretch")
    with r2c1:
        st.subheader("4. Difference (HOT colormap)")
        st.image(_to_rgb(result["diff_heatmap"]), width="stretch")
    with r2c2:
        st.subheader("5. Threshold + morphology")
        st.image(result["cleaned"], width="stretch", clamp=True)
    with r2c3:
        st.subheader("6. Detected candidates")
        st.image(_to_rgb(result["annotated"]), width="stretch")

    if show_overlay:
        with st.expander("Alignment overlay (A in red, B-aligned in cyan)"):
            a3 = cv2.cvtColor(result["a_gray"], cv2.COLOR_GRAY2BGR)
            b3 = cv2.cvtColor(result["b_aligned"], cv2.COLOR_GRAY2BGR)
            a3[:, :, 0] = 0; a3[:, :, 1] = 0          # leave red channel
            b3[:, :, 2] = 0                            # leave green+blue (cyan)
            overlay = cv2.add(a3, b3)
            st.image(_to_rgb(overlay), width="stretch")
            st.caption(f"Alignment ECC score: {result['alignment_score']:.4f} (1.0 = perfect)")

    cands = result["candidates"]
    st.markdown(f"**Detected {len(cands)} candidate object(s).**")
    if ground_truth is not None and cands:
        gx, gy = ground_truth
        # closest detection to ground truth
        dists = [
            (c.idx, np.hypot(c.centroid[0] - gx, c.centroid[1] - gy))
            for c in cands
        ]
        idx, d = min(dists, key=lambda t: t[1])
        st.markdown(f"_Ground truth asteroid in A: ({gx:.1f}, {gy:.1f}). Nearest detection #{idx} is {d:.1f} px away._")

    if cands:
        rows = [
            {
                "#": c.idx,
                "Centroid (x,y)": f"({c.centroid[0]:.1f}, {c.centroid[1]:.1f})",
                "BBox (x,y,w,h)": str(c.bbox),
                "Area (px)": c.area,
                "Mean brightness": round(c.mean_brightness, 1),
            }
            for c in cands
        ]
        st.dataframe(rows, width="stretch")
    else:
        st.info("No candidates. If you expect a moving object, lower Min blob area or threshold, or check that A != B.")


# ---------------- Main panel ----------------
if input_mode == "Synthetic Pair":
    if st.button("Generate new synthetic pair"):
        st.session_state.synth_seed = int(np.random.randint(1, 10_000))
    if "synth_seed" not in st.session_state:
        st.session_state.synth_seed = 42

    a, b, gt = make_sky_pair(
        num_stars=num_stars,
        asteroid_offset=(ast_off_x, ast_off_y),
        telescope_drift_px=drift_px,
        telescope_drift_deg=drift_deg,
        seed=st.session_state.synth_seed,
    )
    st.caption(f"Seed: {st.session_state.synth_seed}. Ground truth asteroid pos in A: ({gt[0]:.1f}, {gt[1]:.1f}).")
    _render(a, b, ground_truth=gt)

elif input_mode == "Upload Pair":
    c1, c2 = st.columns(2)
    fa = c1.file_uploader("Image A", type=["png", "jpg", "jpeg", "tif", "fits", "bmp"])
    fb = c2.file_uploader("Image B", type=["png", "jpg", "jpeg", "tif", "fits", "bmp"])
    if fa is not None and fb is not None:
        a = np.array(Image.open(fa).convert("L"))
        b = np.array(Image.open(fb).convert("L"))
        _render(a, b)
    else:
        st.info("Upload two images of the same star field, or switch to Synthetic Pair.")

else:  # Webcam (two captures via st.camera_input)
    st.markdown(
        "Take two photos a moment apart. Move a small object (e.g. your hand)"
        " between captures — the detector will treat it as the 'asteroid'."
    )
    c1, c2 = st.columns(2)
    snap_a = c1.camera_input("Capture A", key="cam_a")
    snap_b = c2.camera_input("Capture B", key="cam_b")

    if snap_a is not None and snap_b is not None:
        a = np.array(Image.open(snap_a).convert("L"))
        b = np.array(Image.open(snap_b).convert("L"))
        _render(a, b)
    else:
        st.info("Capture both A and B to run the pipeline.")
