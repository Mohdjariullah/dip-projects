"""Streamlit frontend for Traffic Signal Detection.

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

from detector import (
    DEFAULT_HSV_RANGES,
    to_hsv,
    make_color_masks,
    clean_mask,
    annotate,
    colored_mask_overlay,
    count_by_state,
    find_blobs,
    Detection,
)
from synthetic import make_traffic_scene, make_blank_street


try:
    st.set_page_config(page_title="Traffic Signal Detection", layout="wide")
except Exception:
    pass  # parent multipage app already set the page config
st.title("Traffic Signal Detection")
st.caption("Classical DIP pipeline: HSV thresholding -> morphology -> contour & circularity filter.")


# ---------------- Sidebar ----------------
st.sidebar.header("Input")
input_mode = st.sidebar.radio("Source", ["Upload Image", "Synthetic", "Webcam"], index=1)

st.sidebar.header("HSV Ranges")
st.sidebar.caption("OpenCV uses H: 0-179, S/V: 0-255")

def _range_sliders(name: str, defaults: list):
    """Render H/S/V sliders for one color and return a sub-ranges list."""
    with st.sidebar.expander(f"{name.capitalize()} thresholds", expanded=False):
        sub = []
        for i, (lo, hi) in enumerate(defaults):
            st.markdown(f"*Range {i + 1}*")
            h_lo = st.slider(f"H min ({name} {i + 1})", 0, 179, lo[0], key=f"{name}_h_lo_{i}")
            h_hi = st.slider(f"H max ({name} {i + 1})", 0, 179, hi[0], key=f"{name}_h_hi_{i}")
            s_min = st.slider(f"S min ({name} {i + 1})", 0, 255, lo[1], key=f"{name}_s_min_{i}")
            v_min = st.slider(f"V min ({name} {i + 1})", 0, 255, lo[2], key=f"{name}_v_min_{i}")
            sub.append(((h_lo, s_min, v_min), (h_hi, 255, 255)))
        return sub

ranges = {
    "red":    _range_sliders("red",    DEFAULT_HSV_RANGES["red"]),
    "yellow": _range_sliders("yellow", DEFAULT_HSV_RANGES["yellow"]),
    "green":  _range_sliders("green",  DEFAULT_HSV_RANGES["green"]),
}

st.sidebar.header("Blob Filter")
kernel_size  = st.sidebar.slider("Morph kernel size", 1, 15, 5, step=2)
min_area     = st.sidebar.slider("Min blob area (px)", 10, 5000, 80, step=10)
max_area     = st.sidebar.slider("Max blob area (px)", 1000, 100000, 50000, step=500)
min_circ     = st.sidebar.slider("Min circularity (0-1)", 0.0, 1.0, 0.55, step=0.05)


# ---------------- Helpers ----------------
def _to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def _run_pipeline(bgr: np.ndarray):
    hsv = to_hsv(bgr)
    raw = make_color_masks(hsv, ranges)
    cleaned = {c: clean_mask(m, kernel_size) for c, m in raw.items()}
    detections: list[Detection] = []
    for color, mask in cleaned.items():
        for blob in find_blobs(mask, min_area, max_area, min_circ):
            detections.append(Detection(
                state=color, bbox=blob["bbox"],
                area=blob["area"], circularity=blob["circularity"],
            ))
    annotated = annotate(bgr, detections)
    overlay = colored_mask_overlay(cleaned, bgr.shape)
    return hsv, overlay, annotated, detections, cleaned

def _render_grid(bgr: np.ndarray):
    hsv, overlay, annotated, detections, cleaned = _run_pipeline(bgr)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Original (BGR)")
        st.image(_to_rgb(bgr), width="stretch")
        st.subheader("3. Color masks (R+Y+G)")
        st.image(_to_rgb(overlay), width="stretch")
    with col2:
        st.subheader("2. HSV view")
        # display HSV as if it were BGR so the user can see the channel layout
        st.image(_to_rgb(hsv), width="stretch")
        st.subheader("4. Annotated output")
        st.image(_to_rgb(annotated), width="stretch")

    # Per-color individual masks for debugging
    with st.expander("Per-color cleaned masks (debug)"):
        c1, c2, c3 = st.columns(3)
        for col, name in zip((c1, c2, c3), ("red", "yellow", "green")):
            with col:
                st.markdown(f"**{name}**")
                st.image(cleaned[name], width="stretch", clamp=True)

    # Counter
    counts = count_by_state(detections)
    st.markdown(
        f"**Detected — Red: {counts['red']}  Yellow: {counts['yellow']}  Green: {counts['green']}**"
    )

    # Table
    if detections:
        rows = [
            {
                "Signal #": i + 1,
                "State": d.state.upper(),
                "BBox (x,y,w,h)": str(d.bbox),
                "Area (px)": round(d.area, 1),
                "Circularity": round(d.circularity, 3),
            }
            for i, d in enumerate(detections)
        ]
        st.dataframe(rows, width="stretch")
    else:
        st.info("No traffic signals detected. Try lowering Min blob area or Min circularity.")


# ---------------- Main panel ----------------
if input_mode == "Upload Image":
    f = st.file_uploader("Upload a traffic-scene image", type=["png", "jpg", "jpeg", "bmp"])
    if f is not None:
        pil = Image.open(f).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        _render_grid(bgr)
    else:
        st.info("Upload an image, or switch to Synthetic mode for a guaranteed-working demo.")

elif input_mode == "Synthetic":
    c1, c2, c3, c4, c5 = st.columns(5)
    if "synth_bgr" not in st.session_state:
        st.session_state.synth_bgr = make_traffic_scene("red", seed=1)
    if c1.button("Red"):
        st.session_state.synth_bgr = make_traffic_scene("red", seed=1)
    if c2.button("Yellow"):
        st.session_state.synth_bgr = make_traffic_scene("yellow", seed=2)
    if c3.button("Green"):
        st.session_state.synth_bgr = make_traffic_scene("green", seed=3)
    if c4.button("Multi (R+G)"):
        st.session_state.synth_bgr = make_traffic_scene(["red", "green"], seed=4)
    if c5.button("Empty (no signals)"):
        st.session_state.synth_bgr = make_blank_street()
    _render_grid(st.session_state.synth_bgr)

else:  # Webcam — browser-side capture via Streamlit
    st.markdown("Take a photo with your camera; the pipeline runs on the captured frame.")
    snap = st.camera_input("Capture")
    if snap is not None:
        pil = Image.open(snap).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        _render_grid(bgr)
