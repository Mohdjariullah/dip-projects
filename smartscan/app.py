"""Streamlit frontend for SmartScan ID — Document Region Extractor.

Run: streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import cv2
import streamlit as st
from PIL import Image

from pipeline import (
    to_gray, rectify_document, projection_profiles,
    extract_fields, detect_lines, region_grow,
    watershed_segment, watershed_overlay,
)
from synthetic import make_tilted_id, make_straight_id


st.set_page_config(page_title="SmartScan ID — Document Region Extractor", layout="wide")
st.title("SmartScan ID")
st.caption(
    "Module 4 segmentation: Hough rectification → projection-profile fields → "
    "click-to-grow region growing → marker-controlled watershed."
)


# ---------------- Input ----------------
st.sidebar.header("Input")
src = st.sidebar.radio("Source", ["Synthetic: tilted ID", "Synthetic: straight ID", "Upload"], index=0)
if src == "Synthetic: tilted ID":
    bgr = make_tilted_id()
elif src == "Synthetic: straight ID":
    bgr = make_straight_id()
else:
    f = st.sidebar.file_uploader("Upload an ID / receipt photo", type=["png", "jpg", "jpeg"])
    if f is None:
        st.info("Upload an image, or pick a synthetic ID from the sidebar.")
        st.stop()
    pil = Image.open(f).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


tabs = st.tabs(["Rectify + Fields", "Projection Profiles", "Region Growing", "Watershed", "Hough Lines"])


# ---- 1. Rectify + Fields ----
with tabs[0]:
    rectified = rectify_document(bgr)
    fields, annotated = extract_fields(bgr)

    c1, c2 = st.columns(2)
    c1.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Input (possibly tilted)", width="stretch")
    c2.image(cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB),
             caption="Rectified (Hough + perspective warp)", width="stretch")
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
             caption=f"Detected {len(fields)} text-band fields", width="stretch")
    if fields:
        st.dataframe([{
            "Field": f.id, "x": f.bbox[0], "y": f.bbox[1],
            "w": f.bbox[2], "h": f.bbox[3], "height_px": f.height,
        } for f in fields], width="stretch")


# ---- 2. Projection Profiles ----
with tabs[1]:
    rectified = rectify_document(bgr)
    g = to_gray(rectified)
    hp, vp = projection_profiles(g)
    c1, c2 = st.columns(2)
    c1.image(g, caption="Rectified (grayscale)", width="stretch", clamp=True)
    with c2:
        st.line_chart({"horizontal sum (per row)": hp.tolist()},
                      x_label="row index", y_label="ink count")
        st.line_chart({"vertical sum (per col)": vp.tolist()},
                      x_label="column index", y_label="ink count")
    st.caption(
        "Bands in the horizontal profile mark text lines; bands in the vertical "
        "profile mark columns. Smoothing + thresholding the peaks gives the "
        "field boxes shown on the Rectify tab."
    )


# ---- 3. Region Growing ----
with tabs[2]:
    st.subheader("Click-to-grow (interactive)")
    rectified = rectify_document(bgr)
    g = to_gray(rectified)
    h, w = g.shape

    c1, c2, c3 = st.columns(3)
    sx = c1.number_input("Seed x", 0, w - 1, w // 2)
    sy = c2.number_input("Seed y", 0, h - 1, h // 2)
    tol = c3.slider("Tolerance T", 1, 80, 15)
    conn = st.radio("Connectivity", [4, 8], horizontal=True)
    mask = region_grow(g, (int(sx), int(sy)), tolerance=int(tol), connectivity=int(conn))

    overlay = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    overlay[mask > 0] = (0.4 * overlay[mask > 0] +
                        0.6 * np.array([0, 200, 255])).astype(np.uint8)
    cv2.drawMarker(overlay, (int(sx), int(sy)), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)

    c1, c2 = st.columns(2)
    c1.image(g, caption="Rectified grayscale", width="stretch", clamp=True)
    c2.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
             caption=f"Region grown ({(mask > 0).sum()} px)", width="stretch")
    st.caption(
        "BFS-based region growing. Predicate: pixel is within ±T of seed value "
        "AND within ±T of the running region mean — implemented from scratch in "
        "`pipeline.region_grow`."
    )


# ---- 4. Watershed ----
with tabs[3]:
    rectified = rectify_document(bgr)
    dist_t = st.slider("Sure-foreground distance threshold (× max distance)",
                       0.1, 0.9, 0.45, 0.05)
    markers = watershed_segment(rectified, dist_threshold=dist_t)
    overlay = watershed_overlay(rectified, markers)
    n_regions = int(markers.max())

    c1, c2 = st.columns(2)
    c1.image(cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB), caption="Rectified", width="stretch")
    c2.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
             caption=f"Marker-controlled watershed ({n_regions} regions)", width="stretch")
    st.caption(
        "Foreground markers come from peaks of the distance transform on the Otsu "
        "mask. Without the markers, watershed would over-segment every speck "
        "of ink — the marker step is the canonical Module 4 talking point."
    )


# ---- 5. Hough Lines ----
with tabs[4]:
    rectified = rectify_document(bgr)
    g = to_gray(rectified)
    min_len = st.slider("Minimum line length (px)", 20, 300, 60, 5)
    lines = detect_lines(g, min_length=min_len)
    out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for (x1, y1, x2, y2) in lines:
        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
             caption=f"{len(lines)} probabilistic-Hough segments (signature lines, table borders)",
             width="stretch")


st.markdown("---")
st.caption("SmartScan ID · Module 4 · Owner: Asif")
