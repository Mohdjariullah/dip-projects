"""Streamlit frontend for AstroVision — Star & Deep-Sky Object Segmentation.

Run: streamlit run app.py
"""

from __future__ import annotations

import io
import numpy as np
import cv2
import streamlit as st
from PIL import Image

from pipeline import (
    to_gray, asinh_stretch, percentile_stretch,
    sigma_clipped_stats,
    detect_stars, classify_extended, annotate_stars,
    region_grow_extended,
    detect_trails, inpaint_trails,
)
from synthetic import make_star_field, make_satellite_streaked_frame


st.set_page_config(page_title="AstroVision — Star & Sky Segmenter", layout="wide")
st.title("AstroVision")
st.caption(
    "Module 4 multi-stage astrophotography pipeline — point detection (LoG), "
    "line detection (Hough trails), local thresholding (sigma-clipped sky), "
    "and region growing (galaxies / nebulae)."
)


# ---------------- Input ----------------
st.sidebar.header("Input")
src = st.sidebar.radio("Source", ["Synthetic: star field",
                                  "Synthetic: with satellite trail",
                                  "Upload"], index=0)

if src == "Synthetic: star field":
    raw = make_star_field()
elif src == "Synthetic: with satellite trail":
    raw = make_satellite_streaked_frame()
else:
    f = st.sidebar.file_uploader("Upload astrophoto", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if f is None:
        st.info("Upload a frame, or pick a synthetic sample. Astrophotos look black "
                "until they're stretched — that's normal.")
        st.stop()
    pil = Image.open(f).convert("L")
    raw = np.array(pil)


# ---------------- Stretch ----------------
st.sidebar.header("Display stretch")
stretch_kind = st.sidebar.radio("Stretch", ["asinh", "percentile", "none"], horizontal=True)
if stretch_kind == "asinh":
    soft = st.sidebar.slider("asinh softening", 0.005, 0.20, 0.05, 0.005)
    stretched = asinh_stretch(raw, soft=soft)
elif stretch_kind == "percentile":
    lo = st.sidebar.slider("low %", 0.0, 5.0, 1.0, 0.1)
    hi = st.sidebar.slider("high %", 95.0, 100.0, 99.5, 0.1)
    stretched = percentile_stretch(raw, low=lo, high=hi)
else:
    stretched = raw


# Sky stats panel
sky_med, sky_std = sigma_clipped_stats(raw)
st.sidebar.markdown(f"**Sky median:** {sky_med:.2f}  \n**Sky std:** {sky_std:.2f}")


tabs = st.tabs(["Stretch", "Stars", "Extended Objects", "Trail Cleanup", "Catalogue"])


# ---- 1. Stretch ----
with tabs[0]:
    c1, c2 = st.columns(2)
    c1.image(raw, caption="Raw (pre-stretch — often very dark)", width="stretch", clamp=True)
    c2.image(stretched, caption=f"After {stretch_kind} stretch", width="stretch", clamp=True)


# ---- 2. Stars ----
with tabs[1]:
    c1, c2, c3 = st.columns(3)
    k_thresh = c1.slider("Detection threshold (k · sky_std)", 2.0, 12.0, 6.0, 0.5)
    min_dist = c2.slider("Min separation (px)", 2, 12, 4, 1)
    aperture = c3.slider("Photometry aperture (px)", 3, 12, 6, 1)

    stars = detect_stars(raw, k_thresh=k_thresh, min_distance=min_dist, aperture=aperture)
    annotated = annotate_stars(stretched, stars)

    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
             caption=f"Detected {len(stars)} sources "
                     f"(green = star, magenta = galaxy/nebula candidate)",
             width="stretch")


# ---- 3. Extended Objects ----
with tabs[2]:
    st.subheader("Seeded region growing for galaxies / nebulae")
    h, w = raw.shape
    c1, c2, c3 = st.columns(3)
    sx = c1.number_input("Seed x", 0, w - 1, int(w * 0.65))
    sy = c2.number_input("Seed y", 0, h - 1, int(h * 0.55))
    kk = c3.slider("k · sky_std for grow", 0.5, 6.0, 1.5, 0.1)
    mask = region_grow_extended(raw, (int(sx), int(sy)), k_sigma=kk)

    overlay = cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)
    overlay[mask > 0] = (0.4 * overlay[mask > 0] +
                        0.6 * np.array([200, 80, 255])).astype(np.uint8)
    cv2.drawMarker(overlay, (int(sx), int(sy)), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)

    c1, c2 = st.columns(2)
    c1.image(stretched, caption="Stretched", width="stretch", clamp=True)
    c2.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
             caption=f"Region grown ({(mask>0).sum()} px)", width="stretch")
    st.caption(
        "Predicate: I(p) > sky_median + k·sky_std. The sky stats are computed by "
        "iterative sigma-clipping, so single bright stars in the patch don't "
        "skew the cutoff."
    )


# ---- 4. Trail Cleanup ----
with tabs[3]:
    c1, c2 = st.columns(2)
    k_trail = c1.slider("Brightness threshold (k · sky_std)", 2.0, 10.0, 4.0, 0.5)
    min_len = c2.slider("Min trail length (px)", 30, 300, 80, 5)
    trails = detect_trails(raw, k_sigma=k_trail, min_length=min_len)
    annotated_trails = cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in trails:
        cv2.line(annotated_trails, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cleaned = inpaint_trails(raw, trails, thickness=5)
    cleaned_view = asinh_stretch(cleaned, 0.05) if stretch_kind == "asinh" else cleaned

    c1, c2 = st.columns(2)
    c1.image(cv2.cvtColor(annotated_trails, cv2.COLOR_BGR2RGB),
             caption=f"Detected {len(trails)} trail segments", width="stretch")
    c2.image(cleaned_view, caption="Cleaned (Telea inpaint)", width="stretch", clamp=True)


# ---- 5. Catalogue ----
with tabs[4]:
    stars = detect_stars(raw, k_thresh=k_thresh, min_distance=min_dist, aperture=aperture)
    if not stars:
        st.info("No stars detected — try lowering the detection threshold.")
    else:
        rows = [{
            "ID": s.id, "x": round(s.x, 2), "y": round(s.y, 2),
            "peak": round(s.peak, 1), "flux": round(s.flux, 1),
            "fwhm": round(s.fwhm, 2), "ecc": round(s.ellipticity, 3),
            "class": classify_extended(s),
        } for s in stars]
        st.dataframe(rows, width="stretch")

        # CSV download
        import csv
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
        st.download_button("Download catalogue CSV", buf.getvalue(),
                           file_name="astrovision_catalogue.csv", mime="text/csv")


st.markdown("---")
st.caption("AstroVision · Module 4 · Owner: Sudeep")
