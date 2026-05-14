"""Unified multipage entry point for the entire DIP mini-projects portfolio.

Deploys all 8 new projects (and the 2 original ones) as a single Streamlit
app with sidebar navigation. Each page is the project's existing app.py —
no duplication, no copy-pasted UI.

Streamlit Cloud setup:
    Main file path:    streamlit_app.py
    Branch:            main
    Python version:    3.11

Local run:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import sys

import streamlit as st


# Make sure every project subfolder is on sys.path so each app.py can do
# `from pipeline import ...` without caring about the working directory.
_ROOT = os.path.dirname(os.path.abspath(__file__))
for _proj in (
    "radiolens", "docuclean", "agroleaf", "noiselab",
    "smartscan", "formshape", "orbitrestore", "astrovision",
    "traffic_signal_detection", "asteroid_detection",
):
    _path = os.path.join(_ROOT, _proj)
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)


# Landing-page content for when no project is selected yet
def landing() -> None:
    st.title("DIP Mini-Projects Portfolio")
    st.markdown(
        """
        Ten classical Digital Image Processing apps in one deployment.
        Pick a project from the sidebar to launch its interactive demo.
        No deep learning — pure OpenCV + NumPy.

        | Module | Project | Headline algorithm |
        |---|---|---|
        | M2 | **RadioLens** | LUT-based intensity transforms + bit-plane slicing |
        | M2 | **DocuClean Pro** | Sauvola / Niblack via integral-image |
        | M3 | **NoiseLab** | Adaptive median + 7 noise generators |
        | M3 | **OrbitRestore** | Wiener filter (FFT) + dark-channel dehaze |
        | M4 | **SmartScan ID** | Projection profiles + watershed + region growing |
        | M4 | **AstroVision** | LoG point detection + sigma-clipped sky |
        | M5 | **AgroLeaf** | RGB→HSI from scratch + region descriptors |
        | M5 | **FormShape Inspector** | Moore tracing + Fourier descriptors |

        The two original projects from this repo (Traffic Signal Detection
        and Asteroid Detection) are also available in the sidebar.
        """
    )
    st.info(
        "Every project bundles a **synthetic** image generator, so a demo "
        "always works without uploading anything. Look for the 'Source' "
        "radio in each project's sidebar."
    )


pages = {
    "Home": [
        st.Page(landing, title="Portfolio overview", icon="🏠", default=True),
    ],
    "Module 2 — Intensity & Thresholding": [
        st.Page("radiolens/app.py", title="RadioLens — X-Ray Studio", icon="🦴"),
        st.Page("docuclean/app.py", title="DocuClean Pro — Doc Binarizer", icon="📄"),
    ],
    "Module 3 — Restoration": [
        st.Page("noiselab/app.py", title="NoiseLab — Noise & Restoration", icon="📡"),
        st.Page("orbitrestore/app.py", title="OrbitRestore — Satellite Restoration", icon="🛰️"),
    ],
    "Module 4 — Segmentation": [
        st.Page("smartscan/app.py", title="SmartScan ID — Document Fields", icon="🪪"),
        st.Page("astrovision/app.py", title="AstroVision — Astrophotography", icon="🌌"),
        st.Page("traffic_signal_detection/app.py", title="Traffic Signal Detection", icon="🚦"),
        st.Page("asteroid_detection/app.py", title="Asteroid Detection", icon="☄️"),
    ],
    "Module 5 — Colour & Descriptors": [
        st.Page("agroleaf/app.py", title="AgroLeaf — HSI Disease Analyzer", icon="🌿"),
        st.Page("formshape/app.py", title="FormShape — Defect Inspector", icon="⚙️"),
    ],
}

pg = st.navigation(pages, position="sidebar")
pg.run()
