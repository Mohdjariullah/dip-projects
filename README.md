# Digital Image Processing — Mini Projects Portfolio

Classical Digital Image Processing mini projects. Ten standalone Streamlit
applications, all OpenCV + NumPy — **no deep learning, no GPUs, no large
datasets**. Each project is a self-contained subfolder you can run on its
own and deploy individually to Streamlit Cloud.

## Owners

- **Asif** — DocuClean, NoiseLab, SmartScan, AgroLeaf, plus the original
  Traffic-Signal Detection.
- **Sudeep** — RadioLens, OrbitRestore, AstroVision, FormShape, plus the
  original Asteroid Detection.

## Module coverage

| Module | Topic | Asif's projects | Sudeep's projects |
|---|---|---|---|
| **2** | Intensity transforms + thresholding | [docuclean](docuclean/) | [radiolens](radiolens/) |
| **3** | Noise models + restoration | [noiselab](noiselab/) | [orbitrestore](orbitrestore/) |
| **4** | Segmentation (point/line/threshold/region) | [smartscan](smartscan/), [traffic_signal_detection](traffic_signal_detection/) | [astrovision](astrovision/), [asteroid_detection](asteroid_detection/) |
| **5** | Colour models + region/boundary descriptors | [agroleaf](agroleaf/) | [formshape](formshape/) |

## The ten projects

| # | Project | Module | Headline technique | Owner |
|---|---|---|---|---|
| 1 | [docuclean](docuclean/) — Adaptive Document Binarizer | M2 | Sauvola & Niblack (integral-image O(1)) | Asif |
| 2 | [radiolens](radiolens/) — X-Ray Contrast Studio | M2 | Bit-plane slicing + 6 intensity lenses | Sudeep |
| 3 | [noiselab](noiselab/) — Noise + Restoration Suite | M3 | Adaptive median (G&W level-A/level-B) | Asif |
| 4 | [orbitrestore](orbitrestore/) — Satellite Restoration | M3 | Wiener filter in frequency domain | Sudeep |
| 5 | [smartscan](smartscan/) — ID Field Extractor | M4 | Projection profiles + region growing | Asif |
| 6 | [astrovision](astrovision/) — Astrophotography Pipeline | M4 | LoG point detection + sigma-clipped sky | Sudeep |
| 7 | [agroleaf](agroleaf/) — HSI Leaf Disease Analyzer | M5 | RGB→HSI from scratch + region descriptors | Asif |
| 8 | [formshape](formshape/) — Industrial Defect Inspector | M5 | Moore tracing + Fourier descriptors | Sudeep |
| 9 | [traffic_signal_detection](traffic_signal_detection/) — Traffic-Light Classifier | M4 | HSV thresholding + circularity filter | Asif |
| 10 | [asteroid_detection](asteroid_detection/) — Moving-Object Detector | M4 | ECC alignment + image differencing | Sudeep |

## Run any project

Each project is independent. From the project's folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

(On Windows: `py -m streamlit run app.py` if `streamlit` isn't on your PATH.)

Every project bundles a **synthetic image generator** (`synthetic.py`) so a
demo always works without external data. The Streamlit sidebar exposes a
"Synthetic" source so you can show the pipeline on a generated reference
image before swapping in uploaded photos.

## Shared conventions

Every project follows the same structure:

```text
<project>/
├── app.py             # Streamlit UI
├── pipeline.py        # pure OpenCV / NumPy algorithms (no Streamlit imports — testable)
│  ↑ or detector.py for the two existing detection-style projects (same role)
├── synthetic.py       # synthetic test-image generator
├── requirements.txt
├── runtime.txt        # python-3.11 (Streamlit Cloud)
├── README.md
└── samples/           # pre-generated demo images
```

The split exists so the algorithms (`pipeline.py` / `detector.py`) are
importable, unit-testable, and viva-defendable as standalone code; the
Streamlit UI in `app.py` is just a thin wrapper.

## Deploy

Each project is a separate **[Streamlit Community Cloud](https://share.streamlit.io)** deployment:

1. Push this repo to GitHub.
2. share.streamlit.io → New app.
3. Repo: this repo. Branch: main. Main file path: `<project>/app.py`.
4. Advanced → Python 3.11 (some wheels lag on the newest Python).
5. Repeat for each project you want online.

Use `opencv-python-headless` (not `opencv-python`) on Streamlit Cloud — it
ships without GUI libs and is ~5× smaller.

## What's NOT here (on purpose)

- No CNNs, no Torch / TensorFlow, no pretrained networks.
- No giant datasets — every demo runs on a synthetic image or a handful of
  bundled sample PNGs.
- No "histogram equalisation app" or "Canny edge detector app" — every project
  is a multi-stage pipeline that produces a useful real-world output, not a
  one-button textbook demo.

## Suggested build order

If building from scratch, this order minimises rework (each new project reuses
patterns from the previous one):

`radiolens → docuclean → agroleaf → noiselab → smartscan → formshape → orbitrestore → astrovision`

Asif owns 1, 2, 3, 5, 7; Sudeep owns the rest.
