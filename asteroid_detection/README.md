# Asteroid Detection

Classical Digital Image Processing project. Given two images of the same patch
of sky taken at different times, find any object that **moved** — a candidate
asteroid (stars stay fixed, asteroids drift). No deep learning, only OpenCV.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## Pipeline (in `detector.py`)

| Step | Function | DIP technique |
|------|----------|---------------|
| 1 | `preprocess` | Grayscale + Gaussian blur (denoise) |
| 2 | `align` | ECC (`findTransformECC`) intensity-based alignment, with ORB+RANSAC as fallback |
| 3 | `difference` | `absdiff` — stars cancel, movers leave a residue |
| 4 | `threshold` | Manual (default 30) or Otsu's method |
| 5 | `morphological_clean` | Opening — kills 1-pixel cosmic-ray hits |
| 6 | `find_candidates` | `connectedComponentsWithStats` — each blob is a candidate |
| 7 | filter | Min/max area filter |
| 8 | `annotate` | Circle and number each candidate on Image A |

Why alignment? Telescope shifts slightly between exposures. Without alignment,
**every star** would look like it moved when you `absdiff` the two frames.

Why ECC instead of ORB? Star fields have no distinctive corners — every star
looks the same — so ORB feature matching fails. ECC works on pixel intensities
directly and aligns sparse-feature images much better. ORB is kept as a
fallback for natural images (e.g. the webcam mode where you point at a
textured scene).

Threshold choice: with good alignment the difference image is mostly black
with a few bright residues. Otsu's method assumes a balanced bimodal
histogram, so when the foreground is sparse (typical for asteroid detection)
Otsu picks a near-zero threshold and noise dominates. **Manual threshold (~30)
is the default**. Switch to Otsu when the change region is large (e.g. a hand
moving through a webcam frame).

## Demo

The sidebar **Synthetic Pair** mode generates two star-field images with a
known asteroid offset and a tiny telescope drift. The pipeline should recover
the asteroid every time. The "Generate new synthetic pair" button reseeds for
a fresh demo.

The 2×3 image grid shows: Image A → Image B → B aligned to A → difference
heatmap → threshold + morphology → final detection. So the examiner sees every
DIP stage at once.

The candidates table lists centroid, bounding box, area, and mean brightness
of each detection. When using synthetic mode, the app also reports the
distance from the nearest detection to the ground-truth asteroid position.

## Edge cases

- **Identical images** (A = B): difference is all zeros; should detect 0
  candidates and not crash.
- **Bad alignment** (no matching features): `align` returns the original B
  unchanged, so most stars will appear "moved". A real symptom of this in the
  UI: the difference heatmap is bright everywhere instead of just at the
  asteroid.

## Files

- `detector.py` — pure OpenCV pipeline (no Streamlit deps; importable)
- `synthetic.py` — synthetic star-field pair generator
- `app.py` — Streamlit UI
- `samples/` — pre-generated demo images
