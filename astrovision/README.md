# AstroVision — Star & Deep-Sky Object Segmentation

Module 4 mini-project. Classical astrophotography pipeline — **point detection
(LoG), line detection (Hough), local thresholding (sigma-clipped sky),
region growing (galaxies/nebulae)** — in one Streamlit app. Outputs a star
catalogue CSV, removes satellite trails, and segments extended objects.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pipeline (in `pipeline.py`)

| Step | Function | DIP technique |
|------|----------|---------------|
| 0 | `asinh_stretch`, `percentile_stretch` | Display stretch (Module 2 carry-over) |
| 1 | `sigma_clipped_stats` | Robust sky median + std (iterative outlier rejection) |
| 2 | `log_response` | Laplacian-of-Gaussian = canonical Module 4 point detector |
| 3 | `find_peaks` | Local maxima above `k·σ_sky` via `cv2.dilate` |
| 4 | `_subpixel_centroid` | Sub-pixel refinement (parabolic / Gaussian fit) |
| 5 | `detect_stars` | Multi-scale LoG + photometry + ellipticity |
| 6 | `classify_extended` | FWHM + ellipticity → star vs galaxy/nebula |
| 7 | `region_grow_extended` | BFS region growing with sigma-clipped predicate |
| 8 | `detect_trails`, `inpaint_trails` | Hough lines on bright mask → Telea inpaint |

## Why every step maps to Module 4

| Sub-topic | Where it shows up |
|---|---|
| **Point detection** | `log_response` is exactly the Laplacian point detector from the textbook |
| **Line detection** | `detect_trails` uses `cv2.HoughLinesP` on a thresholded bright mask |
| **Local / adaptive thresholding** | `sigma_clipped_stats` produces a robust local-ish threshold; trail & extended detectors both use it |
| **Region growing** | `region_grow_extended` is a hand-written BFS — no `cv2.floodFill` shortcut |

## Sub-pixel centroiding — why bother?

Integer-pixel detection of a Gaussian PSF has a ±0.5 px error in each axis.
A 2-D first-moment centroid on the intensity patch reaches ~0.05 px on bright
stars. This is the difference between astrometry that's usable and astrometry
that's noise — and it's two lines of NumPy, not a deep network.

## Files

- `pipeline.py` — stretch, sky stats, LoG, peaks, centroid, photometry, region grow, trails, inpaint
- `synthetic.py` — synthetic Gaussian-star fields, with optional extended object + satellite trail
- `app.py` — Streamlit UI (Stretch / Stars / Extended / Trails / Catalogue)
- `samples/` — pre-generated star fields

Owner: **Sudeep**.
