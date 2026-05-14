# DocuClean Pro — Adaptive Document Binarizer

Module 2 mini-project. Binarizes degraded documents (faded ink, coffee stains,
uneven illumination, photocopy noise) using five thresholding algorithms side
by side, with **Sauvola and Niblack written from scratch** via the integral-image
box-filter trick.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pipeline (in `pipeline.py`)

| Stage | Function | DIP technique |
|------|----------|---------------|
| Pre-1 | `gamma_correct` | Power-law intensity transform |
| Pre-2 | `remove_shading` | Morphological-opening background estimation, subtraction |
| Thr-1 | `thresh_otsu` | Global thresholding (Otsu, between-class variance) |
| Thr-2 | `thresh_adaptive_gaussian` | Local adaptive mean with Gaussian weights |
| Thr-3 | `thresh_niblack` | `T = μ + k·σ` in a local window |
| Thr-4 | `thresh_sauvola` | `T = μ·(1 + k·(σ/R − 1))` — robust to uneven illumination |
| Thr-5 | `thresh_wolf` | Wolf-Jolion adapt-to-image-min variant |
| Post-1 | `despeckle` | Remove tiny connected components |
| Post-2 | `morph_clean` | Optional closing to repair broken strokes |

## The integral-image trick

Niblack and Sauvola both need the **local mean and standard deviation** in a
window around every pixel. A naive implementation is `O(N·w²)` (N pixels, window
size w). Using `cv2.boxFilter` (mean) and `cv2.sqrBoxFilter` (mean of squares)
on the integral image makes it `O(N)` — independent of window size. This is what
lets the Streamlit window-size slider re-render in real time.

```python
mean    = cv2.boxFilter(g, cv2.CV_32F, (w, w), normalize=True)
mean_sq = cv2.sqrBoxFilter(g, cv2.CV_32F, (w, w), normalize=True)
std     = np.sqrt(np.maximum(mean_sq - mean*mean, 0))
```

## Why these algorithms?

| Algorithm | Strength | Weakness |
|----------|---------|---------|
| Otsu | Fast, parameter-free | Collapses on uneven illumination |
| Adaptive Gaussian | Built-in, decent | Sensitive to window size |
| Niblack | Sharp text edges | Noisy in low-contrast background regions |
| **Sauvola** | Robust to shading; clean background | Slightly slower than Niblack |
| Wolf-Jolion | Sauvola + adapts to image min | Needs whole-image stats (less local) |

## Files

- `pipeline.py` — pure NumPy/OpenCV thresholding algorithms (no Streamlit deps)
- `synthetic.py` — degraded-document generator (faded ink + coffee + shading + noise)
- `app.py` — Streamlit 5-up grid, tuner, diff overlay, metrics
- `samples/` — pre-generated degraded documents

Owner: **Asif**.
