# RadioLens — X-Ray Contrast Studio

Module 2 mini-project. Six **intensity-domain** lenses on a single X-ray —
gamma, log, piecewise contrast stretching, bit-plane slicing, intensity-level
slicing, and Otsu bone isolation. Every transform is a 1-D LUT, so sliders
re-render in real time.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pipeline (in `pipeline.py`)

| Lens | Function | DIP technique |
|------|----------|---------------|
| 1 | `gamma_lut` | Power-law transform `s = 255 · (r/255)^γ` |
| 2 | `log_lut` | Log transform `s = c · log(1 + r)` |
| 3 | `contrast_stretch_lut` | Two-knee piecewise-linear contrast stretching |
| 4 | `all_bit_planes`, `reconstruct_from_planes` | Bit-plane slicing |
| 5 | `intensity_slice` | Intensity-level slicing |
| 6 | `isolate_bone` | Otsu global thresholding + morphological closing |

Each transform is built as a `np.ndarray` LUT and applied via `cv2.LUT` —
this is the only way to keep sliders interactive on large images.

## Why intensity-domain?

Junior radiologists and rural-clinic doctors see X-rays on cheap monitors
where micro-fractures and pneumonia opacities disappear. Re-rendering through
multiple intensity lenses is a zero-cost way to recover diagnostic detail
without segmentation or any DL black box.

## Files

- `pipeline.py` — pure NumPy/OpenCV transforms (no Streamlit deps)
- `synthetic.py` — synthetic chest / hand X-ray generator
- `app.py` — Streamlit UI with 6 lens tabs
- `samples/` — pre-generated demo X-rays

Owner: **Sudeep**.
