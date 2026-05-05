# Traffic Signal Detection

Classical Digital Image Processing project. Detects traffic signals in images
or webcam frames and classifies each as **Red**, **Yellow**, or **Green** —
no deep learning, only OpenCV.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## Pipeline (in `detector.py`)

| Step | Function | DIP technique |
|------|----------|---------------|
| 1 | `to_hsv` | BGR → HSV color-space conversion |
| 2 | `make_color_masks` | HSV `inRange` thresholding (R needs two ranges) |
| 3 | `clean_mask` | Morphological opening + closing |
| 4 | `find_blobs` | `findContours` + area & circularity filter |
| 5 | `detect` | Composes the above and draws bounding boxes |

Why HSV instead of RGB? Hue separates color from brightness, so the same red
bulb is detected whether the photo is bright or dim.

Why circularity? `4πA / P²` is 1.0 for a perfect circle and drops for irregular
shapes — this rejects red shirts, brake lights, etc., which are colored but
not disc-shaped.

## Demo

The sidebar **Synthetic** mode renders a generated traffic-light scene
(`synthetic.py`). Five buttons: Red / Yellow / Green / Multi / Empty. The Empty
button is the edge case — it should detect zero signals and not crash.

The 2x2 image grid shows: Original → HSV → Color masks → Annotated, so the
examiner can see each pipeline stage. HSV sliders in the sidebar update masks
live, which is useful when explaining how thresholding parameters affect the
output.

## Files

- `detector.py` — pure OpenCV pipeline (no Streamlit deps; importable)
- `synthetic.py` — synthetic test-image generator
- `app.py` — Streamlit UI
- `samples/` — pre-generated demo images
