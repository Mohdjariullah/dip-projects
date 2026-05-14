# SmartScan ID — Document Region Extractor

Module 4 mini-project. Rectifies a tilted ID card with Hough lines + perspective
warp, then segments fields using **projection profiles**, **marker-controlled
watershed**, and **interactive region growing** (BFS from a seed pixel).

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pipeline (in `pipeline.py`)

| Step | Function | DIP technique |
|------|----------|---------------|
| 1 | `rectify_document` | Canny → contour approx → 4-point perspective warp |
| 2 | `projection_profiles` | Horizontal / vertical ink sums |
| 3 | `detect_text_lines` | Smooth-and-threshold the H-profile → text-band ranges |
| 4 | `extract_fields` | Compose 1–3 → numbered field bounding boxes |
| 5 | `region_grow` | BFS seeded flood with intensity-tolerance predicate (4/8 conn) |
| 6 | `watershed_segment` + `watershed_overlay` | Marker-controlled watershed |
| 7 | `detect_lines` | Probabilistic Hough — signature lines / table borders |

## Marker-controlled watershed — why?

Naive watershed treats every regional minimum as a basin and over-segments
catastrophically (every speck of ink becomes its own region). Marker-controlled
watershed seeds **only** the high-confidence foreground regions (peaks of the
distance transform inside the Otsu mask). Everything else is `unknown` and
gets assigned to whichever marker's basin it falls into. This is the canonical
Module 4 demonstration.

```text
gray ──► Otsu ──► distance transform ──► peaks > threshold ──► fg markers
                                                                    │
                                            unknown = sure_bg − sure_fg
                                                                    │
                                                          watershed(gray, markers)
```

## Region growing — implementation

`region_grow` is a hand-written BFS — no `cv2.floodFill` magic. Maintains a
running region mean and accepts a neighbour pixel only if it's within `T` of
**both** the seed value **and** the running mean. This stays connected
(unlike k-means) and adapts gracefully to slowly-changing illumination.

## Files

- `pipeline.py` — rectification, profiles, region growing, watershed, Hough
- `synthetic.py` — generates a mock state-ID card on a tilted background
- `app.py` — Streamlit UI with 5 tabs
- `samples/` — pre-generated tilted/straight ID samples

Owner: **Asif**.
