# AgroLeaf — HSI Plant Disease Analyzer

Module 5 mini-project. RGB → HSI conversion written from scratch (textbook
formulas, **not** `cv2.cvtColor`), HSI-space disease segmentation, and per-lesion
region descriptors (area, perimeter, compactness, eccentricity, solidity, Euler
number) with a leaf-level severity score.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pipeline (in `pipeline.py`)

| Step | Function | DIP technique |
|------|----------|---------------|
| 1 | `rgb_to_hsi` | RGB → HSI conversion (Gonzalez & Woods Eq. 6.2-2) |
| 2 | `leaf_mask` | Lab a*-channel + Otsu → bounding mask for the leaf |
| 3 | `healthy_mask` / `disease_mask` | HSI hue+sat thresholding for green tissue; disease = leaf ∩ ¬healthy |
| 4 | `lesion_descriptors` | Per-component area / perimeter / compactness / eccentricity / solidity / Euler |
| 5 | `severity_score` | Affected % + risk class (Healthy / Mild / Moderate / Severe) |
| 6 | `hsv_disease_mask` | Same segmentation in OpenCV HSV for the HSI-vs-HSV comparison panel |

## The RGB → HSI math

$$I = \tfrac{R+G+B}{3} \qquad S = 1 - \tfrac{3}{R+G+B}\min(R,G,B)$$

$$\theta = \cos^{-1}\!\left(\tfrac{0.5\bigl[(R-G)+(R-B)\bigr]}
{\sqrt{(R-G)^2 + (R-B)(G-B)}}\right) \qquad
H = \begin{cases}\theta & B \le G\\ 2\pi-\theta & B > G\end{cases}$$

Implemented in `rgb_to_hsi` with NaN-safe handling on pure black pixels and
`cos⁻¹` argument clipped to `[-1, 1]` for numerical safety.

## Region descriptors

| Descriptor | Formula | Meaning |
|----------|--------|---------|
| Area | pixel count | size |
| Perimeter | `cv2.arcLength` | boundary length |
| Compactness | P² / (4πA) | 1.0 = circle, >1 = elongated/irregular |
| Eccentricity | from central moments, λ₁, λ₂ | 0 = circle, →1 = line |
| Solidity | A / convex-hull-area | 1.0 = convex shape, <1 = concave |
| Euler number | 1 − holes | topology of the lesion |

## Why HSI, not HSV?

HSI's *intensity* is `(R+G+B)/3` — close to perceptual brightness. HSV's *value*
is `max(R,G,B)` — fast but distorts brightness for highly-saturated colours.
For thresholding leaf tissue against lesions under variable lighting, HSI is
more stable. The HSI-vs-HSV tab demonstrates this directly.

## Files

- `pipeline.py` — RGB→HSI, segmentation, descriptors, severity scoring
- `synthetic.py` — generates a green leaf with brown/yellow lesions
- `app.py` — Streamlit UI (HSI Explorer / Disease Map / Descriptors / HSI vs HSV)
- `samples/` — pre-generated leaf images

Owner: **Asif**.
