# FormShape Inspector — Boundary Descriptor Defect Detection

Module 5 mini-project. Industrial-QC tool: extract a part silhouette, trace
its boundary with **Moore neighbour tracing** (from scratch), compute **Freeman
chain codes** (4-conn and 8-conn) and **Fourier descriptors** (FFT of the
boundary as a complex sequence), and decide PASS / FAIL by FD-space distance.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pipeline (in `pipeline.py`)

| Step | Function | DIP technique |
|------|----------|---------------|
| 1 | `silhouette` | Otsu + largest connected component |
| 2 | `moore_boundary` | Moore-neighbour tracing (from scratch) |
| 3a | `chain_code_8`, `chain_code_4` | Freeman chain codes |
| 3b | `chain_first_difference` | Rotation invariance under 45° / 90° |
| 4 | `fourier_descriptors` | FFT of complex boundary, normalised for T/S/R/start-point |
| 5 | `shape_signature` | Radial distance r(θ) vs angle |
| 6 | `fd_distance` | Euclidean distance between FD vectors + per-coef breakdown |

## Moore-neighbour tracing — algorithm

```text
1. Find top-most, left-most foreground pixel s.    boundary ← [s]
2. Record back-direction = WEST (we entered s from the west).
3. Loop:
     a. Walk the 8-neighbourhood clockwise from `back`.
     b. First foreground neighbour found is the next boundary pixel.
     c. The new `back` is the opposite of that step.
     d. If we revisited s, stop.
```

Implemented in `moore_boundary` — viva-defendable single function.

## Fourier descriptor invariances

The boundary is treated as a complex sequence `z[n] = x + j·y`. Take FFT → `F[k]`.

| Invariance | How it's achieved in `fourier_descriptors` |
|---|---|
| Translation | discard `F[0]` (the centroid) |
| Scale | divide by `|F[1]|` |
| Rotation | take magnitudes `|F[k]|` |
| Start point | resample boundary to fixed length before the FFT |

The bar chart in the UI lets you watch the FD coefficients line up between
identical parts at different rotations — that's the headline demonstration.

## Files

- `pipeline.py` — silhouette → Moore → chain codes → FDs → distance
- `synthetic.py` — generates hex-screw, gear, bottle silhouettes (with optional defects)
- `app.py` — Streamlit UI (Boundaries / Chain code / Signature / FDs / Verdict)
- `samples/` — six pre-generated parts (3 reference, 3 defective)

Owner: **Sudeep**.
