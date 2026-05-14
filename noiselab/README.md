# NoiseLab Forensics — Noise Models + Restoration

Module 3 mini-project. Inject any of seven analytical noise models, run any
of eight spatial-domain restoration filters, and judge recovery with PSNR/SSIM/
MSE/MAE. **Adaptive median** is implemented per Gonzalez & Woods (level-A /
level-B); a rule-based **noise-type classifier** picks the right filter from
histogram statistics alone (no ML).

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pipeline (in `pipeline.py`)

### Noise models (7)

| Model | PDF tendency | Function |
|---|---|---|
| Gaussian | symmetric, additive | `add_gaussian` |
| Rayleigh | right-skewed | `add_rayleigh` |
| Gamma (Erlang) | right-skewed, heavy tail | `add_gamma` |
| Exponential | right-skewed | `add_exponential` |
| Uniform | flat | `add_uniform` |
| Salt-and-pepper | impulse spikes at 0/255 | `add_salt_pepper` |
| Speckle | multiplicative Gaussian | `add_speckle` |

### Restoration filters (8)

| Filter | Best for | Function |
|---|---|---|
| Arithmetic mean | uniform / gaussian | `arithmetic_mean` |
| Geometric mean | edge-preserving for gaussian | `geometric_mean` |
| Harmonic mean | salt noise (kills bright outliers) | `harmonic_mean` |
| Contraharmonic mean (Q) | salt OR pepper depending on sign of Q | `contraharmonic_mean` |
| Median | salt-and-pepper up to ~20% | `median_filter` |
| **Adaptive median** | salt-and-pepper up to ~80% | `adaptive_median` |
| Midpoint | uniform / gaussian | `midpoint_filter` |
| Alpha-trimmed mean | mixed gaussian + impulse | `alpha_trimmed_mean` |

### The adaptive median algorithm (Gonzalez & Woods 5.3.3)

**Per pixel:**

1. **Level A** — In a window of size S compute `z_min`, `z_med`, `z_max`.
   If `z_min < z_med < z_max` go to B (median is not an impulse — trust it).
   Else expand `S ← S + 2`; repeat until `S > S_max`, then output `z_med`.
2. **Level B** — Let `z_xy` be the current pixel. If `z_min < z_xy < z_max`,
   output `z_xy` (it's not an impulse, preserve it). Otherwise output `z_med`.

This is the viva-defendable centrepiece: it beats plain median because it
keeps the window small for clean regions and expands it only when needed.

### Noise classifier

Computes histogram skewness, kurtosis, and impulse spikes at intensities 0
and 255, then runs a rule-based scorer. Returns predicted noise type +
recommended filter from `RECOMMENDED_FILTER`.

## Files

- `pipeline.py` — noise generators, restoration filters, metrics, classifier
- `synthetic.py` — synthetic clean landscape reference image
- `app.py` — Streamlit UI (Inject / Classify / Restore / All-filters grid)
- `samples/` — pre-generated clean reference

Owner: **Asif**.
