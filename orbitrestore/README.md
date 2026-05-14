# OrbitRestore — Satellite/Aerial Image Restoration

Module 3 mini-project. Three independent classical restoration pipelines for
satellite & aerial imagery: **frequency-domain deblurring** (inverse, Wiener,
Lucy-Richardson), **CCD-row destriping**, and **dark-channel-prior dehaze**.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pipeline (in `pipeline.py`)

### Deblurring

We model the degradation as `g = h * f + n`. Three classical inverters:

| Method | Frequency-domain expression | Best for |
|---|---|---|
| Inverse (regularised) | `F̂ = G / (H + ε)` | High-SNR cases; explodes if H≈0 |
| Wiener | `F̂ = (H* / (|H|² + K)) · G` | Mixed blur + noise — workhorse |
| Lucy-Richardson | `f_{n+1} = f_n · (h^T * (g / (h * f_n)))` | Poisson noise, e.g. sensor counts |

All three are implemented as full 2-D FFTs in `pipeline.py` (`np.fft.fft2`).
The PSF is zero-padded and `roll`-centred so its phase doesn't translate the
recovered image.

### Destripe

Defective CCD rows cause horizontal stripes. We compute the **per-row mean**
profile, smooth it with a 1-D median filter, and subtract the difference —
a row-level realisation of "spatial mean filter" applied along the time
axis. Cleaner than median-filtering the whole image (which would blur).

### Dehaze (He, Sun, Tang 2009)

Atmospheric scattering model: `I = J·t + A·(1-t)`. Recover `J` from a single
hazy `I`:

1. Compute the **dark channel**: `min over RGB → min over patch`. Hazy
   regions have a brighter dark channel.
2. Estimate atmospheric light `A` from the brightest 0.1 % of dark-channel
   pixels.
3. Transmission: `t(x) = 1 - ω · darkChannel(I/A)`.
4. Recover: `J = (I - A) / max(t, t_min) + A`.

No DL, no training — pure classical CV. Patch size, `ω`, and `t_min` are all
exposed as sliders.

## Files

- `pipeline.py` — PSFs, FFT-based deblur, destripe, dehaze
- `synthetic.py` — synthetic aerial scene + hazy variant
- `app.py` — Streamlit UI (Deblur / Destripe / Dehaze)
- `samples/` — pre-generated aerials

Owner: **Sudeep**.
