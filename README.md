# Digital Image Processing — Two Mini-Projects

College DIP project. Two standalone Python applications using **OpenCV** for the image-processing pipeline and **Streamlit** for the frontend. No deep learning — pure classical DIP techniques.

## Projects

| Project | What it does | Live demo |
|---|---|---|
| [traffic_signal_detection/](traffic_signal_detection/) | Detect & classify traffic lights as Red / Yellow / Green using HSV thresholding, morphology, and contour analysis | _add Streamlit Cloud URL here_ |
| [asteroid_detection/](asteroid_detection/) | Find moving objects between two sky images using ECC alignment, image differencing, and connected components | _add Streamlit Cloud URL here_ |

## Run locally

Install Python 3.10+, then for each project:

```bash
cd traffic_signal_detection      # or asteroid_detection
pip install -r requirements.txt
streamlit run app.py
```

(On Windows you may need `py -m streamlit run app.py` if `streamlit` isn't on your PATH.)

Each project includes a **synthetic test-image generator** so a demo always works without real data.

## Deploy

Both apps are designed for **[Streamlit Community Cloud](https://share.streamlit.io)** (free). Each project is a separate deployment:
1. Push this repo to GitHub
2. Go to share.streamlit.io → New app
3. Repo: this repo, Branch: main, Main file path: `traffic_signal_detection/app.py` (or `asteroid_detection/app.py`)
4. Set Python version to 3.11 in advanced settings (3.14 is too new for some wheels on the cloud)

Repeat for the second app.
