# Dance Practice Overlay (Streamlit)

This folder contains the Streamlit app for selecting a practice segment from a video and rendering a pose overlay with audio.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run streamlit_app.py
```

## Notes

- The app uses `ffmpeg` to merge audio into the rendered segment. If `ffmpeg` is not installed, the video still renders but will have no audio.
- The “UpperBody overlay” mode mirrors the logic from `UpperBody_LiveOverlay.py` in the original project.
