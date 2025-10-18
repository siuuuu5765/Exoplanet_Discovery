# Exoplanet Discovery Hub 🚀
A Streamlit web app that detects and analyzes potential exoplanets from TESS lightcurves.

## Features
- Lightcurve upload & automatic transit detection
- Planet radius estimation
- Phase-folded lightcurve visualization
- AI chatbot (OpenAI-powered)
- Candidate database with export

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Cloud
- Upload to GitHub
- Add your `OPENAI_API_KEY` under `App → Settings → Secrets`
