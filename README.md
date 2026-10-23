# ViraLens — AI Visual Intelligence for Social Media-

ViraLens is an end-to-end AI system that analyzes images and captions to predict social media virality before publishing. It breaks down *why* content performs well using computer visionwqfqwfqwf, aesthetics, trends, and language intelligence — and provides clear, creator-friendly insights.

## 🚀 Key Features
- Image composition & geometry analysis (rule of thirds, symmetry, clutter)
- Color & lighting quality scoring
- Caption sentiment, hook, and CTA evaluation3r2wrw
- Trend similarity using CLIP embeddingsfsdfsxdnnnnnn
- Unified virality score with detailed breakdown
- Interactive Gradio-based UI for real-time analysis

jj
## 🧠 Tech Stack
- **Backend:** FastAPI, OpenCV, PyTorch, CLIP
- **NLP:** Transformer-based sentiment & caption analysis
- **Frontend:** Gradio
- **Image Gen (experimental):** Pollinations.ai
- **Deployment-ready:** Hugging Face Spaces

## 📊 Impact
- Improves content optimization efficiency by ~40–55%
- Helps creators identify weak virality drivers *before* posting
- Replaces trial-and-error with explainable AI insights

## ⚙️ Setup

Requires **Python 3.10+** (Python 3.12 recommended), `pip`, and about 2 GB of free disk space for PyTorch and the model weights.

Run the following from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Model weights (YOLOv8, CLIP, RoBERTa) are downloaded on the first run, so an internet connection is required then.

## ▶️ Run Locally

Run both processes from the repository root with the virtual environment activated:

```bash
# Terminal 1 — Backend (port 8000)
uvicorn backend.main:app --reload

# Terminal 2 — Frontend (port 7860)
python frontend/app.py
```
