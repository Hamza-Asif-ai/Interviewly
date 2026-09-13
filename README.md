---
title: Interviewly
emoji: 🎤
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Interviewly — AI Interview Buddy & Performance Analyzer

A multimodal career-readiness platform with two engines:

1. **Prep Engine** — upload your CV + a target job description + company overview to get
   an exportable prep guide: tailored technical & behavioral questions, STAR model
   answers grounded in *your* CV, company-specific positioning tips, and a skill-gap
   analysis.
2. **Analysis Engine** — record or upload a practice interview video and receive a
   diagnostic scorecard across **Content (40%)**, **Speech (30%)**, and **Visual (30%)**
   with strengths, weaknesses, and concrete improvement tips.

---

## Quick Start

### 1. Local (recommended)

Requires **Python 3.11**.

```bash
# 1. (optional but recommended) create a virtual environment
python -m venv venv
# Windows: venv\Scripts\activate      # macOS/Linux: source venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. configure environment
cp .env.example .env
#   → set OPENAI_API_KEY (required for LLM question generation & review)

# 4. run
python run.py
```

Open <http://localhost:7860>.

> **Note:** the Prep Engine also needs the local Whisper model to not be required; the
> Prep tab only needs `OPENAI_API_KEY`. The Practice tab falls back to deterministic
> scoring when no API key / local Whisper is available.

### 2. Docker

```bash
cp .env.example .env          # set OPENAI_API_KEY
docker compose up --build     # or: docker build -t interviewly . && docker run -p 7860:7860 interviewly
```

Open <http://localhost:7860>.

### 3. Hugging Face Spaces (Docker SDK)

1. Create a new Space with **Docker** as the SDK (or set `sdk: docker`).
2. Push this repository (or its contents) to the Space repo.
3. Add `OPENAI_API_KEY` as a Space secret.
4. The Space builds from the included `Dockerfile` and serves on port `7860`
   (`app_port: 7860` is already set in the metadata at the top of this README).

---

## Demo Notebook

`Interviewly_Demo.ipynb` is a self-contained, zero-error walkthrough that launches the real
backend (`app/main.py`), runs the full pipeline end-to-end, and renders every output
(metrics tables, charts, scorecards, PDF, live API endpoints) as it would appear in the app.
It has been executed end-to-end with `nbconvert` and verifies cleanly.

```bash
# install the notebook runtime + plotting libs (on top of requirements.txt)
pip install jupyter nbformat matplotlib pandas

# run interactively
jupyter notebook Interviewly_Demo.ipynb

# or re-execute headlessly (default per-cell timeout is 30s, so allow more)
jupyter nbconvert --to notebook --execute Interviewly_Demo.ipynb \
  --output Interviewly_Demo_executed.ipynb \
  --ExecutePreprocessor.timeout=1800
```

> Everything degrades gracefully: when the LLM is offline/rate-limited or Whisper is
> missing, the notebook switches to bundled demo data (`data/prep_guide_fallback.json`, a
> pre-saved transcript) so it still runs clean on any machine.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  http://localhost:7860  (Gradio UI, mounted on FastAPI at /)   │
│  ┌─────────────────────────┐   ┌────────────────────────────┐  │
│  │  Tab 1: Prep Engine     │   │  Tab 2: Practice Engine    │  │
│  │  CV + JD + Company      │   │  video + question          │  │
│  └─────────────┬───────────┘   └──────────────┬─────────────┘  │
└────────────────┼──────────────────────────────┼────────────────┘
                 │  HTTP                        │ HTTP multipart
┌────────────────▼──────────────────────────────▼────────────────┐
│                   FastAPI (/prep, /practice)                   │
├────────────────────────────────────────────────────────────────┤
│  PrepEngine          Analysis pipeline                        │
│   - cv_parser         - ffmpeg → 16kHz mono WAV               │
│   - chunk + embed     - Whisper → transcript                   │
│   - ChromaDB RAG      - text_analysis (fillers, STAR, rel.)    │
│   - LLM guide JSON     - audio_analysis (librosa)              │
│   - fpdf2 → PDF       - video_analysis (mediapipe + OpenCV)    │
│                       - scoring (40/30/30) → review via LLM    │
├────────────────────────────────────────────────────────────────┤
│                SQLite (SQLAlchemy)  +  data/                   │
│        users · sessions · prep_guides · practice_videos ·      │
│        analysis_reports                                        │
└────────────────────────────────────────────────────────────────┘
```

---

## API Endpoints

| Method | Endpoint                     | Description                                  |
|--------|------------------------------|----------------------------------------------|
| GET    | `/health`                    | Health check                                |
| POST   | `/prep/generate`             | Generate a prep guide (multipart form)      |
| GET    | `/prep/{id}`                 | Fetch a saved prep guide                    |
| POST   | `/practice/upload`           | Upload a video → analysis scorecard         |
| GET    | `/practice/{video_id}/report`| Fetch a saved analysis report               |

Example (curl):

```bash
curl -X POST http://localhost:7860/prep/generate \
  -F "cv_file=@resume.pdf" \
  -F "job_description=Backend engineer with FastAPI experience" \
  -F "company_overview=Startup building AI interview tools"
```

---

## Scorecard Model

- **Content 40%** — STAR structure, relevance to the question, conciseness, filler-word
  penalty (um, uh, like, basically, you know, matlab, yaani).
- **Speech 30%** — speaking rate (WPM), pause ratio, pitch variation, energy variance,
  clarity proxy.
- **Visual 30%** — face-detection rate, eye contact (frontal head-pose cone), posture
  stability (shoulder tilt variance), head-movement score.

## Tests

```bash
pytest tests/
```

## Project Structure

```
interviewly/
├── app/                    # backend application
│   ├── main.py             # FastAPI app + mounted Gradio UI
│   ├── config.py           # environment settings
│   ├── database.py         # SQLAlchemy engine/session
│   ├── models.py           # ORM models
│   ├── schemas.py          # Pydantic schemas
│   ├── services/           # cv_parser, prep_engine, transcription,
│   │                       # text/audio/video analysis, scoring
│   ├── routers/            # prep.py, practice.py
│   └── ui/gradio_app.py    # two-tab Gradio interface
├── data/                   # uploads & Chroma persistence (gitignored)
├── tests/                  # pytest suite
├── Interviewly_Demo.ipynb  # zero-error end-to-end demo notebook
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── run.py                  # entrypoint (uvicorn on 0.0.0.0:7860)
```

## Privacy

Videos and CVs are processed **locally** (or on your own HF Space), analysed in memory,
and deleted immediately after processing. Only derived metrics and text/JSON outputs are
persisted to the local SQLite database.