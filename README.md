# 🎤 Interviewly — AI Interview Buddy & Performance Analyzer

A multimodal, career-readiness platform that pairs **retrieval-augmented question generation** (RAG) with **speech + computer-vision performance analysis**. Interviewly reads a candidate's actual CV, generates a tailored, hallucination-free interview prep guide grounded strictly in that CV, then scores a recorded practice answer across **Content**, **Speech**, and **Visual** delivery — instantly, at scale, and privacy-first.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)
![Gradio](https://img.shields.io/badge/UI-Gradio-FF7C00?logo=gradio)
![Whisper](https://img.shields.io/badge/Speech-OpenAI%20Whisper-412991?logo=openai)
![MediaPipe](https://img.shields.io/badge/Vision-MediaPipe%20%2B%20OpenCV-00A67E)
![ChromaDB](https://img.shields.io/badge/RAG-ChromaDB-6E56CF)
![Gemini](https://img.shields.io/badge/LLM-Gemini%20(OpenAI--compatible)-4285F4?logo=google)
![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED?logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Deploy-Render-FF3D00?logo=render&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Web%20%2F%20Jupyter%20%2F%20Docker-lightgrey)
![Tests](https://img.shields.io/badge/tests-26%20passing-brightgreen)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Technology Stack](#️-technology-stack)
- [Two Engines](#-two-engines)
- [Fast Mode](#-fast-mode)
- [Pipeline Architecture](#-pipeline-architecture)
- [Configuration](#️-configuration)
- [Anti-Hallucination Design](#-anti-hallucination-design)
- [Testing & Validation](#-testing--validation)
- [System Architecture](#-system-architecture)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Calibration & Tuning](#-calibration--tuning)
- [Known Limitations](#-known-limitations)
- [Troubleshooting](#-troubleshooting)
- [Real-World Applications](#-real-world-applications)
- [Author](#-author)

---

## 📌 Project Overview

Generic question banks don't prepare anyone for a specific role at a specific company, and self-recorded practice answers are hard to judge objectively without a coach in the room. Interviewly closes both gaps:

1. **Prep Engine** — a **RAG pipeline** over the candidate's CV, job description, and company overview writes questions and STAR answers that are *traceable back to real CV facts*, with **zero invented jobs, tools, or metrics**.
2. **Practice Engine** — a recorded practice answer is run through **NLP, acoustic, and computer-vision analysis** to produce an objective, weighted diagnostic scorecard plus concrete coaching feedback.

### 🎯 Objectives

- Generate a personalized interview prep guide grounded ONLY in the candidate's actual CV
- Produce full STAR-format (Situation, Task, Action, Result) model answers aligned to the target JD and company — each capped at 3 sentences for fast, focused delivery
- Analyze a practice interview video across three independent dimensions: **Content**, **Speech**, and **Visual delivery**
- Return a weighted scorecard with concrete strengths, weaknesses, and improvement tips — not just a number
- Never fail a live demo: a **Fast Mode** and a **crash-safe JSON fallback** guarantee a 200 response even when the LLM is slow, rate-limited, or retired
- Export the prep guide as a downloadable PDF

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Web UI | Gradio (mounted on the FastAPI app at `/`) |
| LLM (Q&A generation, JSON repair, scoring review) | Google Gemini via an OpenAI-compatible endpoint, with an automatic model-fallback chain (any OpenAI-compatible base URL works) |
| Speech-to-Text | OpenAI Whisper (local, model size configurable), with an optional Whisper-API fallback |
| RAG / Embeddings | ChromaDB (persistent, per-session collections) + Sentence-Transformers (`all-MiniLM-L6-v2`, lazy-loaded once) |
| Audio Analysis | librosa (VAD via RMS energy, pitch via `pyin`) + soundfile |
| Video Analysis | OpenCV + MediaPipe Holistic (Face Mesh + Pose) |
| Database | SQLite via SQLAlchemy (Users → Sessions → PrepGuides / PracticeVideos → AnalysisReports) |
| Document Parsing | pdfplumber (PyPDF2 fallback), python-docx |
| PDF Export | fpdf2, with a Unicode → latin-1 sanitizer for LLM output |
| Deployment | Docker + docker-compose, one-command deploy on Render |
| Language | Python 3.11 |
| Notebook | Jupyter (`Interviewly_Demo.ipynb`) |

---

## 🎯 Two Engines

### 1️⃣ Prep Engine
**Input:** CV (PDF / DOCX / TXT) + Job Description + Company Overview

**Pipeline:** the CV, JD, and company text are chunked, embedded, and stored in a fresh ChromaDB collection per request; retrieved context plus a section-labeled *"verified facts"* sheet parsed straight from the CV are handed to the LLM under a strict grounding prompt. The entire LLM call is bounded by a **60-second timeout**; if the model 404s (retired), is rate-limited, times out, or returns unusable JSON, the engine transparently falls back to `data/prep_guide_fallback.json` so the user still gets a polished guide instead of an error.

**Output:** an exportable preparation guide containing:
- **5 technical questions** tailored to the JD, each with a short STAR model answer (≤3 sentences)
- **3 behavioral questions** aligned to the company's stated values, each with a short STAR model answer
- **5 company-specific positioning tips**
- **5 skill-gap items** (skills the JD wants that the CV doesn't show)
- A downloadable, styled PDF of the whole guide

### 2️⃣ Practice / Analysis Engine
**Input:** a recorded practice-interview video (upload or webcam)

**Output:** a diagnostic scorecard with:

| Dimension | Weight | Signals Analyzed |
|---|---|---|
| **Content** | 40% | STAR structure, relevance to the prompt, conciseness, filler-word penalty |
| **Speech** | 30% | Speaking rate (WPM), pause ratio, pitch variation, an LLM/heuristic clarity score |
| **Visual** | 30% | Face-detection %, eye-contact %, posture stability, head-movement stability |

Plus: strengths, weaknesses, and improvement tips generated by the LLM (or a deterministic fallback when no API key is configured), each tied to a concrete metric.

---

## ⚡ Fast Mode

Built for live demos and uptime: **a one-click instant-demo toggle on both tabs.**

| Tab | Fast Mode behavior | Latency |
|---|---|---|
| **Prep Engine** | Skips the LLM entirely; serves `data/prep_guide_fallback.json` | **~0.5 s** |
| **Practice Engine** | Skips Whisper/MediaPipe; serves `data/practice_scorecard_fallback.json` (pre-saved demo scorecard + review + transcript) | **< 1 s** |

The same fallback files are used automatically when the LLM times out or fails after retries, so the public URL stays responsive even under API outages.

---

## 🤖 Pipeline Architecture

### Prep Engine Pipeline:
```
CV (PDF/DOCX/TXT) → Text Extraction (pdfplumber → PyPDF2 fallback)
→ Section-Labeled "Verified Facts" Sheet (regex-repairs flattened two-column CV rows)
→ Chunk + Embed (Sentence-Transformers, @lru_cache loaded once) CV / JD / Company text
→ ChromaDB Retrieval (per-session collection)
→ LLM Generation (60 s timeout, JSON-only, model-fallback chain, retry + auto-repair)
→ Guide Validation (rejects empty/placeholder sections)
→ SQLite + PDF export   |   on timeout/failure → prep_guide_fallback.json
```

### Practice Engine Pipeline:
```
Video Upload → ffmpeg Audio Extraction (16 kHz mono WAV)
→ ┌─ Whisper Transcription (local, Urdu-forced) → LLM clean-up/translation to English
   ├─ librosa Audio Analysis (speaking rate, pauses, pitch, clarity)      [parallel]
   └─ OpenCV + MediaPipe Video Analysis (face, eye contact, posture)     [parallel]
→ Weighted Scorecard (Content 40% / Speech 30% / Visual 30%)
→ LLM Review (strengths / weaknesses / improvement tips) → SQLite
→ Practice video + audio deleted immediately after analysis (privacy-first)
```

---

## ⚙️ Configuration

### LLM, Timeout & Prep Engine (`.env` / `app/config.py`)
| Setting | Default | Why |
|---|---|---|
| `OPENAI_API_KEY` | — | Any OpenAI-compatible key (ships configured for Google Gemini's OpenAI-compatible endpoint) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Swap to Gemini's (or any provider's) OpenAI-compatible base URL |
| `LLM_MODEL` | `gemini-2.0-flash` | Render env should set this; see **Model fallback** below |
| `LLM_TIMEOUT` | `60` | Seconds before the Prep Engine abandons the LLM and serves the JSON fallback |
| `PREP_MAX_TOKENS` | `8192` | Caps each LLM response; generous head-room so the reduced 5+3 guide never truncates |
| `PREP_RETRIES` | `2` | Full regeneration attempts if the model returns invalid/truncated JSON |

#### Model fallback chain
`app/services/llm.py` strips any `models/` prefix and tries candidates in order until one returns content:
1. `gemini-2.0-flash` (the configured default)
2. `gemini-2.5-flash`
3. `gemini-1.5-flash-latest`
4. `gemini-1.5-pro-latest`
5. `gemini-flash-latest` (verified on new accounts where the 1.5/2.x line is retired)

If **every** candidate fails, the Prep Engine returns `data/prep_guide_fallback.json` rather than an error.

### Speech (`app/services/transcription.py`, `audio_analysis.py`)
| Setting | Default | Why |
|---|---|---|
| `WHISPER_MODEL` | `small` | Local Whisper model size — `base` for speed, `medium` for accuracy |
| `WHISPER_API_ENABLED` | `false` | When `true`, skips the local model and uses the Whisper API directly |
| VAD threshold | 30th-percentile RMS × 0.6 | Separates voiced speech from silence without a video-dependent noise floor |
| `pyin` pitch range | 50–500 Hz | Human speaking-voice fundamental-frequency range |

### Vision (`app/services/video_analysis.py`)
| Setting | Default | Why |
|---|---|---|
| Sampling rate | 1 frame/sec | Enough temporal resolution for posture/eye-contact trends, not all frames |
| `EYE_CONTACT_YAW_CONE` / `PITCH_CONE` | 25° / 20° | Head-pose cone (via `solvePnP`) counted as "looking at camera" |
| MediaPipe `model_complexity` | 0 | Fastest Holistic variant; keeps CPU-only inference practical on Render |

### Uploads
| Setting | Default | Why |
|---|---|---|
| `MAX_UPLOAD_MB` | 200 | Caps CV and video upload size; streamed to disk in chunks |
| Allowed CV types | `.pdf .docx .doc .txt` | — |
| Allowed video types | `.mp4 .mov .avi .mkv .webm` | — |

---

## 🛡️ Anti-Hallucination Design

The Prep Engine's biggest risk is a confident, fabricated answer. Several layers guard against it:
- **Verified Facts Sheet** — the raw CV is split into labeled sections (`EXPERIENCE`, `SKILLS`, `CERTIFICATIONS`, …) before it ever reaches the LLM, so a certification can't be mistaken for a job.
- **Two-column flattening repair** — CVs from a two-column PDF template often flatten into garbled lines; targeted regexes reconstruct which employer, role, and date belong together instead of letting the LLM guess.
- **Explicit "I don't have direct experience…" fallback** — the system prompt requires this exact phrasing when the CV has no relevant example, rather than inventing one.
- **Guide validation** — every generated guide must have non-empty, non-placeholder content in every required section; a failing guide triggers an automatic regeneration.
- **JSON-repair loop** — invalid JSON (truncated, wrapped in commentary) triggers a scoped re-prompt, then a lenient brace-matching salvage parse, before any fallback is served.

---

## 📊 Testing & Validation

| Test | Method | Result |
|---|---|---|
| Filler-word detection | Word-boundary regex counts on hand-written transcripts (`tests/test_analysis.py`) | Roman-Urdu/English mix (`um`, `like`, `matlab`, `yaani`) counted independently, never double-counted |
| Scorecard weighting | Fixed sample metrics through `build_scorecard()` | Confirms 40/30/30 weighting + penalty terms land in the 0–10 range |
| CV chunking | Long synthetic text through `_chunk_text()` | Chunk size stays within limit with expected overlap |
| CV parsing | Real `.txt`/`.pdf` fixtures + unsupported/missing files | `CVParseError` is raised cleanly, never an unhandled exception |
| Guide reduction | `PREP_SYSTEM_PROMPT` + fallback JSON fixtures | 5 technical + 3 behavioral questions, STAR answers ≤3 sentences |
| Fallback loader | `load_fallback_guide()` on the checked-in JSON | Returns a complete, validated 5+3+5+5 guide instantly |
| PDF export sanitization | Unicode-heavy LLM output through `sanitize_pdf_text()` | Renders safely in fpdf2's latin-1 core fonts |
| End-to-end pipeline | `Interviewly_Demo.ipynb`, plus a live `/prep/generate` call | CV → Prep Guide → Scorecard using the same functions the API calls |

Run the suite with:
```bash
pytest tests/ -v
```

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                Gradio UI  (Prep tab + Practice tab)               │
│        Fast Mode checkboxes → cached JSON demo outputs            │
│                    mounted on FastAPI at "/"                      │
└───────────────────────────────┬────────────────────────────────┘
                                 │
                        FastAPI routers
              ┌──────────────────┴──────────────────┐
              │                                      │
      /prep/generate, /prep/{id}          /practice/upload, /practice/{id}/report
              │                                      │
      PrepEngine (RAG + LLM, 60 s         Transcription + Audio + Video analysis
      timeout → fallback JSON)                      │
              │                                      │
      ChromaDB (embeddings)                 Whisper · librosa · MediaPipe
              │                                      │
              └──────────────┬───────────────────────┘
                              │
                    SQLite (Users, Sessions, PrepGuides,
                       PracticeVideos, AnalysisReports)
```

---

## 🚀 How to Run

### Option A — Docker (recommended for local)
```bash
cp .env.example .env   # add your OPENAI_API_KEY
docker-compose up --build
```
The Dockerfile installs `ffmpeg`, OpenCV system libs, pins `setuptools<82` (required by `openai-whisper`'s build), and installs `openai-whisper` before the rest of `requirements.txt`.

### Option B — Local virtual environment
```bash
python -m venv venv
venv\Scripts\activate          # Windows (macOS/Linux: source venv/bin/activate)
pip install --upgrade pip "setuptools<82" wheel
pip install --no-build-isolation openai-whisper
pip install --no-build-isolation -r requirements.txt
cp .env.example .env           # then paste your Gemini key into OPENAI_API_KEY
```

Configure your key — grab a free Gemini API key at [aistudio.google.com](https://aistudio.google.com/app/apikey). Running **without** a key still works: Prep returns the deterministic fallback guide and Practice falls back to heuristic scoring.

### Run it
```bash
python run.py
# or
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Open **http://localhost:7860** for the Gradio UI, or call the REST API:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Health check (`{"status": "ok"}`, ~50 ms) |
| `/prep/generate` | POST | Upload CV + JD + company overview → generate a prep guide |
| `/prep/{guide_id}` | GET | Fetch a previously generated guide |
| `/practice/upload` | POST | Upload a practice video → full analysis → scorecard |
| `/practice/{video_id}/report` | GET | Fetch a previously saved scorecard |

### Option C — Deploy on Render

1. Push this repo to GitHub.
2. In Render → **New** → **Web Service**, connect the repo.
3. Set:
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install --upgrade pip "setuptools<82" wheel && pip install --no-build-isolation openai-whisper && pip install --no-build-isolation -r requirements.txt`
   - **Start Command:** `gunicorn app.main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 300 --workers 1` (or `python run.py`)
4. Add the **environment variables** from `.env.example` (see the *Render environment variables* note at the end of this README).
5. Render auto-provisions an `X-Forwarded-Proto`/`PORT`; the health check is already mounted at `/health`.

> Tip: Render's free tier is CPU-slim. Whisper/MediaPipe inference is slow at first boot while models download; use **Fast Mode** for demos and expect ~10–30 s for a full practice analysis until caching warms up.

---

## 📁 Project Structure

```
interviewly/
├── run.py                          # Entrypoint: launches FastAPI + Gradio on :7860
├── requirements.txt
├── Dockerfile                      # ffmpeg + OpenCV system deps, pinned setuptools
├── docker-compose.yml
├── .env.example                    # Copy to .env / set as Render env vars
├── Interviewly_Demo.ipynb          # Notebook walkthrough of the full pipeline
├── conftest.py
├── tests/
│   ├── test_prep.py                # CV parsing, chunking, prompts, fallback JSON, PDF
│   └── test_analysis.py            # Scoring, audio/text/video analysis
├── data/
│   ├── prep_guide_fallback.json    # Crash-safe / Fast-Mode prep guide (5+3+5+5)
│   ├── practice_scorecard_fallback.json  # Fast-Mode demo scorecard
│   ├── uploads/                    # Temp CV/video staging (cleared post-processing)
│   ├── pdfs/                       # Exported prep-guide PDFs
│   ├── chroma/                     # ChromaDB persistent embeddings
│   └── interviewly.db              # SQLite database
└── app/
    ├── main.py                     # FastAPI app + Gradio mount + lifespan DB init
    ├── config.py                   # Settings loaded from .env
    ├── database.py                 # SQLAlchemy engine/session setup
    ├── models.py                   # ORM: User, Session, PrepGuide, PracticeVideo, AnalysisReport
    ├── schemas.py                  # Pydantic request/response models
    ├── routers/
    │   ├── prep.py                 # /prep/* endpoints
    │   └── practice.py             # /practice/* endpoints
    ├── services/
    │   ├── cv_parser.py            # PDF/DOCX/TXT → plain text
    │   ├── prep_engine.py          # RAG + grounded LLM guide generation
    │   ├── llm.py                  # OpenAI-compatible client + model fallback + JSON repair
    │   ├── transcription.py        # ffmpeg + Whisper (local/API) + Urdu→English cleanup
    │   ├── text_analysis.py        # Filler words, WPM, STAR/relevance/conciseness
    │   ├── audio_analysis.py       # librosa: speaking rate, pauses, pitch, clarity
    │   ├── video_analysis.py       # OpenCV + MediaPipe: face, eye contact, posture
    │   ├── scoring.py              # Weighted scorecard + LLM review
    │   └── pdf_export.py           # fpdf2 prep-guide export + Unicode sanitizer
    └── ui/
        └── gradio_app.py           # Prep tab + Practice tab + Fast Mode toggles
```

---

## 🎯 Calibration & Tuning

- **Sparse or non-standard CV formats** — if the "verified facts" sheet comes out mostly unlabeled, add the relevant section header (e.g. `Experience`, `Skills`) to `_SECTION_HEADERS` in `prep_engine.py`.
- **Guide keeps failing validation** — raise `PREP_MAX_TOKENS` first; truncation is the most common cause of invalid JSON.
- **Clarity/speech scores look off** — the heuristic bands (120–180 WPM, 0.15–0.35 pause ratio, 20–80 Hz pitch std) in `audio_analysis.py` assume one clear speaker near the mic; noisy recordings shift the VAD threshold.
- **Eye-contact % too strict/lenient** — adjust `EYE_CONTACT_YAW_CONE` / `EYE_CONTACT_PITCH_CONE` in `video_analysis.py` for your camera framing.
- **LLM output too slow on Render's free tier** — cut `PREP_MAX_TOKENS`, tighten `LLM_TIMEOUT`, or switch the demo to Fast Mode.
- **Non-English or mixed-language answers** — Whisper is forced to `language="ur"`; adjust the language heuristic in `transcription.py` for a purely English test set.

---

## ⚠️ Known Limitations

- **Speaking rate is a heuristic.** `speaking_rate_wpm` scales an assumed 140 WPM baseline by the voiced-audio fraction rather than aligning to transcribed words — an approximation, not ground truth.
- **Heuristic scoring is coarser than the LLM path.** Without an API key, STAR/relevance/conciseness use keyword matching rather than LLM grading.
- **Single-speaker, front-facing assumption.** Visual analysis is tuned for one candidate mostly facing the camera; multiple faces or occlusion reduce accuracy.
- **Privacy-first means no replay.** Practice videos and extracted audio are deleted immediately after analysis; only the scorecard and transcript remain.
- **ChromaDB collections aren't cleaned up.** Each prep generation creates a uniquely-named collection that is never deleted, so `data/chroma/` grows with usage.
- **Gemini free-tier quota is small.** Google's free tier bills per-model daily (e.g. 20 requests/day for `gemini-flash-latest`); heavy demos can hit `429`. A paid key or the JSON fallback avoids this.
- **PDF export is latin-1 only.** `sanitize_pdf_text()` transliterates or drops characters fpdf2's core fonts can't render; non-Latin scripts show fine in the web UI but not in the exported PDF.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `404 NOT_FOUND` for `gemini-1.5-flash` / `gemini-2.0-flash` | Models are retired for new accounts. The app auto-shifts through the fallback chain to `gemini-flash-latest`; no action needed. |
| Response takes 3–4 minutes | The LLM fell back across several retired models. Set `LLM_TIMEOUT=20–30`, reduce `PREP_RETRIES`, and use Fast Mode for demos. |
| `429 RESOURCE_EXHAUSTED` from Gemini | Free-tier daily quota exhausted. Use a paid key, or let the JSON fallback serve the guide. |
| Prep guide always shows the cached "ACME Analytics" answers | The LLM failed (404/429/timeout) and the JSON fallback fired. Check `server.log` for the real cause. |
| `pkg_resources` / setuptools error on install | `pip install "setuptools<82"` before installing `openai-whisper`. |
| `ffmpeg not found` | App auto-falls back to bundled `imageio-ffmpeg`; otherwise `apt-get install ffmpeg`. |
| Empty or garbled transcript | Confirm `WHISPER_MODEL` finished downloading and audio isn't silent; set `WHISPER_API_ENABLED=true`. |
| Visual metrics all return 0 | opencv/mediapipe import issue — reinstall the pinned versions (`numpy<2`) from `requirements.txt`. |
| `Report generation timed out on Render` | Render's default proxy timeout — add `--timeout 300` to the gunicorn start command. |
| Notebook `ModuleNotFoundError` | Pick the project's `venv`/`.venv` interpreter in Jupyter, not the global Python. |

---

## 🌍 Real-World Applications

- **University career services** — self-serve interview prep grounded in each student's real CV, at scale
- **Bootcamp / cohort-based programs** — objective, repeatable practice-interview feedback without a coach in every session
- **HR / L&D coaching tools** — a diagnostic layer on top of existing mock-interview programs
- **Individual job seekers** — company-specific prep without generic, one-size-fits-all question banks
- **Recruitment prep platforms / B2B Saas** — rebrandable demo generator with Fast Mode for sales demos

---

## 👨‍💻 Author

**Hamza Asif**  
BS Artificial Intelligence — DUET, Karachi

[![GitHub](https://img.shields.io/badge/GitHub-Hamza--Asif--ai-black?style=flat-square&logo=github)](https://github.com/Hamza-Asif-ai)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Hamza%20Asif-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/hamza-asif-ai)

---

> **Render environment variables** — on your Render Web Service, set: `OPENAI_API_KEY`, `OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`, `LLM_MODEL=gemini-2.0-flash` (fallbacks are automatic), `LLM_TIMEOUT=60`, `PREP_RETRIES=2`, `PREP_MAX_TOKENS=8192`, `WHISPER_MODEL=small`, `WHISPER_API_ENABLED=false`, `MAX_UPLOAD_MB=200`, `PORT` (Render injects it).