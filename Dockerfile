FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for ffmpeg, OpenCV, MediaPipe, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-dejavu-core \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip "setuptools<82" wheel && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 && \
    pip install --no-build-isolation -r requirements.txt

# Pre-download the int8 Whisper and MediaPipe models into the image so the first
# request never needs network access to fetch model weights. The app enforces
# tiny/base only, regardless of the value supplied at build/run time.
ARG WHISPER_MODEL=base
ENV WHISPER_MODEL=${WHISPER_MODEL}
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')" && \
    python -c "import mediapipe as mp; h = mp.solutions.holistic.Holistic(); h.close()"

COPY . .

EXPOSE 7860

CMD ["python", "run.py"]