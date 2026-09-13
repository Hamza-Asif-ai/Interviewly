FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for ffmpeg, OpenCV, MediaPipe, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# FIX: Pin setuptools<82 so pkg_resources is available for openai-whisper build.
# Also install openai-whisper separately BEFORE the rest of requirements.
RUN pip install --upgrade pip "setuptools<82" wheel && \
    pip install --no-build-isolation openai-whisper && \
    pip install --no-build-isolation -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "run.py"]