"""Audio signal analysis using librosa: speaking rate, pauses, pitch, energy."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


def analyze_audio(audio_path: str) -> Dict[str, Any]:
    """Analyse a 16 kHz mono WAV file using librosa.

    Returns:
        speaking_rate_wpm: Estimated words per minute from voice activity.
        pause_ratio: Fraction of total duration spent in silence (>0.3s).
        pitch_mean_hz: Mean fundamental frequency of voiced segments.
        pitch_std_hz: Standard deviation of fundamental frequency.
        energy_variance: Variance of frame-level RMS energy.
        clarity_score: Heuristic 0-10 score combining WPM, pauses and pitch variance.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    import librosa
    import numpy as np

    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    if duration < 0.01:
        return {
            "speaking_rate_wpm": 0.0,
            "pause_ratio": 0.0,
            "pitch_mean_hz": 0.0,
            "pitch_std_hz": 0.0,
            "energy_variance": 0.0,
            "clarity_score": 0.0,
        }

    # Voice Activity Detection using RMS energy
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    threshold = float(np.percentile(rms, 30)) * 0.6
    threshold = max(threshold, 0.001)

    frames = len(rms)
    frame_duration = hop_length / sr
    voiced_frames = int(np.sum(rms > threshold))
    voiced_duration = voiced_frames * frame_duration
    pause_duration = max(duration - voiced_duration, 0.0)

    # Estimated speaking rate: assume a continuous speaker at ~140 words/min.
    # Scale by the voiced fraction of the timestamp so pauses reduce the rate.
    wpm = round(140.0 * (voiced_duration / max(duration, 0.01)), 1)
    wpm = max(0.0, min(wpm, 250.0))

    # Pitch (F0) via pyin
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=50, fmax=500, sr=sr, hop_length=hop_length
    )
    f0 = np.asarray(f0)
    voiced_f0 = f0[~np.isnan(f0)]
    pitch_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    pitch_std = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0

    # Energy variance
    energy_var = float(np.var(rms))

    # Pause ratio
    pause_ratio = round(pause_duration / max(duration, 0.01), 3)

    # Clarity score heuristic (0-10)
    # Good: 120-180 wpm, 0.2-0.35 pause ratio, moderate pitch variation
    clarity = 5.0
    # Speaking rate contribution
    if 120 <= wpm <= 180:
        clarity += 1.5
    elif 90 <= wpm <= 220:
        clarity += 0.7
    else:
        clarity -= 1.0
    # Pause ratio
    if 0.15 <= pause_ratio <= 0.35:
        clarity += 1.5
    elif pause_ratio < 0.10:
        clarity -= 0.5  # too fast / no breathing
    else:
        clarity -= 0.5
    # Pitch variation (reduces monotony penalty)
    if 20 < pitch_std < 80:
        clarity += 1.0
    elif pitch_std < 10:
        clarity -= 0.5
    clarity = round(max(0.0, min(10.0, clarity)), 2)

    return {
        "speaking_rate_wpm": max(0, wpm),
        "pause_ratio": pause_ratio,
        "pitch_mean_hz": round(pitch_mean, 1),
        "pitch_std_hz": round(pitch_std, 2),
        "energy_variance": round(energy_var, 6),
        "clarity_score": clarity,
    }