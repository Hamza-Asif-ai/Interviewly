"""Audio signal analysis using librosa: speaking rate, pauses, pitch, energy."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

PAUSE_THRESHOLD_S = 0.3


def analyze_audio(audio_path: str, word_count: int = 0) -> Dict[str, Any]:
    """Analyse a 16 kHz mono WAV file using librosa.

    Args:
        audio_path: Path to the 16 kHz mono WAV file.
        word_count: Number of words in the transcript (for the WPM estimate).

    Returns:
        speaking_rate_wpm: Transcribed words per minute over voiced speaking time.
        pause_ratio: Fraction of total duration spent in deliberate pauses (>0.3s).
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
            "words_per_minute": 0.0,
            "pause_ratio": 0.0,
            "pause_count": 0,
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
    voiced_mask = rms > threshold
    # Sum the durations of contiguous voiced segments.
    voiced_seconds = 0.0
    run = 0
    for is_voiced in voiced_mask:
        if is_voiced:
            run += 1
        elif run:
            voiced_seconds += run * frame_duration
            run = 0
    if run:
        voiced_seconds += run * frame_duration
    # Only silence long enough to be a deliberate pause (>0.3s) contributes to
    # the pause count and the pause ratio; micro-gaps are part of normal speech.
    pause_count = 0
    pause_seconds = 0.0
    silent_run = 0
    for is_voiced in voiced_mask:
        if is_voiced:
            run_seconds = silent_run * frame_duration
            if run_seconds > PAUSE_THRESHOLD_S:
                pause_count += 1
                pause_seconds += run_seconds
            silent_run = 0
        else:
            silent_run += 1
    trailing_seconds = silent_run * frame_duration
    if trailing_seconds > PAUSE_THRESHOLD_S:
        pause_count += 1
        pause_seconds += trailing_seconds

    # Real speaking rate: transcribed words over voiced (speaking) time, plus
    # the overall transcript WPM over the full clip duration.
    if voiced_seconds > 0:
        wpm = round(word_count / (voiced_seconds / 60.0), 1)
    else:
        wpm = 0.0
    overall_wpm = round(word_count / (duration / 60.0), 1) if word_count else 0.0

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

    # Pause ratio from qualifying pauses only
    pause_ratio = round(pause_seconds / max(duration, 0.01), 3)

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
        "words_per_minute": max(0.0, overall_wpm),
        "pause_ratio": pause_ratio,
        "pause_count": pause_count,
        "pitch_mean_hz": round(pitch_mean, 1),
        "pitch_std_hz": round(pitch_std, 2),
        "energy_variance": round(energy_var, 6),
        "clarity_score": clarity,
    }