"""Video analysis using OpenCV + MediaPipe: face, eye contact, posture, head pose."""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from .timing import StepTimeoutError, step_budget

logger = logging.getLogger(__name__)

MAX_SAMPLE_FPS = 1.0
MAX_FRAME_WIDTH = 320
MAX_FRAME_HEIGHT = 240

# MediaPipe face-mesh landmark indices used for head-pose estimation.
# Sides are the subject's: in an unmirrored image the subject's right eye
# appears on the viewer's left (smaller x).
_NOSE_TIP = 1
_CHIN = 152
_RIGHT_EYE_OUTER = 33
_LEFT_EYE_OUTER = 263
_RIGHT_MOUTH = 61
_LEFT_MOUTH = 291

_POSE_LEFT_SHOULDER = 11
_POSE_RIGHT_SHOULDER = 12

# Canonical 3D facial points (approximate head geometry in cm) used with solvePnP.
# MediaPipe's y axis points down and its z axis points out of the image, hence
# the sign flip relative to a conventional right-handed head model.
_FACE_3D = {
    _NOSE_TIP: (0.0, 0.0, 0.0),
    _CHIN: (0.0, 30.0, 20.0),
    _RIGHT_EYE_OUTER: (-30.0, -20.0, 20.0),
    _LEFT_EYE_OUTER: (30.0, -20.0, 20.0),
    _RIGHT_MOUTH: (-20.0, 20.0, 30.0),
    _LEFT_MOUTH: (20.0, 20.0, 30.0),
}

EYE_CONTACT_YAW_CONE = 25.0  # degrees
EYE_CONTACT_PITCH_CONE = 20.0  # degrees


def _is_eye_contact(yaw: float, pitch: float) -> bool:
    """Whether head yaw and pitch (degrees) fall inside the frontal cone."""
    return (
        abs(yaw) <= EYE_CONTACT_YAW_CONE
        and abs(pitch) <= EYE_CONTACT_PITCH_CONE
    )


def _resize_for_mediapipe(frame: Any) -> Any:
    """Shrink a frame to at most 320x240, preserving its aspect ratio."""
    import cv2

    height, width = frame.shape[:2]
    scale = min(MAX_FRAME_WIDTH / width, MAX_FRAME_HEIGHT / height)
    if scale >= 1.0:
        return frame
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _pose_from_face_landmarks(
    landmarks: Any,
    frame_width: int,
    frame_height: int,
) -> Tuple[float, float]:
    """Estimate head yaw and pitch (degrees) from face-mesh landmarks via solvePnP."""
    import cv2
    import numpy as np

    def lp(index: int) -> Any:
        return landmarks.landmark[index]

    point_order = (
        _NOSE_TIP, _CHIN, _RIGHT_EYE_OUTER, _LEFT_EYE_OUTER,
        _RIGHT_MOUTH, _LEFT_MOUTH,
    )
    points_2d = np.array(
        [
            [lp(i).x * frame_width, lp(i).y * frame_height]
            for i in point_order
        ],
        dtype=np.float64,
    )
    points_3d = np.array([_FACE_3D[i] for i in point_order], dtype=np.float64)
    focal_length = frame_width
    camera_matrix = np.array(
        [
            [focal_length, 0, frame_width / 2],
            [0, focal_length, frame_height / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vec, _ = cv2.solvePnP(
        points_3d,
        points_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    yaw = math.degrees(math.asin(float(np.clip(-rotation_mat[2, 0], -1, 1))))
    pitch = math.degrees(math.atan2(rotation_mat[2, 1], rotation_mat[2, 2]))
    return float(yaw), float(pitch)


def _shoulder_tilt_degrees(landmarks: Any) -> Optional[float]:
    """Return the angle of the shoulder line relative to horizontal, in degrees."""
    left = landmarks.landmark[_POSE_LEFT_SHOULDER]
    right = landmarks.landmark[_POSE_RIGHT_SHOULDER]
    dx = right.x - left.x
    dy = right.y - left.y
    if abs(dx) < 1e-6:
        return None
    return math.degrees(math.atan(dy / dx))


def analyze_video(video_path: str) -> Dict[str, Any]:
    """Analyse a practice video and produce visual-performance metrics.

    Samples one frame per second, then computes:
        face_detection_percent  - % of sampled frames with a detected face
        eye_contact_percent     - % of face frames looking within the frontal cone
        posture_stability_score - 0-10 based on shoulder-tilt variance
        head_movement_score     - 0-10 based on head yaw variance
        frames_analyzed         - number of sampled frames
        face_frames             - number of sampled frames with a detected face
        analysis_available      - False when OpenCV/MediaPipe could not analyse
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    start = time.time()
    budget = step_budget()
    deadline = start + budget
    logger.info(
        "Starting MediaPipe video analysis (max %.0f fps, max %dx%d, hard timeout %.1fs)...",
        MAX_SAMPLE_FPS,
        MAX_FRAME_WIDTH,
        MAX_FRAME_HEIGHT,
        budget,
    )

    try:
        import cv2
        import mediapipe as mp

        mp_holistic = mp.solutions.holistic
    except ImportError as exc:
        logger.warning("OpenCV/MediaPipe unavailable (%s); skipping visual analysis.", exc)
        logger.info("MediaPipe analysis finished in %.2fs", time.time() - start)
        return {
            "face_detection_percent": 0.0,
            "eye_contact_percent": 0.0,
            "posture_stability_score": 0.0,
            "head_movement_score": 0.0,
            "frames_analyzed": 0,
            "face_frames": 0,
            "analysis_available": False,
        }

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = (
        raw_fps if math.isfinite(raw_fps) and 0 < raw_fps <= 1000 else 30.0
    )
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        logger.warning(
            "Video frame count is unknown (%d); sampling by timestamp.", frame_count
        )

    face_frames = 0
    eye_contact_frames = 0
    yaws: List[float] = []
    pitches: List[float] = []
    shoulder_tilts: List[float] = []
    frames_analyzed = 0

    holistic = None
    try:
        holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception as exc:  # noqa: BLE001 - never crash on model init failure
        logger.warning(
            "MediaPipe Holistic init failed (%s); skipping visual analysis.", exc
        )
        capture.release()
        logger.info("MediaPipe analysis finished in %.2fs", time.time() - start)
        return {
            "face_detection_percent": 0.0,
            "eye_contact_percent": 0.0,
            "posture_stability_score": 0.0,
            "head_movement_score": 0.0,
            "frames_analyzed": 0,
            "face_frames": 0,
            "analysis_available": False,
        }

    try:
        frame_step = max(1, int(round(fps / MAX_SAMPLE_FPS)))
        index = 0
        samples = 0
        use_seek = True
        while frame_count <= 0 or index < frame_count:
            if time.time() >= deadline:
                raise StepTimeoutError(
                    f"MediaPipe video analysis exceeded the {budget:.0f}s hard timeout."
                )
            target = samples * frame_step
            if frame_count > 0 and target >= frame_count:
                break
            if use_seek:
                try:
                    seek_ok = bool(capture.set(cv2.CAP_PROP_POS_FRAMES, target))
                except Exception:  # pragma: no cover - backend may reject seeks
                    seek_ok = False
                if not seek_ok:
                    logger.info("Frame seeking unavailable; sampling sequentially.")
                    use_seek = False
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    index = 0
                    continue
            ok, frame = capture.read()
            if not ok:
                break
            if use_seek:
                index = target + 1
            elif index % frame_step:
                index += 1
                continue
            else:
                index += 1

            small = _resize_for_mediapipe(frame)
            height, width = small.shape[:2]
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            samples += 1
            frames_analyzed += 1

            face_landmarks = results.face_landmarks
            if face_landmarks is not None:
                face_frames += 1
                yaw, pitch = _pose_from_face_landmarks(face_landmarks, width, height)
                yaws.append(yaw)
                pitches.append(pitch)
                if _is_eye_contact(yaw, pitch):
                    eye_contact_frames += 1

            pose_landmarks = results.pose_landmarks
            if pose_landmarks is not None:
                tilt = _shoulder_tilt_degrees(pose_landmarks)
                if tilt is not None:
                    shoulder_tilts.append(tilt)
    finally:
        holistic.close()
        capture.release()

    total = max(frames_analyzed, 1)
    face_detection_percent = round(100 * face_frames / total, 1)
    eye_contact_percent = round(
        100 * eye_contact_frames / max(face_frames, 1), 1
    )

    import statistics

    if shoulder_tilts:
        tilt_std = statistics.pstdev(shoulder_tilts)
        posture_stability_score = round(max(0.0, min(10.0, 10 - 2 * tilt_std)), 2)
    else:
        posture_stability_score = 0.0

    if yaws:
        yaw_std = statistics.pstdev(yaws)
        head_movement_score = round(max(0.0, min(10.0, 10 - 1.5 * yaw_std)), 2)
    else:
        head_movement_score = 0.0

    logger.info(
        "Video analysis: %d frames, face=%s%%, eye_contact=%s%%",
        frames_analyzed,
        face_detection_percent,
        eye_contact_percent,
    )
    logger.info("MediaPipe analysis finished in %.2fs", time.time() - start)
    return {
        "face_detection_percent": face_detection_percent,
        "eye_contact_percent": eye_contact_percent,
        "posture_stability_score": posture_stability_score,
        "head_movement_score": head_movement_score,
        "frames_analyzed": frames_analyzed,
        "face_frames": face_frames,
        "analysis_available": True,
    }