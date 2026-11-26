"""Video analysis using OpenCV + MediaPipe: face, eye contact, posture, head pose."""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

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
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    start = time.perf_counter()

    try:
        import cv2
        import mediapipe as mp

        mp_holistic = mp.solutions.holistic
    except ImportError as exc:
        logger.warning("OpenCV/MediaPipe unavailable (%s); skipping visual analysis.", exc)
        return {
            "face_detection_percent": 0.0,
            "eye_contact_percent": 0.0,
            "posture_stability_score": 0.0,
            "head_movement_score": 0.0,
            "frames_analyzed": 0,
        }

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise ValueError(f"Video has no frames: {video_path}")

    sample_indices = {int(i * fps) for i in range(int(frame_count / fps) + 1)}

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
        return {
            "face_detection_percent": 0.0,
            "eye_contact_percent": 0.0,
            "posture_stability_score": 0.0,
            "head_movement_score": 0.0,
            "frames_analyzed": 0,
        }

    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index not in sample_indices:
                index += 1
                continue

            height, width, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
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
            index += 1
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
    logger.info("MediaPipe analysis took %.2fs", time.perf_counter() - start)
    return {
        "face_detection_percent": face_detection_percent,
        "eye_contact_percent": eye_contact_percent,
        "posture_stability_score": posture_stability_score,
        "head_movement_score": head_movement_score,
        "frames_analyzed": frames_analyzed,
    }