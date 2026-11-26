"""Head-pose regression tests using synthetic, projected MediaPipe-like landmarks."""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, ".")

from app.services import video_analysis  # noqa: E402

WIDTH, HEIGHT = 640, 480
_POINT_ORDER = (1, 152, 33, 263, 61, 291)


def _camera_matrix() -> object:
    import numpy as np

    return np.array(
        [
            [WIDTH, 0, WIDTH / 2],
            [0, WIDTH, HEIGHT / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )


def _rotation(axis: str, degrees: float) -> object:
    import numpy as np

    angle = math.radians(degrees)
    cos, sin = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]])
    if axis == "y":
        return np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    if axis == "z":
        return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    raise ValueError(axis)


def _landmarks_for_rotation(rotation: object) -> object:
    """Project the canonical face model, rotated by `rotation`, into the image."""
    import cv2
    import numpy as np

    points_3d = np.array(
        [video_analysis._FACE_3D[index] for index in _POINT_ORDER], dtype=np.float64
    )
    translation = np.array([[0.0], [0.0], [500.0]])
    projected, _ = cv2.projectPoints(
        points_3d,
        cv2.Rodrigues(rotation)[0],
        translation,
        _camera_matrix(),
        np.zeros((4, 1)),
    )
    normalized = [
        SimpleNamespace(x=float(point[0][0]) / WIDTH, y=float(point[0][1]) / HEIGHT)
        for point in projected
    ]
    # MediaPipe exposes 468+ landmarks; only the six PnP points matter here.
    by_index = dict(zip(_POINT_ORDER, normalized))
    return SimpleNamespace(
        landmark=[by_index.get(i, SimpleNamespace(x=0.0, y=0.0)) for i in range(468)]
    )


def _pose(axis: str, degrees: float) -> tuple[float, float]:
    landmarks = _landmarks_for_rotation(_rotation(axis, degrees))
    return video_analysis._pose_from_face_landmarks(landmarks, WIDTH, HEIGHT)


@pytest.mark.parametrize(
    ("axis", "degrees", "expected_yaw", "expected_pitch"),
    [
        ("x", 0, 0.0, 0.0),
        ("y", -45, -45.0, 0.0),
        ("y", 45, 45.0, 0.0),
        ("x", -45, 0.0, -45.0),
        ("x", 45, 0.0, 45.0),
    ],
)
def test_yaw_and_pitch_recovered_from_projected_points(
    axis: str, degrees: float, expected_yaw: float, expected_pitch: float
) -> None:
    yaw, pitch = _pose(axis, degrees)
    assert yaw == pytest.approx(expected_yaw, abs=6)
    assert pitch == pytest.approx(expected_pitch, abs=6)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_forty_five_degree_turns_are_not_eye_contact(axis: str) -> None:
    yaw, pitch = _pose(axis, 45)
    assert not video_analysis._is_eye_contact(yaw, pitch)


def test_front_facing_is_eye_contact() -> None:
    yaw, pitch = _pose("z", 0)
    assert video_analysis._is_eye_contact(yaw, pitch)


@pytest.mark.parametrize("degrees", [-30, 15, 40])
def test_pure_roll_does_not_change_yaw_or_pitch(degrees: float) -> None:
    yaw, pitch = _pose("z", degrees)
    assert abs(yaw) < 6, (degrees, yaw)
    assert abs(pitch) < 6, (degrees, pitch)


def test_mediapipe_landmark_sides_match_canonical_points() -> None:
    # MediaPipe: 33/61 = subject's right, 263/291 = subject's left. In an
    # unmirrored webcam image the subject's right appears on the image's left.
    assert video_analysis._RIGHT_EYE_OUTER == 33
    assert video_analysis._LEFT_EYE_OUTER == 263
    assert video_analysis._RIGHT_MOUTH == 61
    assert video_analysis._LEFT_MOUTH == 291
    assert video_analysis._FACE_3D[33][0] < 0 < video_analysis._FACE_3D[263][0]
    assert video_analysis._FACE_3D[61][0] < 0 < video_analysis._FACE_3D[291][0]
