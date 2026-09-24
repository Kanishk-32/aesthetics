"""
Tests for FaceDetector (unit-level, no real face images).
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.cv.face_detector import FaceDetector, FaceLandmarks


@pytest.fixture
def detector():
    with patch("src.cv.face_detector.FaceLandmarker") as mock_cls:
        mock_landmarker = MagicMock()
        mock_cls.create_from_options.return_value = mock_landmarker
        det = FaceDetector(min_confidence=0.5, min_face_size=50)
        det._mock_landmarker = mock_landmarker
        yield det


def test_no_face_on_blank(detector):
    """A solid-colour frame should return None."""
    detector._mock_landmarker.detect_for_video.return_value = MagicMock(face_landmarks=[])
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detector.detect(blank) is None


def test_init_params(detector):
    assert detector.min_confidence == 0.5
    assert detector.min_face_size == 50


def test_face_detected(detector):
    """Mock a face detection and check the result."""
    h, w = 480, 640
    mock_lm = []
    for i in range(478):
        lm = MagicMock()
        lm.x = 0.3 + 0.001 * (i % 50)
        lm.y = 0.2 + 0.001 * (i % 50)
        lm.z = 0.0
        mock_lm.append(lm)

    detector._mock_landmarker.detect_for_video.return_value = MagicMock(
        face_landmarks=[mock_lm]
    )
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    result = detector.detect(frame)

    assert result is not None
    assert isinstance(result, FaceLandmarks)
    assert result.landmarks.shape == (478, 3)
    assert result.confidence == 0.5


def test_face_too_small_rejected(detector):
    """A face with tiny bbox should be rejected."""
    mock_lm = []
    for i in range(478):
        lm = MagicMock()
        lm.x = 0.5 + 0.0001 * (i % 10)
        lm.y = 0.5 + 0.0001 * (i % 10)
        lm.z = 0.0
        mock_lm.append(lm)

    detector._mock_landmarker.detect_for_video.return_value = MagicMock(
        face_landmarks=[mock_lm]
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(frame)
    assert result is None
