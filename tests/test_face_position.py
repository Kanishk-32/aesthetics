"""
Tests for FacePositionAnalyzer.
"""
import pytest

from src.cv.face_position import FacePositionAnalyzer


@pytest.fixture
def analyzer():
    return FacePositionAnalyzer(
        center_threshold=0.15,
        min_face_size=0.15,
        max_face_size=0.60,
        optimal_face_size=0.35,
    )


FRAME = (480, 640, 3)  # height, width, channels


def test_centered_face(analyzer):
    bbox = (220, 140, 200, 200)  # roughly centered
    result = analyzer.analyze(FRAME, bbox)
    assert result.horizontal == "center"
    assert result.vertical == "center"
    assert result.score > 70


def test_face_too_left(analyzer):
    bbox = (10, 140, 200, 200)
    result = analyzer.analyze(FRAME, bbox)
    assert result.horizontal == "left"


def test_face_too_right(analyzer):
    bbox = (450, 140, 200, 200)
    result = analyzer.analyze(FRAME, bbox)
    assert result.horizontal == "right"


def test_face_too_small(analyzer):
    bbox = (270, 200, 40, 40)  # 40/480 = 0.08 < 0.15
    result = analyzer.analyze(FRAME, bbox)
    assert result.size_status == "too_small"


def test_face_too_large(analyzer):
    bbox = (50, 10, 540, 460)  # 460/480 = 0.96 > 0.60
    result = analyzer.analyze(FRAME, bbox)
    assert result.size_status == "too_large"


def test_guidance_move_right(analyzer):
    bbox = (10, 140, 200, 200)
    result = analyzer.analyze(FRAME, bbox)
    tips = analyzer.guidance(result)
    assert any("right" in t.lower() for t in tips)


def test_guidance_move_closer(analyzer):
    bbox = (270, 200, 40, 40)
    result = analyzer.analyze(FRAME, bbox)
    tips = analyzer.guidance(result)
    assert any("closer" in t.lower() for t in tips)
