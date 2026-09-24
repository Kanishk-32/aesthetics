"""
Tests for LightingAnalyzer.
"""
import numpy as np
import pytest

from src.cv.lighting import LightingAnalyzer


@pytest.fixture
def analyzer():
    return LightingAnalyzer(min_brightness=40, max_brightness=220, optimal_brightness=120)


def _make_frame(brightness: int) -> np.ndarray:
    """Create a uniform BGR frame at the given brightness."""
    return np.full((480, 640, 3), brightness, dtype=np.uint8)


def test_dark_frame(analyzer):
    result = analyzer.analyze(_make_frame(20))
    assert result.exposure == "underexposed"
    assert result.brightness_score < 50


def test_bright_frame(analyzer):
    result = analyzer.analyze(_make_frame(240))
    assert result.exposure == "overexposed"


def test_good_frame(analyzer):
    result = analyzer.analyze(_make_frame(120))
    assert result.exposure == "good"
    assert result.brightness_score > 90


def test_face_region(analyzer):
    frame = _make_frame(100)
    # Make the face region brighter
    frame[100:200, 100:200] = 200
    result = analyzer.analyze(frame, face_bbox=(100, 100, 100, 100))
    assert result.face_brightness > result.brightness


def test_guidance_dark(analyzer):
    result = analyzer.analyze(_make_frame(20))
    tips = analyzer.guidance(result)
    assert any("dark" in t.lower() or "light" in t.lower() for t in tips)


def test_guidance_good(analyzer):
    result = analyzer.analyze(_make_frame(120))
    tips = analyzer.guidance(result)
    assert any("✓" in t for t in tips)
