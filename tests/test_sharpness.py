"""
Tests for SharpnessAnalyzer.
"""
import numpy as np
import pytest

from src.cv.sharpness import SharpnessAnalyzer


@pytest.fixture
def analyzer():
    return SharpnessAnalyzer(blur_threshold=100)


def test_blurry_frame(analyzer):
    # A uniform image has zero Laplacian variance → blurry
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    result = analyzer.analyze(frame)
    assert result.is_blurry
    assert result.score < 10


def test_sharp_frame(analyzer):
    # Random noise has high Laplacian variance → sharp
    rng = np.random.RandomState(42)
    frame = rng.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    result = analyzer.analyze(frame)
    assert not result.is_blurry
    assert result.score > 50


def test_face_region(analyzer):
    rng = np.random.RandomState(42)
    frame = rng.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    result = analyzer.analyze(frame, face_bbox=(100, 100, 200, 200))
    assert result.laplacian_var > 0


def test_guidance_blurry(analyzer):
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    result = analyzer.analyze(frame)
    tips = analyzer.guidance(result)
    assert any("blurry" in t.lower() for t in tips)


def test_guidance_sharp(analyzer):
    rng = np.random.RandomState(42)
    frame = rng.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    result = analyzer.analyze(frame)
    tips = analyzer.guidance(result)
    assert any("sharp" in t.lower() for t in tips)
