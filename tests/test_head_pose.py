"""
Tests for HeadPoseEstimator.
"""
import pytest

from src.cv.head_pose import HeadPose, HeadPoseEstimator


@pytest.fixture
def estimator():
    return HeadPoseEstimator(yaw_threshold=15, pitch_threshold=10, roll_threshold=8)


# --- is_good ----------------------------------------------------------

def test_good_pose(estimator):
    assert estimator.is_good(HeadPose(yaw=5, pitch=3, roll=2))


def test_bad_yaw(estimator):
    assert not estimator.is_good(HeadPose(yaw=20, pitch=0, roll=0))


def test_bad_pitch(estimator):
    assert not estimator.is_good(HeadPose(yaw=0, pitch=15, roll=0))


def test_bad_roll(estimator):
    assert not estimator.is_good(HeadPose(yaw=0, pitch=0, roll=12))


# --- guidance ---------------------------------------------------------

def test_guidance_all_good(estimator):
    tips = estimator.guidance(HeadPose(yaw=0, pitch=0, roll=0))
    assert all(t.startswith("✓") for t in tips)


def test_guidance_turn_left(estimator):
    tips = estimator.guidance(HeadPose(yaw=20, pitch=0, roll=0))
    assert any("left" in t for t in tips)


def test_guidance_turn_right(estimator):
    tips = estimator.guidance(HeadPose(yaw=-20, pitch=0, roll=0))
    assert any("right" in t for t in tips)


def test_guidance_chin_up(estimator):
    tips = estimator.guidance(HeadPose(yaw=0, pitch=15, roll=0))
    assert any("chin" in t.lower() for t in tips)
