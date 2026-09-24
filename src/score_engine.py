"""
Score fusion engine.
Combines individual CV scores into a single Photo Readiness score and
generates an ordered list of guidance messages.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .cv.face_position import FacePositionAnalyzer, PositionResult
from .cv.head_pose import HeadPose, HeadPoseEstimator
from .cv.lighting import LightingAnalyzer, LightingResult
from .cv.sharpness import SharpnessAnalyzer, SharpnessResult


@dataclass
class AnalysisResult:
    """Full analysis output for one frame / photo."""
    face_detected: bool = False
    head_pose: HeadPose = None
    lighting: LightingResult = None
    sharpness: SharpnessResult = None
    position: PositionResult = None
    pose_score: float = 0.0
    overall_score: float = 0.0
    guidance: list[str] = field(default_factory=list)


class ScoreEngine:
    """Weighted fusion of sub-scores + feedback aggregation."""

    def __init__(
        self,
        head_pose_est: HeadPoseEstimator,
        lighting_ana: LightingAnalyzer,
        sharpness_ana: SharpnessAnalyzer,
        position_ana: FacePositionAnalyzer,
        weights: dict[str, float] | None = None,
    ):
        self.head_pose_est = head_pose_est
        self.lighting_ana = lighting_ana
        self.sharpness_ana = sharpness_ana
        self.position_ana = position_ana

        self.weights = weights or {
            "head_pose": 0.20,
            "lighting": 0.25,
            "sharpness": 0.25,
            "position": 0.30,
        }

    def compute(self, result: AnalysisResult) -> AnalysisResult:
        """Fill in *overall_score* and *guidance* from the sub-results."""
        if not result.face_detected:
            result.guidance = ["⚠️ No face detected"]
            result.overall_score = 0.0
            return result

        # Sub-scores (each 0-100)
        yaw_score = max(0.0, 100 - abs(result.head_pose.yaw) * (100 / self.head_pose_est.yaw_threshold))
        pitch_score = max(0.0, 100 - abs(result.head_pose.pitch) * (100 / self.head_pose_est.pitch_threshold))
        roll_score = max(0.0, 100 - abs(result.head_pose.roll) * (100 / self.head_pose_est.roll_threshold))
        pose_score = (yaw_score + pitch_score + roll_score) / 3
        result.pose_score = round(pose_score, 1)

        scores = {
            "head_pose": pose_score,
            "lighting": result.lighting.brightness_score,
            "sharpness": result.sharpness.score,
            "position": result.position.score,
        }

        # Weighted average
        total = sum(self.weights[k] * scores[k] for k in self.weights)
        result.overall_score = round(total, 1)

        # Guidance — collect tips, put warnings first
        tips: list[str] = []
        tips.extend(self.head_pose_est.guidance(result.head_pose))
        tips.extend(self.lighting_ana.guidance(result.lighting))
        tips.extend(self.sharpness_ana.guidance(result.sharpness))
        tips.extend(self.position_ana.guidance(result.position))

        # Warnings on top, then OK messages
        warnings = [t for t in tips if not t.startswith("✓")]
        oks = [t for t in tips if t.startswith("✓")]
        result.guidance = warnings + oks

        return result
