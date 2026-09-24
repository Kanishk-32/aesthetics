"""
Head-pose estimation from MediaPipe facial landmarks.
Outputs yaw, pitch, roll in degrees.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class HeadPose:
    """Head orientation in degrees."""
    yaw: float    # positive → looking right
    pitch: float  # positive → looking down
    roll: float   # positive → tilted clockwise


class HeadPoseEstimator:
    """
    Estimate head pose from 468 facial landmarks using key-point geometry.

    No external 3-D model is needed — we derive angles from the relative
    positions of eyes, nose, and chin.
    """

    # MediaPipe Face Mesh indices
    NOSE_TIP = 1
    FOREHEAD = 10
    CHIN = 152
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263

    def __init__(
        self,
        yaw_threshold: float = 15.0,
        pitch_threshold: float = 10.0,
        roll_threshold: float = 8.0,
    ):
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.roll_threshold = roll_threshold

    def estimate(self, landmarks: np.ndarray) -> HeadPose:
        """
        Compute yaw / pitch / roll from landmark array (468, 3).
        """
        nose = landmarks[self.NOSE_TIP]
        forehead = landmarks[self.FOREHEAD]
        chin = landmarks[self.CHIN]
        left_eye = landmarks[self.LEFT_EYE_OUTER]
        right_eye = landmarks[self.RIGHT_EYE_OUTER]

        eye_center = (left_eye + right_eye) / 2
        eye_width = np.linalg.norm(right_eye[:2] - left_eye[:2])

        # Yaw — horizontal nose offset relative to eye span
        nose_offset = (nose[0] - eye_center[0]) / max(eye_width, 1)
        yaw = float(np.degrees(np.arcsin(np.clip(nose_offset, -1, 1))) * 1.5)

        # Pitch — vertical nose position relative to face height
        face_height = max(chin[1] - forehead[1], 1)
        nose_ratio = (nose[1] - eye_center[1]) / face_height
        pitch = float((nose_ratio - 0.3) * 60)

        # Roll — angle of the inter-eye line
        eye_vec = right_eye[:2] - left_eye[:2]
        roll = float(np.degrees(np.arctan2(eye_vec[1], eye_vec[0])))

        return HeadPose(yaw=yaw, pitch=pitch, roll=roll)

    def is_good(self, pose: HeadPose) -> bool:
        """True when yaw, pitch, and roll are all within threshold."""
        return (
            abs(pose.yaw) <= self.yaw_threshold
            and abs(pose.pitch) <= self.pitch_threshold
            and abs(pose.roll) <= self.roll_threshold
        )

    def guidance(self, pose: HeadPose) -> list[str]:
        """Return plain-English suggestions based on the current pose."""
        tips: list[str] = []

        if abs(pose.yaw) > self.yaw_threshold:
            direction = "left" if pose.yaw > 0 else "right"
            tips.append(f"Turn your face slightly {direction}")
        else:
            tips.append("✓ Head angle (left/right) is good")

        if abs(pose.pitch) > self.pitch_threshold:
            tips.append("Lift your chin" if pose.pitch > 0 else "Lower your chin")
        else:
            tips.append("✓ Head angle (up/down) is good")

        if abs(pose.roll) > self.roll_threshold:
            direction = "left" if pose.roll > 0 else "right"
            tips.append(f"Tilt your head slightly {direction}")
        else:
            tips.append("✓ Head tilt is good")

        return tips
