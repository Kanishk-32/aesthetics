"""
Face position and composition analysis.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class PositionResult:
    """Face position analysis output."""
    center: tuple[float, float]  # (x, y) normalised 0-1
    face_size: float             # face height / frame height
    horizontal: str              # "left" | "center" | "right"
    vertical: str                # "top" | "center" | "bottom"
    size_status: str             # "too_small" | "good" | "too_large"
    score: float                 # composition score 0-100


class FacePositionAnalyzer:
    """Check whether the face is well-framed inside the camera view."""

    def __init__(
        self,
        center_threshold: float = 0.15,
        min_face_size: float = 0.15,
        max_face_size: float = 0.60,
        optimal_face_size: float = 0.35,
    ):
        self.center_threshold = center_threshold
        self.min_face_size = min_face_size
        self.max_face_size = max_face_size
        self.optimal_face_size = optimal_face_size

    def analyze(
        self,
        frame_shape: tuple[int, ...],
        face_bbox: tuple[int, int, int, int],
    ) -> PositionResult:
        """
        Analyze face position within the frame.

        Args:
            frame_shape: (height, width, ...) of the image.
            face_bbox: (x, y, w, h) bounding box.
        """
        fh, fw = frame_shape[:2]
        x, y, w, h = face_bbox

        cx = (x + w / 2) / fw
        cy = (y + h / 2) / fh
        face_size = h / fh

        # Horizontal
        if cx < 0.5 - self.center_threshold:
            horizontal = "left"
        elif cx > 0.5 + self.center_threshold:
            horizontal = "right"
        else:
            horizontal = "center"

        # Vertical
        if cy < 0.5 - self.center_threshold:
            vertical = "top"
        elif cy > 0.5 + self.center_threshold:
            vertical = "bottom"
        else:
            vertical = "center"

        # Size
        if face_size < self.min_face_size:
            size_status = "too_small"
        elif face_size > self.max_face_size:
            size_status = "too_large"
        else:
            size_status = "good"

        # Score
        dist = np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
        size_dev = abs(face_size - self.optimal_face_size)
        score = max(0.0, min(100.0, 100 - dist * 50 - size_dev * 100))

        return PositionResult(
            center=(cx, cy),
            face_size=face_size,
            horizontal=horizontal,
            vertical=vertical,
            size_status=size_status,
            score=score,
        )

    def guidance(self, result: PositionResult) -> list[str]:
        tips: list[str] = []

        # Horizontal
        if result.horizontal == "left":
            tips.append("Move right →")
        elif result.horizontal == "right":
            tips.append("← Move left")
        else:
            tips.append("✓ Centered horizontally")

        # Vertical
        if result.vertical == "top":
            tips.append("Move down ↓")
        elif result.vertical == "bottom":
            tips.append("Move up ↑")
        else:
            tips.append("✓ Centered vertically")

        # Distance
        if result.size_status == "too_small":
            tips.append("Move closer — face is too small")
        elif result.size_status == "too_large":
            tips.append("Move back — face is too close")
        else:
            tips.append("✓ Good distance")

        return tips
