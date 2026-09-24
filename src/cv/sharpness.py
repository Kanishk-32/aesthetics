"""
Sharpness / blur detection via Laplacian variance.
Handles both full-frame and face-crop analysis with appropriate thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SharpnessResult:
    """Sharpness analysis output."""
    laplacian_var: float    # raw Laplacian variance
    score: float            # 0-100
    is_blurry: bool


class SharpnessAnalyzer:
    """
    Detect blur using the variance of the Laplacian.

    Note: Face crops naturally have lower variance than full frames because
    skin is smooth. We use a lower threshold for face regions (15.0) vs
    full frames (30.0).
    """

    def __init__(
        self,
        blur_threshold: float = 20.0,
        face_blur_threshold: float = 15.0,
    ):
        self.blur_threshold = blur_threshold
        self.face_blur_threshold = face_blur_threshold

    def analyze(
        self,
        frame: np.ndarray,
        face_bbox: tuple[int, int, int, int] | None = None,
    ) -> SharpnessResult:
        """
        Compute sharpness of *frame* (or the face region inside it).
        When *face_bbox* is given, we crop to the face and use a lower
        threshold since human skin has naturally low edge variance.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        is_face_crop = face_bbox is not None

        if is_face_crop:
            x, y, w, h = face_bbox
            # Bound crop to image dimensions
            ih, iw = gray.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(iw, x + w), min(ih, y + h)
            if (x2 - x1) > 10 and (y2 - y1) > 10:
                gray = gray[y1:y2, x1:x2]

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        threshold = self.face_blur_threshold if is_face_crop else self.blur_threshold

        # Score mapping:
        # - At threshold: score = 60 (acceptable)
        # - At 2x threshold: score = 85 (good)
        # - At 4x threshold+: score = 100 (excellent)
        # - Below threshold: scales down to 0
        if lap_var >= threshold:
            # 60 to 100 range
            excess = lap_var - threshold
            score = min(100.0, 60.0 + (excess / (threshold * 3)) * 40.0)
        else:
            # 0 to 60 range
            score = max(0.0, (lap_var / threshold) * 60.0)

        return SharpnessResult(
            laplacian_var=lap_var,
            score=round(score, 1),
            is_blurry=lap_var < threshold,
        )

    def guidance(self, result: SharpnessResult) -> list[str]:
        if result.is_blurry:
            return ["⚠️ Image is blurry — hold the camera steadier"]
        return ["✓ Image is sharp"]
