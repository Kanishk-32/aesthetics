"""
Lighting analysis — brightness, contrast, and exposure assessment.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LightingResult:
    """Lighting analysis output."""
    brightness: float       # mean pixel intensity (0-255)
    brightness_score: float # 0-100 (100 = optimal)
    contrast: float         # std-dev of pixel intensity
    exposure: str           # "underexposed" | "good" | "overexposed"
    face_brightness: float  # brightness of the face region only


class LightingAnalyzer:
    """Evaluate lighting quality from a BGR frame."""

    def __init__(
        self,
        min_brightness: int = 40,
        max_brightness: int = 220,
        optimal_brightness: int = 120,
    ):
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.optimal_brightness = optimal_brightness

    def analyze(
        self,
        frame: np.ndarray,
        face_bbox: tuple[int, int, int, int] | None = None,
    ) -> LightingResult:
        """
        Analyze lighting in *frame*.

        Args:
            frame: BGR image.
            face_bbox: Optional (x, y, w, h) to restrict analysis to
                       the face region.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        if face_bbox is not None:
            x, y, w, h = face_bbox
            face_brightness = float(np.mean(gray[y : y + h, x : x + w]))
        else:
            face_brightness = brightness

        if face_brightness < self.min_brightness:
            exposure = "underexposed"
        elif face_brightness > self.max_brightness:
            exposure = "overexposed"
        else:
            exposure = "good"

        score = self._score(face_brightness)

        return LightingResult(
            brightness=brightness,
            brightness_score=score,
            contrast=contrast,
            exposure=exposure,
            face_brightness=face_brightness,
        )

    def _score(self, brightness: float) -> float:
        """0-100 score that peaks at *optimal_brightness*."""
        distance = abs(brightness - self.optimal_brightness)
        max_dist = max(
            self.optimal_brightness - self.min_brightness,
            self.max_brightness - self.optimal_brightness,
        )
        return max(0.0, min(100.0, 100 * (1 - distance / max_dist)))

    def guidance(self, result: LightingResult) -> list[str]:
        tips: list[str] = []
        if result.exposure == "underexposed":
            tips.append("⚠️ Too dark — move toward a light source")
        elif result.exposure == "overexposed":
            tips.append("⚠️ Too bright — move away from the light")
        else:
            tips.append("✓ Lighting is good")

        if result.contrast < 30:
            tips.append("⚠️ Low contrast — try a different lighting angle")
        return tips
