"""
Face detection and landmark extraction using MediaPipe Tasks API (1.0+).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

_DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "checkpoints" / "face_landmarker.task"


@dataclass
class FaceLandmarks:
    landmarks: np.ndarray       # (478, 3) — x, y, z per landmark
    bbox: tuple[int, int, int, int]  # x, y, width, height
    confidence: float


class FaceDetector:
    """Detect faces and extract landmarks via MediaPipe FaceLandmarker."""

    def __init__(
        self,
        min_confidence: float = 0.7,
        min_face_size: int = 100,
        model_path: str | None = None,
    ):
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size

        model = model_path or str(_DEFAULT_MODEL)
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=model,
                delegate=BaseOptions.Delegate.CPU,  # Force CPU to avoid Metal crashes on macOS
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._ts = 0

    def detect(self, frame: np.ndarray) -> FaceLandmarks | None:
        """
        Detect a face and return its landmarks.

        Args:
            frame: BGR image from OpenCV.

        Returns:
            FaceLandmarks if a face is found, None otherwise.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._ts += 33  # ~30 fps timestamps
        result = self._landmarker.detect_for_video(mp_image, self._ts)

        if not result.face_landmarks:
            return None

        face = result.face_landmarks[0]
        h, w = frame.shape[:2]

        landmarks = np.array(
            [[lm.x * w, lm.y * h, lm.z] for lm in face],
            dtype=np.float64,
        )

        xs, ys = landmarks[:, 0], landmarks[:, 1]
        pad = 20
        x_min = max(0, int(xs.min()) - pad)
        y_min = max(0, int(ys.min()) - pad)
        x_max = min(w, int(xs.max()) + pad)
        y_max = min(h, int(ys.max()) + pad)
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        if bbox[2] < self.min_face_size or bbox[3] < self.min_face_size:
            return None

        return FaceLandmarks(
            landmarks=landmarks, bbox=bbox, confidence=self.min_confidence
        )

    def draw(self, frame: np.ndarray, result: FaceLandmarks) -> np.ndarray:
        out = frame.copy()
        x, y, w, h = result.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for pt in result.landmarks[::10]:
            cv2.circle(out, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)
        return out

    def close(self):
        self._landmarker.close()
