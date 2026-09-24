#!/usr/bin/env python3
"""
Live webcam demo — runs all CV modules and overlays scores + guidance.

Usage:
    python main.py

Press 'q' to quit, 's' to save the current frame.
"""
import sys

import cv2
import numpy as np

from src.cv import (
    FaceDetector,
    FacePositionAnalyzer,
    HeadPoseEstimator,
    LightingAnalyzer,
    SharpnessAnalyzer,
)
from src.score_engine import AnalysisResult, ScoreEngine
from src.utils import load_config

# ── colours ───────────────────────────────────────────────────────────
GREEN = (0, 220, 80)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
BAR_BG = (50, 50, 50)


def draw_hud(frame: np.ndarray, result: AnalysisResult) -> np.ndarray:
    """Draw a heads-up display with scores and guidance on the frame."""
    out = frame.copy()
    h, _w = out.shape[:2]

    # Semi-transparent sidebar
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (320, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    y = 30
    _text(out, "AI PHOTO COACH", 10, y, scale=0.7, color=GREEN, thick=2)
    y += 35

    if not result.face_detected:
        _text(out, "No face detected", 10, y, color=RED)
        return out

    # Score bars
    bars = {
        "Head Pose": result.pose_score,
        "Lighting": result.lighting.brightness_score if result.lighting else 0,
        "Sharpness": result.sharpness.score if result.sharpness else 0,
        "Position": result.position.score if result.position else 0,
    }
    for label, score in bars.items():
        _text(out, label, 10, y, scale=0.5, color=GRAY)
        y += 20
        _bar(out, 10, y, 200, 14, score)
        _text(out, f"{score:.0f}", 220, y + 11, scale=0.4, color=WHITE)
        y += 22

    # Overall
    y += 10
    _text(out, f"READINESS: {result.overall_score:.0f}%", 10, y,
          scale=0.7, color=GREEN if result.overall_score >= 85 else RED, thick=2)
    y += 35

    # Status
    if result.overall_score >= 85:
        _text(out, "READY TO CAPTURE", 10, y, color=GREEN, thick=2)
    else:
        _text(out, "ADJUSTING...", 10, y, color=RED)
    y += 30

    # Guidance
    y += 10
    _text(out, "Guidance:", 10, y, scale=0.5, color=GRAY)
    y += 20
    for tip in result.guidance[:6]:
        color = GREEN if tip.startswith("✓") else RED
        _text(out, tip, 15, y, scale=0.45, color=color)
        y += 18

    return out


def _text(img, text, x, y, scale=0.55, color=WHITE, thick=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                cv2.LINE_AA)


def _bar(img, x, y, w, h, pct):
    cv2.rectangle(img, (x, y), (x + w, y + h), BAR_BG, -1)
    fill = int(w * max(0, min(pct, 100)) / 100)
    if fill > 0:
        color = GREEN if pct >= 70 else (0, 180, 255) if pct >= 40 else RED
        cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), GRAY, 1)


def main():
    # Load config
    try:
        cfg = load_config()
    except FileNotFoundError:
        cfg = {}

    # Initialise modules
    face_det = FaceDetector(
        min_confidence=cfg.get("face_detection", {}).get("min_confidence", 0.7),
        min_face_size=cfg.get("face_detection", {}).get("min_face_size", 100),
    )
    head_pose = HeadPoseEstimator(
        yaw_threshold=cfg.get("head_pose", {}).get("yaw_threshold", 15),
        pitch_threshold=cfg.get("head_pose", {}).get("pitch_threshold", 10),
        roll_threshold=cfg.get("head_pose", {}).get("roll_threshold", 8),
    )
    lighting = LightingAnalyzer(
        min_brightness=cfg.get("lighting", {}).get("min_brightness", 40),
        max_brightness=cfg.get("lighting", {}).get("max_brightness", 220),
        optimal_brightness=cfg.get("lighting", {}).get("optimal_brightness", 120),
    )
    sharpness = SharpnessAnalyzer(
        blur_threshold=cfg.get("sharpness", {}).get("blur_threshold", 20),
        face_blur_threshold=cfg.get("sharpness", {}).get("face_blur_threshold", 15),
    )
    position = FacePositionAnalyzer()
    engine = ScoreEngine(head_pose, lighting, sharpness, position)

    skip = cfg.get("camera", {}).get("process_every_n_frames", 3)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open webcam")
        sys.exit(1)

    print("AI Photo Coach — press 'q' to quit, 's' to save a frame")

    frame_idx = 0
    last_result = AnalysisResult()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only run analysis every N frames for performance
        if frame_idx % skip == 0:
            result = AnalysisResult()
            face = face_det.detect(frame)

            if face is not None:
                result.face_detected = True
                result.head_pose = head_pose.estimate(face.landmarks)
                result.lighting = lighting.analyze(frame, face.bbox)
                result.sharpness = sharpness.analyze(frame, face.bbox)
                result.position = position.analyze(frame.shape, face.bbox)

                # Draw face bbox
                x, y, w, h = face.bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)

            engine.compute(result)
            last_result = result

        # Always draw HUD (reuse last result on skipped frames)
        display = draw_hud(frame, last_result)
        cv2.imshow("AI Photo Coach", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"capture_{frame_idx}.jpg"
            cv2.imwrite(fname, frame)
            print(f"Saved {fname}")

        frame_idx += 1

    cap.release()
    face_det.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
