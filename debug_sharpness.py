#!/usr/bin/env python3
"""Debug script to measure actual Laplacian variance from your webcam."""
import sys

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: cannot open webcam")
    sys.exit(1)

print("Measuring sharpness from your webcam...")
print("Press 'q' to quit, 's' to sample 10 frames\n")

samples = []
sampling = False
sample_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if sampling:
        samples.append(lap_var)
        sample_count += 1
        print(f"Sample {sample_count}/10: {lap_var:.1f}")
        if sample_count >= 10:
            sampling = False
            print("\n--- Statistics ---")
            print(f"Min:    {min(samples):.1f}")
            print(f"Max:    {max(samples):.1f}")
            print(f"Mean:   {np.mean(samples):.1f}")
            print(f"Median: {np.median(samples):.1f}")
            print(f"\nRecommended blur_threshold: {np.median(samples) * 0.4:.1f}")
            print("(40% of median — frames below this are genuinely blurry)\n")

    # Draw current value
    text = f"Laplacian var: {lap_var:.1f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Press 's' to sample 10 frames", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.imshow("Sharpness Debug", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and not sampling:
        print("Sampling...")
        sampling = True
        sample_count = 0
        samples = []

cap.release()
cv2.destroyAllWindows()
