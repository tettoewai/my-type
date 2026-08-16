#!/usr/bin/env python3
"""Debug script to see detailed measurements."""

import cv2
import mediapipe as mp
from pathlib import Path
from modules import body_analyzer

pose = mp.solutions.pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    smooth_landmarks=True
)

images_dir = Path(__file__).parent / "images"
image_files = sorted(images_dir.glob("test_image*.jpg"))[:3]  # First 3 images

print("\n" + "="*100)
print("DETAILED BODY MEASUREMENTS")
print("="*100)

for image_path in image_files:
    image = cv2.imread(str(image_path))
    if image is None:
        continue
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = body_analyzer.analyze_body(image_rgb, pose)
    
    print(f"\n{image_path.name}:")
    print(f"  Shoulder width: {result['shoulder_width']}px")
    print(f"  Hip width: {result['hip_width']}px")
    print(f"  Raw ratio: {result['raw_ratio']}")
    print(f"  Perspective factor: {result['perspective_factor']}")
    print(f"  Corrected ratio: {result['ratio']}")
    print(f"  Shape: {result['shape']} ({result['confidence']}%)")

print("\n" + "="*100 + "\n")

pose.close()
