#!/usr/bin/env python3
"""Test script to analyze all images in the images folder."""

import cv2
import mediapipe as mp
from pathlib import Path
from modules import body_analyzer

# Initialize MediaPipe Pose
pose = mp.solutions.pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=True,   # required for mask-based body width measurement
)

images_dir = Path(__file__).parent / "images"
image_files = sorted(images_dir.glob("test_image*.jpg"))

print("\n" + "="*80)
print("BODY ANALYZER TEST - ALL IMAGES")
print("="*80)

for image_path in image_files:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"\n{image_path.name}: FAILED TO LOAD")
        continue
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = body_analyzer.analyze_body(image_rgb, pose)
    
    print(f"\n{image_path.name}:")
    if result.get('error'):
        print(f"  Error: {result['error']}")
    else:
        print(f"  Shape: {result['shape']}")
        print(f"  Ratio: {result['ratio']}")
        print(f"  Confidence: {result['confidence']}%")
        if result.get('pose_warning'):
            print(f"  Warning: {result['pose_warning']}")
        for w in result.get('pose_warnings', []):
            print(f"  Note: {w}")

print("\n" + "="*80 + "\n")

pose.close()
