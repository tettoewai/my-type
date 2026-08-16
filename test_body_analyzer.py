#!/usr/bin/env python3
"""Test script for body analyzer improvements."""

import cv2
import mediapipe as mp
from PIL import Image
from pathlib import Path
from modules import body_analyzer

# Load the test image
test_image_path = Path(__file__).parent / "test_image.jpg"

if not test_image_path.exists():
    print(f"Test image not found at {test_image_path}")
    print("Please save the test image as test_image.jpg in the project root")
    exit(1)

# Load image
image = cv2.imread(str(test_image_path))
if image is None:
    print(f"Failed to load image from {test_image_path}")
    exit(1)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize MediaPipe Pose
pose = mp.solutions.pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    smooth_landmarks=True
)

# Analyze body shape
result = body_analyzer.analyze_body(image_rgb, pose)

print("\n" + "="*60)
print("BODY ANALYZER TEST RESULTS")
print("="*60)
for key, value in result.items():
    if key == "scores":
        print(f"\n{key}:")
        for shape, score in value.items():
            print(f"  {shape}: {score:.4f}")
    else:
        print(f"{key}: {value}")
print("="*60 + "\n")

pose.close()
