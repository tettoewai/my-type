#!/usr/bin/env python3
"""Comprehensive body shape analyzer with extended shape support."""

import cv2
import mediapipe as mp
from pathlib import Path
from modules import body_analyzer

pose = mp.solutions.pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    smooth_landmarks=True
)

print("\n" + "="*100)
print("COMPREHENSIVE BODY SHAPE ANALYSIS - ALL SHAPES SUPPORTED")
print("="*100)

# Test with a few representative images
test_images = [
    ("images/test_image_2.jpg", "Inverted Triangle (Broad shoulders)"),
    ("images/test_image_5.jpg", "Pear (Wide hips)"),
    ("images/test_image_1.jpg", "Hourglass (Balanced curves)"),
]

print("\n📊 FEMALE BODY SHAPES SUPPORTED:")
print("  • Hourglass - Balanced shoulder/hip with defined waist")
print("  • Rectangle - Straight silhouette, minimal waist")
print("  • Pear - Wider hips than shoulders")
print("  • Inverted Triangle - Wider shoulders than hips")

print("\n" + "-"*100)

for image_path, description in test_images:
    image = cv2.imread(image_path)
    if image is None:
        continue
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Female classification
    result_female = body_analyzer.analyze_body(image_rgb, pose, gender="female")
    
    # Male classification (optional)
    result_male = body_analyzer.analyze_body(image_rgb, pose, gender="male")
    
    print(f"\n{Path(image_path).name} - {description}")
    print(f"  Female Shape:     {result_female['shape']} ({result_female['confidence']}%)")
    print(f"  Male Equivalent:  {result_male['shape']}")
    print(f"  Ratio: {result_female['ratio']} | Elbow Ratio: {result_female['elbow_ratio']}")

print("\n" + "="*100)
print("\n🎯 MALE BODY SHAPES MAPPING:")
print("  • Trapezoid ← Hourglass (Broad shoulders with taper)")
print("  • V-Taper ← Inverted Triangle (Extremely broad shoulders)")
print("  • Rectangle ← Rectangle (Straight silhouette)")
print("  • Triangle ← Pear (Wide lower body)")

print("\n" + "="*100 + "\n")

pose.close()
