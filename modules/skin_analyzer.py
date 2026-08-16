"""Skin undertone classification using MediaPipe landmarks + CIELAB color space.

Builds a binary mask over the inner cheeks using a set of facial landmarks,
extracts those pure-skin pixels, converts them to the LAB color space and
classifies Warm / Cool / Neutral from the mean A* and B* channel values.
"""

from __future__ import annotations

import numpy as np
import cv2

CHEEK_POLYGON_INDICES = [
    234, 227, 116, 117, 118, 119, 120, 121, 122, 123,
    352, 425, 426, 427, 428, 429, 430, 431, 432, 433,
]


def _build_skin_mask(image: np.ndarray, landmarks) -> np.ndarray:
    height, width = image.shape[:2]
    points = []
    for index in CHEEK_POLYGON_INDICES:
        lm = landmarks.landmark[index]
        x = int(round(lm.x * width))
        y = int(round(lm.y * height))
        points.append((min(max(x, 0), width - 1), min(max(y, 0), height - 1)))

    mask = np.zeros((height, width), dtype=np.uint8)
    contour = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillConvexPoly(mask, contour, 255)
    return mask


def analyze_skin(image: np.ndarray, face_mesh) -> dict:
    """Classify skin undertone from an RGB image.

    Parameters
    ----------
    image : np.ndarray
        RGB image (H, W, 3).
    face_mesh : mediapipe.solutions.face_mesh.FaceMesh
        Pre-initialized FaceMesh instance.

    Returns
    -------
    dict
        Contains the predicted undertone plus raw mean A* / B* values.
    """
    results = face_mesh.process(image)
    if not results.multi_face_landmarks:
        return {"error": "No face detected"}

    landmarks = results.multi_face_landmarks[0]
    mask = _build_skin_mask(image, landmarks)

    skin_pixels = image[mask > 0]
    if skin_pixels.size == 0:
        return {"error": "Unable to extract skin region"}

    MIN_SKIN_PIXELS = 500
    if skin_pixels.size < MIN_SKIN_PIXELS:
        return {
            "error": f"Insufficient skin region ({skin_pixels.size} pixels < {MIN_SKIN_PIXELS} minimum)"
        }

    lab = cv2.cvtColor(skin_pixels.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)

    a_values = lab[:, :, 1].flatten()
    b_values = lab[:, :, 2].flatten()

    a_mean = float(np.median(a_values))
    b_mean = float(np.median(b_values))
    a_std = float(np.std(a_values))
    b_std = float(np.std(b_values))

    if a_mean > 130 and b_mean > 140:
        undertone = "Warm"
    elif a_mean < 125 and b_mean < 130:
        undertone = "Cool"
    else:
        undertone = "Neutral"

    return {
        "error": None,
        "undertone": undertone,
        "a_mean": round(a_mean, 2),
        "b_mean": round(b_mean, 2),
        "a_std": round(a_std, 2),
        "b_std": round(b_std, 2),
    }
