"""Face shape detection and fuzzy confidence scoring using MediaPipe FaceMesh.

Extracts key facial geometry (jaw width, face length, jaw angle and
forehead-to-jaw ratio) from the 468-point landmark mesh and scores each of
the five classic face shapes (Oval, Round, Square, Heart, Diamond) with a
rule-based fuzzy confidence model.
"""

from __future__ import annotations

import math

import numpy as np

LANDMARK_CHIN = 152
LANDMARK_LEFT_JAW = 234
LANDMARK_RIGHT_JAW = 454
LANDMARK_FOREHEAD = 10
LANDMARK_LEFT_TEMPLE = 54
LANDMARK_RIGHT_TEMPLE = 284

FACE_SHAPES = ["Oval", "Round", "Square", "Heart", "Diamond"]


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _angle_at(vertex: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Return the angle (degrees) formed at `vertex` by points p1 and p2."""
    v1 = p1 - vertex
    v2 = p2 - vertex
    cos_angle = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def analyze_face(image: np.ndarray, face_mesh) -> dict:
    """Analyze the dominant face in an RGB image.

    Parameters
    ----------
    image : np.ndarray
        RGB image (H, W, 3). Larger images should be pre-resized by helpers.
    face_mesh : mediapipe.solutions.face_mesh.FaceMesh
        Pre-initialized FaceMesh instance (cache at app level).

    Returns
    -------
    dict
        Contains the predicted shape, raw features, all fuzzy scores and the
        top-3 shapes normalized to 100%.
    """
    height, width = image.shape[:2]
    results = face_mesh.process(image)

    if not results.multi_face_landmarks:
        return {"error": "No face detected"}

    landmarks = results.multi_face_landmarks[0]

    def point(index: int) -> np.ndarray:
        lm = landmarks.landmark[index]
        return np.array([lm.x * width, lm.y * height])

    chin = point(LANDMARK_CHIN)
    forehead = point(LANDMARK_FOREHEAD)
    left_jaw = point(LANDMARK_LEFT_JAW)
    right_jaw = point(LANDMARK_RIGHT_JAW)
    left_temple = point(LANDMARK_LEFT_TEMPLE)
    right_temple = point(LANDMARK_RIGHT_TEMPLE)

    jaw_width = _distance(left_jaw, right_jaw)
    face_length = _distance(chin, forehead)
    temple_width = _distance(left_temple, right_temple)
    jaw_angle = _angle_at(chin, left_jaw, right_jaw)

    w_to_l = jaw_width / face_length if face_length > 0 else 0.0
    forehead_to_jaw = temple_width / jaw_width if jaw_width > 0 else 0.0
    jaw_to_forehead = jaw_width / temple_width if temple_width > 0 else 0.0

    scores: dict[str, float] = {}
    scores["Round"] = _clip(100 - abs(w_to_l - 0.85) * 400)
    scores["Oval"] = _clip(100 - abs(w_to_l - 0.72) * 400)

    if w_to_l > 0.80 and jaw_angle > 125:
        scores["Square"] = _clip((w_to_l - 0.80) * 200 + (jaw_angle - 125) + 50)
    else:
        scores["Square"] = 0.0

    if forehead_to_jaw > 1.1 and w_to_l < 0.80:
        scores["Heart"] = _clip((forehead_to_jaw - 1.1) * 200 + (0.80 - w_to_l) * 200 + 50)
    else:
        scores["Heart"] = 0.0

    if jaw_to_forehead > 1.1 and w_to_l < 0.80:
        scores["Diamond"] = _clip((jaw_to_forehead - 1.1) * 200 + (0.80 - w_to_l) * 200 + 50)
    else:
        scores["Diamond"] = 0.0

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_3 = ranked[:3]
    total = sum(score for _, score in top_3)
    normalized = [{"shape": name, "confidence": (score / total) if total > 0 else 0.0}
                  for name, score in top_3]

    predicted = top_3[0][0] if top_3 else "Unknown"

    return {
        "error": None,
        "shape": predicted,
        "scores": scores,
        "top_shapes": normalized,
        "features": {
            "jaw_width": round(jaw_width, 2),
            "face_length": round(face_length, 2),
            "temple_width": round(temple_width, 2),
            "jaw_angle": round(jaw_angle, 2),
            "w_to_l": round(w_to_l, 4),
            "forehead_to_jaw": round(forehead_to_jaw, 4),
        },
    }
