"""Body shape estimation using MediaPipe Pose landmarks + segmentation mask.

Primary approach (when enable_segmentation=True is used)
---------------------------------------------------------
Measures actual pixel body widths directly from the segmentation mask at
three anatomical levels:

  shoulder_w  : pixel width at the EXACT shoulder-landmark Y row
  waist_w     : minimum pixel width in the 35–65 % zone of the torso,
                skipped if the elbows are in that zone (arm contamination)
  hip_w       : pixel width at the EXACT hip-landmark Y row

Two key ratios and one profile shape flag are derived:
  s2h        = shoulder_w / hip_w
  w2h        = waist_w    / hip_w   (None if waist unmeasurable)
  mid_widens = True when the body profile is widest in the 20–55 % zone
               (bust/chest bulge typical of Inverted Triangle silhouettes)

Fallback (when no segmentation mask is available)
-------------------------------------------------
Uses the raw landmark shoulder/hip ratio (empirical range 1.48–2.24) with
recalibrated fuzzy membership centres for this range.
"""

from __future__ import annotations

import numpy as np

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26

LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16

BODY_SHAPES = ["Inverted Triangle", "Pear", "Hourglass", "Rectangle"]

_VIS_THRESHOLD = 0.5
_VIS_RELAXED   = 0.3
_SEG_THRESHOLD = 0.5


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _visibility(landmark, threshold: float = _VIS_THRESHOLD) -> bool:
    return bool(landmark.visibility > threshold)


def _fuzzy(value: float, center: float, half_width: float) -> float:
    """Triangular fuzzy membership: 1.0 at center, 0.0 at center ± half_width."""
    return float(np.clip(1.0 - abs(value - center) / half_width, 0.0, 1.0))


# ── Segmentation mask helpers ─────────────────────────────────────────────────

def _row_width(mask: np.ndarray, y: int) -> float:
    """Foreground pixel width at row y. Returns 0 if empty or out of range."""
    h, w = mask.shape
    if not (0 <= y < h):
        return 0.0
    row = mask[y, :] > _SEG_THRESHOLD
    if not row.any():
        return 0.0
    left  = int(np.argmax(row))
    right = int(w - 1 - np.argmax(row[::-1]))
    return float(max(right - left, 0))


def _extract_mask_measurements(
    mask: np.ndarray,
    y_shoulder: int, y_hip: int, y_elbow: int,
) -> tuple[float, float | None, float, bool]:
    """
    Returns (shoulder_w, waist_w_or_None, hip_w, mid_widens).

    shoulder_w  – exact pixel width at the shoulder-landmark Y row
    waist_w     – minimum width in 35–65 % torso zone; None if arms are
                  in that zone (elbow Y falls within it)
    hip_w       – exact pixel width at the hip-landmark Y row
    mid_widens  – True when the 20–55 % zone is wider than both ends by
                  ≥ 8 % (Inverted Triangle / bust-dominant silhouette)
    """
    shoulder_w = _row_width(mask, y_shoulder)
    hip_w      = _row_width(mask, y_hip)

    if shoulder_w <= 0 or hip_w <= 0:
        return 0.0, None, 0.0, False

    torso = max(y_hip - y_shoulder, 1)

    # Sample 50 evenly spaced rows across the torso
    n = 50
    ys = np.linspace(y_shoulder, y_hip, n, dtype=int)
    profile = np.array([_row_width(mask, int(y)) for y in ys])

    # ── Waist: minimum in 35–65 % zone with row-level arm exclusion ──────────
    i_w_s = int(n * 0.35)
    i_w_e = int(n * 0.65)

    elbow_rel = (y_elbow - y_shoulder) / torso
    pcts = np.linspace(0, 1, n)

    # Exclude only the rows that are within ±8 % of the torso around the elbow Y
    # to avoid arm contamination while keeping the rest of the waist zone.
    elbow_lo = elbow_rel - 0.08
    elbow_hi = elbow_rel + 0.08
    waist_vals = [
        profile[idx]
        for idx in range(i_w_s, i_w_e)
        if not (elbow_lo <= pcts[idx] <= elbow_hi) and profile[idx] > 0
    ]

    if not waist_vals:
        waist_w = None
    else:
        waist_w = float(np.percentile(waist_vals, 5))

    # ── Mid-widening: Inverted Triangle / bust-dominant pattern ───────────────
    i_m_s = int(n * 0.20)
    i_m_e = int(n * 0.55)
    mid_zone = profile[i_m_s:i_m_e]
    valid_mid = mid_zone[mid_zone > 0]
    if len(valid_mid) > 0:
        mid_max = float(np.max(valid_mid))
        mid_widens = (mid_max > shoulder_w * 1.08 and mid_max > hip_w * 1.05)
    else:
        mid_widens = False

    return shoulder_w, waist_w, hip_w, mid_widens


# ── Female classifier ─────────────────────────────────────────────────────────

def _classify_female(
    shoulder_w: float, waist_w: float | None, hip_w: float,
    mid_widens: bool, lm_ratio: float,
) -> dict[str, float]:
    """Return fuzzy shape scores for female body types."""
    if hip_w <= 0:
        # Degenerate: no valid measurements
        return {"Pear": 0.1, "Rectangle": 0.25, "Hourglass": 0.25, "Inverted Triangle": 0.1}

    s2h = shoulder_w / hip_w
    w2h = waist_w / hip_w if waist_w is not None else None

    # ── Case 1: Clear Hourglass — waist noticeably narrower than hips ────────
    # W/H < 0.88 AND waist narrower than shoulder (eliminates dress-flare noise)
    hourglass_waist = (
        w2h is not None
        and w2h < 0.88
        and (waist_w or 0) < shoulder_w * 0.92
    )

    if hourglass_waist:
        waist_strength = max(0.0, (0.88 - w2h) / 0.20)
        scores = {
            "Pear":              _fuzzy(s2h, 0.85, 0.20) * 0.35,
            "Rectangle":         _fuzzy(s2h, 1.05, 0.25) * 0.25,
            "Hourglass":         min(1.0, 0.62 + waist_strength * 0.38),
            "Inverted Triangle": _fuzzy(s2h, 1.42, 0.28) * 0.45,
        }
        return scores

    # ── Case 2: Inverted Triangle — torso widens toward bust/chest ───────────
    if mid_widens:
        inv_score = min(1.0, 0.55 + max(0.0, lm_ratio - 1.55) * 0.25)
        scores = {
            "Pear":              0.08,
            "Rectangle":         0.22 if s2h < 1.05 else 0.08,
            "Hourglass":         0.12,
            "Inverted Triangle": max(0.52, inv_score),
        }
        return scores

    # ── Case 3: General scoring on s2h axis with w2h modulation ──────────────
    # Calibrated s2h ranges from pixel measurements:
    #   Pear            ~0.52–0.95   (hips at least as wide as shoulders)
    #   Rectangle       ~0.85–1.60   (wide range; clothing & angle vary)
    #   Hourglass       ~0.90–1.30   (shoulders ≥ hips, uses waist signal)
    #   Inv Triangle    ~1.10–1.60   (shoulders clearly wider)
    scores = {
        "Pear":              _fuzzy(s2h, 0.85, 0.22),
        "Rectangle":         _fuzzy(s2h, 1.05, 0.28),
        "Hourglass":         _fuzzy(s2h, 1.10, 0.26),
        "Inverted Triangle": _fuzzy(s2h, 1.42, 0.30),
    }

    # Waist-to-hip modulation
    if w2h is not None:
        if w2h < 0.93:
            # Moderate waist narrowing: lean Hourglass over Rectangle
            scores["Hourglass"] = min(1.0, scores["Hourglass"] * 1.30)
            scores["Rectangle"] = max(0.0, scores["Rectangle"] * 0.65)
        elif 0.93 <= w2h <= 1.08:
            # Straight/rectangular torso; suppress Hourglass
            scores["Rectangle"] = min(1.0, scores["Rectangle"] * 1.25)
            scores["Hourglass"] = max(0.0, scores["Hourglass"] * 0.45)
        # w2h > 1.08: dress flare or clothing — don't suppress Hourglass further
        # (could be Hourglass with flared skirt — trust LM ratio instead)

    # LM ratio as cross-check for high shoulder or hip dominance
    if lm_ratio > 1.85 and s2h > 1.10:
        # High LM + high pixel S/H → Hourglass or Inverted Triangle
        scores["Hourglass"]          = min(1.0, scores["Hourglass"]          * 1.20)
        scores["Inverted Triangle"]  = min(1.0, scores["Inverted Triangle"]  * 1.20)
    if lm_ratio > 1.90:
        scores["Inverted Triangle"]  = min(1.0, scores["Inverted Triangle"]  * 1.25)

    # Clear boundary overrides
    if s2h > 1.40:
        scores["Inverted Triangle"]  = min(1.0, scores["Inverted Triangle"]  * 1.35)
        scores["Pear"]               = max(0.0, scores["Pear"]               * 0.20)
    if s2h < 0.82:
        scores["Pear"]               = min(1.0, scores["Pear"]               * 1.45)
        scores["Inverted Triangle"]  = max(0.0, scores["Inverted Triangle"]  * 0.15)
        scores["Hourglass"]          = max(0.0, scores["Hourglass"]          * 0.50)

    return scores


def _classify_female_lm(lm_ratio: float) -> dict[str, float]:
    """Landmark-only fallback for female shapes (ratio range ~1.48–2.24)."""
    scores = {
        "Pear":              _fuzzy(lm_ratio, 1.57, 0.22),
        "Rectangle":         _fuzzy(lm_ratio, 1.65, 0.22),
        "Hourglass":         _fuzzy(lm_ratio, 1.72, 0.25),
        "Inverted Triangle": _fuzzy(lm_ratio, 1.95, 0.32),
    }
    return scores


# ── Male classifier ───────────────────────────────────────────────────────────

def _classify_male(
    shoulder_w: float, hip_w: float, lm_ratio: float,
) -> dict[str, float]:
    s2h = shoulder_w / hip_w if hip_w > 0 else lm_ratio / 1.5
    scores = {
        "Triangle":  _fuzzy(s2h, 0.78, 0.22),
        "Rectangle": _fuzzy(s2h, 0.98, 0.22),
        "Trapezoid": _fuzzy(s2h, 1.22, 0.28),
        "V-Taper":   _fuzzy(s2h, 1.52, 0.35),
    }
    return scores


def _classify_male_lm(lm_ratio: float) -> dict[str, float]:
    scores = {
        "Triangle":  _fuzzy(lm_ratio, 1.52, 0.25),
        "Rectangle": _fuzzy(lm_ratio, 1.65, 0.25),
        "Trapezoid": _fuzzy(lm_ratio, 1.82, 0.28),
        "V-Taper":   _fuzzy(lm_ratio, 2.05, 0.35),
    }
    return scores


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_body(image: np.ndarray, pose, pose_results=None, gender: str = "female") -> dict:
    """Estimate body shape from an RGB image.

    For best results, initialise Pose with ``enable_segmentation=True``::

        mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=True,
        )

    Args:
        image: RGB image array (H × W × 3, uint8)
        pose: MediaPipe Pose instance
        pose_results: Pre-computed pose results (optional)
        gender: "female" or "male"

    Returns:
        dict with keys: error, shape, s_to_h_ratio, w_to_h_ratio,
        raw_lm_ratio, shoulder_width, hip_width, confidence, scores,
        pose_warnings.
    """
    height, width = image.shape[:2]
    results = pose_results if pose_results is not None else pose.process(image)

    if not results.pose_landmarks:
        return {"error": "No person detected"}

    landmarks = results.pose_landmarks.landmark

    def point(index: int) -> np.ndarray:
        lm = landmarks[index]
        return np.array([lm.x * width, lm.y * height])

    left_shoulder  = point(LEFT_SHOULDER)
    right_shoulder = point(RIGHT_SHOULDER)
    left_hip       = point(LEFT_HIP)
    right_hip      = point(RIGHT_HIP)
    left_knee      = point(LEFT_KNEE)
    right_knee     = point(RIGHT_KNEE)
    left_elbow     = point(LEFT_ELBOW)
    right_elbow    = point(RIGHT_ELBOW)
    left_wrist     = point(LEFT_WRIST)
    right_wrist    = point(RIGHT_WRIST)

    mid_shoulder = (left_shoulder + right_shoulder) / 2.0
    mid_hip      = (left_hip      + right_hip)      / 2.0
    mid_knee     = (left_knee     + right_knee)     / 2.0
    mid_elbow    = (left_elbow    + right_elbow)    / 2.0

    pose_warnings: list[str] = []

    # ── Visibility gates ──────────────────────────────────────────────────────
    if not (_visibility(landmarks[LEFT_SHOULDER]) and _visibility(landmarks[RIGHT_SHOULDER])):
        return {"error": "Shoulders not fully visible"}

    if not (_visibility(landmarks[LEFT_HIP]) and _visibility(landmarks[RIGHT_HIP])):
        if not (_visibility(landmarks[LEFT_HIP], _VIS_RELAXED) and
                _visibility(landmarks[RIGHT_HIP], _VIS_RELAXED)):
            return {"error": "Hips not visible – please upload a photo showing at least the waist"}
        pose_warnings.append("Hip landmarks have low confidence; result may be less accurate")

    knees_visible = (
        _visibility(landmarks[LEFT_KNEE]) and _visibility(landmarks[RIGHT_KNEE])
    )

    if mid_shoulder[1] >= mid_hip[1]:
        return {"error": "Person appears not upright; please upload a standing full-body photo"}

    if knees_visible and mid_hip[1] >= mid_knee[1]:
        pose_warnings.append("Person may be sitting or crouching; accuracy may be reduced")

    # ── Landmark measurements (always, as fallback / cross-check) ─────────────
    lm_shoulder_w = _distance(left_shoulder, right_shoulder)
    lm_hip_w      = _distance(left_hip,      right_hip)
    lm_ratio      = lm_shoulder_w / lm_hip_w if lm_hip_w > 0 else 1.7

    y_shoulder = int(mid_shoulder[1])
    y_hip      = int(mid_hip[1])
    y_elbow    = int(mid_elbow[1])

    # ── Segmentation mask measurements (primary) ──────────────────────────────
    mask = getattr(results, "segmentation_mask", None)
    shoulder_px = waist_px = hip_px = 0.0
    waist_px_val: float | None = None
    mid_widens = False
    using_mask = False

    if mask is not None and mask.shape[0] > 0 and y_hip > y_shoulder:
        shoulder_px, waist_px_val, hip_px, mid_widens = _extract_mask_measurements(
            mask, y_shoulder, y_hip, y_elbow
        )
        if shoulder_px > 10 and hip_px > 10:
            using_mask = True
            waist_px = waist_px_val if waist_px_val is not None else 0.0
            if waist_px_val is None:
                pose_warnings.append("Waist hidden by arms; waist measurement skipped")
        else:
            pose_warnings.append("Segmentation mask unreliable; using landmark ratios only")

    if not using_mask:
        pose_warnings.append("No segmentation mask; estimate based on landmark ratios only")

    # ── Classify ──────────────────────────────────────────────────────────────
    if using_mask:
        if gender.lower() == "male":
            scores = _classify_male(shoulder_px, hip_px, lm_ratio)
        else:
            scores = _classify_female(shoulder_px, waist_px_val, hip_px, mid_widens, lm_ratio)
        display_s2h = shoulder_px / hip_px if hip_px > 0 else 0.0
        display_w2h = (waist_px_val / hip_px) if (waist_px_val and hip_px > 0) else None
    else:
        if gender.lower() == "male":
            scores = _classify_male_lm(lm_ratio)
        else:
            scores = _classify_female_lm(lm_ratio)
        display_s2h = lm_ratio
        display_w2h = None

    body_shape = max(scores, key=scores.get)
    selected_confidence = scores[body_shape]

    if pose_warnings:
        selected_confidence *= 0.85

    return {
        "error": None,
        "shape": body_shape,
        "s_to_h_ratio": round(display_s2h, 4),
        "w_to_h_ratio": round(display_w2h, 4) if display_w2h is not None else None,
        "raw_lm_ratio": round(lm_ratio, 4),
        # Legacy compatibility
        "ratio": round(display_s2h, 4),
        "raw_ratio": round(lm_ratio, 4),
        "shoulder_width": round(shoulder_px if using_mask else lm_shoulder_w, 2),
        "hip_width": round(hip_px if using_mask else lm_hip_w, 2),
        "confidence": round(selected_confidence * 100.0, 1),
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "pose_warnings": pose_warnings,
    }
