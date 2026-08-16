"""Rule-based expert system that maps detected features to styling advice.

Hardcoded look-up tables connect face shape -> hairstyles, skin undertone ->
makeup (or grooming) palettes and body shape -> outfit strategies. The
``gender`` parameter tunes the result set, e.g. makeup advice is replaced by
skincare/grooming guidance for men.
"""

from __future__ import annotations

HAIRSTYLE_RULES = {
    "Oval": "Blunt Bob, Long Layers, Middle Part",
    "Round": "High Ponytail, Long Curls, Side-Swept Bangs",
    "Square": "Soft Layers, Wispy Bangs, Lob (Long Bob)",
    "Heart": "Side Part, Chin-Length Bob, Layers starting at collarbone",
    "Diamond": "Side-Swept Bangs, Volume at Chin, Updo with loose strands",
}

MAKEUP_RULES = {
    "Warm": "Peach/Coral Blush, Gold Shimmer Eyeshadow, Warm Nude Lipstick",
    "Cool": "Rose/Pink Blush, Silver Glitter Eyeshadow, Berry/Mauve Lipstick",
    "Neutral": "Taupe/Mauve Blush, Bronze Shimmer Eyeshadow, Rosy Nude Lipstick",
}

SKINCARE_RULES = {
    "Warm": "Skip makeup - clean groomed skin: gentle cleanser, moisturizer, daily SPF",
    "Cool": "Skip makeup - clean groomed skin: hydrating cleanser, light moisturizer, SPF",
    "Neutral": "Skip makeup - clean groomed skin: balanced routine, moisturizer, daily SPF",
}

OUTFIT_RULES = {
    "Inverted Triangle": "A-Line Skirts, Wide-Leg Pants, V-Necks, Drop Sleeves",
    "Pear": "A-Line Dresses, Dark bottoms/Light tops, Statement Necklaces, Off-Shoulder Tops",
    "Hourglass": "Wrap Dresses, High-Waisted Skirts, Belts, Fitted Knits",
    "Rectangle": "Peplum Tops, Layered Necklaces, Belted Waists, Ruffles",
}

MALE_OUTFIT_RULES = {
    "Triangle": "Structured shoulders, horizontal-stripe tops, avoid tight-fitting hips",
    "Rectangle": "Layered fits, textured knits, lightly padded blazers",
    "Trapezoid": "Classic tailored suits, V-necks, well-fitted shoulders",
    "V-Taper": "Slim-fit shirts, tapered trousers, fitted blazers",
}

DEFAULT_FALLBACK = "Classic tailored pieces that highlight your favorite features"


def get_recommendations(face_shape: str, skin_tone: str, body_shape: str, gender: str = "female") -> dict:
    """Return a formatted dict of styling recommendations for the three traits.

    Unknown or missing values gracefully fall back to a generic suggestion so
    the UI never renders an empty recommendation card. When ``gender`` is
    "male", the makeup card is replaced by skincare/grooming advice and
    outfit suggestions use male body-shape terminology.
    """
    face_shape = face_shape if isinstance(face_shape, str) else ""
    skin_tone = skin_tone if isinstance(skin_tone, str) else ""
    body_shape = body_shape if isinstance(body_shape, str) else ""
    is_male = gender.lower() == "male"

    if is_male:
        return {
            "Hairstyle": {
                "rule": HAIRSTYLE_RULES.get(face_shape, DEFAULT_FALLBACK),
                "source": face_shape or "Unknown face shape",
            },
            "Skincare": {
                "rule": SKINCARE_RULES.get(skin_tone, DEFAULT_FALLBACK),
                "source": skin_tone or "Unknown undertone",
            },
            "Outfit": {
                "rule": MALE_OUTFIT_RULES.get(body_shape, DEFAULT_FALLBACK),
                "source": body_shape or "Unknown body shape",
            },
        }

    return {
        "Hairstyle": {
            "rule": HAIRSTYLE_RULES.get(face_shape, DEFAULT_FALLBACK),
            "source": face_shape or "Unknown face shape",
        },
        "Makeup": {
            "rule": MAKEUP_RULES.get(skin_tone, DEFAULT_FALLBACK),
            "source": skin_tone or "Unknown undertone",
        },
        "Outfit": {
            "rule": OUTFIT_RULES.get(body_shape, DEFAULT_FALLBACK),
            "source": body_shape or "Unknown body shape",
        },
    }
