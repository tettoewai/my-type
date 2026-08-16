"""Image pre-processing helpers: resize, normalization, and safe decoding."""

from __future__ import annotations

import numpy as np
import cv2
from PIL import Image

MAX_IMAGE_DIMENSION = 1024


def load_image(image_bytes: bytes) -> np.ndarray | None:
    """Decode raw uploaded bytes into an RGB numpy array (BGR/HSV unaffected by order).

    Returns None if the image cannot be decoded.
    """
    try:
        img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def resize_to_max_dimension(image: np.ndarray, max_dim: int = MAX_IMAGE_DIMENSION) -> np.ndarray:
    """Downscale the image so that its longest side is <= max_dim.

    Keeps aspect ratio intact. MediaPipe prefers reasonably-sized inputs
    for both speed and landmark accuracy.
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image
    scale = max_dim / float(longest)
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0, 1] float range for MediaPipe consumption."""
    return image.astype(np.float32) / 255.0


def pil_to_bytes(pil_image: Image.Image, fmt: str = "JPEG", quality: int = 90) -> bytes:
    """Convert a PIL image into a byte string (useful for caching)."""
    buffer = __import__("io").BytesIO()
    pil_image.save(buffer, format=fmt, quality=quality)
    return buffer.getvalue()


def to_pil(image: np.ndarray) -> Image.Image:
    """Convert a numpy RGB image to a PIL Image."""
    return Image.fromarray(image)
