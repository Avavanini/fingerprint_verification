"""
normalize.py — Image loading and normalization utilities.

Functions:
    load_image(path) -> np.ndarray
    normalize(image) -> np.ndarray
"""

import cv2
import numpy as np
from pathlib import Path


# Supported image formats
SUPPORTED_FORMATS = {'.tif', '.tiff', '.bmp', '.png', '.jpg', '.jpeg', '.wsq'}


def load_image(image_path: str, target_size: tuple = (300, 300)) -> np.ndarray:
    """
    Load a fingerprint image as grayscale and resize to target dimensions.

    Handles .tif, .bmp, .png, .jpg formats. Resizing preserves aspect ratio
    by padding with black if needed.

    Args:
        image_path: Path to the fingerprint image file.
        target_size: Tuple of (width, height) to resize to. None to skip resize.

    Returns:
        Grayscale image as numpy array (uint8).

    Raises:
        FileNotFoundError: If image file does not exist.
        ValueError: If image cannot be loaded or format is unsupported.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format '{suffix}'. Supported: {SUPPORTED_FORMATS}")

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    if target_size is not None:
        tw, th = target_size
        h, w = image.shape[:2]

        # Compute scale to fit within target while preserving aspect ratio
        scale = min(tw / w, th / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Pad to exact target_size with black
        canvas = np.zeros((th, tw), dtype=np.uint8)
        y_offset = (th - new_h) // 2
        x_offset = (tw - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        image = canvas

    return image


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Apply zero-mean, unit-variance normalization to the image.

    Args:
        image: Grayscale image as numpy array.

    Returns:
        Normalized image as float64 numpy array.
    """
    image = image.astype(np.float64)
    mean = np.mean(image)
    std = np.std(image)

    if std < 1e-6:
        return image - mean

    return (image - mean) / std
