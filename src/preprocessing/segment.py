"""
segment.py — Foreground/background segmentation for fingerprint images.

Functions:
    segment(image, block_size, threshold) -> mask
    apply_mask(image, mask) -> masked_image
"""

import cv2
import numpy as np


def segment(image: np.ndarray, block_size: int = 16,
            variance_threshold: float = 100) -> np.ndarray:
    """
    Perform block-wise variance segmentation to separate foreground (ridges)
    from background.

    Each non-overlapping block is classified as foreground if its local variance
    exceeds the threshold. Morphological closing + opening are applied to fill
    small holes and remove isolated noise blocks.

    Args:
        image: Grayscale fingerprint image (uint8 or float).
        block_size: Size of blocks for variance computation.
        variance_threshold: Minimum variance to classify block as foreground.

    Returns:
        Binary mask where 255 = foreground, 0 = background (same shape as input).
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    img_float = image.astype(np.float64)

    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = img_float[i:i + block_size, j:j + block_size]
            variance = np.var(block)
            if variance > variance_threshold:
                mask[i:i + block_size, j:j + block_size] = 255

    # Morphological cleanup: close small holes, then remove isolated noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block_size, block_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a foreground mask to an image, setting background pixels to white (255).

    Args:
        image: Grayscale image (uint8).
        mask: Binary mask (255 = foreground).

    Returns:
        Masked image with background set to 255.
    """
    result = np.full_like(image, 255, dtype=np.uint8)
    if image.dtype != np.uint8:
        img_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img_uint8 = image
    result[mask == 255] = img_uint8[mask == 255]
    return result
