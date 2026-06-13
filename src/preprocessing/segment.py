"""
segment.py — Foreground/background segmentation for fingerprint images.

Functions:
    segment(image, block_size, threshold) -> mask
"""

import numpy as np


def segment(image: np.ndarray, block_size: int = 16, variance_threshold: float = 100) -> np.ndarray:
    """
    Perform block-wise variance segmentation to separate foreground (ridges)
    from background.
    
    Args:
        image: Grayscale fingerprint image (uint8 or float).
        block_size: Size of blocks for variance computation.
        variance_threshold: Minimum variance to classify block as foreground.
        
    Returns:
        Binary mask where 255 = foreground, 0 = background.
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
    
    return mask
