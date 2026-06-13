"""
thin.py — Binarization and skeletonization of fingerprint images.

Functions:
    binarize(image, method) -> binary_image
    thin(image) -> skeleton
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize


def binarize(image: np.ndarray, method: str = "otsu",
             block_size: int = 15, c: int = 5) -> np.ndarray:
    """
    Binarize a grayscale fingerprint image.
    
    Args:
        image: Grayscale image (uint8).
        method: "otsu" for Otsu's thresholding, "adaptive" for local adaptive.
        block_size: Block size for adaptive thresholding.
        c: Constant subtracted from mean in adaptive thresholding.
        
    Returns:
        Binary image (0 and 255).
    """
    if method == "otsu":
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, c
        )
    else:
        raise ValueError(f"Unknown binarization method: {method}")
    
    return binary


def thin(binary_image: np.ndarray) -> np.ndarray:
    """
    Skeletonize a binary fingerprint image to produce 1-pixel-wide ridges.
    
    Args:
        binary_image: Binary image (0 and 255).
        
    Returns:
        Thinned/skeletonized image (0 and 255).
    """
    # Convert to boolean for skeletonize
    bool_image = (binary_image > 0).astype(bool)
    skeleton = skeletonize(bool_image)
    return (skeleton.astype(np.uint8)) * 255
