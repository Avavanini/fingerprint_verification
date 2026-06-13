"""
thin.py — Binarization and skeletonization of fingerprint images.

Functions:
    binarize(image, method) -> binary_image
    thin(image) -> skeleton
    clean_skeleton(image) -> cleaned_skeleton
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
        block_size: Block size for adaptive thresholding (must be odd).
        c: Constant subtracted from mean in adaptive thresholding.

    Returns:
        Binary image (0 and 255, uint8).
    """
    # Ensure uint8
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if method == "otsu":
        _, binary = cv2.threshold(image, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
        binary_image: Binary image (0 and 255, uint8).

    Returns:
        Thinned/skeletonized image (0 and 255, uint8).
        White (255) = ridge pixels, Black (0) = background.
    """
    # skeletonize expects white ridges on black background
    # Fingerprints are typically dark ridges on white background, so we invert
    bool_image = (binary_image == 0)  # ridges are dark (0) in the binary image
    skeleton = skeletonize(bool_image)
    return (skeleton.astype(np.uint8)) * 255


def clean_skeleton(skeleton: np.ndarray, min_branch_length: int = 10) -> np.ndarray:
    """
    Remove short spurious branches and isolated noise from a skeleton image
    using connected component analysis.

    Args:
        skeleton: Skeletonized binary image (0 and 255).
        min_branch_length: Minimum size (area in pixels) of components to keep.

    Returns:
        Cleaned skeleton image (0 and 255, uint8).
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        skeleton, connectivity=8
    )

    cleaned = np.zeros_like(skeleton)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_branch_length:
            cleaned[labels == label] = 255

    return cleaned
