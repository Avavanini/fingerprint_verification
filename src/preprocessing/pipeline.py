"""
pipeline.py — Unified fingerprint preprocessing pipeline.

Chains all preprocessing steps:
    1. Load & resize image
    2. Normalize
    3. Segment (foreground mask)
    4. Enhance ridges (Gabor / fingerprint_enhancer)
    5. Binarize
    6. Thin / Skeletonize
    7. Clean skeleton

Functions:
    preprocess(image_path, ...) -> dict
    quality_check(image) -> (bool, float)
"""

import cv2
import numpy as np
from pathlib import Path

from .normalize import load_image, normalize
from .segment import segment, apply_mask
from .enhance import enhance_ridges
from .thin import binarize, thin, clean_skeleton


def quality_check(image: np.ndarray, min_variance: float = 200.0) -> tuple:
    """
    Variance-based quality check. Images with very low variance are likely
    blank, over-exposed, or otherwise unusable.

    Args:
        image: Grayscale image (uint8).
        min_variance: Minimum pixel variance to pass quality check.

    Returns:
        Tuple of (passed: bool, variance: float).
    """
    variance = float(np.var(image.astype(np.float64)))
    return variance >= min_variance, variance


def preprocess(image_path: str, target_size: tuple = (300, 300),
               block_size: int = 16, variance_threshold: float = 100.0,
               binarize_method: str = "otsu",
               min_quality_variance: float = 200.0,
               use_enhancer_library: bool = True,
               return_intermediates: bool = False) -> dict:
    """
    Full preprocessing pipeline for a fingerprint image.

    Steps:
        1. Load image (grayscale, resized)
        2. Quality check (variance-based)
        3. Normalize (zero-mean, unit-variance)
        4. Segment (foreground mask via block variance)
        5. Enhance ridges (fingerprint_enhancer or Gabor bank)
        6. Binarize (Otsu or adaptive threshold)
        7. Thin / skeletonize (1-pixel ridges)
        8. Clean skeleton (remove short spurious branches)

    Args:
        image_path: Path to the fingerprint image.
        target_size: (width, height) to resize to. None to skip.
        block_size: Block size for variance segmentation.
        variance_threshold: Threshold for foreground classification.
        binarize_method: "otsu" or "adaptive".
        min_quality_variance: Minimum variance for quality check.
        use_enhancer_library: Use fingerprint_enhancer if available.
        return_intermediates: If True, return intermediate images for visualization.

    Returns:
        Dictionary with keys:
            - 'skeleton': Final thinned skeleton image (uint8, 0/255)
            - 'quality_passed': Whether the image passed quality check
            - 'quality_variance': Variance value
            - 'intermediates': dict of intermediate images (if return_intermediates=True)
    """
    result = {
        'skeleton': None,
        'quality_passed': False,
        'quality_variance': 0.0,
        'intermediates': {},
    }

    # Step 1: Load
    original = load_image(image_path, target_size=target_size)

    # Step 2: Quality check
    passed, variance = quality_check(original, min_quality_variance)
    result['quality_passed'] = passed
    result['quality_variance'] = variance

    if not passed:
        if return_intermediates:
            result['intermediates']['original'] = original
        return result

    # Step 3: Normalize
    normalized = normalize(original)

    # Step 4: Segment
    mask = segment(original, block_size=block_size,
                   variance_threshold=variance_threshold)
    masked = apply_mask(original, mask)

    # Step 5: Enhance
    enhanced = enhance_ridges(masked, use_library=use_enhancer_library)

    # Step 6: Binarize
    binary = binarize(enhanced, method=binarize_method)

    # Step 7: Thin
    skeleton = thin(binary)

    # Step 8: Clean
    skeleton_clean = clean_skeleton(skeleton, min_branch_length=10)

    result['skeleton'] = skeleton_clean

    if return_intermediates:
        result['intermediates'] = {
            'original': original,
            'normalized': normalized,
            'mask': mask,
            'masked': masked,
            'enhanced': enhanced,
            'binary': binary,
            'skeleton_raw': skeleton,
            'skeleton_clean': skeleton_clean,
        }

    return result
