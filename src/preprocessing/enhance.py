"""
enhance.py — Ridge enhancement using fingerprint-enhancer library or Gabor filter bank.

The fingerprint_enhancer library implements Hong, Wan, & Jain's method:
  1. Normalize the image
  2. Estimate local ridge orientation (gradient-based)
  3. Estimate local ridge frequency
  4. Apply contextual Gabor filtering

If fingerprint_enhancer is not available, falls back to a basic Gabor filter bank.

Functions:
    enhance_ridges(image) -> enhanced_image
"""

import cv2
import numpy as np

try:
    import fingerprint_enhancer
    _HAS_FP_ENHANCER = True
except ImportError:
    _HAS_FP_ENHANCER = False


def enhance_ridges(image: np.ndarray, use_library: bool = True,
                   ksize: int = 31, sigma: float = 4.0,
                   lambd: float = 10.0, gamma: float = 0.5,
                   num_orientations: int = 8) -> np.ndarray:
    """
    Enhance fingerprint ridges for cleaner binarization and minutiae detection.

    Uses the fingerprint_enhancer library (recommended) which performs:
    - Ridge orientation estimation
    - Ridge frequency estimation
    - Contextual Gabor filtering

    Falls back to a basic multi-orientation Gabor filter bank if the library
    is not available.

    Args:
        image: Grayscale fingerprint image (uint8).
        use_library: If True, use fingerprint_enhancer library when available.
        ksize: Size of the Gabor kernel (fallback mode).
        sigma: Std dev of the Gaussian envelope (fallback mode).
        lambd: Wavelength of the sinusoidal factor (fallback mode).
        gamma: Spatial aspect ratio (fallback mode).
        num_orientations: Number of filter orientations (fallback mode).

    Returns:
        Enhanced image (uint8, 0-255).
    """
    if use_library and _HAS_FP_ENHANCER:
        return _enhance_with_library(image)
    else:
        return _enhance_with_gabor(image, ksize, sigma, lambd, gamma, num_orientations)


def _enhance_with_library(image: np.ndarray) -> np.ndarray:
    """Enhance using the fingerprint_enhancer library."""
    # Ensure uint8 input
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    enhanced = fingerprint_enhancer.enhance_fingerprint(image)

    # The library may return bool or float; convert to uint8
    if enhanced.dtype == bool or enhanced.dtype == np.bool_:
        enhanced = (enhanced.astype(np.uint8)) * 255
    elif enhanced.dtype != np.uint8:
        enhanced = enhanced.astype(np.float64)
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return enhanced


def _enhance_with_gabor(image: np.ndarray, ksize: int, sigma: float,
                        lambd: float, gamma: float,
                        num_orientations: int) -> np.ndarray:
    """Fallback: enhance using a basic Gabor filter bank."""
    img_float = image.astype(np.float64)
    accumulated = np.zeros_like(img_float)

    for i in range(num_orientations):
        theta = i * np.pi / num_orientations
        kernel = cv2.getGaborKernel(
            (ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_64F
        )
        filtered = cv2.filter2D(img_float, cv2.CV_64F, kernel)
        accumulated = np.maximum(accumulated, filtered)

    enhanced = cv2.normalize(accumulated, None, 0, 255, cv2.NORM_MINMAX)
    return enhanced.astype(np.uint8)
