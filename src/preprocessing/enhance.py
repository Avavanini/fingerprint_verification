"""
enhance.py — Ridge enhancement using Gabor filter bank.

Functions:
    enhance_ridges(image) -> enhanced_image
"""

import cv2
import numpy as np


def enhance_ridges(image: np.ndarray, ksize: int = 31, sigma: float = 4.0,
                   lambd: float = 10.0, gamma: float = 0.5,
                   num_orientations: int = 8) -> np.ndarray:
    """
    Enhance fingerprint ridges using a bank of Gabor filters at multiple orientations.
    
    Args:
        image: Grayscale fingerprint image.
        ksize: Size of the Gabor kernel.
        sigma: Standard deviation of the Gaussian envelope.
        lambd: Wavelength of the sinusoidal factor.
        gamma: Spatial aspect ratio.
        num_orientations: Number of filter orientations (evenly spaced across pi).
        
    Returns:
        Enhanced image with improved ridge contrast.
    """
    img_float = image.astype(np.float64)
    accumulated = np.zeros_like(img_float)
    
    for i in range(num_orientations):
        theta = i * np.pi / num_orientations
        kernel = cv2.getGaborKernel(
            (ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_64F
        )
        filtered = cv2.filter2D(img_float, cv2.CV_64F, kernel)
        accumulated = np.maximum(accumulated, filtered)
    
    # Normalize to 0-255 range
    enhanced = cv2.normalize(accumulated, None, 0, 255, cv2.NORM_MINMAX)
    return enhanced.astype(np.uint8)
