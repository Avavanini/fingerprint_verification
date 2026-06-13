"""
normalize.py — Image loading and normalization utilities.

Functions:
    load_image(path) -> np.ndarray
    normalize(image) -> np.ndarray
"""

import cv2
import numpy as np
from pathlib import Path


def load_image(image_path: str, target_size: tuple = (300, 300)) -> np.ndarray:
    """
    Load a fingerprint image as grayscale and resize to target dimensions.
    
    Args:
        image_path: Path to the fingerprint image file.
        target_size: Tuple of (width, height) to resize to.
        
    Returns:
        Grayscale image as numpy array.
        
    Raises:
        FileNotFoundError: If image file does not exist.
        ValueError: If image cannot be loaded.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    if target_size:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
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
    
    if std == 0:
        return image - mean
    
    return (image - mean) / std
