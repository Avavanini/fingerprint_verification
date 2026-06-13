"""
Feature Extraction module for fingerprint verification.

Contains methods for minutiae detection (Crossing Number), descriptor 
computation (sector-based ridge density), and deep learning embeddings.
"""

from .minutiae import detect_minutiae, compute_orientations, visualize_minutiae
from .descriptor import compute_descriptors, create_template, save_template, load_template

__all__ = [
    'detect_minutiae',
    'compute_orientations',
    'visualize_minutiae',
    'compute_descriptors',
    'create_template',
    'save_template',
    'load_template',
]
