"""
hybrid.py — Hybrid matching engine that fuses deep embeddings with classical minutiae.

Combines the Siamese CNN embedding cosine similarity with the classical
minutiae alignment + pairing score using a weighted linear fusion:

    hybrid_score = α × embedding_score + (1 - α) × minutiae_score

This approach compensates for the CNN's weaknesses on heavily distorted
images (Medium/Hard SOCOFing alterations) by leveraging the structural
robustness of minutiae-based matching.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Optional, Tuple

from src.preprocessing.pipeline import preprocess
from src.feature_extraction.minutiae import detect_minutiae
from src.feature_extraction.descriptor import compute_descriptors, create_template
from src.matching.align import align_templates
from src.matching.matcher import match_minutiae
from src.matching.scorer import compute_minutiae_score, compute_embedding_score

logger = logging.getLogger(__name__)

# Default fusion weight: 60% CNN + 40% minutiae
DEFAULT_ALPHA = 0.6


def extract_minutiae_template(img_gray: np.ndarray) -> Optional[dict]:
    """
    Run the full classical preprocessing + feature extraction pipeline
    on a grayscale fingerprint image and return a minutiae template.

    Pipeline:
        1. Preprocess (normalize → segment → enhance → binarize → thin)
        2. Detect minutiae (Crossing Number algorithm)
        3. Compute local ridge descriptors
        4. Package into a template dict

    Args:
        img_gray: Grayscale fingerprint image (uint8).

    Returns:
        Template dict with 'minutiae' and 'num_minutiae' keys,
        or None if preprocessing/extraction fails.
    """
    try:
        # Save to a temp buffer and preprocess via the pipeline
        # The pipeline expects a file path, so we write to a temp file
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, img_gray)

        try:
            result = preprocess(tmp_path, target_size=(300, 300))
        finally:
            os.unlink(tmp_path)

        if not result['quality_passed'] or result['skeleton'] is None:
            logger.warning("Image failed quality check or skeleton extraction.")
            return None

        skeleton = result['skeleton']

        # Detect minutiae
        minutiae = detect_minutiae(skeleton)

        if len(minutiae) < 2:
            logger.warning(f"Too few minutiae detected ({len(minutiae)}). Skipping.")
            return None

        # Compute descriptors
        descriptors = compute_descriptors(skeleton, minutiae)

        # Create template
        template = create_template(minutiae, descriptors)

        return template

    except Exception as e:
        logger.error(f"Minutiae extraction failed: {e}")
        return None


def extract_minutiae_template_from_path(img_path: str) -> Optional[dict]:
    """
    Convenience wrapper that loads an image from disk and extracts
    the minutiae template.

    Args:
        img_path: Path to the fingerprint image file.

    Returns:
        Template dict, or None if extraction fails.
    """
    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        logger.error(f"Failed to load image: {img_path}")
        return None
    return extract_minutiae_template(img_gray)


def compute_classical_minutiae_score(template1: dict, template2: dict,
                                     spatial_tol: float = 20.0,
                                     angular_tol: float = 0.5236) -> float:
    """
    Compute the classical minutiae matching score between two templates.

    Steps:
        1. Align template1 (probe) to template2 (gallery) using descriptor matching.
        2. Pair aligned minutiae within spatial and angular tolerances.
        3. Compute score as 2*matched / (probe_count + gallery_count).

    Args:
        template1: Probe minutiae template.
        template2: Gallery minutiae template.
        spatial_tol: Maximum pixel distance for a valid minutia pair (default 20px).
        angular_tol: Maximum angular difference in radians (default 30°).

    Returns:
        float: Minutiae match score in [0.0, 1.0].
    """
    if not template1 or not template2:
        return 0.0

    if not template1.get('minutiae') or not template2.get('minutiae'):
        return 0.0

    # Align probe to gallery
    aligned_probe, dx, dy, dtheta = align_templates(template1, template2)

    # Match aligned minutiae
    pairs = match_minutiae(aligned_probe, template2,
                           spatial_tol=spatial_tol,
                           angular_tol=angular_tol)

    # Compute score
    probe_count = len(template1['minutiae'])
    gallery_count = len(template2['minutiae'])

    score = compute_minutiae_score(pairs, probe_count, gallery_count)
    return score


def compute_hybrid_score(emb1: np.ndarray, emb2: np.ndarray,
                         template1: Optional[dict], template2: Optional[dict],
                         alpha: float = DEFAULT_ALPHA) -> Tuple[float, float, float]:
    """
    Compute the hybrid matching score by fusing deep embedding similarity
    with classical minutiae matching.

    Formula:
        hybrid_score = α × embedding_score + (1 - α) × minutiae_score

    If minutiae templates are unavailable for either image, falls back
    to embedding-only scoring (hybrid_score = embedding_score).

    Args:
        emb1: 128-D embedding of the probe image.
        emb2: 128-D embedding of the gallery image.
        template1: Minutiae template for the probe (or None).
        template2: Minutiae template for the gallery (or None).
        alpha: Fusion weight for the embedding score [0.0, 1.0].

    Returns:
        Tuple of (hybrid_score, embedding_score, minutiae_score).
        minutiae_score is 0.0 if templates are unavailable.
    """
    # Deep embedding score (always available)
    embedding_score = compute_embedding_score(emb1, emb2)

    # Classical minutiae score (may not be available)
    if template1 is not None and template2 is not None:
        minutiae_score = compute_classical_minutiae_score(template1, template2)
        hybrid_score = alpha * embedding_score + (1 - alpha) * minutiae_score
    else:
        # Fallback: embedding only
        minutiae_score = 0.0
        hybrid_score = embedding_score

    return float(hybrid_score), float(embedding_score), float(minutiae_score)
