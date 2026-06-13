"""
scorer.py — Scoring functions for matching engines.

Computes similarity scores for classical minutiae matches and deep embeddings.
"""

import numpy as np


def compute_minutiae_score(matched_pairs: list, probe_count: int, gallery_count: int) -> float:
    """
    Compute a similarity score between 0.0 and 1.0 based on matched minutiae.
    
    The score is computed using the formula:
    Score = (2 * matched_count) / (probe_count + gallery_count)
    
    Args:
        matched_pairs: List of matched minutiae pairs from matcher.
        probe_count: Total number of minutiae in the probe template.
        gallery_count: Total number of minutiae in the gallery template.
        
    Returns:
        float: Similarity score [0.0, 1.0]
    """
    if probe_count == 0 or gallery_count == 0:
        return 0.0
        
    matched_count = len(matched_pairs)
    score = (2.0 * matched_count) / (probe_count + gallery_count)
    
    # Cap at 1.0 (though mathematically it shouldn't exceed 1.0 if matched <= min(probe, gallery))
    return min(1.0, max(0.0, score))


def compute_embedding_score(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two deep learning embeddings.
    
    Args:
        emb1: 1D numpy array representing the first embedding.
        emb2: 1D numpy array representing the second embedding.
        
    Returns:
        float: Cosine similarity score [-1.0, 1.0], usually mapped to [0.0, 1.0] for comparison.
    """
    if len(emb1) == 0 or len(emb2) == 0:
        return 0.0
        
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
        
    similarity = dot_product / (norm1 * norm2)
    
    # Normalize from [-1, 1] to [0, 1]
    return float((similarity + 1.0) / 2.0)
