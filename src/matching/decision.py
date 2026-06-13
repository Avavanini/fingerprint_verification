"""
decision.py — Decision module for fingerprint matching.

Implements threshold-based decision making and evaluation metrics
like False Accept Rate (FAR), False Reject Rate (FRR), and Equal Error Rate (EER).
"""

import numpy as np
from typing import List, Tuple, Dict


def make_decision(score: float, threshold: float) -> bool:
    """
    Make a binary match decision based on a score and a threshold.
    
    Args:
        score: The similarity score [0.0, 1.0].
        threshold: The operating threshold.
        
    Returns:
        bool: True if score >= threshold (MATCH), False otherwise.
    """
    return score >= threshold


def compute_rates(genuine_scores: List[float], impostor_scores: List[float], thresholds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FAR and FRR across a range of thresholds.
    
    FAR (False Accept Rate) = Impostors accepted / Total Impostors
    FRR (False Reject Rate) = Genuine rejected / Total Genuine
    
    Args:
        genuine_scores: List of scores from genuine pairs.
        impostor_scores: List of scores from impostor pairs.
        thresholds: Array of threshold values to evaluate.
        
    Returns:
        Tuple of (far_array, frr_array) corresponding to the thresholds.
    """
    gen = np.array(genuine_scores)
    imp = np.array(impostor_scores)
    
    far = np.zeros_like(thresholds)
    frr = np.zeros_like(thresholds)
    
    for i, t in enumerate(thresholds):
        # Impostors falsely accepted
        if len(imp) > 0:
            far[i] = np.sum(imp >= t) / len(imp)
        else:
            far[i] = 0.0
            
        # Genuine falsely rejected
        if len(gen) > 0:
            frr[i] = np.sum(gen < t) / len(gen)
        else:
            frr[i] = 0.0
            
    return far, frr


def find_eer(far: np.ndarray, frr: np.ndarray, thresholds: np.ndarray) -> Tuple[float, float]:
    """
    Find the Equal Error Rate (EER) and its corresponding threshold.
    
    EER is the point where FAR == FRR (or the closest approximation).
    
    Args:
        far: Array of False Accept Rates.
        frr: Array of False Reject Rates.
        thresholds: Array of corresponding thresholds.
        
    Returns:
        Tuple of (eer_value, operating_threshold)
    """
    # Find the index where the difference between FAR and FRR is minimized
    diff = np.abs(far - frr)
    min_idx = np.argmin(diff)
    
    # The EER can be approximated as the average of FAR and FRR at that index
    eer_value = (far[min_idx] + frr[min_idx]) / 2.0
    operating_threshold = thresholds[min_idx]
    
    return float(eer_value), float(operating_threshold)


def evaluate_system(genuine_scores: List[float], impostor_scores: List[float], num_points: int = 100) -> Dict[str, float]:
    """
    Evaluate the matching system and return key metrics.
    
    Args:
        genuine_scores: List of scores from genuine pairs.
        impostor_scores: List of scores from impostor pairs.
        num_points: Number of thresholds to sample between 0.0 and 1.0.
        
    Returns:
        Dictionary containing EER and the optimal threshold.
    """
    thresholds = np.linspace(0.0, 1.0, num_points)
    far, frr = compute_rates(genuine_scores, impostor_scores, thresholds)
    eer, opt_thresh = find_eer(far, frr, thresholds)
    
    return {
        "eer": eer,
        "optimal_threshold": opt_thresh
    }
