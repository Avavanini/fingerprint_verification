"""
Matching Engine module for fingerprint verification.

Contains functions for aligning and matching classical minutiae templates,
as well as scoring functions for both classical and deep embedding features.
"""

from .align import align_templates
from .matcher import match_minutiae
from .scorer import compute_minutiae_score, compute_embedding_score
from .decision import make_decision, compute_rates, find_eer, evaluate_system

__all__ = [
    'align_templates',
    'match_minutiae',
    'compute_minutiae_score',
    'compute_embedding_score',
    'make_decision',
    'compute_rates',
    'find_eer',
    'evaluate_system',
]
