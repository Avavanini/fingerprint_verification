"""
Matching Engine module for fingerprint verification.

Contains functions for aligning and matching classical minutiae templates,
scoring functions for both classical and deep embedding features,
and hybrid fusion of deep embeddings with classical minutiae matching.
"""

from .align import align_templates
from .matcher import match_minutiae
from .scorer import compute_minutiae_score, compute_embedding_score
from .decision import make_decision, compute_rates, find_eer, evaluate_system
from .hybrid import compute_hybrid_score, extract_minutiae_template, extract_minutiae_template_from_path

__all__ = [
    'align_templates',
    'match_minutiae',
    'compute_minutiae_score',
    'compute_embedding_score',
    'make_decision',
    'compute_rates',
    'find_eer',
    'evaluate_system',
    'compute_hybrid_score',
    'extract_minutiae_template',
    'extract_minutiae_template_from_path',
]
