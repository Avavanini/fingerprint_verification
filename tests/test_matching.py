"""
test_matching.py — Unit tests for the matching engine.
"""

import os
import sys
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.matching import (
    align_templates,
    match_minutiae,
    compute_minutiae_score,
    compute_embedding_score
)

# ===== Fixtures =====

@pytest.fixture
def mock_templates():
    # Probe template (shifted by x+10, y-5, rotated by 0.0)
    probe = {
        'minutiae': [
            {'x': 110, 'y': 95, 'type': 'ending', 'orientation': 0.1, 'descriptor': [0.1, 0.2, 0.3]},
            {'x': 210, 'y': 195, 'type': 'bifurcation', 'orientation': 0.5, 'descriptor': [0.5, 0.5, 0.5]}
        ]
    }
    
    # Gallery template (original position)
    gallery = {
        'minutiae': [
            {'x': 100, 'y': 100, 'type': 'ending', 'orientation': 0.1, 'descriptor': [0.1, 0.2, 0.3]},
            {'x': 200, 'y': 200, 'type': 'bifurcation', 'orientation': 0.5, 'descriptor': [0.5, 0.5, 0.5]},
            {'x': 300, 'y': 300, 'type': 'ending', 'orientation': 1.0, 'descriptor': [0.9, 0.9, 0.9]}
        ]
    }
    
    return probe, gallery

# ===== Align Tests =====

class TestAlignTemplates:
    def test_alignment_translation(self, mock_templates):
        probe, gallery = mock_templates
        aligned, dx, dy, dtheta = align_templates(probe, gallery)
        
        # Best match is the first minutia (exact descriptor match)
        # Probe is at (110, 95), gallery at (100, 100)
        # Translation applied to probe should be dx=-10, dy=5
        assert dx == pytest.approx(-10.0)
        assert dy == pytest.approx(5.0)
        assert dtheta == pytest.approx(0.0)
        
        # Check aligned coordinates of first probe minutia
        assert aligned[0]['x'] == pytest.approx(100.0)
        assert aligned[0]['y'] == pytest.approx(100.0)

# ===== Matcher Tests =====

class TestMatchMinutiae:
    def test_match_minutiae_perfect_alignment(self, mock_templates):
        probe, gallery = mock_templates
        aligned, _, _, _ = align_templates(probe, gallery)
        
        pairs = match_minutiae(aligned, gallery, spatial_tol=5.0, angular_tol=0.1)
        
        assert len(pairs) == 2
        assert pairs[0][0]['type'] == pairs[0][1]['type']
        
    def test_match_minutiae_out_of_tolerance(self, mock_templates):
        probe, gallery = mock_templates
        # Pass unaligned probe
        pairs = match_minutiae(probe['minutiae'], gallery, spatial_tol=5.0, angular_tol=0.1)
        
        # Spatial diff is sqrt(10^2 + 5^2) = 11.18, which is > 5.0
        assert len(pairs) == 0

# ===== Scorer Tests =====

class TestScoring:
    def test_compute_minutiae_score(self):
        # probe=2, gallery=3, matched=2
        score = compute_minutiae_score([("a", "b"), ("c", "d")], 2, 3)
        assert score == pytest.approx((2.0 * 2) / (2 + 3)) # 4/5 = 0.8
        
        # probe=0
        assert compute_minutiae_score([], 0, 5) == 0.0
        
    def test_compute_embedding_score(self):
        emb1 = np.array([1.0, 0.0])
        emb2 = np.array([1.0, 0.0])
        assert compute_embedding_score(emb1, emb2) == pytest.approx(1.0)
        
        emb3 = np.array([-1.0, 0.0])
        assert compute_embedding_score(emb1, emb3) == pytest.approx(0.0) # Cosine is -1.0, mapped to 0.0
        
        emb4 = np.array([0.0, 1.0])
        assert compute_embedding_score(emb1, emb4) == pytest.approx(0.5) # Cosine is 0.0, mapped to 0.5


# ===== Hybrid Scoring Tests =====

from src.matching.hybrid import compute_hybrid_score

class TestHybridScoring:
    def test_hybrid_score_with_both_components(self):
        """Verify the weighted fusion formula: hybrid = α*emb + (1-α)*min."""
        emb1 = np.array([1.0, 0.0])
        emb2 = np.array([1.0, 0.0])
        
        # Create minimal templates that will produce a known minutiae score
        # Two identical templates → perfect alignment → all pairs match
        template1 = {
            'minutiae': [
                {'x': 100, 'y': 100, 'type': 'ending', 'orientation': 0.1, 'descriptor': [0.5, 0.5]},
                {'x': 200, 'y': 200, 'type': 'bifurcation', 'orientation': 0.5, 'descriptor': [0.3, 0.7]}
            ]
        }
        template2 = {
            'minutiae': [
                {'x': 100, 'y': 100, 'type': 'ending', 'orientation': 0.1, 'descriptor': [0.5, 0.5]},
                {'x': 200, 'y': 200, 'type': 'bifurcation', 'orientation': 0.5, 'descriptor': [0.3, 0.7]}
            ]
        }
        
        alpha = 0.6
        hybrid, emb_s, min_s = compute_hybrid_score(emb1, emb2, template1, template2, alpha=alpha)
        
        # Embedding score should be 1.0 (identical vectors)
        assert emb_s == pytest.approx(1.0)
        
        # Minutiae score should be > 0 (identical templates)
        assert min_s > 0.0
        
        # Hybrid should follow the formula
        expected = alpha * emb_s + (1 - alpha) * min_s
        assert hybrid == pytest.approx(expected)

    def test_hybrid_score_fallback_no_templates(self):
        """When minutiae templates are None, hybrid falls back to embedding-only."""
        emb1 = np.array([1.0, 0.0])
        emb2 = np.array([0.0, 1.0])
        
        hybrid, emb_s, min_s = compute_hybrid_score(emb1, emb2, None, None, alpha=0.6)
        
        # Should fall back to embedding-only
        assert min_s == 0.0
        assert hybrid == pytest.approx(emb_s)
        assert emb_s == pytest.approx(0.5)  # Cosine 0.0 → mapped to 0.5

    def test_hybrid_score_fallback_one_template_missing(self):
        """When only one template is None, hybrid falls back to embedding-only."""
        emb1 = np.array([1.0, 0.0])
        emb2 = np.array([1.0, 0.0])
        
        template = {
            'minutiae': [
                {'x': 100, 'y': 100, 'type': 'ending', 'orientation': 0.1, 'descriptor': [0.5]}
            ]
        }
        
        hybrid, emb_s, min_s = compute_hybrid_score(emb1, emb2, template, None, alpha=0.6)
        
        assert min_s == 0.0
        assert hybrid == pytest.approx(emb_s)

    def test_hybrid_alpha_extremes(self):
        """Alpha=1.0 should be pure embedding, alpha=0.0 should be pure minutiae."""
        emb1 = np.array([1.0, 0.0])
        emb2 = np.array([1.0, 0.0])
        
        template1 = {
            'minutiae': [
                {'x': 100, 'y': 100, 'type': 'ending', 'orientation': 0.1, 'descriptor': [0.5, 0.5]},
            ]
        }
        template2 = {
            'minutiae': [
                {'x': 100, 'y': 100, 'type': 'ending', 'orientation': 0.1, 'descriptor': [0.5, 0.5]},
            ]
        }
        
        # Alpha = 1.0 → pure embedding
        hybrid_1, emb_s, min_s = compute_hybrid_score(emb1, emb2, template1, template2, alpha=1.0)
        assert hybrid_1 == pytest.approx(emb_s)
        
        # Alpha = 0.0 → pure minutiae
        hybrid_0, emb_s, min_s = compute_hybrid_score(emb1, emb2, template1, template2, alpha=0.0)
        assert hybrid_0 == pytest.approx(min_s)
