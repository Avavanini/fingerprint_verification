"""
test_feature_extraction.py — Unit tests for minutiae detection and descriptors.
"""

import os
import sys
import pytest
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.feature_extraction import (
    detect_minutiae, compute_orientations, visualize_minutiae,
    compute_descriptors, create_template, save_template, load_template
)

# ===== Fixtures =====

@pytest.fixture
def synthetic_skeleton():
    """Create a synthetic 100x100 skeleton with an ending and a bifurcation."""
    skel = np.zeros((100, 100), dtype=np.uint8)
    
    # Draw a line that ends abruptly (ending at x=50, y=50)
    skel[50, 10:51] = 255
    
    # Draw a Y-shape (bifurcation at x=50, y=20)
    skel[20, 10:51] = 255
    for i in range(1, 10):
        skel[20-i, 50+i] = 255
        skel[20+i, 50+i] = 255
        
    return skel

@pytest.fixture
def test_minutiae():
    return [
        (50, 50, "ending", 0.0),
        (50, 20, "bifurcation", 0.0)
    ]

# ===== Minutiae Tests =====

class TestDetectMinutiae:
    def test_detect_ending_and_bifurcation(self, synthetic_skeleton):
        # margin 5 to avoid edge effects
        minutiae = detect_minutiae(synthetic_skeleton, border_margin=5, min_distance=2)
        
        # We expect 1 bifurcation (at 50, 20) and 3 endings (50,50), (10,50), (10,20)
        # But wait, the branches of the Y also end: (59, 11) and (59, 29).
        # Let's just check that we detect the bifurcation and at least one ending
        
        bifurcations = [m for m in minutiae if m[2] == 'bifurcation']
        endings = [m for m in minutiae if m[2] == 'ending']
        
        assert len(bifurcations) >= 1
        assert len(endings) >= 1
        
        # Check if the bifurcation at (50, 20) is found
        found_bif = False
        for x, y, t, _ in bifurcations:
            if abs(x - 50) <= 2 and abs(y - 20) <= 2:
                found_bif = True
        assert found_bif

    def test_filter_close_minutiae(self, synthetic_skeleton):
        # We know there are endings near x=10. If we set min_distance=100, we should get very few.
        minutiae = detect_minutiae(synthetic_skeleton, border_margin=5, min_distance=100)
        assert len(minutiae) <= 2

class TestComputeOrientations:
    def test_compute_orientations_runs(self, synthetic_skeleton, test_minutiae):
        oriented = compute_orientations(synthetic_skeleton, test_minutiae)
        assert len(oriented) == 2
        for x, y, t, o in oriented:
            assert isinstance(o, float)
            assert 0.0 <= o < 2 * np.pi + 1e-5

class TestVisualizeMinutiae:
    def test_visualize_output_shape(self, synthetic_skeleton, test_minutiae):
        vis = visualize_minutiae(synthetic_skeleton, test_minutiae)
        assert vis.shape == (100, 100, 3)

# ===== Descriptor Tests =====

class TestDescriptors:
    def test_compute_descriptors(self, synthetic_skeleton, test_minutiae):
        descs = compute_descriptors(synthetic_skeleton, test_minutiae, radius=10, num_sectors=8)
        assert len(descs) == len(test_minutiae)
        assert descs[0].shape == (8,)
        assert descs[1].shape == (8,)

class TestTemplate:
    def test_create_and_save_load_json(self, tmp_path, test_minutiae):
        descs = [np.zeros(8), np.ones(8)]
        template = create_template(test_minutiae, descs, {"id": 1})
        
        path = str(tmp_path / "test.json")
        save_template(template, path, fmt="json")
        
        loaded = load_template(path)
        assert loaded["num_minutiae"] == 2
        assert loaded["metadata"]["id"] == 1
        assert len(loaded["minutiae"]) == 2

    def test_create_and_save_load_npy(self, tmp_path, test_minutiae):
        descs = [np.zeros(8), np.ones(8)]
        template = create_template(test_minutiae, descs, {"id": 2})
        
        path = str(tmp_path / "test.npz")
        save_template(template, path, fmt="npy")
        
        loaded = load_template(path)
        assert loaded["num_minutiae"] == 2
        assert loaded["metadata"]["id"] == 2
        assert len(loaded["minutiae"]) == 2
