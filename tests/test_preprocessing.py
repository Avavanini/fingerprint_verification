"""
test_preprocessing.py — Unit tests for the fingerprint preprocessing pipeline.

Tests each preprocessing function for correct output types, shapes, and basic behavior.
"""

import os
import sys
import pytest
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.normalize import load_image, normalize
from src.preprocessing.segment import segment, apply_mask
from src.preprocessing.enhance import enhance_ridges
from src.preprocessing.thin import binarize, thin, clean_skeleton
from src.preprocessing.pipeline import preprocess, quality_check


# ===== Fixtures =====

@pytest.fixture
def sample_image():
    """Create a synthetic fingerprint-like image for testing."""
    img = np.random.randint(50, 200, (300, 300), dtype=np.uint8)
    for i in range(0, 300, 6):
        img[i:i+3, 50:250] = np.random.randint(0, 50, (3, 200), dtype=np.uint8)
    return img


@pytest.fixture
def blank_image():
    """Create a uniform (blank) image for quality check testing."""
    return np.full((300, 300), 128, dtype=np.uint8)


@pytest.fixture
def real_image_path():
    """Return path to a real SOCOFing image if available."""
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw',
                        'SOCOFing', 'Real', '1__M_Left_index_finger.BMP')
    if os.path.exists(path):
        return path
    return None


# ===== normalize.py tests =====

class TestLoadImage:
    def test_load_real_image(self, real_image_path):
        if real_image_path is None:
            pytest.skip("SOCOFing dataset not available")
        img = load_image(real_image_path, target_size=(300, 300))
        assert img.shape == (300, 300)
        assert img.dtype == np.uint8

    def test_load_no_resize(self, real_image_path):
        if real_image_path is None:
            pytest.skip("SOCOFing dataset not available")
        img = load_image(real_image_path, target_size=None)
        assert img.dtype == np.uint8
        assert len(img.shape) == 2

    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_image("nonexistent_file.bmp")


class TestNormalize:
    def test_output_type(self, sample_image):
        result = normalize(sample_image)
        assert result.dtype == np.float64

    def test_zero_mean(self, sample_image):
        result = normalize(sample_image)
        assert abs(np.mean(result)) < 1e-10

    def test_unit_variance(self, sample_image):
        result = normalize(sample_image)
        assert abs(np.std(result) - 1.0) < 1e-10

    def test_blank_image(self, blank_image):
        result = normalize(blank_image)
        assert np.all(result == 0)


# ===== segment.py tests =====

class TestSegment:
    def test_output_shape(self, sample_image):
        mask = segment(sample_image)
        assert mask.shape == sample_image.shape

    def test_output_binary(self, sample_image):
        mask = segment(sample_image)
        unique_vals = set(np.unique(mask))
        assert unique_vals.issubset({0, 255})

    def test_blank_gives_no_foreground(self, blank_image):
        mask = segment(blank_image, variance_threshold=100)
        assert np.sum(mask) == 0


class TestApplyMask:
    def test_masked_background_is_white(self, sample_image):
        mask = np.zeros_like(sample_image)
        mask[100:200, 100:200] = 255
        result = apply_mask(sample_image, mask)
        assert np.all(result[0:50, 0:50] == 255)


# ===== enhance.py tests =====

class TestEnhanceRidges:
    def test_output_shape(self, sample_image):
        result = enhance_ridges(sample_image)
        assert result.shape == sample_image.shape

    def test_output_dtype(self, sample_image):
        result = enhance_ridges(sample_image)
        assert result.dtype == np.uint8


# ===== thin.py tests =====

class TestBinarize:
    def test_otsu(self, sample_image):
        result = binarize(sample_image, method="otsu")
        assert result.dtype == np.uint8
        unique_vals = set(np.unique(result))
        assert unique_vals.issubset({0, 255})

    def test_adaptive(self, sample_image):
        result = binarize(sample_image, method="adaptive")
        assert result.dtype == np.uint8

    def test_invalid_method(self, sample_image):
        with pytest.raises(ValueError):
            binarize(sample_image, method="invalid")


class TestThin:
    def test_output_shape(self, sample_image):
        binary = binarize(sample_image)
        result = thin(binary)
        assert result.shape == binary.shape

    def test_output_binary(self, sample_image):
        binary = binarize(sample_image)
        result = thin(binary)
        unique_vals = set(np.unique(result))
        assert unique_vals.issubset({0, 255})


class TestCleanSkeleton:
    def test_removes_small_fragments(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[10, 10] = 255  # tiny 1-pixel fragment
        img[50, 20:80] = 255  # long line (60px)
        cleaned = clean_skeleton(img, min_branch_length=5)
        assert cleaned[10, 10] == 0
        assert np.sum(cleaned[50, 20:80]) > 0


# ===== pipeline.py tests =====

class TestQualityCheck:
    def test_good_image_passes(self, sample_image):
        passed, variance = quality_check(sample_image)
        assert passed is True
        assert variance > 0

    def test_blank_image_fails(self, blank_image):
        passed, variance = quality_check(blank_image)
        assert passed is False
        assert variance < 1.0


class TestPreprocess:
    def test_full_pipeline_real(self, real_image_path):
        if real_image_path is None:
            pytest.skip("SOCOFing dataset not available")
        result = preprocess(real_image_path, return_intermediates=True)
        assert result['quality_passed'] is True
        assert result['skeleton'] is not None
        assert result['skeleton'].shape == (300, 300)
        assert result['skeleton'].dtype == np.uint8
        assert 'original' in result['intermediates']
        assert 'enhanced' in result['intermediates']
        assert 'skeleton_clean' in result['intermediates']

    def test_pipeline_rejects_blank(self, tmp_path):
        blank = np.full((300, 300), 128, dtype=np.uint8)
        path = str(tmp_path / "blank.bmp")
        cv2.imwrite(path, blank)
        result = preprocess(path)
        assert result['quality_passed'] is False
        assert result['skeleton'] is None
