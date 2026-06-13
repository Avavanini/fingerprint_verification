"""
test_siamese.py — Unit tests for the Siamese CNN embedding pipeline.
"""

import os
import sys
import pytest
import torch
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction.dataset import SOCOFingSiameseDataset
from src.feature_extraction.embedding import SiameseNetwork, ContrastiveLoss, extract_embedding

class TestSiameseNetwork:
    def test_forward_once_output_shape(self):
        model = SiameseNetwork(embedding_dim=128, pretrained=False)
        # Create a dummy image tensor (Batch, Channels, Height, Width)
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model.forward_once(dummy_input)
        
        assert output.shape == (2, 128)
        
    def test_forward_pair_output_shape(self):
        model = SiameseNetwork(embedding_dim=64, pretrained=False)
        dummy_input1 = torch.randn(2, 3, 224, 224)
        dummy_input2 = torch.randn(2, 3, 224, 224)
        
        out1, out2 = model(dummy_input1, dummy_input2)
        
        assert out1.shape == (2, 64)
        assert out2.shape == (2, 64)

class TestContrastiveLoss:
    def test_loss_computation(self):
        criterion = ContrastiveLoss(margin=2.0)
        
        # Two identical embeddings -> distance 0
        out1 = torch.tensor([[1.0, 1.0]])
        out2 = torch.tensor([[1.0, 1.0]])
        
        # Genuine pair (label=1)
        label_gen = torch.tensor([1.0])
        loss_gen = criterion(out1, out2, label_gen)
        # Should be 0 since distance is 0 and they are a genuine pair
        assert loss_gen.item() == pytest.approx(0.0, abs=1e-4)
        
        # Impostor pair (label=0)
        label_imp = torch.tensor([0.0])
        loss_imp = criterion(out1, out2, label_imp)
        # Since distance is 0, loss should be max(margin - 0, 0)^2 = 4.0
        assert loss_imp.item() == pytest.approx(4.0, abs=1e-4)

class TestDataset:
    def test_dataset_initialization(self):
        data_dir = str(PROJECT_ROOT / "data" / "raw" / "SOCOFing")
        if not os.path.exists(os.path.join(data_dir, "Real")):
            pytest.skip("SOCOFing dataset not available")
            
        dataset = SOCOFingSiameseDataset(data_dir=data_dir, num_pairs=10)
        assert len(dataset) == 10
        
        img1, img2, label = dataset[0]
        # Check shapes: ResNet expects 3 channels. Our dataset converts grayscale to 3 channel RGB.
        # However, dataset returns (C, H, W) tensor only if default transform is applied
        assert img1.shape[0] == 3
        assert img2.shape[0] == 3
        assert label.item() in [0.0, 1.0]

class TestExtractEmbedding:
    def test_extract_embedding_helper(self):
        model = SiameseNetwork(embedding_dim=128, pretrained=False)
        dummy_input = torch.randn(3, 224, 224) # Single image (C, H, W)
        
        device = torch.device('cpu')
        emb = extract_embedding(model, dummy_input, device)
        
        assert emb.shape == (1, 128)
