"""
embedding.py — Deep learning embedding extraction using Siamese CNN.

Implements the SiameseNetwork and ContrastiveLoss for training and feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple


class SiameseNetwork(nn.Module):
    """
    Siamese CNN architecture using ResNet-18 as a backbone.
    Outputs a dense embedding vector (e.g., 128-D) for contrastive learning.
    """
    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super(SiameseNetwork, self).__init__()
        
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # We replace the final fully connected layer to output our embedding dimension
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim)
        )

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single image."""
        output = self.backbone(x)
        return output

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a pair of images."""
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    
    Based on:
    L = Y * D^2 + (1 - Y) * max(margin - D, 0)^2
    where D is the Euclidean distance between outputs, and Y is the label
    (1 for genuine/similar, 0 for impostor/dissimilar).
    """
    def __init__(self, margin: float = 2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output1: Embedding of the first image (B, D).
            output2: Embedding of the second image (B, D).
            label: 1 for Genuine, 0 for Impostor (B,).
            
        Returns:
            Scalar loss value.
        """
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        
        # Reshape label to match distance shape
        label = label.view(-1, 1)
        
        loss_contrastive = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive


def extract_embedding(model: nn.Module, image_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Helper to extract the embedding vector for a single image tensor.
    
    Args:
        model: Trained SiameseNetwork.
        image_tensor: Input image tensor (C, H, W) or (B, C, H, W).
        device: CPU or CUDA.
        
    Returns:
        Embedding tensor (1, D) if single image, or (B, D).
    """
    model.eval()
    
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)
        
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        embedding = model.forward_once(image_tensor)
        
    return embedding
