"""
dataset.py — PyTorch Dataset for Siamese Network training using SOCOFing.

Generates genuine pairs (Real + Altered of same finger) and impostor pairs
(Real + Real of different fingers) for contrastive learning.
"""

import os
import random
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from typing import Tuple, List, Dict


class SOCOFingSiameseDataset(Dataset):
    """
    Dataset that dynamically generates image pairs from the SOCOFing dataset.
    
    Returns:
        tuple: (image1, image2, label)
        label is 1 for Genuine pairs (same finger), 0 for Impostor pairs.
    """
    
    def __init__(self, data_dir: str, transform=None, num_pairs: int = 1000):
        """
        Args:
            data_dir: Path to SOCOFing dataset root (contains 'Real' and 'Altered' dirs).
            transform: PyTorch transforms to apply to images.
            num_pairs: Total number of pairs to generate (half genuine, half impostor).
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.num_pairs = num_pairs
        
        # Determine paths
        self.real_dir = self.data_dir / 'Real'
        self.altered_dir = self.data_dir / 'Altered'
        
        if not self.real_dir.exists() or not self.altered_dir.exists():
            raise FileNotFoundError(f"Ensure '{self.real_dir}' and '{self.altered_dir}' exist.")
            
        self.real_files = sorted([f for f in os.listdir(self.real_dir) if f.endswith('.BMP')])
        
        # Map: subject_finger_key -> list of file paths
        # Key format: "SubjectID_Gender_Finger" e.g., "100_M_Left_index"
        self.real_map: Dict[str, str] = {}
        self.altered_map: Dict[str, List[str]] = {}
        
        self._build_maps()
        self.keys = list(self.real_map.keys())
        
        # Pre-generate pair instructions for consistency during an epoch
        self.pairs = self._generate_pairs()

    def _build_maps(self):
        """Map each unique finger to its Real image and Altered variants."""
        for f in self.real_files:
            # "100__M_Left_index_finger.BMP" -> "100_M_Left_index"
            parts = f.replace('.BMP', '').split('__')
            if len(parts) != 2: continue
            
            sub_id = parts[0]
            rest = parts[1].replace('_finger', '')
            key = f"{sub_id}_{rest}"
            
            self.real_map[key] = str(self.real_dir / f)
            self.altered_map[key] = []
            
        # Scan altered directories
        for diff in ['Altered-Easy', 'Altered-Medium', 'Altered-Hard']:
            d = self.altered_dir / diff
            if not d.exists(): continue
            
            for f in os.listdir(d):
                if not f.endswith('.BMP'): continue
                
                parts = f.replace('.BMP', '').split('__')
                if len(parts) != 2: continue
                
                sub_id = parts[0]
                # "M_Left_index_finger_CR" -> "M_Left_index"
                rest = parts[1].split('_finger_')[0] 
                key = f"{sub_id}_{rest}"
                
                if key in self.altered_map:
                    self.altered_map[key].append(str(d / f))

    def _generate_pairs(self) -> List[Tuple[str, str, int]]:
        """
        Generate instructions for pairs: (path1, path2, label).
        label = 1 (Genuine), label = 0 (Impostor).
        """
        pairs = []
        half = self.num_pairs // 2
        
        # Genuine pairs
        genuine_count = 0
        while genuine_count < half:
            key = random.choice(self.keys)
            if not self.altered_map[key]: continue # Skip if no altered versions
            
            img1 = self.real_map[key]
            img2 = random.choice(self.altered_map[key])
            pairs.append((img1, img2, 1))
            genuine_count += 1
            
        # Impostor pairs
        impostor_count = 0
        while impostor_count < self.num_pairs - half:
            key1, key2 = random.sample(self.keys, 2)
            img1 = self.real_map[key1]
            img2 = self.real_map[key2]
            pairs.append((img1, img2, 0))
            impostor_count += 1
            
        random.shuffle(pairs)
        return pairs

    def _load_image(self, path: str) -> torch.Tensor:
        """Load image as grayscale tensor."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load {path}")
            
        # Convert to RGB because ResNet backbone expects 3 channels
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        if self.transform:
            img = self.transform(img)
        else:
            # Default transform if none provided
            img = transforms.ToTensor()(img)
        
        return img

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path1, path2, label = self.pairs[idx]
        
        img1 = self._load_image(path1)
        img2 = self._load_image(path2)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)
