"""
evaluate.py — Full evaluation of the matching engine.

Computes scores for genuine and impostor pairs, calculates FAR, FRR, EER, and AUC.
Saves the results to results/evaluation_report.csv.
"""

import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision import transforms
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction.embedding import SiameseNetwork, extract_embedding
from src.matching.scorer import compute_embedding_score
from src.matching.decision import compute_rates, find_eer

DATA_DIR = PROJECT_ROOT / 'data' / 'raw' / 'SOCOFing'
MODEL_PATH = PROJECT_ROOT / 'models' / 'siamese_model.pth'
RESULTS_DIR = PROJECT_ROOT / 'results'

def get_embedding(model, transform, device, img_path):
    img_cv = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img_cv is None: return None
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
    tensor = transform(img_rgb)
    emb = extract_embedding(model, tensor, device).cpu().numpy()[0]
    return emb

def main():
    print("=== Phase 5: Evaluation & Benchmarking ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = SiameseNetwork(embedding_dim=128, pretrained=False).to(device)
    
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded trained weights from {MODEL_PATH}")
    else:
        print(f"Warning: Model weights not found at {MODEL_PATH}. Using untrained model.")
        
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    genuine_scores = []
    impostor_scores = []
    
    real_dir = DATA_DIR / 'Real'
    altered_dir = DATA_DIR / 'Altered' / 'Altered-Easy'
    
    if not real_dir.exists() or not altered_dir.exists():
        print("Dataset not found. Please ensure SOCOFing dataset is available.")
        sys.exit(1)
        
    # Take a subset of 100 subjects for evaluation
    real_files = sorted(os.listdir(real_dir))[:100]
    
    print(f"Evaluating {len(real_files)} genuine and {len(real_files)} impostor pairs...")
    
    for f in tqdm(real_files):
        real_path = real_dir / f
        alt_name = f.replace('.BMP', '_CR.BMP')
        alt_path = altered_dir / alt_name
        
        # Genuine Score
        emb1 = get_embedding(model, transform, device, real_path)
        emb2 = get_embedding(model, transform, device, alt_path)
        
        if emb1 is not None and emb2 is not None:
            score = compute_embedding_score(emb1, emb2)
            genuine_scores.append(score)
            
        # Impostor Score
        idx = real_files.index(f)
        next_f = real_files[(idx + 1) % len(real_files)]
        imp_path = real_dir / next_f
        
        emb3 = get_embedding(model, transform, device, imp_path)
        if emb1 is not None and emb3 is not None:
            score = compute_embedding_score(emb1, emb3)
            impostor_scores.append(score)
            
    print(f"\nCollected {len(genuine_scores)} genuine and {len(impostor_scores)} impostor scores.")
    
    # Calculate FAR, FRR, EER
    thresholds = np.linspace(0.0, 1.0, 200)
    far, frr = compute_rates(genuine_scores, impostor_scores, thresholds)
    eer_val, opt_thresh = find_eer(far, frr, thresholds)
    
    # Calculate ROC and AUC
    y_true = [1] * len(genuine_scores) + [0] * len(impostor_scores)
    y_scores = genuine_scores + impostor_scores
    
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    print(f"\nEvaluation Results:")
    print(f"EER: {eer_val:.4f}")
    print(f"Optimal Threshold: {opt_thresh:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    
    # Log to CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / 'evaluation_report.csv'
    
    df = pd.DataFrame([{
        'Model': 'Siamese CNN (ResNet-18)',
        'Dataset': 'SOCOFing (Subset 100)',
        'Genuine Pairs': len(genuine_scores),
        'Impostor Pairs': len(impostor_scores),
        'EER': eer_val,
        'Optimal Threshold': opt_thresh,
        'AUC': roc_auc
    }])
    
    df.to_csv(report_path, index=False)
    print(f"\nResults saved to {report_path}")

if __name__ == "__main__":
    main()
