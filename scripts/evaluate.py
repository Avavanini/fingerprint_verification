"""
evaluate.py — Full evaluation of the matching engine.

Computes scores for genuine and impostor pairs, calculates FAR, FRR, EER, and AUC.
Supports three matching modes: embedding-only, minutiae-only, and hybrid fusion.
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
import argparse

# Add project root to path
PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction.embedding import SiameseNetwork, extract_embedding
from src.matching.scorer import compute_embedding_score
from src.matching.decision import compute_rates, find_eer
from src.matching.hybrid import (
    compute_hybrid_score,
    extract_minutiae_template_from_path,
    compute_classical_minutiae_score
)

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
    parser = argparse.ArgumentParser(description="Evaluate the matching engine.")
    parser.add_argument('--difficulty', type=str, choices=['Easy', 'Medium', 'Hard', 'All'], default='Easy',
                        help="Difficulty level to evaluate (Easy, Medium, Hard, or All)")
    parser.add_argument('--mode', type=str, choices=['embedding', 'minutiae', 'hybrid'], default='embedding',
                        help="Matching mode: embedding-only, minutiae-only, or hybrid fusion")
    parser.add_argument('--alpha', type=float, default=0.6,
                        help="Fusion weight for hybrid mode (0.0 = minutiae only, 1.0 = embedding only)")
    args = parser.parse_args()

    print(f"=== Phase 5: Evaluation & Benchmarking ({args.difficulty}, mode={args.mode}, alpha={args.alpha}) ===")
    
    # Load deep model (needed for embedding and hybrid modes)
    device = None
    model = None
    transform_fn = None
    
    if args.mode in ('embedding', 'hybrid'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        model = SiameseNetwork(embedding_dim=128, pretrained=False).to(device)
        
        if MODEL_PATH.exists():
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"Loaded trained weights from {MODEL_PATH}")
        else:
            print(f"Warning: Model weights not found at {MODEL_PATH}. Using untrained model.")
            
        model.eval()
        
        transform_fn = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    genuine_scores = []
    impostor_scores = []
    
    # For detailed reporting in hybrid mode
    genuine_emb_scores = []
    genuine_min_scores = []
    
    real_dir = DATA_DIR / 'Real'
    
    difficulties = ['Easy', 'Medium', 'Hard'] if args.difficulty == 'All' else [args.difficulty]
    
    if not real_dir.exists():
        print("Dataset not found. Please ensure SOCOFing dataset is available.")
        sys.exit(1)
    
    # Take a subset of 100 subjects for evaluation
    real_files = sorted(os.listdir(real_dir))[:100]
    
    # Cache embeddings and templates to avoid redundant computation
    emb_cache = {}
    template_cache = {}
    
    def get_cached_embedding(path):
        key = str(path)
        if key not in emb_cache:
            emb_cache[key] = get_embedding(model, transform_fn, device, path)
        return emb_cache[key]
    
    def get_cached_template(path):
        key = str(path)
        if key not in template_cache:
            template_cache[key] = extract_minutiae_template_from_path(str(path))
        return template_cache[key]
    
    for diff in difficulties:
        altered_dir = DATA_DIR / 'Altered' / f'Altered-{diff}'
        if not altered_dir.exists():
            print(f"Directory {altered_dir} not found. Skipping.")
            continue
            
        print(f"\nEvaluating {diff} difficulty (mode={args.mode})...")
        
        for f in tqdm(real_files, desc=f"Evaluating {diff}"):
            real_path = real_dir / f
            
            # Get real image features based on mode
            real_emb = None
            real_template = None
            
            if args.mode in ('embedding', 'hybrid'):
                real_emb = get_cached_embedding(real_path)
                if real_emb is None: continue
            
            if args.mode in ('minutiae', 'hybrid'):
                real_template = get_cached_template(real_path)
                if args.mode == 'minutiae' and real_template is None: continue
            
            # Genuine Scores
            for suffix in ['_CR.BMP', '_Obl.BMP', '_Zcut.BMP']:
                alt_name = f.replace('.BMP', suffix)
                alt_path = altered_dir / alt_name
                
                if not alt_path.exists():
                    continue
                
                if args.mode == 'embedding':
                    alt_emb = get_cached_embedding(alt_path)
                    if alt_emb is not None:
                        score = compute_embedding_score(real_emb, alt_emb)
                        genuine_scores.append(score)
                
                elif args.mode == 'minutiae':
                    alt_template = get_cached_template(alt_path)
                    if alt_template is not None:
                        score = compute_classical_minutiae_score(real_template, alt_template)
                        genuine_scores.append(score)
                
                elif args.mode == 'hybrid':
                    alt_emb = get_cached_embedding(alt_path)
                    alt_template = get_cached_template(alt_path)
                    
                    if alt_emb is not None:
                        hybrid, emb_s, min_s = compute_hybrid_score(
                            real_emb, alt_emb,
                            real_template, alt_template,
                            alpha=args.alpha
                        )
                        genuine_scores.append(hybrid)
                        genuine_emb_scores.append(emb_s)
                        genuine_min_scores.append(min_s)
            
            # Impostor Score (Compare real with next real)
            idx = real_files.index(f)
            next_f = real_files[(idx + 1) % len(real_files)]
            imp_path = real_dir / next_f
            
            if args.mode == 'embedding':
                imp_emb = get_cached_embedding(imp_path)
                if imp_emb is not None:
                    score = compute_embedding_score(real_emb, imp_emb)
                    impostor_scores.append(score)
            
            elif args.mode == 'minutiae':
                imp_template = get_cached_template(imp_path)
                if imp_template is not None and real_template is not None:
                    score = compute_classical_minutiae_score(real_template, imp_template)
                    impostor_scores.append(score)
            
            elif args.mode == 'hybrid':
                imp_emb = get_cached_embedding(imp_path)
                imp_template = get_cached_template(imp_path)
                
                if imp_emb is not None:
                    hybrid, _, _ = compute_hybrid_score(
                        real_emb, imp_emb,
                        real_template, imp_template,
                        alpha=args.alpha
                    )
                    impostor_scores.append(hybrid)
                
    if not genuine_scores or not impostor_scores:
        print("Not enough scores collected to perform evaluation.")
        sys.exit(1)
            
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
    
    print(f"\nEvaluation Results ({args.mode} mode):")
    print(f"EER: {eer_val:.4f}")
    print(f"Optimal Threshold: {opt_thresh:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    
    if args.mode == 'hybrid' and genuine_emb_scores:
        avg_emb = np.mean(genuine_emb_scores)
        avg_min = np.mean(genuine_min_scores)
        print(f"\nComponent Breakdown (genuine pairs):")
        print(f"  Avg Embedding Score: {avg_emb:.4f}")
        print(f"  Avg Minutiae Score:  {avg_min:.4f}")
    
    # Log to CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / 'evaluation_report.csv'
    
    row = {
        'Model': f'Siamese CNN (ResNet-18) [{args.mode}]',
        'Dataset': f'SOCOFing (Subset 100, {args.difficulty})',
        'Mode': args.mode,
        'Alpha': args.alpha if args.mode == 'hybrid' else 'N/A',
        'Genuine Pairs': len(genuine_scores),
        'Impostor Pairs': len(impostor_scores),
        'EER': eer_val,
        'Optimal Threshold': opt_thresh,
        'AUC': roc_auc
    }
    
    df = pd.DataFrame([row])
    
    # Append to existing report if it exists
    if report_path.exists():
        existing = pd.read_csv(report_path)
        df = pd.concat([existing, df], ignore_index=True)
    
    df.to_csv(report_path, index=False)
    print(f"\nResults saved to {report_path}")

if __name__ == "__main__":
    main()
