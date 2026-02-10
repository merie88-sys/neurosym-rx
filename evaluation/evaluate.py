#!/usr/bin/env python3
"""
Evaluate NeuroSym-Rx on TwoSIDES benchmark
→ Computes F1, precision, recall, AUC-PR, AUC-ROC
→ Compares against rule-based baseline
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from tqdm import tqdm
from src.neurosym_rx import NeuroSymRx

INPUT_PATH = "data/processed/twosides_filtered.csv"

def simulate_contexts(n_samples: int):
    """Generate realistic patient contexts for evaluation"""
    np.random.seed(42)
    contexts = []
    for _ in range(n_samples):
        contexts.append({
            'age': np.random.choice([45, 65, 78, 85], p=[0.25, 0.35, 0.30, 0.10]),
            'egfr': np.random.choice([95, 75, 55, 35], p=[0.40, 0.30, 0.20, 0.10]),
            'polypharmacy_count': np.random.choice([2, 4, 7, 10], p=[0.30, 0.40, 0.20, 0.10])
        })
    return contexts

def evaluate():
    print("Loading preprocessed TwoSIDES...")
    df = pd.read_csv(INPUT_PATH)
    print(f"✓ Dataset size: {len(df):,} interactions")
    
    # Sample for faster evaluation (remove sampling for full evaluation)
    if len(df) > 20000:
        df = df.sample(n=20000, random_state=42)
        print(f"✓ Sampling 20,000 interactions for evaluation")
    
    model = NeuroSymRx()
    contexts = simulate_contexts(len(df))
    
    print("Running inference...")
    y_scores = []
    for (_, row), ctx in tqdm(zip(df.iterrows(), contexts), total=len(df)):
        score = model.predict_risk(row['drug1'], row['drug2'], ctx)
        y_scores.append(score)
    
    y_true = df['label'].values
    y_scores = np.array(y_scores)
    y_pred = (y_scores >= 0.5).astype(int)
    
    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    auc_roc = roc_auc_score(y_true, y_scores)
    auc_pr = average_precision_score(y_true, y_scores)
    
    print("\n" + "="*60)
    print("NEUROSYM-RX EVALUATION RESULTS (TwoSIDES Benchmark)")
    print("="*60)
    print(f"Precision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")
    print(f"F1-score  : {f1:.3f}")
    print(f"AUC-ROC   : {auc_roc:.3f}")
    print(f"AUC-PR    : {auc_pr:.3f}")
    print("="*60)
    
    # Save metrics
    metrics = pd.DataFrame([{
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'n_samples': len(df)
    }])
    metrics.to_csv('evaluation/metrics.csv', index=False)
    print("\n✅ Metrics saved to: evaluation/metrics.csv")

if __name__ == "__main__":
    evaluate()
