#!/usr/bin/env python3
"""
Evaluate NeuroSym-Rx on TwoSIDES benchmark
→ Computes F1, precision, recall
→ Compares against rule-based baseline
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
# ⚠️ REPLACE WITH YOUR ACTUAL IMPLEMENTATION ⚠️
# from src.neurosym_rx import NeuroSymRx

def simulate_neurosym_prediction(drug1, drug2, context):
    """Placeholder — replace with your actual model"""
    high_risk = [('warfarin', 'amiodarone'), ('warfarin', 'aspirin')]
    pair = tuple(sorted([drug1.lower(), drug2.lower()]))
    base = 0.85 if pair in [tuple(sorted(p)) for p in high_risk] else 0.25
    if context and context.get('age', 0) > 75:
        base *= 1.3
    return min(max(base, 0.0), 1.0)

def evaluate():
    df = pd.read_csv("data/processed/twosides_filtered.csv")
    if len(df) > 20000:
        df = df.sample(n=20000, random_state=42)
    
    y_true = df['label'].values
    y_scores = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        score = simulate_neurosym_prediction(row['drug1'], row['drug2'], None)
        y_scores.append(score)
    
    y_pred = (np.array(y_scores) >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    print("\n" + "="*60)
    print("NEUROSYM-RX EVALUATION (TwoSIDES Benchmark)")
    print("="*60)
    print(f"Precision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")
    print(f"F1-score  : {f1:.3f}")
    print("="*60)

if __name__ == "__main__":
    evaluate()
