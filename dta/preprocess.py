#!/usr/bin/env python3
"""
Preprocess TwoSIDES: filter high-confidence interactions
→ count ≥ 5 (statistical reliability)
→ mean_ratio ≥ 1.5 (clinical relevance)
"""

import pandas as pd
import os

INPUT_PATH = "data/raw/TWOSIDES.csv"
OUTPUT_PATH = "data/processed/twosides_filtered.csv"

def preprocess():
    print("Loading TwoSIDES...")
    df = pd.read_csv(INPUT_PATH)
    total = len(df)
    
    # Apply filters
    df = df[df['count'] >= 5]
    df = df[df['mean_ratio'] >= 1.5]
    df['label'] = (df['mean_ratio'] >= 2.0).astype(int)
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"✓ Raw interactions: {total:,}")
    print(f"✓ After filtering: {len(df):,} high-confidence interactions")
    print(f"✅ Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess()
