#!/usr/bin/env python3
"""
Utility functions for NeuroSym-Rx

Author: Anonymous (for double-blind review)
"""

import json
import numpy as np
from typing import Dict, List, Tuple

def load_drugbank_rules(rules_path: str = "drugbank_rules.json") -> Dict[Tuple[str, str], float]:
    """
    Load DrugBank interaction rules from JSON file.
    
    Args:
        rules_path: Path to JSON file containing drug interaction rules
    
    Returns:
        Dictionary with (drug1, drug2) tuple keys and severity scores (0.0-1.0)
    """
    with open(rules_path, 'r') as f:
        data = json.load(f)
    
    rules = {}
    for interaction in data['interactions']:
        drug1 = interaction['drug1'].lower()
        drug2 = interaction['drug2'].lower()
        severity = interaction['severity']
        rules[(drug1, drug2)] = severity
        rules[(drug2, drug1)] = severity  # Symmetric
    
    return rules

def compute_interaction_score(drug1: str, drug2: str, rules: Dict) -> float:
    """
    Compute interaction severity score using DrugBank rules.
    
    Args:
        drug1: First drug name
        drug2: Second drug name
        rules: Dictionary of drug interaction rules
    
    Returns:
        Severity score (0.0 = no interaction, 1.0 = critical interaction)
    """
    key = (drug1.lower(), drug2.lower())
    
    if key in rules:
        return rules[key]
    else:
        # No known interaction - return low baseline risk
        return 0.15

def create_patient_context(age: int = 65, renal_function: float = 75.0, 
                          polypharmacy_count: int = 5) -> Dict:
    """
    Create standardized patient context for evaluation.
    
    Args:
        age: Patient age in years
        renal_function: eGFR value (mL/min/1.73m²)
        polypharmacy_count: Number of concurrent medications
    
    Returns:
        Dictionary with patient-specific factors
    """
    return {
        'age': age,
        'renal_function': renal_function,
        'polypharmacy_count': polypharmacy_count,
        'therapeutic_history': []  # Empty for TwoSIDES benchmark
    }

def apply_clinical_modifiers(base_score: float, context: Dict) -> float:
    """
    Apply clinical modifiers for personalized risk stratification.
    
    Args:
        base_score: Base interaction risk score (0.0-1.0)
        context: Patient context dictionary
    
    Returns:
        Modified risk score with clinical factors applied
    """
    modified_score = base_score
    
    # Age modifier: Elderly patients (>75) have higher risk
    if context.get('age', 50) > 75:
        modified_score *= 1.15
    
    # Renal function modifier: Impaired renal function increases risk
    renal_function = context.get('renal_function', 90.0)
    if renal_function < 60:  # Moderate impairment
        modified_score *= 1.20
    elif renal_function < 30:  # Severe impairment
        modified_score *= 1.35
    
    # Polypharmacy modifier: More medications = higher risk
    polypharmacy = context.get('polypharmacy_count', 0)
    if polypharmacy >= 8:
        modified_score *= 1.10
    
    return np.clip(modified_score, 0.0, 1.0)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    Compute standard classification metrics.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_pred: Predicted scores (0.0-1.0)
        threshold: Decision threshold for binary classification
    
    Returns:
        Dictionary of metrics
    """
    y_binary = (y_pred >= threshold).astype(int)
    
    tp = np.sum((y_binary == 1) & (y_true == 1))
    fp = np.sum((y_binary == 1) & (y_true == 0))
    fn = np.sum((y_binary == 0) & (y_true == 1))
    tn = np.sum((y_binary == 0) & (y_true == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }
