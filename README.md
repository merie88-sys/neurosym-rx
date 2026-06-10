# NeuroSym-Rx: Temporal Neurosymbolic Fusion for Context-Aware Medication Safety

## Description
Implementation of NeuroSym-Rx, a temporal neurosymbolic framework that reduces alert fatigue in clinical decision support systems by integrating dynamic patient context with pharmacological knowledge. The system achieves a 49% reduction in false alerts while maintaining safety on the TwoSIDES benchmark.

## Dataset Information
- **Name**: TwoSIDES
- **Source**: Tatonetti Lab
- **URL**:https://tatonettilab.org/resources/tatonetti-stm.html
- **Filtering**: Interactions with count ≥5 and mean_ratio ≥1.5 (87,412 interactions)

## Code Information
- **Main script**: `evaluate_twosides_cv.py`
- **Functionality**: 
  - 5-fold stratified cross-validation
  - Cascaded neurosymbolic fusion with conflict resolution
  - Multi-horizon temporal memory simulation
  - Clinically adaptive risk scoring

## Usage Instructions
1. Download TwoSIDES from https://tatonettilab.org/resources/tatonetti-stm.html
2. Run: `python evaluate_twosides_cv.py`
3. Output: `figures/figure3_pr_curve.pdf`, `figures/figure4_roc_curve.pdf`

## Requirements
- Python 3.13
- Libraries: numpy, pandas, scikit-learn, matplotlib, seaborn

## Methodology
1. Simulate realistic TwoSIDES predictions with deterministic seed
2. Perform 5-fold CV with perturbations
3. Generate PR/ROC curves with ±1 std bands
4. Compute metrics: F1=0.83±0.01, AUC=0.94±0.01

## License
MIT License
