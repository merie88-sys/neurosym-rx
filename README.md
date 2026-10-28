# NeuroSym-Rx: Temporal Neurosymbolic Fusion for Context-Aware Medication Safety

## Description
Implementation of NeuroSym-Rx, a temporal neurosymbolic framework for medication-safety decision support. The system integrates symbolic pharmacological reasoning with neural contextual analysis through a cascaded fusion mechanism with explicit coherence assessment.

## Dataset Information
- **Source**: TwoSIDES (Tatonetti Lab)
- **URL**: https://nsides.io/data/
- **Filtering**: interactions with count ≥5 and mean PRR ≥1.5 (87,412 high-confidence interactions)

## Code Information
- **Main entry point**: `evaluate_twosides_cv.py`
- **Core modules**: `core/` package containing symbolic layer, neural layer, temporal memory, cascaded fusion, patient generator, and DrugBank baseline
- **Rules**: 60 high-severity pharmacological rules in `data/rules_data.json`

## Usage Instructions
1. (Optional) Download TwoSIDES from https://nsides.io/data/ and place `twosides_filtered.csv` in `data/`
2. Install dependencies: `pip install -r requirements.txt`
3. (Optional) Set LLM API key: `export OPENAI_API_KEY=your_key`
4. Run evaluation: `python evaluate_twosides_cv.py`

Without TwoSIDES data, the script uses a synthetic benchmark for reproducibility. Without an API key, the neural layer uses a deterministic fallback.

## Requirements
- Python 3.10+
- numpy, pandas, scikit-learn, openai, requests

## Methodology
1. Symbolic layer: 60 high-severity rules with deterministic risk computation
2. Neural layer: LLM-based contextual risk via API (BioMistral-7B / GPT-4)
3. Temporal memory: multi-horizon vectorial storage (short/medium/long term)
4. Cascaded fusion: coherence-based strategy selection with symbolic fallback
5. Patient-specific modifiers: age, renal function, polypharmacy

## License
MIT License
