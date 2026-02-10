# NeuroSym-Rx: Cascaded Neurosymbolic Fusion for Context-Aware Medication Safety

Official implementation of NeuroSym-Rx...

## 📝 Citation

If you use this work, please cite:

> Meriem KERMANI. (2026). NeuroSym-Rx: Cascaded Neurosymbolic Fusion for 
> Context-Aware Medication Safety. Zenodo. DOI: 10.5281/zenodo.18598830

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/your-username/neurosym-rx.git
cd neurosym-rx

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download TwoSIDES dataset (official source: nsides.io)
python data/download_twosides.py

# 4. Preprocess: filter high-confidence interactions
python data/preprocess.py

# 5. Run evaluation on TwoSIDES benchmark
python evaluation/evaluate.py
