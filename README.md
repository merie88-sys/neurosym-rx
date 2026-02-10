# NeuroSym-Rx: Cascaded Neurosymbolic Fusion for Context-Aware Medication Safety

Official implementation of NeuroSym-Rx, a temporal neurosymbolic framework for personalized drug interaction detection and risk assessment.

## 📌 Key Innovations

- ✅ **Cascaded fusion mechanism** with explicit coherence assessment between symbolic and neural evidence streams
- ✅ **Multi-horizon vectorial temporal memory** (short/medium/long-term therapeutic history)
- ✅ **Clinically adaptive alert thresholds** modulated by age, renal function, and polypharmacy burden
- ✅ **49% reduction in false alerts** while maintaining safety standards (TwoSIDES benchmark)

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- 4 GB RAM minimum
- Internet connection (for dataset download)

### Installation
```bash
git clone https://github.com/your-username/neurosym-rx.git
cd neurosym-rx
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
