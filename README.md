# OHCA Classifier: Automated Out-of-Hospital Cardiac Arrest Identification

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/monajm36/ohca-classifier-v11)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A transformer-based deep learning system for automatically identifying Out-of-Hospital Cardiac Arrest (OHCA) cases from clinical notes, with specific focus on reducing false positives from in-hospital cardiac arrest (IHCA) cases.

## 🎯 Key Results

| Metric | V9 (Baseline) | V10 (+ Location) | V11 (+ Temporal) |
|--------|---------------|------------------|------------------|
| **Sensitivity** | 96.1% | 84.2% | **92.1%** |
| **Specificity** | 69.6% | 89.6% | **89.4%** |
| **F1-Score** | 0.732 | 0.814 | **0.856** |
| **AUC-ROC** | 0.932 | 0.938 | **0.956** |
| **IHCA False Positives** | 88 (64.2%) | 39 (28.5%) | **39 (28.5%)** |

**Validation:** 647 manually annotated clinical notes from UChicago C19 dataset

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/monajm36/ohca-classifier.git
cd ohca-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Using the Pre-trained Model

```python
from predict_v11 import predict_ohca

# Example clinical note
note = """
Patient found unresponsive at home by family. 911 called.
EMS arrived and initiated CPR. ROSC achieved in field.
Transported to ED.
"""

# Predict
result = predict_ohca(note, threshold=0.14)

print(f"Prediction: {result['prediction']}")  # 'OHCA' or 'Non-OHCA'
print(f"Probability: {result['probability']:.2%}")
print(f"Features: {result['features']}")
```

### Download Pre-trained Model

```bash
# Model is available on Hugging Face
# https://huggingface.co/monajm36/ohca-classifier-v11

# Or use the download script
python scripts/download_model.py
```

## 📊 Model Architecture

**V11: Temporal + Location-Aware OHCA Classifier**

```
Input Clinical Note
       ↓
┌──────────────────────┐
│  Text Processing     │
│  - Extract sections  │
│  - Tokenize (512)    │
└──────────────────────┘
       ↓
┌──────────────────────────────────────┐
│  Feature Extraction                  │
├──────────────────────────────────────┤
│  1. BERT Embeddings (768)            │
│  2. Location Features (2)            │
│     • OHCA indicators (22 phrases)   │
│     • IHCA indicators (25 phrases)   │
│  3. Temporal Features (5)            │
│     • Arrest timing score            │
│     • First location (in/out)        │
│     • Movement patterns              │
└──────────────────────────────────────┘
       ↓
┌──────────────────────┐
│  MLP Classifier      │
│  775 → 512 → 256 → 2 │
└──────────────────────┘
       ↓
  OHCA Probability
```

**Base Model:** microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract

## 📁 Repository Structure

```
ohca-classifier/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation
│
├── models/                           # Model code
│   ├── __init__.py
│   ├── v9_bert_classifier.py        # V9: Baseline BERT
│   ├── v10_location_aware.py        # V10: + Location features
│   └── v11_temporal_location.py     # V11: + Temporal features
│
├── training/                         # Training scripts
│   ├── train_v9.py
│   ├── train_v10.py
│   └── train_v11.py
│
├── prediction/                       # Prediction scripts
│   ├── predict_v9.py
│   ├── predict_v10.py
│   └── predict_v11.py
│
├── features/                         # Feature extraction
│   ├── __init__.py
│   ├── text_processing.py           # Section extraction
│   ├── location_features.py         # Location indicators
│   └── temporal_features.py         # Temporal features
│
├── evaluation/                       # Evaluation scripts
│   ├── compare_models.py            # V9 vs V10 vs V11
│   ├── threshold_optimization.py   # Find optimal thresholds
│   └── error_analysis.py           # Analyze false positives
│
├── scripts/                          # Utility scripts
│   ├── download_model.py            # Download from HF
│   └── prepare_data.py              # Data preprocessing
│
├── notebooks/                        # Jupyter notebooks
│   ├── model_comparison.ipynb
│   ├── feature_analysis.ipynb
│   └── demo.ipynb
│
├── docs/                            # Documentation
│   ├── COMPREHENSIVE_REPORT.pdf     # Full technical report
│   ├── model_architecture.md
│   ├── feature_engineering.md
│   └── training_guide.md
│
└── tests/                           # Unit tests
    ├── test_features.py
    ├── test_models.py
    └── test_predictions.py
```

## 🔬 Methodology

### Progressive Feature Engineering

Our approach built three successive models, each addressing specific challenges:

**V9: Baseline BERT Classifier**
- Pure semantic understanding
- Issue: 65% of false positives were IHCA cases
- Learning: Model confused arrest terminology regardless of location

**V10: Location-Aware Classifier**
- Added 2 location features (22 OHCA + 25 IHCA indicators)
- Reduced IHCA false positives by 55.7%
- Issue: Lost sensitivity (96.1% → 84.2%)
- Learning: Location helps but temporal context missing

**V11: Temporal + Location-Aware Classifier**
- Added 5 temporal features (timing, movement patterns)
- Recovered sensitivity (84.2% → 92.1%)
- Maintained high specificity (89.4%)
- Learning: Temporal sequence crucial for disambiguation

### Feature Categories

**Location Features (2):**
1. OHCA indicator count: home, EMS, scene, field, bystander, ambulance...
2. IHCA indicator count: floor, ICU, ward, room, bed, code blue...

**Temporal Features (5):**
1. Arrest timing score: "before arrival" vs "during hospitalization" phrases
2. First location outside: Binary indicator of first location mentioned
3. First location inside: Binary indicator of first location mentioned
4. Movement outside→inside: Count of transition patterns
5. Movement inside→inside: Count of transition patterns

## 📈 Training

### Data

- **Training Set**: MIMIC-III clinical notes
  - 330 notes total (47 OHCA, 283 Non-OHCA)
  - Split: 70% train / 15% validation / 15% test
  - Average note length: 13,042 characters

- **Validation Set**: UChicago C19 dataset
  - 647 manually annotated cases
  - 203 OHCA (31.4%)
  - 137 IHCA (21.2%)
  - 307 Non-arrest (47.4%)

### Train Your Own Model

```bash
# Train V11 model
python training/train_v11.py \
    --data_path /path/to/mimic_labelled_binary.csv \
    --output_dir ./models/v11_output \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --num_epochs 5

# Train with custom hyperparameters
python training/train_v11.py \
    --config configs/custom_config.json
```

## 🎯 Threshold Selection

V11 offers flexible threshold tuning for different clinical scenarios:

| Use Case | Threshold | Sensitivity | Specificity | F1 | When to Use |
|----------|-----------|-------------|-------------|-----|-------------|
| **Screening** | 0.14 | 92.1% | 89.4% | 0.856 | Maximize recall |
| **Balanced** | 0.74 | 82.3% | 93.2% | 0.831 | General use |
| **Research** | 0.85 | 75.4% | 95.0% | 0.810 | High precision needed |

```python
# Use different thresholds
result_screening = predict_ohca(note, threshold=0.14)  # High sensitivity
result_balanced = predict_ohca(note, threshold=0.74)   # Balanced
result_research = predict_ohca(note, threshold=0.85)   # High specificity
```

## 📊 Evaluation

### Compare All Models

```bash
# Run comprehensive comparison
python evaluation/compare_models.py \
    --data_path /path/to/c19_validation.csv \
    --output_dir ./results

# Generate comparison plots
python evaluation/compare_models.py --plot
```

### Error Analysis

```bash
# Analyze false positives and false negatives
python evaluation/error_analysis.py \
    --model_path models/v11_output/final_model \
    --data_path /path/to/validation.csv
```

## 🔍 Example Predictions

### True OHCA Case
```
Patient found unresponsive at home. Family called 911.
EMS arrived, started CPR. ROSC in field.
Transported to ED.

→ Prediction: OHCA (98.5%)
→ Key features:
   - Location: home (OHCA +1), EMS (+1)
   - Temporal: "found at home", "ROSC in field"
   - Movement: outside→inside
```

### True IHCA Case
```
Patient admitted to medical floor for pneumonia.
On hospital day 3, found unresponsive in bed.
Code blue called. CPR initiated on floor.

→ Prediction: Non-OHCA (2.3% OHCA probability)
→ Key features:
   - Location: floor (+1), bed (+1)
   - Temporal: "hospital day 3", "admitted"
   - Movement: inside→inside
```

### Challenging Case (Mixed Signals)
```
Patient arrested at home. EMS called, ROSC achieved.
Admitted to ICU. On hospital day 2, arrested again.

→ V9: OHCA (85%) - sees "home" and "EMS"
→ V10: Uncertain (52%) - conflicting locations
→ V11: OHCA (78%) - temporal features indicate primary arrest was OHCA
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas for improvement:
- [ ] Additional temporal features
- [ ] Multi-institution validation
- [ ] Support for non-English notes
- [ ] Real-time deployment pipeline
- [ ] Explainability visualizations

## 📝 Citation

If you use this code or model in your research, please cite:

```bibtex
@misc{moukaddem2025ohca,
  author = {Moukaddem, Mona},
  title = {OHCA Classifier: Automated Out-of-Hospital Cardiac Arrest Identification 
           using Temporal and Location-Aware Deep Learning},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/monajm36/ohca-classifier}},
  note = {Model available at \url{https://huggingface.co/monajm36/ohca-classifier-v11}}
}
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MIMIC-III Database**: Johnson et al., Scientific Data (2016)
- **UChicago C19 Dataset**: Validation data source
- **BiomedNLP-PubMedBERT**: Microsoft Research
- **Hugging Face**: Model hosting and transformers library

## 📧 Contact

**Mona Moukaddem**
- GitHub: [@monajm36](https://github.com/monajm36)
- Hugging Face: [@monajm36](https://huggingface.co/monajm36)
- Email: [your-email@example.com]

## 🔗 Links

- 🤗 [Pre-trained Model (Hugging Face)](https://huggingface.co/monajm36/ohca-classifier-v11)
- 📊 [Interactive Demo](https://huggingface.co/spaces/monajm36/ohca-demo) *(coming soon)*
- 📄 [Technical Report](docs/COMPREHENSIVE_REPORT.pdf)
- 📝 [Blog Post](https://your-blog.com/ohca-classifier) *(coming soon)*

---

**Built with ❤️ for improving cardiac arrest research and patient care**
