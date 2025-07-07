# ChronoForget: Machine Unlearning for Multimodal Clinical AI

This repository contains the implementation of ChronoForget, a novel framework for machine unlearning in healthcare settings that addresses class imbalance, longitudinal dependencies, and fairness-aware forgetting in multimodal clinical models.

ChronoForget builds upon the Forget-MI baseline method, which focuses on unlearning both unimodal and joint modality representations from trained models. We extend Forget-MI by incorporating:
- Dynamic uncertainty-aware loss weighting
- Focal loss with adaptive focusing
- Temporal consistency constraints
- Contrastive margin losses
- Teacher-student distillation for retention

This makes ChronoForget more robust, equitable, and effective in real-world clinical AI systems governed by privacy regulations such as GDPR and HIPAA.

---

## 📁 Repository Structure

```
chronoforget/
│
├── data/
│   ├── loaders.py        # Dataset-specific dataloaders (MIMIC-III, NIH ChestX-ray14, ADNI, ISIC Skin, eICU)
│
├── models/
│   ├── base_model.py     # Base model definitions (ResNet, BERT, LSTM, etc.)
│   ├── forget_mi.py      # Forget-MI baseline implementation
│   ├── chronoforget.py   # ChronoForget methodology
│
├── losses/
│   ├── forget_losses.py  # Uncertainty-aware reweighting, focal loss, contrastive margin loss
│   ├── retain_losses.py  # Teacher-student distillation, embedding drift
│   ├── temporal_losses.py# Cross-time forgetting and sequence-aware propagation
│
├── trainers/
│   ├── trainer.py        # Training loop with joint objective optimization
│
├── utils/
│   ├── metrics.py        # Evaluation metrics (AUC, F1, MIA score, fairness metrics)
│
├── notebooks/
│   └── visualization.ipynb # Visualization of results (ROC/PR curves, radar plots, leakage analysis)
│
├── scripts/
│   └── run_experiments.py # End-to-end training and evaluation pipeline
│
└── README.md             # This file
```

---

## 🧪 Supported Datasets

ChronoForget is evaluated on five multimodal clinical datasets:

| Dataset | Modalities | Temporal? | Class Imbalance? | Source |
|--------|------------|----------|------------------|--------|
| MIMIC-III | EHR (tabular), Notes (text) | ✅ Yes | ✅ Yes | [Johnson et al., *Scientific Data*, 2019] |
| NIH ChestX-ray14 | X-ray images, Labels | ❌ No | ✅ Yes | [NIH ChestX-ray14 Dataset] |
| ADNI | MRI/CT scans, Cognitive scores, Genetic data | ✅ Yes | ✅ Yes | [Alzheimer’s Disease Neuroimaging Initiative] |
| ISIC Skin Lesion Archive | Dermoscopic images, Metadata | ❌ No | ✅ Yes | [ISIC Challenge] |
| eICU Collaborative Research Database | Vitals (time-series), Imaging (optional), Notes | ✅ Yes | ✅ Yes | [Pollard et al., *Scientific Data*, 2018] |

These datasets are chosen due to their relevance to clinical AI and represent diverse challenges including:
- Longitudinal patient records
- High-dimensional imaging and genomic data
- Severe class imbalance (e.g., rare pathologies)

---

## 🔬 Methodology Overview

### Base Model: Forget-MI

Forget-MI introduces a multimodal unlearning strategy where both unimodal and joint modality embeddings are modified to erase specific patient information while preserving performance on retained data.

Key components:
- Unimodal Forgetting Loss: Pushes away individual modality embeddings.
- Multimodal Forgetting Loss: Breaks cross-modality associations.
- Unimodal Retention Loss: Preserves original unimodal knowledge.
- Multimodal Retention Loss: Preserves joint representation learning.

### ChronoForget – Our Contribution

ChronoForget extends Forget-MI with novel enhancements:
- Uncertainty-Aware Reweighting: Dynamically adjusts sample weights during unlearning based on predictive uncertainty and class rarity.
- Focal Loss with Adaptive Focusing: Focuses on hard-to-forget samples using uncertainty-guided intensity modulation.
- Contrastive Margin Losses: Maintains inter-class separation post-unlearning, especially for rare classes.
- Temporal Consistency Loss: Propagates forgetting across time steps to remove longitudinal traces.
- Teacher-Student Distillation Loss: Ensures minimal utility degradation on non-forgotten data.

ChronoForget outperforms existing methods like NegGrad+, SCRUB, CF-k, EU-k, and MultiDelete, achieving lower Membership Inference Attack (MIA) scores, stronger forget set degradation, and superior fairness preservation.

---

## 🛠️ Installation

```bash
git clone https://github.com/SaeedIqbal/chronoforget.git
cd chronoforget
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- PyTorch 2.x
- scikit-learn
- pandas
- numpy
- transformers (for text encoders)
- matplotlib / plotly (for visualization)

---

## 🚀 Usage

To train and evaluate ChronoForget:

```bash
python scripts/run_experiments.py --dataset MIMIC-III --method ChronoForget --forget_percent 5
```

Available options:
- `--dataset`: MIMIC-III, NIH ChestX-ray14, ADNI, ISIC Skin, eICU
- `--method`: ChronoForget, Forget-MI, NegGrad+, SCRUB, CF-k, EU-k, MultiDelete, Retrain
- `--forget_percent`: 3%, 6%, or 10%
- `--visualize`: Generate ROC/PR curves and radar plots

---

## 📊 Evaluation Metrics

The following metrics are used to assess unlearning effectiveness:

- Forget Set Performance: AUC ↓, F1 ↓
- Test Set Performance: AUC ↑, F1 ↑
- Embedding Drift: Cosine Similarity ↓, Euclidean Distance ↑
- Privacy Leakage: Membership Inference Attack (MIA) score ↓
- Fairness Preservation: Equal Opportunity Difference ↓, Demographic Parity ↓

ChronoForget achieves the best balance between strong forgetting and high test utility while maintaining fairness and privacy compliance.

---

## 📄 References

### Papers Cited

1. Hardan, S. et al. (2025). "Forget-MI: Machine Unlearning for Forgetting Multimodal Information in Healthcare Settings." *arXiv preprint arXiv:2506.23145*  
2. Cheng, J. & Amiri, H. (2024). "MultiDelete for Multimodal Machine Unlearning." *European Conference on Computer Vision*
3. Goel, S. et al. (2022). "Evaluating Inexact Unlearning Requires Revisiting Forgetting." *CoRR abs/2201.06640*
4. Triantafillou, E. et al. (2024). "Are We Making Progress in Unlearning?" *NeurIPS Competition Track*
5. Zhou, J. et al. (2023). "A Unified Method to Revoke Private Patient Data in Intelligent Healthcare with Audit to Forget." *Nature Communications*

### Dataset References

- MIMIC-CXR: Johnson, K. et al. (2019). “MIMIC-CXR, a de-identified publicly available database of chest radiographs.” *Scientific Data*
- NIH ChestX-ray14: Wang, X. et al. (2017). “ChestX-ray14: A multi-label dataset for automated pneumonia detection and classification.”
- ADNI: Alzheimer's Disease Neuroimaging Initiative (ADNI), www.loni.usc.edu
- ISIC Skin Lesion Archive: Codella, N. et al. (2019). “Skin Lesion Analysis Toward Melanoma Detection,” *IEEE Access*
- eICU Collaborative Research Database: Pollard, T. et al. (2018). “eICU: An expanded ICU dataset,” *Scientific Data*

---

## 📝 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 📨 Contact

For questions or contributions, please open an issue or reach out via email at saeediqbalkhattak@gmail.com

---

## 🏷️ Acknowledgements

We thank the authors of Forget-MI and related works for establishing the foundation for multimodal unlearning in healthcare. We also acknowledge funding sources and supporting institutions where applicable.
