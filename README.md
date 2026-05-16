# 🎵 AudioGuard
### Speech · Music · Environmental Noise — Classical ML Audio Classifier

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/Librosa-0.10%2B-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3%2B-f7931e?style=flat-square&logo=scikit-learn"/>
  <img src="https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=flat-square&logo=streamlit"/>
  <img src="https://img.shields.io/badge/Dataset-MUSAN-4ECDC4?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/>
</p>

---

## Overview

**AudioGuard** classifies audio signals into three categories — **speech**, **music**, and **environmental noise** — using exclusively hand-crafted acoustic features and classical machine learning. No deep learning required.

The pipeline follows a rigorous structure with clean module separation:

```
Data Collection → Feature Extraction → Preprocessing → Training → Evaluation → Deployment
```

---

## Repository Structure

```
AudioGuard/
│
├── app.py                        ← Streamlit deployment app
├── train.py                      ← Main training entry point
│
├── src/
│   ├── config.py                 ← Central configuration (paths, hyperparams, constants)
│   ├── data_loader.py            ← Dataset scanning & stratified splitting
│   ├── feature_extraction.py     ← MFCC + Spectral + ZCR + Energy features
│   ├── models.py                 ← SVM / KNN / Gradient Boosting definitions
│   ├── training.py               ← Feature pipeline + model training + artifact saving
│   ├── evaluation.py             ← Metrics, confusion matrices, ROC, SNR robustness
│   └── utils.py                  ← Logger and shared helpers
│
├── models/                       ← Saved .pkl artefacts (after train.py)
│   ├── svm_model.pkl
│   ├── knn_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── scaler.pkl
│   ├── selector.pkl
│   ├── label_encoder.pkl
│   ├── feature_names.pkl
│   └── selected_features.pkl
│
├── outputs/
│   ├── evaluation/               ← Confusion matrices, ROC, PR, SNR, model comparison
│   └── explainability/           ← Feature importance plots
│
├── .streamlit/config.toml        ← Streamlit dark theme
├── requirements.txt
└── README.md
```

---

## Dataset

**MUSAN — Music, Speech, and Noise Corpus**
- **Kaggle:** [`dogrose/musan-dataset`](https://www.kaggle.com/datasets/dogrose/musan-dataset)
- **Classes:** `speech` · `music` · `noise`
- **Size:** ~109 hours of audio, WAV format
- **License:** Creative Commons CC BY 4.0

> Originally published: Snyder, D., Chen, G. & Povey, D. (2015). *MUSAN: A Music, Speech, and Noise Corpus.* arXiv:1510.08484

---

## Feature Engineering

**153 raw features** are extracted per 3-second audio segment, then reduced to the **top 80** via Mutual Information selection.

| Feature Group | Details | Dims |
|---|---|---|
| MFCC (20 coeff) | mean · variance · skewness | 60 |
| Δ-MFCC (20 coeff) | mean · variance | 40 |
| Δ²-MFCC (20 coeff) | mean · variance | 40 |
| Spectral Centroid | mean · variance | 2 |
| Spectral Bandwidth | mean · variance | 2 |
| Spectral Flux | mean · variance | 2 |
| Spectral Rolloff | mean · variance | 2 |
| Short-Time Energy | mean · variance · max | 3 |
| Zero Crossing Rate | mean · variance | 2 |
| **Total raw** | | **153** |
| **After MI selection** | | **→ 80** |

---

## Models & Pipeline

```
Raw audio (.wav/.mp3)
    │
    ▼ load_and_preprocess()
Pre-emphasis + peak normalise (3s clip @ 22050 Hz)
    │
    ▼ extract_all_features()
153-dim feature vector
    │
    ▼ StandardScaler
Scaled features
    │
    ▼ SelectKBest (mutual_info_classif, k=80)
80-dim selected features
    │
    ├─▶ SVM (RBF, C=10)
    ├─▶ KNN (k=7, Euclidean)
    └─▶ Gradient Boosting (200 trees, lr=0.1)
```

| Model | Notes |
|---|---|
| **Gradient Boosting** ⭐ | Best overall Macro F1 — used by default in app |
| **SVM (RBF)** | Most robust under SNR degradation |
| **KNN (k=7)** | Fastest inference, Euclidean distance |

---

## Setup & Usage

### 1 · Clone

```bash
git clone https://github.com/<your-username>/AudioGuard.git
cd AudioGuard
```

### 2 · Install dependencies

```bash
pip install -r requirements.txt
# ffmpeg is needed for MP3 support:
# Ubuntu: sudo apt-get install ffmpeg
# macOS:  brew install ffmpeg
```

### 3 · Download MUSAN

```bash
# Requires ~/.kaggle/kaggle.json  (Kaggle API token)
kaggle datasets download -d dogrose/musan-dataset -p data --unzip
```

Directory should look like:
```
data/musan/
├── music/
├── speech/
└── noise/
```

### 4 · Train

```bash
python train.py
```

Training runs in ~15–30 min depending on hardware. All artefacts are saved to `models/` and plots to `outputs/`.

### 5 · Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) and upload any `.wav` / `.mp3` / `.flac` / `.ogg` file.

---

## Evaluation Outputs

After `python train.py`, the following plots are saved to `outputs/`:

| File | Description |
|---|---|
| `evaluation/cm_*.png` | 3×3 confusion matrices per model |
| `evaluation/roc_*.png` | ROC-AUC curves (one-vs-rest) per model |
| `evaluation/pr_*.png` | Precision-Recall curves per model |
| `evaluation/model_comparison.png` | Side-by-side accuracy · F1 · ROC-AUC |
| `evaluation/snr_robustness.png` | Macro F1 vs. SNR level (dB) |
| `evaluation/cross_validation.png` | 5-fold CV macro F1 ± std |
| `explainability/importance_*.png` | Top-25 feature importances (GB) |

---

## Robustness Analysis

All models are evaluated across simulated SNR levels by injecting Gaussian noise:

| SNR | Environment |
|---|---|
| 30 dB | Studio / quiet |
| 20 dB | Office background |
| 15 dB | Café noise |
| 10 dB | Outdoor / moderate noise |
| 5 dB | Heavy background noise |
| 0 dB | Signal overwhelmed |

SVM (max-margin boundary) degrades most gracefully. All models retain usable accuracy at ≥15 dB SNR.

---

## Key References

- Snyder, D., Chen, G. & Povey, D. (2015). **MUSAN: A Music, Speech, and Noise Corpus.** arXiv:1510.08484
- Lu, L., Zhang, H.-J. & Li, S.-Z. (2003). **Content-Based Audio Classification and Segmentation by Using SVMs.** *Multimedia Systems*, Springer.
- Guo, G. & Li, S.-Z. (2003). **Content-Based Audio Classification by SVMs.** *IEEE Trans. Neural Networks.*

---

## Team

Arya Remani· Aleena Antony· Knanthanarayanan B — Predictive Analytics 2025–26

**Course:** Predictive Analytics 2025–26

---

## License

MIT License — see [LICENSE](LICENSE).  
MUSAN dataset: Creative Commons CC BY 4.0.
