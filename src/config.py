"""
config.py — Central Configuration
====================================
Single source of truth for all paths, hyperparameters, feature
definitions, and constants used across every module in AudioGuard.

Environment Variables:
  MUSAN_DATA_DIR : Path to the MUSAN dataset folder
                   Defaults to ./data if not set.

CLASSIFICATION MODE: MULTICLASS (3 audio types)
  speech · music · noise
"""

import os
from pathlib import Path

# ─────────────────────────────── Paths ───────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR   = Path(os.getenv("MUSAN_DATA_DIR", PROJECT_ROOT / "data"))
MODEL_DIR  = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

for _d in [MODEL_DIR, OUTPUT_DIR / "eda", OUTPUT_DIR / "evaluation",
           OUTPUT_DIR / "explainability"]:
    _d.mkdir(parents=True, exist_ok=True)

# ───────────────────── Model Artifact Paths ─────────────────────
SVM_MODEL_PATH     = MODEL_DIR / "svm_model.pkl"
KNN_MODEL_PATH     = MODEL_DIR / "knn_model.pkl"
GB_MODEL_PATH      = MODEL_DIR / "gradient_boosting_model.pkl"
SCALER_PATH        = MODEL_DIR / "scaler.pkl"
SELECTOR_PATH      = MODEL_DIR / "selector.pkl"
LABEL_ENC_PATH     = MODEL_DIR / "label_encoder.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
SEL_FEATURES_PATH  = MODEL_DIR / "selected_features.pkl"

# ───────────────────── Dataset Constants ─────────────────────
RANDOM_STATE   = 42
SAMPLE_RATE    = 22050
CLIP_DURATION  = 20.0          # seconds per clip
N_SAMPLES      = int(SAMPLE_RATE * CLIP_DURATION)

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# MUSAN: 3 audio classes
AUDIO_CLASSES        = ["music", "noise", "speech"]
AUDIO_CLASSES_SORTED = sorted(AUDIO_CLASSES)   # alphabetical → label index
N_CLASSES            = len(AUDIO_CLASSES)       # 3

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}

# ───────────────────── Feature Extraction ─────────────────────
N_MFCC       = 20
FRAME_LENGTH = 2048
HOP_LENGTH   = 512

# Number of features after MI selection
N_FEATURES_TO_SELECT = 80

# ALL 153 raw feature names — exact same order produced by extract_all_features()
# mfcc_i_{mean,var,skew}   (i = 1..20) → 60
# delta_mfcc_i_{mean,var}  (i = 1..20) → 40
# delta2_mfcc_i_{mean,var} (i = 1..20) → 40
# spectral group             → 8
# energy group               → 3
# zcr group                  → 2
#                              ═══ 153 total
ALL_FEATURE_NAMES: list = (
    [f"mfcc_{i}_{s}" for i in range(1, N_MFCC + 1) for s in ("mean", "var", "skew")]
    + [f"delta_mfcc_{i}_{s}" for i in range(1, N_MFCC + 1) for s in ("mean", "var")]
    + [f"delta2_mfcc_{i}_{s}" for i in range(1, N_MFCC + 1) for s in ("mean", "var")]
    + [
        "centroid_mean", "centroid_var",
        "bandwidth_mean", "bandwidth_var",
        "flux_mean",      "flux_var",
        "rolloff_mean",   "rolloff_var",
        "energy_mean",    "energy_var",    "energy_max",
        "zcr_mean",       "zcr_var",
    ]
)
assert len(ALL_FEATURE_NAMES) == 153, f"Expected 153, got {len(ALL_FEATURE_NAMES)}"

# ───────────────────── Model Hyperparameters ─────────────────────

SVM_PARAMS = {
    "C":            10.0,
    "kernel":       "rbf",
    "gamma":        "scale",
    "probability":  True,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "max_iter":     5000,
    "decision_function_shape": "ovr",
}

KNN_PARAMS = {
    "n_neighbors": 7,
    "metric":      "euclidean",
    "n_jobs":      -1,
}

GB_PARAMS = {
    "n_estimators":  200,
    "max_depth":     4,
    "learning_rate": 0.1,
    "subsample":     0.8,
    "random_state":  RANDOM_STATE,
}

# ───────────────────── Visualization ─────────────────────
COLORS = {
    "bg_dark":  "#0D1117",
    "bg_card":  "#161B22",
    "grid":     "#21262D",
    "border":   "#30363D",
    "text":     "#E6EDF3",
    "subtext":  "#8B949E",
    "speech":   "#4ECDC4",
    "music":    "#FF6B6B",
    "noise":    "#FFE66D",
    "primary":  "#58A6FF",
    "success":  "#3FB950",
    "warning":  "#FF7B72",
    "purple":   "#D2A8FF",
}

# class colour palette (indexed by sorted class order: music=0, noise=1, speech=2)
CLASS_COLORS = ["#FF6B6B", "#FFE66D", "#4ECDC4"]

FIGSIZE_LARGE  = (14, 6)
FIGSIZE_MEDIUM = (10, 5)
FIGSIZE_SMALL  = (7,  4)
DPI            = 130
