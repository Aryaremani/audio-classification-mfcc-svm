"""
training.py — Stage 4: Feature Pipeline & Model Training
=========================================================
MULTICLASS: y = class_idx (0-2). SVM, KNN, GB all support multiclass natively.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.config import (
    SVM_MODEL_PATH, KNN_MODEL_PATH, GB_MODEL_PATH,
    SCALER_PATH, SELECTOR_PATH, LABEL_ENC_PATH,
    FEATURE_NAMES_PATH, SEL_FEATURES_PATH,
    N_FEATURES_TO_SELECT, RANDOM_STATE, AUDIO_CLASSES_SORTED,
)
from src.feature_extraction import extract_all_features, get_feature_names
from src.models import build_svm, build_knn, build_gradient_boosting


def mutual_info_score_func(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return mutual_info_classif(X, y, random_state=RANDOM_STATE)


# ══════════════════════════════════════════════════════════
# Batch Feature Extraction
# ══════════════════════════════════════════════════════════

def extract_features_batch(filepaths: List[str], verbose: bool = True) -> np.ndarray:
    """Extract features for every file in *filepaths*, returning shape (N, 153)."""
    features = []
    n   = len(filepaths)
    t0  = time.time()
    n_feat = len(get_feature_names())   # 153

    for i, fp in enumerate(filepaths):
        try:
            vec = extract_all_features(fp)
        except Exception as exc:
            print(f"  [WARN] Skipping {fp}: {exc}")
            vec = np.zeros(n_feat, dtype=np.float32)
        features.append(vec)
        if verbose and (i + 1) % 100 == 0:
            elapsed   = time.time() - t0
            rate      = (i + 1) / elapsed
            remaining = (n - i - 1) / max(rate, 1e-6)
            print(f"  [{i+1:4d}/{n}] {rate:.1f} file/s  ~{remaining:.0f}s remaining")

    arr = np.vstack(features).astype(np.float32)
    print(f"  Feature matrix: {arr.shape}  ({time.time()-t0:.1f}s total)")
    return arr


# ══════════════════════════════════════════════════════════
# Preprocessing Pipeline
# ══════════════════════════════════════════════════════════

def build_preprocessing_pipeline(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[StandardScaler, SelectKBest]:
    """Fit StandardScaler + Mutual-Information SelectKBest on training data."""
    print("\n[Pipeline] Fitting StandardScaler …")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    k = min(N_FEATURES_TO_SELECT, X_scaled.shape[1])
    print(f"[Pipeline] SelectKBest top-{k} (mutual_info_classif) …")
    selector = SelectKBest(
        score_func=mutual_info_score_func,
        k=k,
    )
    selector.fit(X_scaled, y_train)
    print(f"[Pipeline] Selected {selector.get_support().sum()} features "
          f"from {X_scaled.shape[1]}.")
    return scaler, selector


def apply_preprocessing(X: np.ndarray, scaler, selector) -> np.ndarray:
    """Scale then select features on unseen data."""
    return selector.transform(scaler.transform(X))


# ══════════════════════════════════════════════════════════
# Main Training Entry
# ══════════════════════════════════════════════════════════

def train_all_models(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
    """
    Full training pipeline:
      1. Extract features for train + val sets
      2. Fit scaler + MI feature selector
      3. Train SVM, KNN, Gradient Boosting
      4. Save all artefacts to models/

    Returns a dict of artefacts (models + preprocessors + metadata).
    """
    print("\n" + "=" * 60)
    print("  PHASE 1: FEATURE EXTRACTION")
    print("=" * 60)

    print(f"\n[Train] {len(train_df)} audio clips …")
    X_train_raw = extract_features_batch(train_df["filepath"].tolist())
    y_train     = train_df["label"].to_numpy()

    print(f"\n[Val]   {len(val_df)} audio clips …")
    X_val_raw = extract_features_batch(val_df["filepath"].tolist())
    y_val     = val_df["label"].to_numpy()

    all_feature_names = get_feature_names()

    print("\n" + "=" * 60)
    print("  PHASE 2: PREPROCESSING")
    print("=" * 60)

    scaler, selector = build_preprocessing_pipeline(X_train_raw, y_train)
    X_train = apply_preprocessing(X_train_raw, scaler, selector)
    X_val   = apply_preprocessing(X_val_raw,   scaler, selector)

    support               = selector.get_support()
    selected_feature_names = [n for n, s in zip(all_feature_names, support) if s]

    # Label encoder (class_idx ↔ class_name)
    le = LabelEncoder()
    le.fit(AUDIO_CLASSES_SORTED)

    print("\n" + "=" * 60)
    print("  PHASE 3: MODEL TRAINING  (MULTICLASS — 3 audio types)")
    print("=" * 60)

    # ── SVM ──────────────────────────────────────────────────────
    print("\n[SVM] Training …")
    svm = build_svm()
    t0  = time.time()
    svm.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    val_acc = (svm.predict(X_val) == y_val).mean()
    print(f"  Val accuracy: {val_acc:.4f}")

    # ── KNN ──────────────────────────────────────────────────────
    print("\n[KNN] Training …")
    knn = build_knn()
    t0  = time.time()
    knn.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    val_acc = (knn.predict(X_val) == y_val).mean()
    print(f"  Val accuracy: {val_acc:.4f}")

    # ── Gradient Boosting ─────────────────────────────────────────
    print("\n[GB] Training …")
    gb  = build_gradient_boosting()
    t0  = time.time()
    gb.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    val_acc = (gb.predict(X_val) == y_val).mean()
    print(f"  Val accuracy: {val_acc:.4f}")

    # ── Save artefacts ─────────────────────────────────────────────
    print("\n[Save] Persisting artefacts to models/ …")
    joblib.dump(svm,      SVM_MODEL_PATH,     compress=3)
    joblib.dump(knn,      KNN_MODEL_PATH,     compress=3)
    joblib.dump(gb,       GB_MODEL_PATH,      compress=3)
    joblib.dump(scaler,   SCALER_PATH,        compress=3)
    joblib.dump(selector, SELECTOR_PATH,      compress=3)
    joblib.dump(le,       LABEL_ENC_PATH,     compress=3)
    joblib.dump(all_feature_names,      FEATURE_NAMES_PATH)
    joblib.dump(selected_feature_names, SEL_FEATURES_PATH)

    print(f"  ✅ svm_model.pkl")
    print(f"  ✅ knn_model.pkl")
    print(f"  ✅ gradient_boosting_model.pkl")
    print(f"  ✅ scaler.pkl  selector.pkl  label_encoder.pkl  feature_names.pkl")

    return {
        "svm":      svm,
        "knn":      knn,
        "gb":       gb,
        "scaler":   scaler,
        "selector": selector,
        "encoder":  le,
        "all_feature_names":      all_feature_names,
        "selected_feature_names": selected_feature_names,
    }
