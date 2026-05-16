#!/usr/bin/env python3
"""
train.py — Main Training Entry Point
======================================
Run this script to train all models on the MUSAN audio dataset.

Usage:
    python train.py
    MUSAN_DATA_DIR=/path/to/musan python train.py

The script:
  1. Scans the MUSAN dataset directory  (music / speech / noise)
  2. Extracts MFCC + Delta + Delta² + Spectral + ZCR + Energy features
  3. Scales with StandardScaler, selects top-80 via Mutual Information
  4. Trains SVM (RBF), KNN (k=7), and Gradient Boosting
  5. Evaluates on the held-out test set
  6. Saves all artefacts to ./models/
  7. Saves evaluation plots to ./outputs/
"""

from __future__ import annotations

import sys
import time

import numpy as np

from src.config import (
    SCALER_PATH, SELECTOR_PATH, FEATURE_NAMES_PATH,
    SVM_MODEL_PATH, KNN_MODEL_PATH, GB_MODEL_PATH,
)
from src.data_loader import load_dataset, split_dataset
from src.training import (
    train_all_models,
    extract_features_batch,
    apply_preprocessing,
)
from src.evaluation import evaluate_all_models


def main():
    t_start = time.time()
    print("\n" + "═" * 60)
    print("  AudioGuard — Training Pipeline")
    print("  MUSAN: Music · Speech · Noise")
    print("═" * 60)

    # ── 1. Load & split ───────────────────────────────────
    try:
        df = load_dataset()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        print(
            "\nTo fix:\n"
            "  Make sure you have your Kaggle API credentials set up:\n"
            "    export KAGGLE_USERNAME=your_username\n"
            "    export KAGGLE_KEY=your_api_key\n"
            "  Or place kaggle.json at ~/.kaggle/kaggle.json\n"
            "  Then re-run: python train.py\n"
        )
        sys.exit(1)

    train_df, val_df, test_df = split_dataset(df)

    # ── 2. Train (includes feature extraction + saving) ───
    artefacts = train_all_models(train_df, val_df)

    # ── 3. Evaluate on test set ───────────────────────────
    print("\n" + "═" * 60)
    print("  TEST SET EVALUATION")
    print("═" * 60)

    scaler   = artefacts["scaler"]
    selector = artefacts["selector"]
    selected_feat_names = artefacts["selected_feature_names"]

    print(f"\n[Test] Extracting features for {len(test_df)} clips …")
    X_test_raw = extract_features_batch(test_df["filepath"].tolist())
    y_test     = test_df["label"].to_numpy()
    X_test     = apply_preprocessing(X_test_raw, scaler, selector)

    metrics_df = evaluate_all_models(
        models_dict         = artefacts,
        X_test              = X_test,
        y_test              = y_test,
        selected_feature_names = selected_feat_names,
        run_snr             = True,
    )

    # ── 4. Done ───────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n[Done] Total time: {elapsed / 60:.1f} min")
    print("  Models saved to: ./models/")
    print("  Plots  saved to: ./outputs/")
    print("\n  Launch the app:\n    streamlit run app.py\n")

    return metrics_df


if __name__ == "__main__":
    main()
