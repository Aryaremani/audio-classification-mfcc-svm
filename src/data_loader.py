"""
data_loader.py — Stage 1: Dataset Loading & Scanning
======================================================
MULTICLASS MODE: label = class_idx (0-2), one per audio type.

Handles the MUSAN folder layout:
  data/musan/music/<subdir>/<file>.wav
  data/musan/speech/<subdir>/<file>.wav
  data/musan/noise/<subdir>/<file>.wav

Class is determined by the top-level subfolder name inside musan/.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_DIR, RANDOM_STATE,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    AUDIO_CLASSES, AUDIO_CLASSES_SORTED,
    AUDIO_EXTS,
)

# class_name → integer label (0..2, alphabetical)
CLASS_TO_IDX = {cls: i for i, cls in enumerate(AUDIO_CLASSES_SORTED)}
IDX_TO_CLASS = {i: cls for cls, i in CLASS_TO_IDX.items()}


def _collect_musan(musan_root: Path) -> pd.DataFrame:
    """
    Walk the MUSAN directory tree.
    Expects musan_root/<class_name>/... structure.
    """
    rows: List[Dict] = []
    for cls in AUDIO_CLASSES:
        class_dir = musan_root / cls
        if not class_dir.exists():
            continue
        for root_s, _, files in os.walk(class_dir):
            for fname in sorted(files):
                if Path(fname).suffix.lower() in AUDIO_EXTS:
                    rows.append({
                        "filepath":   str(Path(root_s) / fname),
                        "class_name": cls,
                        "label":      CLASS_TO_IDX[cls],
                    })
    return pd.DataFrame(rows)


def _scan_root(root: Path) -> pd.DataFrame:
    """Try multiple MUSAN layout variants."""
    # Layout A — data/musan/
    for candidate in [root / "musan", root / "MUSAN", root]:
        df = _collect_musan(candidate)
        if len(df) > 0:
            print(f"[DataLoader] Found MUSAN at: {candidate}")
            return df
    return pd.DataFrame()


def load_dataset(data_dir=None) -> pd.DataFrame:
    """
    Downloads the MUSAN dataset via kagglehub (if not cached),
    then scans and returns a DataFrame with columns:
      filepath, class_name, label
    """
    import kagglehub
    print("[DataLoader] Downloading/verifying MUSAN dataset via kagglehub...")
    path = kagglehub.dataset_download("dogrose/musan-dataset")
    print(f"[DataLoader] Dataset path: {path}")

    root = Path(path)
    df = _scan_root(root)

    if df.empty:
        raise FileNotFoundError(
            f"No audio files found under {root}.\n"
            "Expected: <path>/musan/<class>/<files>.wav\n"
            "Classes: music, speech, noise"
        )

    print(f"[DataLoader] Found {len(df)} audio files.")
    for cls in AUDIO_CLASSES_SORTED:
        n = (df["class_name"] == cls).sum()
        print(f"  {cls:10s}: {n:5d} files")

    return df.reset_index(drop=True)


def split_dataset(df: pd.DataFrame):
    """
    Stratified split → (train_df, val_df, test_df).
    Applies a per-class cap of 500 clips to keep training time manageable.
    """
    MAX_PER_CLASS = 500

    # Cap per class
    capped = (
        df.groupby("class_name", group_keys=False)
          .apply(lambda g: g.sample(min(len(g), MAX_PER_CLASS), random_state=RANDOM_STATE))
          .reset_index(drop=True)
    )

    # Train / temp split
    train_df, temp_df = train_test_split(
        capped,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=capped["label"],
        random_state=RANDOM_STATE,
    )

    # Val / test split
    relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df["label"],
        random_state=RANDOM_STATE,
    )

    print(f"\n[Split] Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
