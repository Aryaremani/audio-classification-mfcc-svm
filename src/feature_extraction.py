"""
feature_extraction.py — Stage 2: Audio Feature Extraction
===========================================================
Extracts three families of acoustic features from mono audio clips:

1. MFCC         — 20 coefficients × {mean, var, skew}         = 60 dims
2. Delta-MFCC   — 20 × {mean, var}                            = 40 dims
3. Delta²-MFCC  — 20 × {mean, var}                            = 40 dims
4. Spectral     — centroid, bandwidth, flux, rolloff × {mean,var} = 8 dims
5. Energy (STE) — RMS mean, var, max                           =  3 dims
6. ZCR          — zero-crossing rate mean, var                 =  2 dims
                                                          ─────────────
                                                    Total : 153 features

Each audio file yields a single 1-D float32 feature vector.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List

import numpy as np
from scipy.stats import skew

try:
    import librosa
    import librosa.effects
    import librosa.feature
except ImportError:  # pragma: no cover
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "librosa"])
    import librosa
    import librosa.effects
    import librosa.feature

from src.config import (
    SAMPLE_RATE, CLIP_DURATION, N_SAMPLES,
    N_MFCC, FRAME_LENGTH, HOP_LENGTH,
    ALL_FEATURE_NAMES,
)

# ══════════════════════════════════════════════════════════
# Audio I/O & Preprocessing
# ══════════════════════════════════════════════════════════

def load_and_preprocess(filepath: str,
                        sr: int = SAMPLE_RATE,
                        duration: float = CLIP_DURATION) -> np.ndarray:
    """
    Load a .wav / .mp3 / .flac / .ogg file, resample to `sr`, pad/trim
    to `duration` seconds, apply pre-emphasis, and peak-normalise.

    Returns
    -------
    y : np.ndarray, shape (N_SAMPLES,), float32
    """
    n = int(sr * duration)
    y, _ = librosa.load(filepath, sr=sr, mono=True, duration=duration)
    y    = librosa.effects.preemphasis(y, coef=0.97)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    else:
        y = y[:n]
    peak = np.max(np.abs(y))
    if peak > 1e-8:
        y = y / peak
    return y.astype(np.float32)


def preprocess_array(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Preprocess an already-loaded numpy array (used in the Streamlit app).
    Applies pre-emphasis and peak-normalises.
    """
    y = librosa.effects.preemphasis(y.astype(np.float32), coef=0.97)
    n = int(sr * CLIP_DURATION)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    else:
        y = y[:n]
    peak = np.max(np.abs(y))
    if peak > 1e-8:
        y = y / peak
    return y


# ══════════════════════════════════════════════════════════
# Feature Families
# ══════════════════════════════════════════════════════════

def _stat(arr: np.ndarray):
    """Return (mean, variance, skewness) of a 1-D array."""
    return float(np.mean(arr)), float(np.var(arr)), float(skew(arr))


def extract_mfcc_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    MFCC + Delta + Delta² features.
    Returns shape (60 + 40 + 40,) = (140,) float32.
    """
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                   n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    parts: List[float] = []
    for i in range(N_MFCC):
        parts.extend(_stat(mfcc[i]))              # mean, var, skew
    for i in range(N_MFCC):
        m, v, _ = _stat(delta[i])
        parts.extend([m, v])                      # mean, var only
    for i in range(N_MFCC):
        m, v, _ = _stat(delta2[i])
        parts.extend([m, v])

    return np.array(parts, dtype=np.float32)


def extract_spectral_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Spectral centroid, bandwidth, flux, rolloff — mean & var each.
    Returns shape (8,) float32.
    """
    cent = librosa.feature.spectral_centroid(y=y, sr=sr,
                                              n_fft=FRAME_LENGTH,
                                              hop_length=HOP_LENGTH)[0]
    bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr,
                                               n_fft=FRAME_LENGTH,
                                               hop_length=HOP_LENGTH)[0]
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr,
                                             n_fft=FRAME_LENGTH,
                                             hop_length=HOP_LENGTH)[0]
    stft  = np.abs(librosa.stft(y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH))
    flux  = np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))

    feats: List[float] = []
    for arr in [cent, bw, flux, roll]:
        m, v, _ = _stat(arr)
        feats.extend([m, v])
    return np.array(feats, dtype=np.float32)


def extract_energy_zcr_features(y: np.ndarray) -> np.ndarray:
    """
    Short-Time Energy (RMS): mean, var, max.
    Zero Crossing Rate: mean, var.
    Returns shape (5,) float32.
    """
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH,
                               hop_length=HOP_LENGTH)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LENGTH,
                                              hop_length=HOP_LENGTH)[0]
    energy_m, energy_v, _ = _stat(rms)
    zcr_m, zcr_v, _       = _stat(zcr)
    return np.array([
        energy_m, energy_v, float(np.max(rms)),
        zcr_m, zcr_v,
    ], dtype=np.float32)


# ══════════════════════════════════════════════════════════
# Full Feature Vector (public API)
# ══════════════════════════════════════════════════════════

def extract_all_features(filepath: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load *filepath*, run preprocessing, and return the full 153-dim
    feature vector in the canonical order defined by ALL_FEATURE_NAMES.

    Parameters
    ----------
    filepath : str   Path to any .wav / .mp3 / .flac / .ogg file.

    Returns
    -------
    np.ndarray, shape (153,), float32
    """
    y = load_and_preprocess(filepath, sr=sr)
    return _extract_from_array(y, sr)


def extract_features_from_array(y: np.ndarray,
                                 sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Same as extract_all_features() but accepts a pre-loaded numpy array
    (after load_and_preprocess / preprocess_array has been called).

    Returns
    -------
    np.ndarray, shape (153,), float32
    """
    return _extract_from_array(y, sr)


def _extract_from_array(y: np.ndarray, sr: int) -> np.ndarray:
    """Internal: extract 153 features from a pre-processed float32 array."""
    mfcc_f     = extract_mfcc_features(y, sr)      # 140
    spectral_f = extract_spectral_features(y, sr)   #   8
    energy_f   = extract_energy_zcr_features(y)     #   5
    vec = np.concatenate([mfcc_f, spectral_f, energy_f]).astype(np.float32)
    assert len(vec) == len(ALL_FEATURE_NAMES), \
        f"Feature count mismatch: {len(vec)} != {len(ALL_FEATURE_NAMES)}"
    return vec


def get_feature_names() -> List[str]:
    """Return the ordered list of all 153 feature names."""
    return list(ALL_FEATURE_NAMES)


def add_awgn(y: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add Additive White Gaussian Noise to *y* at the given SNR (dB).
    Used for robustness evaluation (Stage 8).
    """
    signal_power = float(np.mean(y ** 2)) + 1e-12
    noise_power  = signal_power / (10 ** (snr_db / 10.0))
    noise        = np.random.randn(len(y)) * np.sqrt(noise_power)
    return (y + noise).astype(np.float32)
