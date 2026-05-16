#!/usr/bin/env python3
"""
AudioGuard — Audio Signal Classification App
=============================================
Classifies uploaded audio into one of 3 classes:
  speech · music · noise

Run:  streamlit run app.py
"""

from __future__ import annotations

import io
import os
import tempfile
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import streamlit as st

try:
    import librosa
    import librosa.display
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "librosa"])
    import librosa
    import librosa.display

from src.config import (
    SVM_MODEL_PATH, KNN_MODEL_PATH, GB_MODEL_PATH,
    SCALER_PATH, SELECTOR_PATH, LABEL_ENC_PATH,
    FEATURE_NAMES_PATH, SEL_FEATURES_PATH,
    COLORS, CLASS_COLORS, AUDIO_CLASSES_SORTED, N_CLASSES,
    SAMPLE_RATE, CLIP_DURATION, FRAME_LENGTH, HOP_LENGTH,
    ALL_FEATURE_NAMES,
)
from src.feature_extraction import (
    load_and_preprocess,
    extract_features_from_array,
)
from src.data_loader import CLASS_TO_IDX, IDX_TO_CLASS

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AudioClassifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0D1117;
    color: #E6EDF3;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
.metric-card {
    border: 1px solid rgba(255,255,255,.1);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    background: linear-gradient(180deg, rgba(22,27,34,.95), rgba(13,17,23,.98));
    box-shadow: 0 8px 18px rgba(0,0,0,.25);
    text-align: center;
}
.cls-badge {
    border-radius: 8px;
    padding: 5px 16px;
    font-weight: bold;
    font-size: 1.05rem;
    font-family: 'DM Mono', monospace;
    display: inline-block;
}
.small-note { color: #8B949E; font-size: .88rem; }
.wordmark {
    font-family: 'DM Mono', monospace;
    font-size: .70rem;
    letter-spacing: .3em;
    color: #444;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

CLASS_BADGE = {
    "speech": ("#4ECDC4", "#0d2e2c"),
    "music":  ("#FF6B6B", "#2e0f0f"),
    "noise":  ("#FFE66D", "#2e2908"),
}

MODEL_KEYS   = {"svm": "SVM", "knn": "KNN", "gb": "Gradient Boosting"}
MODEL_LABELS = {"SVM": "svm", "KNN": "knn", "Gradient Boosting": "gb"}

# ── Artefact loading ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts() -> Dict:
    required = [
        (SVM_MODEL_PATH,     "svm_model.pkl"),
        (KNN_MODEL_PATH,     "knn_model.pkl"),
        (GB_MODEL_PATH,      "gradient_boosting_model.pkl"),
        (SCALER_PATH,        "scaler.pkl"),
        (SELECTOR_PATH,      "selector.pkl"),
        (LABEL_ENC_PATH,     "label_encoder.pkl"),
        (SEL_FEATURES_PATH,  "selected_features.pkl"),
    ]
    missing = [name for path, name in required if not Path(path).exists()]
    if missing:
        return {"error": f"Missing model files: {', '.join(missing)}. "
                         "Run `python train.py` first."}
    return dict(
        svm      = joblib.load(SVM_MODEL_PATH),
        knn      = joblib.load(KNN_MODEL_PATH),
        gb       = joblib.load(GB_MODEL_PATH),
        scaler   = joblib.load(SCALER_PATH),
        selector = joblib.load(SELECTOR_PATH),
        encoder  = joblib.load(LABEL_ENC_PATH),
        sel_feat = joblib.load(SEL_FEATURES_PATH),
    )


# ── Inference ────────────────────────────────────────────────────────────────
def _preprocess(filepath: str, scaler, selector) -> np.ndarray:
    """Load → extract 153 raw features → scale → select → (1, 80) array."""
    y      = load_and_preprocess(filepath)
    raw    = extract_features_from_array(y)
    scaled = scaler.transform(raw.reshape(1, -1))
    return selector.transform(scaled)


def predict_single(filepath: str, artifacts: Dict,
                   selected_models: List[str]) -> pd.DataFrame:
    X    = _preprocess(filepath, artifacts["scaler"], artifacts["selector"])
    rows = []
    for key in selected_models:
        model = artifacts[key]
        probs = model.predict_proba(X)[0]
        idx   = int(np.argmax(probs))
        label = IDX_TO_CLASS[idx]
        rows.append({
            "Model":            MODEL_KEYS[key],
            "Predicted Class":  label,
            "Confidence":       float(probs[idx]),
            **{cls: float(probs[i]) for i, cls in enumerate(AUDIO_CLASSES_SORTED)},
        })
    return pd.DataFrame(rows)


# ── Visualisation helpers ─────────────────────────────────────────────────────
def _fig_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_waveform(y: np.ndarray, sr: int) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 2))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")
    t = np.linspace(0, len(y) / sr, len(y))
    ax.plot(t, y, color="#2A2A3A", linewidth=0.5, alpha=0.7)
    ax.fill_between(t, y, color="#3A3A55", alpha=0.3)
    ax.set_xlabel("Time (s)", fontsize=8, color="#8B949E")
    ax.tick_params(colors="#444", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#21262D")
    fig.tight_layout(pad=0.4)
    return _fig_bytes(fig)


def plot_mel_spectrogram(y: np.ndarray, sr: int) -> bytes:
    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel",
                                    ax=ax, cmap="magma", hop_length=HOP_LENGTH)
    ax.set_xlabel("Time (s)", fontsize=8, color="#8B949E")
    ax.set_ylabel("Hz",        fontsize=8, color="#8B949E")
    ax.tick_params(colors="#444", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#21262D")
    plt.colorbar(img, ax=ax, format="%+2.0f dB").ax.tick_params(labelsize=7, colors="#444")
    fig.tight_layout(pad=0.4)
    return _fig_bytes(fig)


def plot_mfcc_heatmap(y: np.ndarray, sr: int) -> bytes:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20,
                                  n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    img = librosa.display.specshow(mfcc, x_axis="time", ax=ax,
                                    cmap="coolwarm", hop_length=HOP_LENGTH, sr=sr)
    ax.set_ylabel("MFCC coeff", fontsize=8, color="#8B949E")
    ax.set_xlabel("Time (s)",   fontsize=8, color="#8B949E")
    ax.tick_params(colors="#444", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#21262D")
    plt.colorbar(img, ax=ax).ax.tick_params(labelsize=7, colors="#444")
    fig.tight_layout(pad=0.4)
    return _fig_bytes(fig)


def plot_confidence_bars(row: pd.Series) -> bytes:
    classes = AUDIO_CLASSES_SORTED
    probs   = [float(row[c]) for c in classes]
    colors  = CLASS_COLORS[:len(classes)]

    fig, ax = plt.subplots(figsize=(6, 2.5))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")

    bars = ax.barh(classes, probs, color=colors, alpha=0.85,
                   edgecolor="#30363D", linewidth=1)
    for bar, v in zip(bars, probs):
        ax.text(min(v + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
                f"{v:.1%}", va="center", fontsize=10,
                color="#E6EDF3", fontweight="bold")
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Probability", color="#8B949E", fontsize=9)
    ax.tick_params(colors="#8B949E", labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#21262D")
    ax.xaxis.grid(True, color="#21262D", linewidth=0.5)
    fig.tight_layout(pad=0.5)
    return _fig_bytes(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="wordmark">Audio · ML · Signal</div>',
                unsafe_allow_html=True)
    st.markdown("## 🎵 AudioClassifier")
    st.markdown("**Audio Signal Classifier**  \nSpeech · Music · Noise")
    st.divider()

    st.markdown("### Models")
    use_svm = st.checkbox("SVM (RBF)",          value=True)
    use_knn = st.checkbox("KNN (k=7)",           value=True)
    use_gb  = st.checkbox("Gradient Boosting ⭐", value=True)

    selected_keys = []
    if use_svm: selected_keys.append("svm")
    if use_knn: selected_keys.append("knn")
    if use_gb:  selected_keys.append("gb")

    st.divider()
    st.markdown("### Dataset")
    st.markdown(
        "**[MUSAN](https://www.kaggle.com/datasets/dogrose/musan-dataset)**  \n"
        "Music · Speech · Noise  \n"
        "~109 h · CC BY 4.0"
    )
    st.divider()
    st.markdown("### Features")
    st.markdown(
        "153-dim vector:  \n"
        "· MFCC ×20 (mean, var, skew)  \n"
        "· Δ-MFCC & Δ²-MFCC  \n"
        "· Spectral Centroid/BW/Flux  \n"
        "· Short-Time Energy · ZCR  \n"
        "→ Top-80 via Mutual Info"
    )
    st.divider()
    st.caption("Run `python train.py` to (re)train.")

# ══════════════════════════════════════════════════════════════════════════════
# Main area
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="wordmark">Speech · Music · Noise — Classical ML</div>',
            unsafe_allow_html=True)
st.markdown("# 🎵 AudioClassifier")
st.markdown(
    "Upload any audio file to classify it as **speech**, **music**, or "
    "**environmental noise** using MFCC features + classical ML — no deep learning."
)

# ── Load models ───────────────────────────────────────────────────────────────
artifacts = load_artifacts()

if "error" in artifacts:
    st.error(artifacts["error"])
    st.info(
        "**To train the models:**\n"
        "```bash\n"
        "# 1. Download MUSAN\n"
        "kaggle datasets download -d dogrose/musan-dataset -p data --unzip\n\n"
        "# 2. Train\n"
        "python train.py\n\n"
        "# 3. Relaunch app\n"
        "streamlit run app.py\n"
        "```"
    )
    st.stop()

# ── Upload ────────────────────────────────────────────────────────────────────
st.divider()
col_up, col_info = st.columns([2, 1])
with col_up:
    uploaded = st.file_uploader(
        "Drop a WAV, MP3, FLAC, or OGG file",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        label_visibility="visible",
    )
with col_info:
    st.markdown("**Supported formats**")
    st.markdown("WAV · MP3 · FLAC · OGG · M4A")
    st.markdown("*First 20 seconds are used.*")

if uploaded is None:
    st.markdown(
        '<p class="small-note">👆 Upload an audio file to get started.</p>',
        unsafe_allow_html=True,
    )
    st.stop()

if not selected_keys:
    st.warning("Select at least one model in the sidebar.")
    st.stop()

# ── Save upload to temp file ──────────────────────────────────────────────────
suffix  = Path(uploaded.name).suffix
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

# ── Audio playback ────────────────────────────────────────────────────────────
st.divider()
st.subheader("▶ Playback")
uploaded.seek(0)
st.audio(uploaded)

# ── Run inference ─────────────────────────────────────────────────────────────
with st.spinner("Extracting features & classifying …"):
    t0    = time.time()
    try:
        results_df = predict_single(tmp_path, artifacts, selected_keys)
        y_audio    = load_and_preprocess(tmp_path)
    except Exception as exc:
        os.unlink(tmp_path)
        st.error(f"Classification failed: {exc}")
        st.stop()
    elapsed = time.time() - t0

os.unlink(tmp_path)

# ── Signal Visualisations ─────────────────────────────────────────────────────
st.divider()
tab_wave, tab_mel, tab_mfcc = st.tabs(["Waveform", "Mel Spectrogram", "MFCC Heatmap"])

with tab_wave:
    st.image(plot_waveform(y_audio, SAMPLE_RATE), use_container_width=True)
with tab_mel:
    st.image(plot_mel_spectrogram(y_audio, SAMPLE_RATE), use_container_width=True)
with tab_mfcc:
    st.image(plot_mfcc_heatmap(y_audio, SAMPLE_RATE), use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────────────
st.divider()
st.subheader("🔖 Classification Results")

for _, row in results_df.iterrows():
    label  = row["Predicted Class"]
    conf   = row["Confidence"]
    model  = row["Model"]
    color, bg = CLASS_BADGE.get(label, ("#58A6FF", "#0d1f33"))

    with st.container():
        left, right = st.columns([1.4, 1])
        with left:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="small-note" style="margin-bottom:.3rem">{model}</div>'
                f'<div class="cls-badge" style="background:{bg};color:{color};">'
                f'{label.upper()}</div>'
                f'<div style="margin-top:.6rem;font-size:1.4rem;font-weight:500;">'
                f'{conf:.1%}</div>'
                f'<div class="small-note">confidence</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with right:
            st.image(plot_confidence_bars(row), use_container_width=True)

    st.markdown("")

# ── Per-class probability table ───────────────────────────────────────────────
with st.expander("📊 Full probability table"):
    display_df = results_df[["Model", "Predicted Class", "Confidence"]
                             + AUDIO_CLASSES_SORTED].copy()
    for cls in AUDIO_CLASSES_SORTED:
        display_df[cls] = display_df[cls].map(lambda v: f"{v:.4f}")
    display_df["Confidence"] = display_df["Confidence"].map(lambda v: f"{v:.4f}")
    st.dataframe(display_df, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
cols = st.columns(3)
cols[0].metric("Inference time", f"{elapsed:.2f}s")
cols[1].metric("Feature dims",   "153 → 80")
cols[2].metric("Models active",  str(len(selected_keys)))

st.markdown(
    '<p class="small-note" style="text-align:center;margin-top:1rem;">'
    "AudioGuard · MFCC + SVM / KNN / Gradient Boosting · MUSAN Dataset · CC BY 4.0"
    "</p>",
    unsafe_allow_html=True,
)
