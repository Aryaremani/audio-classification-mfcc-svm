#!/usr/bin/env python3
"""
AudioGuard — Gradio version for Hugging Face Spaces
Classifies audio into: speech · music · noise
"""

from __future__ import annotations

import io
import os
import tempfile
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import gradio as gr
import librosa
import librosa.display

from src.config import (
    SVM_MODEL_PATH, KNN_MODEL_PATH, GB_MODEL_PATH,
    SCALER_PATH, SELECTOR_PATH, LABEL_ENC_PATH,
    FEATURE_NAMES_PATH, SEL_FEATURES_PATH,
    COLORS, CLASS_COLORS, AUDIO_CLASSES_SORTED,
    SAMPLE_RATE, FRAME_LENGTH, HOP_LENGTH,
)
from src.feature_extraction import load_and_preprocess, extract_features_from_array
from src.data_loader import IDX_TO_CLASS

# ── Load artifacts once at startup ────────────────────────────────────────────
def load_artifacts():
    required = [
        (SVM_MODEL_PATH,    "svm_model.pkl"),
        (KNN_MODEL_PATH,    "knn_model.pkl"),
        (GB_MODEL_PATH,     "gradient_boosting_model.pkl"),
        (SCALER_PATH,       "scaler.pkl"),
        (SELECTOR_PATH,     "selector.pkl"),
        (LABEL_ENC_PATH,    "label_encoder.pkl"),
        (SEL_FEATURES_PATH, "selected_features.pkl"),
    ]
    missing = [name for path, name in required if not Path(path).exists()]
    if missing:
        raise RuntimeError(f"Missing model files: {', '.join(missing)}. Run train.py first.")
    return dict(
        svm      = joblib.load(SVM_MODEL_PATH),
        knn      = joblib.load(KNN_MODEL_PATH),
        gb       = joblib.load(GB_MODEL_PATH),
        scaler   = joblib.load(SCALER_PATH),
        selector = joblib.load(SELECTOR_PATH),
        encoder  = joblib.load(LABEL_ENC_PATH),
    )

ARTIFACTS = load_artifacts()

MODEL_MAP = {"SVM (RBF)": "svm", "KNN (k=7)": "knn", "Gradient Boosting ⭐": "gb"}
CLASS_BADGE_COLOR = {"speech": "#4ECDC4", "music": "#FF6B6B", "noise": "#FFE66D"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    return Image.open(buf)

def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 2))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")
    t = np.linspace(0, len(y) / sr, len(y))
    ax.plot(t, y, color="#4ECDC4", linewidth=0.6, alpha=0.8)
    ax.fill_between(t, y, color="#4ECDC4", alpha=0.15)
    ax.set_xlabel("Time (s)", fontsize=8, color="#8B949E")
    ax.tick_params(colors="#444", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#21262D")
    fig.tight_layout(pad=0.4)
    return _fig_to_pil(fig)

def plot_mel(y, sr):
    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel",
                                   ax=ax, cmap="magma", hop_length=HOP_LENGTH)
    ax.set_xlabel("Time (s)", fontsize=8, color="#8B949E")
    ax.set_ylabel("Hz", fontsize=8, color="#8B949E")
    ax.tick_params(colors="#444", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#21262D")
    plt.colorbar(img, ax=ax, format="%+2.0f dB").ax.tick_params(labelsize=7, colors="#444")
    fig.tight_layout(pad=0.4)
    return _fig_to_pil(fig)

def plot_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20,
                                  n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    img = librosa.display.specshow(mfcc, x_axis="time", ax=ax,
                                   cmap="coolwarm", hop_length=HOP_LENGTH, sr=sr)
    ax.set_ylabel("MFCC coeff", fontsize=8, color="#8B949E")
    ax.set_xlabel("Time (s)", fontsize=8, color="#8B949E")
    ax.tick_params(colors="#444", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#21262D")
    plt.colorbar(img, ax=ax).ax.tick_params(labelsize=7, colors="#444")
    fig.tight_layout(pad=0.4)
    return _fig_to_pil(fig)

def plot_confidence(classes, probs):
    colors = [CLASS_BADGE_COLOR.get(c, "#888") for c in classes]
    fig, ax = plt.subplots(figsize=(6, 2.5))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")
    bars = ax.barh(classes, probs, color=colors, alpha=0.85,
                   edgecolor="#30363D", linewidth=1)
    for bar, v in zip(bars, probs):
        ax.text(min(v + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
                f"{v:.1%}", va="center", fontsize=10,
                color="#E6EDF3", fontweight="bold")
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Probability", color="#8B949E", fontsize=9)
    ax.tick_params(colors="#8B949E", labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#21262D")
    ax.xaxis.grid(True, color="#21262D", linewidth=0.5)
    fig.tight_layout(pad=0.5)
    return _fig_to_pil(fig)

# ── Main prediction function ──────────────────────────────────────────────────
def classify(audio_path, use_svm, use_knn, use_gb):
    if audio_path is None:
        return (
            "⚠️ Please upload an audio file.",
            None, None, None,
            None, None, None,
            None,
        )

    selected = []
    if use_svm: selected.append("svm")
    if use_knn: selected.append("knn")
    if use_gb:  selected.append("gb")

    if not selected:
        return (
            "⚠️ Select at least one model.",
            None, None, None,
            None, None, None,
            None,
        )

    t0 = time.time()
    y = load_and_preprocess(audio_path)
    raw    = extract_features_from_array(y)
    scaled = ARTIFACTS["scaler"].transform(raw.reshape(1, -1))
    X      = ARTIFACTS["selector"].transform(scaled)

    label_map = {"SVM (RBF)": "svm", "KNN (k=7)": "knn", "Gradient Boosting ⭐": "gb"}
    key_to_display = {v: k for k, v in label_map.items()}

    rows = []
    conf_plots = []
    for key in selected:
        model  = ARTIFACTS[key]
        probs  = model.predict_proba(X)[0]
        idx    = int(np.argmax(probs))
        label  = IDX_TO_CLASS[idx]
        conf   = float(probs[idx])
        rows.append({
            "Model": key_to_display[key],
            "Prediction": label.upper(),
            "Confidence": f"{conf:.1%}",
            **{c: f"{float(probs[i]):.4f}" for i, c in enumerate(AUDIO_CLASSES_SORTED)},
        })
        conf_plots.append(plot_confidence(AUDIO_CLASSES_SORTED,
                                          [float(p) for p in probs]))

    elapsed = time.time() - t0

    df = pd.DataFrame(rows)

    # Summary text
    top = rows[0]
    summary = (
        f"### Result\n"
        f"**{top['Model']}** → **{top['Prediction']}** with {top['Confidence']} confidence\n\n"
        f"⏱ Inference: {elapsed:.2f}s &nbsp;|&nbsp; Features: 153 → 80 &nbsp;|&nbsp; Models: {len(selected)}"
    )

    wave_img = plot_waveform(y, SAMPLE_RATE)
    mel_img  = plot_mel(y, SAMPLE_RATE)
    mfcc_img = plot_mfcc(y, SAMPLE_RATE)

    # Return up to 3 confidence plots (pad with None)
    p1 = conf_plots[0] if len(conf_plots) > 0 else None
    p2 = conf_plots[1] if len(conf_plots) > 1 else None
    p3 = conf_plots[2] if len(conf_plots) > 2 else None

    return summary, wave_img, mel_img, mfcc_img, p1, p2, p3, df

# ── Gradio UI ─────────────────────────────────────────────────────────────────
CSS = """
body { background: #0D1117 !important; }
.gradio-container { background: #0D1117 !important; color: #E6EDF3 !important; font-family: 'DM Sans', sans-serif; }
h1 { color: #4ECDC4 !important; letter-spacing: -0.03em; }
.label-wrap span { color: #8B949E !important; font-size: 13px !important; }
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="AudioGuard") as demo:

    gr.Markdown("""
# 🎵 AudioGuard
**Speech · Music · Environmental Noise** — Classical ML Audio Classifier  
153-dim MFCC pipeline · SVM / KNN / Gradient Boosting · MUSAN dataset
""")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Upload Audio",
                type="filepath",
                sources=["upload"],
            )
            gr.Markdown("**Supported:** WAV · MP3 · FLAC · OGG · M4A")

            gr.Markdown("### Models")
            use_svm = gr.Checkbox(label="SVM (RBF)",           value=True)
            use_knn = gr.Checkbox(label="KNN (k=7)",           value=True)
            use_gb  = gr.Checkbox(label="Gradient Boosting ⭐", value=True)

            run_btn = gr.Button("🔍 Classify", variant="primary")

            gr.Markdown("""
---
**Features**  
153-dim vector:  
· MFCC ×20 (mean, var, skew)  
· Δ-MFCC & Δ²-MFCC  
· Spectral Centroid / BW / Flux  
· Short-Time Energy · ZCR  
→ Top-80 via Mutual Info

**Dataset**  
[MUSAN](https://huggingface.co/datasets/audiofolder) · ~109h · CC BY 4.0
""")

        with gr.Column(scale=2):
            summary_md = gr.Markdown("Upload a file and click **Classify** to get started.")

            with gr.Tab("Waveform"):
                wave_out = gr.Image(label="Waveform", show_label=False)
            with gr.Tab("Mel Spectrogram"):
                mel_out  = gr.Image(label="Mel Spectrogram", show_label=False)
            with gr.Tab("MFCC Heatmap"):
                mfcc_out = gr.Image(label="MFCC Heatmap", show_label=False)

    gr.Markdown("### Confidence Breakdown")
    with gr.Row():
        conf1 = gr.Image(label="SVM",              show_label=True)
        conf2 = gr.Image(label="KNN",              show_label=True)
        conf3 = gr.Image(label="Gradient Boosting",show_label=True)

    gr.Markdown("### Full Probability Table")
    table_out = gr.Dataframe(label="Results", interactive=False)

    run_btn.click(
        fn=classify,
        inputs=[audio_input, use_svm, use_knn, use_gb],
        outputs=[summary_md, wave_out, mel_out, mfcc_out, conf1, conf2, conf3, table_out],
    )

    gr.Markdown("""
---
<center><small>AudioGuard · Predictive Analytics 2025–26 · Aaron · Aleena · Krishnendu · MUSAN CC BY 4.0</small></center>
""")

if __name__ == "__main__":
    demo.launch()
