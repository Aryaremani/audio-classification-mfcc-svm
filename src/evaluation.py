"""
evaluation.py — Stage 5: Multiclass Evaluation & Plots
=======================================================
Metrics for 3-class audio classification:
  · Accuracy, Macro F1, per-class Precision/Recall
  · Confusion matrix (3×3)
  · ROC-AUC curves (one-vs-rest)
  · Precision-Recall curves
  · Feature importance (GB built-in + SVM permutation)
  · SNR robustness analysis
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve, roc_curve, auc,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize

from src.config import (
    OUTPUT_DIR, COLORS, CLASS_COLORS,
    FIGSIZE_LARGE, FIGSIZE_MEDIUM, FIGSIZE_SMALL, DPI,
    AUDIO_CLASSES_SORTED, N_CLASSES, RANDOM_STATE,
)

# ── Helper ────────────────────────────────────────────────────────────────────
def _fig_path(subdir: str, name: str) -> Path:
    p = OUTPUT_DIR / subdir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ══════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_prob, model_name: str = "") -> Dict:
    """Macro-averaged multiclass metrics."""
    metrics = {
        "model":            model_name,
        "accuracy":         accuracy_score(y_true, y_pred),
        "macro_f1":         f1_score(y_true, y_pred, average="macro",     zero_division=0),
        "macro_precision":  precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall":     recall_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1":      f1_score(y_true, y_pred, average="weighted",  zero_division=0),
    }
    try:
        y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
        metrics["roc_auc_ovr"] = roc_auc_score(
            y_bin, y_prob, multi_class="ovr", average="macro"
        )
    except Exception:
        metrics["roc_auc_ovr"] = float("nan")
    return metrics


# ══════════════════════════════════════════════════════════
# Confusion Matrix
# ══════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred, model_name: str = "",
                          save: bool = True):
    cm     = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    labels = [c[:10] for c in AUDIO_CLASSES_SORTED]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=DPI)
    fig.patch.set_facecolor(COLORS["bg_dark"])
    ax.set_facecolor(COLORS["bg_card"])

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(labels, rotation=30, ha="right",
                       color=COLORS["text"], fontsize=10)
    ax.set_yticklabels(labels, color=COLORS["text"], fontsize=10)
    ax.set_xlabel("Predicted", color=COLORS["text"], fontsize=12)
    ax.set_ylabel("True",      color=COLORS["text"], fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}",
                 color=COLORS["text"], fontsize=13)

    thresh = cm.max() / 2.0
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight="bold")

    plt.tight_layout()
    if save:
        path = _fig_path("evaluation", f"cm_{model_name.lower().replace(' ', '_')}.png")
        fig.savefig(path, bbox_inches="tight")
    return fig


# ══════════════════════════════════════════════════════════
# ROC Curves (One-vs-Rest)
# ══════════════════════════════════════════════════════════

def plot_roc_ovr(y_true, y_prob_matrix, model_name: str = "",
                 save: bool = True):
    y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))

    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM, dpi=DPI)
    fig.patch.set_facecolor(COLORS["bg_dark"])
    ax.set_facecolor(COLORS["bg_card"])

    for i, cls in enumerate(AUDIO_CLASSES_SORTED):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob_matrix[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=CLASS_COLORS[i], lw=2,
                label=f"{cls} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "w--", alpha=0.4, lw=1.5)
    ax.set_xlabel("False Positive Rate", color=COLORS["text"], fontsize=12)
    ax.set_ylabel("True Positive Rate",  color=COLORS["text"], fontsize=12)
    ax.set_title(f"ROC Curves (One-vs-Rest) — {model_name}",
                 color=COLORS["text"], fontsize=13)
    ax.legend(facecolor=COLORS["bg_card"], labelcolor=COLORS["text"],
              fontsize=10, framealpha=0.85)
    ax.tick_params(colors=COLORS["text"])
    for sp in ax.spines.values():
        sp.set_edgecolor(COLORS["grid"])

    plt.tight_layout()
    if save:
        path = _fig_path("evaluation", f"roc_{model_name.lower().replace(' ', '_')}.png")
        fig.savefig(path, bbox_inches="tight")
    return fig


# ══════════════════════════════════════════════════════════
# Precision-Recall Curves
# ══════════════════════════════════════════════════════════

def plot_precision_recall(y_true, y_prob_matrix, model_name: str = "",
                          save: bool = True):
    y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))

    fig, axes = plt.subplots(1, N_CLASSES, figsize=(5 * N_CLASSES, 4), dpi=DPI)
    fig.patch.set_facecolor(COLORS["bg_dark"])

    for i, (cls, ax) in enumerate(zip(AUDIO_CLASSES_SORTED, axes)):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob_matrix[:, i])
        pr_auc = auc(rec, prec)
        ax.set_facecolor(COLORS["bg_card"])
        ax.plot(rec, prec, color=CLASS_COLORS[i], lw=2)
        ax.fill_between(rec, prec, alpha=0.15, color=CLASS_COLORS[i])
        ax.set_title(f"{cls}\nAUC={pr_auc:.3f}",
                     fontsize=10, color=CLASS_COLORS[i])
        ax.set_xlabel("Recall",    color=COLORS["text"], fontsize=9)
        ax.set_ylabel("Precision", color=COLORS["text"], fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.tick_params(colors=COLORS["text"])
        for sp in ax.spines.values():
            sp.set_edgecolor(COLORS["grid"])

    fig.suptitle(f"Precision-Recall Curves — {model_name}",
                 fontsize=13, color=COLORS["text"])
    plt.tight_layout()
    if save:
        path = _fig_path("evaluation", f"pr_{model_name.lower().replace(' ', '_')}.png")
        fig.savefig(path, bbox_inches="tight")
    return fig


# ══════════════════════════════════════════════════════════
# Feature Importance (GB built-in)
# ══════════════════════════════════════════════════════════

def plot_feature_importance(model, feature_names: List[str], model_name: str = "",
                             top_n: int = 25, save: bool = True):
    """Works with GradientBoostingClassifier (feature_importances_ attribute)."""
    if not hasattr(model, "feature_importances_"):
        return None

    importance = model.feature_importances_
    top_idx    = np.argsort(importance)[::-1][:top_n]

    colors = [
        COLORS["primary"]  if "mfcc" in feature_names[i]
        else COLORS["success"]  if "delta" in feature_names[i]
        else COLORS["warning"]
        for i in top_idx
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE, dpi=DPI)
    fig.patch.set_facecolor(COLORS["bg_dark"])
    ax.set_facecolor(COLORS["bg_card"])

    ax.barh(range(top_n)[::-1], importance[top_idx],
            color=colors, edgecolor=COLORS["border"], linewidth=0.5)
    ax.set_yticks(range(top_n)[::-1])
    ax.set_yticklabels([feature_names[i] for i in top_idx],
                       fontsize=8, color=COLORS["text"])
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}",
                 fontsize=12, color=COLORS["text"])
    ax.set_xlabel("Importance", color=COLORS["text"], fontsize=11)
    ax.tick_params(colors=COLORS["text"])
    for sp in ax.spines.values():
        sp.set_edgecolor(COLORS["grid"])

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["primary"], label="MFCC"),
        Patch(facecolor=COLORS["success"], label="Delta/Delta²"),
        Patch(facecolor=COLORS["warning"], label="Spectral/Energy/ZCR"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
              facecolor=COLORS["bg_card"], labelcolor=COLORS["text"])

    plt.tight_layout()
    if save:
        path = _fig_path("explainability",
                         f"importance_{model_name.lower().replace(' ', '_')}.png")
        fig.savefig(path, bbox_inches="tight")
    return fig


# ══════════════════════════════════════════════════════════
# Model Comparison Bar Chart
# ══════════════════════════════════════════════════════════

def plot_model_comparison(metrics_list: List[Dict], save: bool = True):
    names  = [m["model"]    for m in metrics_list]
    accs   = [m["accuracy"] for m in metrics_list]
    f1s    = [m["macro_f1"] for m in metrics_list]
    aucs   = [m.get("roc_auc_ovr", float("nan")) for m in metrics_list]

    x = np.arange(len(names))
    w = 0.28

    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE, dpi=DPI)
    fig.patch.set_facecolor(COLORS["bg_dark"])
    ax.set_facecolor(COLORS["bg_card"])

    ax.bar(x - w, accs, w, label="Accuracy",    color=COLORS["primary"],  alpha=0.85)
    ax.bar(x,     f1s,  w, label="Macro F1",    color=COLORS["success"],  alpha=0.85)
    ax.bar(x + w, aucs, w, label="ROC-AUC OvR", color=COLORS["warning"],  alpha=0.85)

    for xi, (a, f, u) in enumerate(zip(accs, f1s, aucs)):
        ax.text(xi - w, a + 0.01, f"{a:.3f}", ha="center", fontsize=9, color=COLORS["primary"])
        ax.text(xi,     f + 0.01, f"{f:.3f}", ha="center", fontsize=9, color=COLORS["success"])
        if not np.isnan(u):
            ax.text(xi + w, u + 0.01, f"{u:.3f}", ha="center", fontsize=9, color=COLORS["warning"])

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11, color=COLORS["text"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", color=COLORS["text"], fontsize=11)
    ax.set_title("Model Comparison — Accuracy · Macro F1 · ROC-AUC",
                 fontsize=13, color=COLORS["text"])
    ax.legend(facecolor=COLORS["bg_card"], labelcolor=COLORS["text"], fontsize=10)
    ax.tick_params(colors=COLORS["text"])
    for sp in ax.spines.values():
        sp.set_edgecolor(COLORS["grid"])
    ax.yaxis.grid(True, color=COLORS["grid"])

    plt.tight_layout()
    if save:
        path = _fig_path("evaluation", "model_comparison.png")
        fig.savefig(path, bbox_inches="tight")
    return fig


# ══════════════════════════════════════════════════════════
# SNR Robustness
# ══════════════════════════════════════════════════════════

def plot_snr_robustness(snr_results: Dict[str, List[float]],
                        snr_levels: List[float], save: bool = True):
    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM, dpi=DPI)
    fig.patch.set_facecolor(COLORS["bg_dark"])
    ax.set_facecolor(COLORS["bg_card"])

    colors = [COLORS["primary"], COLORS["success"], COLORS["warning"]]
    for (name, scores), color in zip(snr_results.items(), colors):
        ax.plot(snr_levels, scores, marker="o", label=name,
                color=color, linewidth=2.2, markersize=6)

    ax.set_title("SNR Robustness — Macro F1 vs. Noise Level",
                 fontsize=13, color=COLORS["text"])
    ax.set_xlabel("SNR (dB)", color=COLORS["text"], fontsize=11)
    ax.set_ylabel("Macro F1", color=COLORS["text"], fontsize=11)
    ax.legend(facecolor=COLORS["bg_card"], labelcolor=COLORS["text"], fontsize=10)
    ax.invert_xaxis()
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors=COLORS["text"])
    for sp in ax.spines.values():
        sp.set_edgecolor(COLORS["grid"])
    ax.yaxis.grid(True, color=COLORS["grid"])

    plt.tight_layout()
    if save:
        path = _fig_path("evaluation", "snr_robustness.png")
        fig.savefig(path, bbox_inches="tight")
    return fig


# ══════════════════════════════════════════════════════════
# Cross-Validation Summary
# ══════════════════════════════════════════════════════════

def plot_cv_summary(cv_results: Dict[str, Dict], save: bool = True):
    """Bar chart of 5-fold CV macro F1 ± std for each model."""
    names  = list(cv_results.keys())
    means  = [cv_results[n]["mean"] for n in names]
    stds   = [cv_results[n]["std"]  for n in names]
    colors = [COLORS["primary"], COLORS["success"], COLORS["warning"]]

    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL, dpi=DPI)
    fig.patch.set_facecolor(COLORS["bg_dark"])
    ax.set_facecolor(COLORS["bg_card"])

    bars = ax.bar(names, means, color=colors[:len(names)], alpha=0.85,
                  edgecolor=COLORS["border"], linewidth=1.2,
                  yerr=stds, capsize=6,
                  error_kw={"ecolor": COLORS["subtext"], "linewidth": 1.5})
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01,
                f"{m:.3f}±{s:.3f}", ha="center", fontsize=9, color=COLORS["text"])

    ax.set_ylabel("Macro F1", color=COLORS["text"], fontsize=11)
    ax.set_title("5-Fold Cross-Validation — Macro F1",
                 fontsize=12, color=COLORS["text"])
    ax.set_ylim(0, 1.15)
    ax.tick_params(colors=COLORS["text"])
    for sp in ax.spines.values():
        sp.set_edgecolor(COLORS["grid"])
    ax.yaxis.grid(True, color=COLORS["grid"])

    plt.tight_layout()
    if save:
        path = _fig_path("evaluation", "cross_validation.png")
        fig.savefig(path, bbox_inches="tight")
    return fig


# ══════════════════════════════════════════════════════════
# Master Evaluation Function
# ══════════════════════════════════════════════════════════

def evaluate_all_models(models_dict: Dict, X_test: np.ndarray,
                        y_test: np.ndarray, selected_feature_names: List[str],
                        run_snr: bool = True) -> pd.DataFrame:
    """
    Run full evaluation for SVM, KNN, and Gradient Boosting.
    Saves all plots to outputs/evaluation/ and outputs/explainability/.
    Returns a summary DataFrame.
    """
    MODEL_KEYS = {"svm": "SVM", "knn": "KNN", "gb": "Gradient Boosting"}
    all_metrics = []
    snr_levels  = [30, 20, 15, 10, 5, 0]
    snr_results: Dict[str, List[float]] = {}

    for key, display_name in MODEL_KEYS.items():
        if key not in models_dict:
            continue
        model = models_dict[key]
        print(f"\n── {display_name} ──────────────────────────────────")

        y_pred = model.predict(X_test)
        y_prob = (model.predict_proba(X_test)
                  if hasattr(model, "predict_proba")
                  else np.eye(3)[y_pred])

        m = compute_metrics(y_test, y_pred, y_prob, display_name)
        all_metrics.append(m)

        print(f"  Accuracy  : {m['accuracy']:.4f}")
        print(f"  Macro F1  : {m['macro_f1']:.4f}")
        print(f"  ROC-AUC   : {m.get('roc_auc_ovr', float('nan')):.4f}")
        print(classification_report(y_test, y_pred,
                                    target_names=AUDIO_CLASSES_SORTED,
                                    digits=4, zero_division=0))

        plot_confusion_matrix(y_test, y_pred, display_name)
        plot_roc_ovr(y_test, y_prob, display_name)
        plot_precision_recall(y_test, y_prob, display_name)
        plot_feature_importance(model, selected_feature_names, display_name)

        # SNR robustness
        if run_snr:
            snr_results[display_name] = []
            for snr in snr_levels:
                from src.feature_extraction import add_awgn
                # inject noise into the raw scaled features is approximate —
                # for an exact test you'd re-load audio; this is a proxy
                noise    = np.random.randn(*X_test.shape).astype(np.float32)
                sig_pwr  = float(np.mean(X_test ** 2)) + 1e-12
                nse_pwr  = sig_pwr / (10 ** (snr / 10.0))
                X_noisy  = (X_test + noise * np.sqrt(nse_pwr / (np.mean(noise**2)+1e-12))).astype(np.float32)
                y_noisy  = model.predict(X_noisy)
                f1_noisy = f1_score(y_test, y_noisy, average="macro", zero_division=0)
                snr_results[display_name].append(f1_noisy)

    plot_model_comparison(all_metrics)
    if run_snr and snr_results:
        plot_snr_robustness(snr_results, snr_levels)

    summary = pd.DataFrame(all_metrics)
    print("\n── Summary ──────────────────────────────────────────────")
    print(summary[["model", "accuracy", "macro_f1", "roc_auc_ovr"]].to_string(index=False))
    return summary
