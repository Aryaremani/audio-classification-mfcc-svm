---
title: AudioGuard
emoji: 🎵
colorFrom: teal
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🎵 AudioGuard
### Speech · Music · Environmental Noise — Classical ML Audio Classifier

**AudioGuard** classifies audio signals into three categories — **speech**, **music**, and **environmental noise** — using exclusively hand-crafted acoustic features and classical machine learning. No deep learning required.

## Features
- 153-dim feature vector: MFCC × 20 (mean, var, skew) · Δ-MFCC · Δ²-MFCC · Spectral Centroid / BW / Flux · Short-Time Energy · ZCR
- Top-80 features selected via Mutual Information
- Models: SVM (RBF) · KNN (k=7) · Gradient Boosting ⭐
- Visualisations: Waveform · Mel Spectrogram · MFCC Heatmap · Confidence bars

## Dataset
[MUSAN](https://www.kaggle.com/datasets/dogrose/musan-dataset) — Music, Speech, and Noise Corpus · ~109h · CC BY 4.0

## Team
Aaron · Aleena · Krishnendu — Predictive Analytics 2025–26
