from flask import Flask, request, jsonify
import os
import numpy as np
import joblib
import librosa

app = Flask(__name__)

# -----------------------------
# Load model files
# -----------------------------
MODEL_PATH = "models/gradient_boosting_model.pkl"
SCALER_PATH = "models/scaler.pkl"
SELECTOR_PATH = "models/selector.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
selector = joblib.load(SELECTOR_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(audio_path):

    y, sr = librosa.load(audio_path, sr=22050)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.var(mfcc, axis=1)
    ])

    return features.reshape(1, -1)

# -----------------------------
# Home Route
# -----------------------------
@app.route("/")
def home():
    return """
    <h1>🎵 Audio Classification API</h1>
    <p>Upload audio file using POST /predict</p>
    """

# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"})

    file = request.files["audio"]

    if file.filename == "":
        return jsonify({"error": "No selected file"})

    temp_path = f"/tmp/{file.filename}"
    file.save(temp_path)

    try:
        features = extract_features(temp_path)

        scaled = scaler.transform(features)

        selected = selector.transform(scaled)

        prediction = model.predict(selected)

        predicted_label = label_encoder.inverse_transform(prediction)[0]

        probabilities = model.predict_proba(selected)[0]

        confidence = float(np.max(probabilities))

        return jsonify({
            "prediction": predicted_label,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run()
