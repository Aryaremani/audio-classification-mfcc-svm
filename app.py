from flask import Flask, request, jsonify
import os
import numpy as np
import joblib

app = Flask(__name__)

# -----------------------------
# Home Route
# -----------------------------
@app.route("/")
def home():
    return """
    <html>
        <head>
            <title>Audio Classification</title>
            <style>
                body{
                    font-family: Arial;
                    background:#0d1117;
                    color:white;
                    text-align:center;
                    padding-top:50px;
                }

                .box{
                    width:400px;
                    margin:auto;
                    background:#161b22;
                    padding:30px;
                    border-radius:10px;
                }

                input{
                    margin-top:20px;
                }

                button{
                    margin-top:20px;
                    padding:10px 20px;
                    border:none;
                    border-radius:5px;
                    background:#238636;
                    color:white;
                    cursor:pointer;
                }
            </style>
        </head>

        <body>

            <div class="box">
                <h1>🎵 Audio Classification System</h1>

                <p>Upload an audio file</p>

                <form action="/predict" method="post" enctype="multipart/form-data">

                    <input type="file" name="audio" required>

                    <br>

                    <button type="submit">
                        Predict
                    </button>

                </form>

            </div>

        </body>
    </html>
    """

# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Check file
        if "audio" not in request.files:
            return "No audio file uploaded"

        file = request.files["audio"]

        if file.filename == "":
            return "No selected file"

        # Create temp folder
        temp_path = f"/tmp/{file.filename}"

        # Save uploaded file
        file.save(temp_path)

        # --------------------------------------------------
        # SIMPLE DUMMY PREDICTION
        # --------------------------------------------------
        # Replace later with your ML prediction code
        # --------------------------------------------------

        prediction = "Speech"

        confidence = 0.95

        return f"""
        <html>
            <body style="font-family:Arial;background:#0d1117;color:white;text-align:center;padding-top:50px;">

                <h1>Prediction Result</h1>

                <h2>{prediction}</h2>

                <h3>Confidence: {confidence}</h3>

                <br>

                <a href="/" style="color:#58a6ff;">
                    Upload Another File
                </a>

            </body>
        </html>
        """

    except Exception as e:

        return f"""
        <html>
            <body style="font-family:Arial;background:#0d1117;color:white;text-align:center;padding-top:50px;">

                <h1>Error</h1>

                <p>{str(e)}</p>

            </body>
        </html>
        """

# -----------------------------
# Vercel Entry
# -----------------------------
app = app

# -----------------------------
# Run Local
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
