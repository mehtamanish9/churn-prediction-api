"""
Flask REST API serving the churn prediction model.

Endpoints:
  GET  /              -> health check
  POST /predict        -> churn prediction given customer features

Run locally:
  python app.py
  # then in another terminal:
  curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{"tenure": 5, "MonthlyCharges": 85.5, "Contract": "Month-to-month",
         "InternetService": "Fiber optic", "TechSupport": "No", "PaperlessBilling": "Yes"}'
"""

from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")
COLUMNS_PATH = os.path.join(os.path.dirname(__file__), "feature_columns.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(COLUMNS_PATH)

REQUIRED_FIELDS = [
    "tenure", "MonthlyCharges", "Contract",
    "InternetService", "TechSupport", "PaperlessBilling",
]


def preprocess(payload: dict) -> pd.DataFrame:
    """Turn a raw JSON payload into the exact feature matrix the model expects."""
    row = {
        "tenure": payload["tenure"],
        "MonthlyCharges": payload["MonthlyCharges"],
        "Contract": payload["Contract"],
        "InternetService": payload["InternetService"],
        "TechSupport": payload["TechSupport"],
        "PaperlessBilling": payload["PaperlessBilling"],
    }
    df = pd.DataFrame([row])
    df["TotalCharges"] = df["tenure"] * df["MonthlyCharges"]
    df = pd.get_dummies(df, columns=["Contract", "InternetService", "TechSupport", "PaperlessBilling"])

    # Ensure every column the model was trained on exists (fill missing dummy cols with 0)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]  # exact column order
    return df


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Churn prediction API is running."})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        X = preprocess(data)
        X_scaled = scaler.transform(X)
        churn_probability = float(model.predict_proba(X_scaled)[0][1])
        prediction = "Churn" if churn_probability >= 0.5 else "No Churn"
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({
        "prediction": prediction,
        "churn_probability": round(churn_probability, 4),
        "input": data,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
