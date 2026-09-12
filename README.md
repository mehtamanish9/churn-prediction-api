# customer-churn-api

A REST API that serves a Logistic Regression model for predicting customer
churn, built on the Telco Customer Churn dataset. Extends a notebook-based
ML pipeline into a deployable Flask service with a live `/predict` endpoint.

## Features
- Logistic Regression model trained on 7,043 telecom customer records
- Feature engineering: derived `TotalCharges`, one-hot encoded categoricals
- Class imbalance handled via balanced class weighting
- REST API with JSON request/response and input validation
- Validation ROC-AUC: 0.836

## Project structure
```
churn-api/
├── train_model.py         # Loads data, trains and saves the model
├── app.py                 # Flask API (loads model, serves /predict)
├── model.pkl               # Trained model
├── scaler.pkl               # Feature scaler
├── feature_columns.pkl       # Column order used at inference time
├── telco.db                 # SQLite database (customers table, source data)
├── requirements.txt
├── Procfile                 # For Render/Railway deployment
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
python train_model.py     # trains on telco.db, produces model.pkl
python app.py
```

Note: in production (see Procfile), `train_model.py` runs automatically
before the server starts, so the model is always trained fresh with
whatever scikit-learn version is actually installed in that environment.
This avoids version-mismatch errors between training and serving.

The API runs on `http://localhost:5000` by default.

## API Reference

### `GET /`
Health check.
```json
{ "status": "ok", "message": "Churn prediction API is running." }
```

### `POST /predict`
Returns a churn prediction for a given customer.

**Request body:**
```json
{
  "tenure": 2,
  "MonthlyCharges": 95.5,
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "TechSupport": "No",
  "PaperlessBilling": "Yes"
}
```

**Response:**
```json
{
  "prediction": "Churn",
  "churn_probability": 0.8719,
  "input": { ... }
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 2, "MonthlyCharges": 95.5, "Contract": "Month-to-month",
       "InternetService": "Fiber optic", "TechSupport": "No", "PaperlessBilling": "Yes"}'
```

## Retraining
To retrain on updated data, replace `telco.db` with a new SQLite database
containing a `customers` table with the same schema, then run:
```
python train_model.py
```
This regenerates `model.pkl`, `scaler.pkl`, and `feature_columns.pkl`.
`app.py` requires no changes.

## Deployment

**Render**
1. Push this repo to GitHub
2. Create a new Web Service on render.com, connect the repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`

**Railway**
1. Push this repo to GitHub
2. Create a new project on railway.app, deploy from the repo
3. Railway auto-detects the `Procfile`

## Tech Stack
Python, Flask, scikit-learn, pandas, NumPy, SQLite, Gunicorn
