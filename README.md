# Customer Churn Prediction API

A Flask REST API that serves a Logistic Regression churn prediction model,
built as an extension of the original Customer Churn Prediction project.

## What this adds to the original project
- Wraps the trained model behind a live `/predict` endpoint
- Turns a notebook-based ML pipeline into a deployable service
- Gives you a real, working example of model deployment to talk about in interviews

## Project structure
```
churn-api/
├── train_model.py       # Trains and saves the model
├── app.py                # Flask API (loads model, serves /predict)
├── model.pkl              # Saved trained model
├── scaler.pkl              # Saved feature scaler
├── feature_columns.pkl      # Saved column order (needed for consistent inference)
├── requirements.txt
├── Procfile               # For Render/Railway deployment
└── README.md
```

## Model details
`train_model.py` loads the real Telco Customer Churn dataset from
`telco.db` (a SQLite database, `customers` table, 7,043 rows) — the same
dataset used in the original Customer Churn Prediction project. It applies
the same feature engineering (TotalCharges derived feature, one-hot
encoding, balanced class weighting) and trains a Logistic Regression model.

**Validation ROC-AUC: 0.836** — consistent with the original project's 0.84.

If you want to retrain on updated data later, just replace `telco.db` with
a new database (same `customers` table schema) and re-run:
```
python train_model.py
```
This overwrites `model.pkl`, `scaler.pkl`, and `feature_columns.pkl` — the
API code in `app.py` does not need to change.

## Run locally
```bash
pip install -r requirements.txt
python train_model.py     # trains on telco.db, creates model.pkl
python app.py
```

Test it:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 2, "MonthlyCharges": 95.5, "Contract": "Month-to-month",
       "InternetService": "Fiber optic", "TechSupport": "No", "PaperlessBilling": "Yes"}'
```

Expected response:
```json
{
  "prediction": "Churn",
  "churn_probability": 0.9617,
  "input": { ... }
}
```

## Deploying for free (so you have a live link for your resume)

### Option A: Render
1. Push this folder to a new GitHub repo
2. Go to render.com → New → Web Service → connect your repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Deploy — you'll get a live URL like `https://churn-api-xxxx.onrender.com`

### Option B: Railway
1. Push this folder to a new GitHub repo
2. Go to railway.app → New Project → Deploy from GitHub repo
3. Railway auto-detects the Procfile and deploys
4. You'll get a live URL you can share

## What to add to your resume once deployed
Under your Customer Churn Prediction project, add a line like:

> Deployed the trained model as a REST API (Flask, hosted on Render) with a
> `/predict` endpoint accepting customer features and returning churn
> probability in real time.

This is a completely accurate, non-fabricated addition once you've actually
deployed it — it directly demonstrates API development, which several job
postings you're targeting explicitly ask for.
