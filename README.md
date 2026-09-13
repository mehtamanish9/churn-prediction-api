<div align="center">

# ⚡ Telecom Customer Churn Prediction REST API

A production-ready REST API built with **Flask**, **Scikit-Learn**, and **Gunicorn** to serve real-time customer churn probability inferences from customer account, tenure, and service features.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Framework: Flask](https://img.shields.io/badge/Framework-Flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Analysis Repository](https://img.shields.io/badge/Analysis%20Notebooks-Customer_Churn_analysis-orange)](https://github.com/mehtamanish9/Customer_Churn_analysis)

</div>

---

## 📌 Overview

This service packages a trained **Logistic Regression classification pipeline** into an inference API. It handles incoming customer payloads, applies the exact preprocessing & scaling transforms used during model training, and returns the predicted outcome along with continuous churn probabilities.

### 🌟 Key Highlights
* **Automated Data Transformation**: Dynamically computes missing derived attributes (e.g. estimated `TotalCharges = tenure × MonthlyCharges` if absent) and aligns one-hot encoded dummy columns at inference time.
* **Balanced Classification**: Leverages balanced class weighting (`ROC-AUC: 0.836`) to handle churn imbalance without artificial oversampling.
* **Production Deployment Ready**: Pre-configured with `Procfile` and `gunicorn` for 1-click deployments on **Render**, **Railway**, or **Heroku**.
* **Zero-Downtime Training Hook**: Automatic training invocation before server start prevents scikit-learn version serialization mismatches across environments.

---

## 📁 Repository Structure

```text
churn-prediction-api/
├── app.py                 # Flask REST service with validation & /predict endpoint
├── train_model.py         # Pipeline script: trains on telco.db, saves model & scaler
├── telco.db               # SQLite database containing training customer records
├── model.pkl              # Serialized Logistic Regression model
├── scaler.pkl             # StandardScaler for continuous numerical features
├── feature_columns.pkl    # Exact feature schema & column ordering for inference
├── Procfile               # Production startup command (Render / Railway / Heroku)
├── requirements.txt       # Dependencies (Flask, scikit-learn, pandas, gunicorn)
├── LICENSE                # MIT License
└── README.md              # Documentation
```

---

## 🔌 API Endpoint Documentation

### 1. Health Check
* **Endpoint**: `GET /`
* **Response**:
```json
{
  "status": "ok",
  "message": "Churn prediction API is running."
}
```

---

### 2. Predict Churn Probability
* **Endpoint**: `POST /predict`
* **Headers**: `Content-Type: application/json`

#### Request Body Schema
```json
{
  "tenure": 3,
  "MonthlyCharges": 89.85,
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "TechSupport": "No",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check"
}
```

#### Successful Response (`200 OK`)
```json
{
  "prediction": "Churn",
  "churn_probability": 0.8412,
  "status": "success",
  "input": {
    "tenure": 3,
    "MonthlyCharges": 89.85,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "TechSupport": "No",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
  }
}
```

---

## 💻 Quickstart & Testing Locally

### 1. Clone and install
```bash
git clone https://github.com/mehtamanish9/churn-prediction-api.git
cd churn-prediction-api
pip install -r requirements.txt
```

### 2. Train and launch
```bash
python train_model.py
python app.py
```
*The API will start at `http://localhost:5000`.*

### 3. Send a test request (cURL)
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 2,
    "MonthlyCharges": 95.5,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "TechSupport": "No",
    "PaperlessBilling": "Yes"
  }'
```

### 4. Python `requests` example
```python
import requests

url = "http://localhost:5000/predict"
payload = {
    "tenure": 12,
    "MonthlyCharges": 65.0,
    "Contract": "One year",
    "InternetService": "DSL",
    "TechSupport": "Yes",
    "PaperlessBilling": "No"
}

response = requests.post(url, json=payload)
print(response.json())
```

---

## ☁️ Deployment Instructions

### Deploy to Render
1. Create a **New Web Service** connected to your GitHub repository.
2. **Build Command**: `pip install -r requirements.txt`
3. **Start Command**: `gunicorn app:app`
4. Deploy!

---

## 👨‍💻 Author

**Manish Mehta**  
* GitHub: [@mehtamanish9](https://github.com/mehtamanish9)  
* LinkedIn: [linkedin.com/in/manish-mehta04](https://www.linkedin.com/in/manish-mehta04)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

