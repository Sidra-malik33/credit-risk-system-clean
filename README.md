# 💳 Credit Risk Prediction API

A Machine Learning-powered API that predicts loan approval risk using financial data.

## 🚀 Live API

👉 https://sidnad-credit-risk-system-api.hf.space/docs#/default/predict_predict_post

## 📌 Features

* FastAPI-based REST API
* ML model (Scikit-learn)
* Real-time prediction endpoint
* Deployed on Hugging Face Spaces

## 🔍 API Endpoints

### Home

`GET /`
Returns API status

### Predict

`POST /predict`

#### Example Input:

```json
{
  "income": 50000,
  "loan_amount": 200000,
  "credit_history": 1
}
```

#### Output:
{
  "probability_of_default": 0.4969414198717352,
  "decision": "Review"
}

## 🛠 Tech Stack

* Python
* FastAPI
* Scikit-learn
* Hugging Face Spaces
