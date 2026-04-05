import pandas as pd
import joblib

# Load artifacts
model = joblib.load("loan_model.pkl")
columns = joblib.load("columns.pkl")

# Same feature engineering (MUST match training)
def feature_engineering(df):
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Income_to_Loan'] = df['TotalIncome'] / (df['LoanAmount'] + 1)
    df['EMI'] = df['LoanAmount'] / (df['Loan_Amount_Term'] + 1)
    return df

# Business logic
def loan_decision(prob):
    if prob < 0.3:
        return "Approve"
    elif prob < 0.6:
        return "Review"
    else:
        return "Reject"

# Main prediction function
def predict_loan(data: dict):
    df = pd.DataFrame([data])

    # Feature engineering
    df = feature_engineering(df)

    # Encoding
    df = pd.get_dummies(df)

    # Align columns
    df = df.reindex(columns=columns, fill_value=0)

    # Prediction
    prob = model.predict_proba(df)[0][1]
    decision = loan_decision(prob)

    return {
        "probability_of_default": float(prob),
        "decision": decision
    }