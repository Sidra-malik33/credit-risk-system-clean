import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# ==============================
# 1. Load Dataset
# ==============================
df = pd.read_csv('training_dataset.csv')

print(df.head())
print(df['Loan_Status'].value_counts())

# ==============================
# 2. Preprocessing
# ==============================
df.ffill(inplace=True)
df = df.dropna()

df = df.drop('Loan_ID', axis=1)

# ==============================
# 3. Feature Engineering (IMPORTANT 🔥)
# ==============================
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['Income_to_Loan'] = df['TotalIncome'] / (df['LoanAmount'] + 1)
df['EMI'] = df['LoanAmount'] / (df['Loan_Amount_Term'] + 1)

# ==============================
# 4. Split Features / Target
# ==============================
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Convert target to binary (VERY IMPORTANT)
y = y.map({'Y': 1, 'N': 0})

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Save column names for deployment
joblib.dump(X.columns.tolist(), "columns.pkl")

# ==============================
# 5. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 6. Pipeline
# ==============================
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        class_weight='balanced',
        random_state=42
    ))
])
# Train
pipeline.fit(X_train, y_train)

# ==============================
# 7. Evaluation
# ==============================
y_prob = pipeline.predict_proba(X_test)[:, 1]

# 🔥 Custom Threshold (IMPORTANT)
threshold = 0.7
y_pred = (y_prob > threshold).astype(int)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

fp = confusion_matrix(y_test, y_pred)[0][1]
tn = confusion_matrix(y_test, y_pred)[0][0]

print("False Approval Rate:", fp / (fp + tn))
# Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ==============================
# 8. Business Decision Logic (FinTech Style)
# ==============================
def loan_decision(prob):
    if prob < 0.3:
        return "Approve"
    elif prob < 0.6:
        return "Review"
    else:
        return "Reject"

# Example decisions
sample_probs = y_prob[:5]
decisions = [loan_decision(p) for p in sample_probs]

print("\nSample Decisions:")
for p, d in zip(sample_probs, decisions):
    print(f"PD: {round(p,3)} → {d}")

# Feature Importance (Explainability 🔥)
model = pipeline.named_steps['model']
importances = model.feature_importances_

for name, val in zip(X.columns, importances):
    print(name, val)



# ==============================
# 9. Save Model
# ==============================
joblib.dump(pipeline, "loan_model.pkl")

print("\nModel and columns saved successfully ✅")