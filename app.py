from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_loan

app = FastAPI()

# Input schema
class LoanRequest(BaseModel):
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    Property_Area: str

@app.get("/")
def home():
    return {"message": "Loan Prediction API Running 🚀"}

@app.post("/predict")
def predict(request: LoanRequest):
    data = request.dict()
    result = predict_loan(data)
    return result