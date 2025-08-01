from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)  # Allow frontend access

# Load model and encoders
model = joblib.load("loan_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
categorical_cols = joblib.load("categorical_cols.pkl")

@app.route('/')
def home():
    return "Loan Approval Model is Live!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    
    for col in categorical_cols:
        df[col] = label_encoders[col].transform(df[col])

    prediction = model.predict(df)[0]
    return jsonify({"approved": bool(prediction)})
