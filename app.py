from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os
import requests
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

def download_model_file():
    url = "https://drive.google.com/file/d/1bd7uWT6jjheIgxtvli-ZiqWiFz15r3ie/view?usp=sharing"
    if not os.path.exists("loan_model.pkl"):
        print("Downloading model...")
        r = requests.get(url)
        with open("loan_model.pkl", "wb") as f:
            f.write(r.content)

download_model_file()
