from fastapi import FastAPI
import joblib
import pandas as pd
import json

app = FastAPI()

# Load model
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load feature list
with open("models/features.json") as f:
    feature_list = json.load(f)


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API"}


@app.post("/predict")
def predict(data: dict):

    try:

        # Create dataframe
        df = pd.DataFrame([data])

        # Add missing columns
        for feature in feature_list:
            if feature not in df.columns:
                df[feature] = 0

        # Ensure correct order
        df = df[feature_list]

        # Scale
        df_scaled = scaler.transform(df)

        # Predict
        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0][1]

        return {
            "prediction": int(prediction),
            "probability": float(probability)
        }

    except Exception as e:

        return {
            "error": str(e)
        }
