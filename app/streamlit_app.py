import streamlit as st
import pandas as pd
import joblib
import json
import requests
import matplotlib.pyplot as plt

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Customer Churn ML System",
    layout="wide"
)

st.title("Customer Churn Prediction Dashboard")

# ---------------------------------------------------
# API URL (RENDER)
# ---------------------------------------------------

API_URL = "https://customer-churn-ml-system.onrender.com/predict"

# ---------------------------------------------------
# LOAD MODEL FILES
# ---------------------------------------------------

model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

with open("models/features.json") as f:
    feature_list = json.load(f)

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

df = pd.read_csv("data/telco_churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df["AvgCharges"] = df["TotalCharges"] / (df["tenure"] + 1)

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0,12,24,48,72],
    labels=["0-1yr","1-2yr","2-4yr","4-6yr"]
)

# ---------------------------------------------------
# BUSINESS KPIs
# ---------------------------------------------------

st.header("Business KPIs")

col1, col2, col3 = st.columns(3)

total_customers = len(df)
avg_monthly = df["MonthlyCharges"].mean()
avg_tenure = df["tenure"].mean()

col1.metric("Total Customers", total_customers)
col2.metric("Avg Monthly Charges", round(avg_monthly,2))
col3.metric("Avg Tenure (Months)", round(avg_tenure,2))

st.divider()

# ---------------------------------------------------
# CHURN PREDICTION
# ---------------------------------------------------

st.header("Predict Customer Churn")

col1, col2, col3 = st.columns(3)

tenure = col1.slider("Tenure (Months)",0,72,12)
monthly = col2.slider("Monthly Charges",0,150,70)
total = col3.slider("Total Charges",0,9000,2000)

if st.button("Predict Churn Risk"):

    input_data = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    try:

        response = requests.post(API_URL, json=input_data)

        if response.status_code == 200:

            result = response.json()

            prob = result["probability"]

            st.metric("Churn Probability", round(prob,2))

            if prob > 0.6:
                st.error("High Risk Customer")
            elif prob > 0.4:
                st.warning("Medium Risk")
            else:
                st.success("Low Risk Customer")

        else:
            st.error("API error: " + str(response.status_code))

    except Exception as e:
        st.error(f"API connection failed: {e}")

st.divider()

# ---------------------------------------------------
# CUSTOMER RISK INSIGHTS
# ---------------------------------------------------

st.header("Customer Risk Insights")

col1, col2 = st.columns(2)

# ---------------- FEATURE IMPORTANCE ----------------

with col1:

    st.subheader("Top Features Causing Churn")

    try:

        fi = pd.read_csv("models/feature_importance.csv")

        top_features = fi.head(8)

        fig, ax = plt.subplots(figsize=(4,3))

        ax.barh(top_features["feature"], top_features["importance"])

        ax.invert_yaxis()

        plt.tight_layout()

        st.pyplot(fig)

    except:
        st.warning("Feature importance file not found.")

# ---------------- HIGH RISK SEGMENTATION ----------------

with col2:

    st.subheader("High Risk Customers")

    try:

        df_encoded = pd.get_dummies(df, drop_first=True)

        for col in feature_list:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        df_encoded = df_encoded[feature_list]

        df_scaled = scaler.transform(df_encoded)

        probs = model.predict_proba(df_scaled)[:,1]

        df["Churn_Risk"] = probs

        high_risk = df[df["Churn_Risk"] > 0.6]

        st.metric("High Risk Customers", len(high_risk))

        st.dataframe(
            high_risk[
                [
                    "tenure",
                    "MonthlyCharges",
                    "Churn_Risk"
                ]
            ]
            .sort_values("Churn_Risk", ascending=False)
            .head(10),
            height=250
        )

    except Exception as e:
        st.error(f"Segmentation error: {e}")
