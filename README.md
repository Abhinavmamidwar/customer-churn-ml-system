# Customer Churn Prediction ML System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange)
![Framework](https://img.shields.io/badge/API-FastAPI-green)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)
![Deployment](https://img.shields.io/badge/Deployment-Render%20%7C%20Streamlit%20Cloud-purple)

An **end-to-end Machine Learning system** that predicts telecom customer churn and provides business insights through an interactive dashboard.

This project demonstrates a **complete ML pipeline from data preprocessing to production deployment**.

---

# Live Demo

**Streamlit Dashboard**

https://customer-churn-ml-system.streamlit.app/

**FastAPI Prediction API**

https://customer-churn-ml-system.onrender.com/docs

---

# Project Overview

Customer churn is a critical problem for telecom companies. Retaining existing customers is significantly cheaper than acquiring new ones.

This project builds a **machine learning system that predicts the probability of a customer leaving a telecom service**, enabling companies to take proactive retention actions.

---

# System Architecture

User Input
⬇
Streamlit Dashboard
⬇
FastAPI Prediction API
⬇
Trained ML Model (XGBoost)
⬇
Churn Risk Prediction

---

# Features

### Churn Prediction

Predicts the probability that a telecom customer will churn.

### Business KPI Dashboard

Displays key metrics such as:

* Total Customers
* Average Monthly Charges
* Average Customer Tenure

### Churn Risk Classification

Customers are categorized into:

* Low Risk
* Medium Risk
* High Risk

### Feature Importance Analysis

Visualizes the **top features responsible for customer churn**.

### High-Risk Customer Segmentation

Identifies customers with **high churn probability**, helping businesses focus retention efforts.

---

# Machine Learning Pipeline

## Dataset

Telco Customer Churn Dataset

Key features include:

* Tenure
* Monthly Charges
* Total Charges
* Contract Type
* Internet Service
* Payment Method
* Customer Demographics

---

## Feature Engineering

Created additional features:

* Average Charges per Month
* Tenure Groups

Handled missing values and encoded categorical variables.

---

## Handling Class Imbalance

Used **SMOTE (Synthetic Minority Oversampling Technique)** to balance churn vs non-churn customers.

---

## Model Training

Models evaluated:

* Logistic Regression
* Random Forest
* XGBoost

Final selected model:

**XGBoost Classifier**

Performance:

Accuracy ≈ 79%
ROC-AUC ≈ 0.83

---

# Tech Stack

Python
Pandas
NumPy
Scikit-Learn
XGBoost
SMOTE
FastAPI
Streamlit
Matplotlib
Seaborn
Docker (Local Development)
Render (API Deployment)
Streamlit Cloud (Dashboard Deployment)

---

# Project Structure

```
customer-churn-ml-system

data/
    telco_churn.csv

models/
    churn_model.pkl
    scaler.pkl
    features.json
    feature_importance.csv

src/
    train_model.py
    api.py
    feature_importance.py

app.py
requirements.txt
docker-compose.yml
README.md
```

---

# Run Locally

Clone the repository

```
git clone https://github.com/Abhinavmamidwar/customer-churn-ml-system.git
```

Install dependencies

```
pip install -r requirements.txt
```

Train the model

```
python src/train_model.py
```

Run the API

```
uvicorn src.api:app --reload
```

Run the dashboard

```
streamlit run app.py
```

---

# Business Value

Customer churn leads to **billions in revenue loss for telecom companies** every year.

This system helps businesses:

* Identify high-risk customers
* Understand churn drivers
* Improve customer retention strategies
* Reduce revenue loss

---

# Future Improvements

* Customer retention recommendation engine
* Real-time churn monitoring
* Automated model retraining pipeline
* Cloud-based ML workflow

---

# Author

Abhinav Mamidwar

Aspiring Data Scientist / Machine Learning Engineer

GitHub
https://github.com/Abhinavmamidwar
