# Customer Churn Prediction System

An end-to-end **Machine Learning system** that predicts customer churn for a telecom company and provides actionable business insights through an interactive dashboard.

This project demonstrates a **production-style ML workflow**, including data preprocessing, feature engineering, model training, API deployment, and an analytics dashboard.

---

## Project Overview

Customer churn prediction helps businesses identify customers who are likely to stop using their services. Early identification enables companies to implement **retention strategies** and reduce revenue loss.

This system:

* Trains multiple machine learning models
* Handles class imbalance using SMOTE
* Deploys the best model using FastAPI
* Provides a Streamlit dashboard for prediction and customer risk analysis

---

## Features

* End-to-end ML pipeline
* Feature engineering and preprocessing
* Class imbalance handling using SMOTE
* Model comparison (Logistic Regression, Random Forest, XGBoost)
* Hyperparameter tuning using GridSearchCV
* ROC-AUC evaluation and confusion matrix
* Feature importance visualization
* Customer churn risk segmentation
* REST API using FastAPI
* Interactive analytics dashboard using Streamlit

---

## Tech Stack

**Programming Language**

* Python

**Machine Learning**

* Scikit-learn
* XGBoost
* Imbalanced-learn (SMOTE)

**Data Processing**

* Pandas
* NumPy

**Visualization**

* Matplotlib
* Seaborn

**Deployment**

* FastAPI
* Streamlit
* Docker (optional)

---

## Project Architecture

```
customer-churn-ml-system
│
├── data
│   └── telco_churn.csv
│
├── models
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   ├── feature_importance.csv
│   └── features.json
│
├── src
│   └── train_model.py
│
├── api
│   └── main.py
│
├── app
│   └── streamlit_app.py
│
├── requirements.txt
└── README.md
```

---

## Machine Learning Workflow

### 1. Data Preprocessing

* Removed unnecessary columns
* Handled missing values
* Converted data types

### 2. Feature Engineering

Created additional features such as:

* Average charges per month
* Tenure group segmentation

### 3. Handling Class Imbalance

Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance churn classes.

### 4. Model Training

The following models were trained:

* Logistic Regression
* Random Forest (with GridSearch tuning)
* XGBoost

### 5. Model Evaluation

Evaluation metrics used:

* Accuracy
* ROC-AUC
* Confusion Matrix
* ROC Curve

Best model performance:

```
ROC-AUC: ~0.83
Accuracy: ~0.76
```

---

## Dashboard Features

The Streamlit dashboard includes:

### Business KPIs

* Total customers
* Average monthly charges
* Average tenure

### Churn Prediction

Predict churn probability for a new customer.

### Feature Importance

Shows the most important factors contributing to churn.

### High-Risk Customer Segmentation

Identifies customers with a high probability of churn.

---

## Running the Project

### 1. Clone Repository

```
git clone https://github.com/yourusername/customer-churn-ml-system.git
cd customer-churn-ml-system
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Train the Model

```
python src/train_model.py
```

### 4. Start FastAPI Server

```
uvicorn api.main:app --reload
```

API will run at:

```
http://127.0.0.1:8000
```

Interactive API docs:

```
http://127.0.0.1:8000/docs
```

### 5. Run Dashboard

```
streamlit run app/streamlit_app.py
```

---

## Example API Request

```
POST /predict
```

Request body:

```
{
 "tenure": 12,
 "MonthlyCharges": 70,
 "TotalCharges": 2000
}
```

Response:

```
{
 "prediction": 0,
 "probability": 0.23
}
```

---

## Dataset

Telco Customer Churn Dataset.

Features include:

* Tenure
* Monthly charges
* Contract type
* Payment method
* Internet service
* Customer demographics

---

## Future Improvements

Possible enhancements:

* Batch churn prediction
* Model monitoring
* CI/CD pipeline
* Cloud deployment (AWS / GCP)
* Automated retraining pipeline

---

## Author

**Abhinav B Mamidwar**

Aspiring Data Scientist with interests in:

* Machine Learning
* Deep Learning
* NLP
* Data Analytics

---

## License

This project is for educational and portfolio purposes.
