import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve

from imblearn.over_sampling import SMOTE
import xgboost as xgb

import seaborn as sns
import matplotlib.pyplot as plt


# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("data/telco_churn.csv")


# -----------------------------
# Data Cleaning
# -----------------------------

df = df.drop("customerID", axis=1)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())


# -----------------------------
# Feature Engineering
# -----------------------------

df["AvgCharges"] = df["TotalCharges"] / (df["tenure"] + 1)

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"]
)


# -----------------------------
# Encoding
# -----------------------------

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

df = pd.get_dummies(df, drop_first=True)


# -----------------------------
# Split Features / Target
# -----------------------------

X = df.drop("Churn", axis=1)

y = df["Churn"]


# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# Feature Scaling
# -----------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# -----------------------------
# Handle Class Imbalance
# -----------------------------

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("After SMOTE:", X_train_smote.shape)


# -----------------------------
# Logistic Regression
# -----------------------------

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train_smote, y_train_smote)

log_preds = log_model.predict(X_test)

log_acc = accuracy_score(y_test, log_preds)

print("LogisticRegression:", log_acc)


# -----------------------------
# Random Forest + GridSearch
# -----------------------------

rf_model = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 15, 20],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    rf_model,
    param_grid,
    cv=3,
    n_jobs=-1
)

grid_search.fit(X_train_smote, y_train_smote)

best_rf = grid_search.best_estimator_

rf_preds = best_rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_preds)

print("RandomForest (GridSearch):", rf_acc)

print("Best RF Parameters:", grid_search.best_params_)


# -----------------------------
# XGBoost Model
# -----------------------------

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(X_train_smote, y_train_smote)

xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

xgb_preds = (xgb_probs > 0.4).astype(int)

xgb_acc = accuracy_score(y_test, xgb_preds)

xgb_auc = roc_auc_score(y_test, xgb_probs)

print("XGBoost:", xgb_acc)

print("XGBoost ROC-AUC:", xgb_auc)


# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_test, xgb_preds)

plt.figure(figsize=(6, 4))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


# -----------------------------
# ROC Curve
# -----------------------------

fpr, tpr, thresholds = roc_curve(y_test, xgb_probs)

plt.figure(figsize=(6, 4))

plt.plot(fpr, tpr, label="XGBoost (AUC = %.3f)" % xgb_auc)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()


# -----------------------------
# Save Feature Importance
# -----------------------------

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": xgb_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

feature_importance.to_csv(
    "models/feature_importance.csv",
    index=False
)

print("Feature importance saved")


# -----------------------------
# Save Model + Scaler
# -----------------------------

joblib.dump(xgb_model, "models/churn_model.pkl")

joblib.dump(scaler, "models/scaler.pkl")


# -----------------------------
# Save Feature Order
# -----------------------------

feature_list = list(X.columns)

with open("models/features.json", "w") as f:
    json.dump(feature_list, f)

print("Feature list saved")

print("Model training complete and saved.")
