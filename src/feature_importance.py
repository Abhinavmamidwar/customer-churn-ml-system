import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from preprocess import load_data


# -----------------------------
# Load Dataset
# -----------------------------

df = load_data()


# -----------------------------
# Feature Engineering
# -----------------------------

df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

df["ChargesPerTenure"] = df["MonthlyCharges"] * df["tenure"]


# -----------------------------
# One Hot Encoding
# -----------------------------

df = pd.get_dummies(df, drop_first=True)


# -----------------------------
# Split Features & Target
# -----------------------------

X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]


# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# Feature Scaling
# -----------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# -----------------------------
# Logistic Regression
# -----------------------------

log_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

log_model.fit(X_train, y_train)

log_preds = log_model.predict(X_test)

log_acc = accuracy_score(y_test, log_preds)

print("LogisticRegression:", log_acc)


# -----------------------------
# Random Forest + GridSearch
# -----------------------------

rf = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)

param_grid = {

    "n_estimators": [200, 300, 400],

    "max_depth": [5, 10, 15],

    "min_samples_split": [2, 5, 10]

}

grid_rf = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_

rf_preds = best_rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_preds)

print("RandomForest (GridSearch):", rf_acc)

print("Best RF Parameters:", grid_rf.best_params_)


# -----------------------------
# XGBoost
# -----------------------------

xgb_model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)

xgb_preds = xgb_model.predict(X_test)

xgb_acc = accuracy_score(y_test, xgb_preds)

print("XGBoost:", xgb_acc)


# -----------------------------
# Choose Best Model
# -----------------------------

models = {
    "Logistic": (log_model, log_acc),
    "RandomForest": (best_rf, rf_acc),
    "XGBoost": (xgb_model, xgb_acc)
}

best_model = None
best_score = 0

for name, (model, score) in models.items():

    if score > best_score:

        best_score = score

        best_model = model


# -----------------------------
# Save Best Model
# -----------------------------

joblib.dump(best_model, "models/churn_model.pkl")

print("\nBest Model Saved")
