import joblib
import pandas as pd
import matplotlib.pyplot as plt

from preprocess import load_data


df = load_data()

df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn_Yes", axis=1)

model = joblib.load("models/churn_model.pkl")

importances = model.feature_importances_

features = X.columns

feat_imp = pd.Series(importances, index=features)

feat_imp.nlargest(10).plot(kind="barh")

plt.title("Top 10 Important Features")

plt.show()
