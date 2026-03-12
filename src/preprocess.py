import pandas as pd

def load_data():

    df = pd.read_csv("data/telco_churn.csv")

    # convert to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # fix missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    # drop customerID
    df.drop("customerID", axis=1, inplace=True)

    return df
