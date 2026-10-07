import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_PATH = "../data/RO_Detail.csv"
MODEL_PATH = "../model.pkl"

EVAL_START = "2026-09-01"
EVAL_END = "2026-10-01"


def load_data(path):
    df = pd.read_csv(path)
    df["created_date"] = pd.to_datetime(df["created_date"])
    return df


def build_hourly_demand(df):
    df = df[df["has_appointment"] == True]

    hourly = (
        df.groupby(
            ["created_date", "created_hour"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "demand"})
    )

    return hourly


def predict(model, row):
    return model.get(
        (row["weekday"], row["created_hour"]),
        np.nan,
    )


if __name__ == "__main__":
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    df = load_data(DATA_PATH)
    hourly = build_hourly_demand(df)

    hourly["date"] = hourly["created_date"]
    hourly["weekday"] = hourly["date"].dt.weekday

    test = hourly[
        (hourly["date"] >= EVAL_START) &
        (hourly["date"] < EVAL_END)
    ].copy()

    test["prediction"] = test.apply(
        lambda row: predict(model, row),
        axis=1,
    )

    test = test.dropna(subset=["prediction"])

    mae = mean_absolute_error(
        test["demand"],
        test["prediction"],
    )

    rmse = np.sqrt(
        mean_squared_error(
            test["demand"],
            test["prediction"],
        )
    )

    print(f"Evaluation period: {EVAL_START} to {EVAL_END}")
    print(f"Test rows: {len(test):,}")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
