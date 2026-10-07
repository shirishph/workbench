import pandas as pd

from data_loader import load_ro_detail
from model_features import build_model_features
from time_split import split_by_time


def predict_baseline(train, test):
    historical_mean = (
        train.groupby(["weekday", "created_hour"])["demand"]
        .mean()
        .rename("prediction")
        .reset_index()
    )

    predictions = test.merge(
        historical_mean,
        on=["weekday", "created_hour"],
        how="left",
    )

    return predictions


if __name__ == "__main__":
    df = load_ro_detail("../data/RO_Detail.csv")

    features = build_model_features(df)

    train, test = split_by_time(features)

    predictions = predict_baseline(train, test)

    print(predictions[
        [
            "date",
            "created_hour",
            "weekday",
            "demand",
            "prediction",
        ]
    ].head(20).to_string(index=False))
