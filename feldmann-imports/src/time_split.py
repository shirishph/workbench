import pandas as pd

from data_loader import load_ro_detail
from model_features import build_model_features


TEST_FRACTION = 0.20


def split_by_time(df):
    dates = sorted(df["date"].unique())

    split_index = int(len(dates) * (1 - TEST_FRACTION))
    split_date = dates[split_index]

    train = df[df["date"] < split_date].copy()
    test = df[df["date"] >= split_date].copy()

    return train, test


if __name__ == "__main__":
    df = load_ro_detail("../data/RO_Detail.csv")

    features = build_model_features(df)

    train, test = split_by_time(features)

    print(f"Train rows: {len(train):,}")
    print(f"Train range: {train['date'].min()} → {train['date'].max()}")
    print()
    print(f"Test rows: {len(test):,}")
    print(f"Test range: {test['date'].min()} → {test['date'].max()}")
