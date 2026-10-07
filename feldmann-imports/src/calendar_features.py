import pandas as pd

from data_loader import load_ro_detail
from hourly_demand import build_hourly_demand


def add_calendar_features(hourly):
    hourly = hourly.copy()

    hourly["date"] = pd.to_datetime(hourly["created_date"])

    hourly["weekday"] = hourly["date"].dt.weekday
    hourly["is_weekend"] = (hourly["weekday"] >= 5).astype(int)
    hourly["month"] = hourly["date"].dt.month
    hourly["day_of_month"] = hourly["date"].dt.day

    return hourly


if __name__ == "__main__":
    df = load_ro_detail("data/mini_RO_Detail.csv")

    hourly = build_hourly_demand(df)
    dataset = add_calendar_features(hourly)

    print(dataset.to_string(index=False))
