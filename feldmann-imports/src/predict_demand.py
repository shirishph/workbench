import sys
import pandas as pd
from data_loader import load_ro_detail
from hourly_demand import build_hourly_demand


def predict_demand(df, target_date):
    hourly = build_hourly_demand(df)
    hourly["date"] = pd.to_datetime(hourly["created_date"])

    target = pd.Timestamp(target_date)

    history = hourly[
        (hourly["date"] < target) &
        (hourly["date"].dt.weekday == target.weekday()) &
        (hourly["created_hour"].between(7, 18))
    ]

    predictions = (
        history
        .groupby("created_hour")["demand"]
        .mean()
        .round()
        .astype(int)
        .reset_index(name="appointment_cap")
    )

    return predictions


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_demand.py YYYY-MM-DD")
        sys.exit(1)

    df = load_ro_detail("../data/RO_Detail.csv")
    predictions = predict_demand(df, sys.argv[1])

    print(predictions.to_string(index=False))
