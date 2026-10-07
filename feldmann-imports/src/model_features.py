import pandas as pd
from hourly_demand import build_hourly_demand
from calendar_features import add_calendar_features

def build_model_features(df):
    hourly = build_hourly_demand(df)
    hourly = add_calendar_features(hourly)

    return hourly[
        [
            "date",
            "created_hour",
            "weekday",
            "is_weekend",
            "month",
            "day_of_month",
            "demand",
        ]
    ].copy()
