import pandas as pd


def build_hourly_demand(df):
    hourly = (
        df.groupby(["created_date", "created_hour"], as_index=False)
        .size()
        .rename(columns={"size": "demand"})
    )

    return hourly.sort_values(
        ["created_date", "created_hour"]
    ).reset_index(drop=True)
