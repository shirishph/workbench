import pickle
import pandas as pd


DATA_PATH = "../data/RO_Detail.csv"
MODEL_PATH = "../model.pkl"

EVAL_START = "2026-09-01"
EVAL_END = "2026-10-01"


def load_data(path):
    df = pd.read_csv(path)
    df["created_date"] = pd.to_datetime(df["created_date"])
    return df


def build_hourly_demand(df):
    df = df[
        (df["has_appointment"] == True) &
        (df["created_date"] >= EVAL_START) &
        (df["created_date"] < EVAL_END)
    ]

    return (
        df.groupby(
            ["created_date", "created_hour"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "appointments"})
    )


if __name__ == "__main__":
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    df = load_data(DATA_PATH)
    hourly = build_hourly_demand(df)

    # Randomly select one September observation
    selected = hourly.sample(n=1).iloc[0]

    selected_date = selected["created_date"]
    selected_hour = int(selected["created_hour"])
    actual = int(selected["appointments"])

    prediction = model.get(
        (selected_date.weekday(), selected_hour)
    )

    # Show all September data for the selected date
    day_data = hourly[
        hourly["created_date"] == selected_date
    ].copy()

    day_data = day_data.sort_values("created_hour")

    print()
    print("HUMAN EVALUATION")
    print("=" * 50)

    print(
        f"Selected September date: {selected_date.date()}"
    )
    print(
        f"Selected hour:           {selected_hour}:00"
    )

    print()
    print("SEPTEMBER DATA — SELECTED DATE")
    print("-" * 50)

    for _, row in day_data.iterrows():
        hour = int(row["created_hour"])
        appointments = int(row["appointments"])

        if hour == selected_hour:
            print(
                f">>> {hour:02d}:00  "
                f"APPOINTMENTS: {appointments}  <<<"
            )
        else:
            print(
                f"    {hour:02d}:00  "
                f"appointments: {appointments}"
            )

    print()
    print("SELECTED HOUR")
    print("-" * 50)
    print(f"Actual appointments:   {actual}")
    print(f"Predicted appointments: {prediction:.2f}")

    print()
    print(
        f"Prediction error:      "
        f"{actual - prediction:.2f}"
    )
