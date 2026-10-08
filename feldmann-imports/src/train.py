import pickle
import pandas as pd


DATA_PATH = "../data/RO_Detail.csv"
MODEL_PATH = "../model.pkl"
TRAIN_END = "2026-09-01"


EXPECTED_COLUMNS = [
    "ro_key",
    "created_at",
    "created_date",
    "created_hour",
    "weekday",
    "status",
    "ro_type",
    "has_appointment",
    "appointment_at",
    "promise_at",
    "closed_at",
    "days_open",
    "service_mode",
    "department",
    "advisor",
    "tech_count",
    "job_count",
    "non_voided_job_count",
    "billed_hours",
    "scheduled_hours",
    "has_oil_change",
    "warranty_claim_count",
    "customer_pay",
    "internal_pay",
    "warranty_pay",
    "is_comeback",
    "source",
    "external_source",
]


def load_data(path):
    df = pd.read_csv(path)

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["created_at"] = pd.to_datetime(df["created_at"])
    df["created_date"] = pd.to_datetime(df["created_date"])

    # Exclude sales-driven repair orders
    df = df[df["ro_type"] != "SALES"].copy()

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

    return hourly.sort_values(
        ["created_date", "created_hour"]
    ).reset_index(drop=True)


def train(df):
    hourly = build_hourly_demand(df)

    hourly["date"] = pd.to_datetime(hourly["created_date"])

    train_data = hourly[
        hourly["date"] < TRAIN_END
    ].copy()

    train_data["weekday"] = train_data["date"].dt.weekday

    model = (
        train_data
        .groupby(["weekday", "created_hour"])["demand"]
        .mean()
        .to_dict()
    )

    return model


if __name__ == "__main__":
    df = load_data(DATA_PATH)

    model = train(df)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved: {MODEL_PATH}")
    print(f"Entries: {len(model)}")
