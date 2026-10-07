from pathlib import Path

import pandas as pd


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


def load_ro_detail(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    # Parse date/time fields
    datetime_columns = [
        "created_at",
        "appointment_at",
        "promise_at",
        "closed_at",
    ]

    for column in datetime_columns:
        df[column] = pd.to_datetime(df[column], errors="coerce")

    df["created_date"] = pd.to_datetime(
        df["created_date"], errors="coerce"
    )

    # Basic validation
    if df["ro_key"].isna().any():
        raise ValueError("ro_key contains missing values")

    if df["created_at"].isna().any():
        raise ValueError("created_at contains invalid/missing values")

    return df


if __name__ == "__main__":
    # df = load_ro_detail("data/RO_Detail.csv")
    df = load_ro_detail("data/mini_RO_Detail.csv")

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Date range: {df['created_at'].min()} → {df['created_at'].max()}")
    print()
    print(df.dtypes)
