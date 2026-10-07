from sklearn.ensemble import RandomForestRegressor

from data_loader import load_ro_detail
from model_features import build_model_features
from time_split import split_by_time


FEATURE_COLUMNS = [
    "created_hour",
    "weekday",
    "is_weekend",
    "month",
    "day_of_month",
]

def train_model(train):
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        train[FEATURE_COLUMNS],
        train["demand"],
    )

    return model


if __name__ == "__main__":
    df = load_ro_detail("../data/dev_RO_Detail.csv")

    features = build_model_features(df)

    train, test = split_by_time(features)

    train = train.dropna(subset=FEATURE_COLUMNS)
    test = test.dropna(subset=FEATURE_COLUMNS)

    model = train_model(train)

    test["prediction"] = model.predict(
        test[FEATURE_COLUMNS]
    )

    print(
        test[
            ["date", "created_hour", "demand", "prediction"]
        ].head(20).to_string(index=False)
    )
