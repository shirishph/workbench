import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_loader import load_ro_detail
from model_features import build_model_features
from time_split import split_by_time
from ml_model import FEATURE_COLUMNS, train_model


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

    actual = test["demand"]
    predicted = test["prediction"]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    print(f"Test rows: {len(test):,}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
