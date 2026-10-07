import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_loader import load_ro_detail
from model_features import build_model_features
from time_split import split_by_time
from baseline_model import predict_baseline


if __name__ == "__main__":
    df = load_ro_detail("../data/RO_Detail.csv")

    features = build_model_features(df)

    train, test = split_by_time(features)

    predictions = predict_baseline(train, test)

    predictions = predictions.dropna(subset=["prediction"])

    actual = predictions["demand"]
    predicted = predictions["prediction"]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    print(f"Test rows: {len(predictions):,}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
