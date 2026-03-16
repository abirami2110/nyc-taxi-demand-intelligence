import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.config import DATA_PROCESSED, MODELS
from src.utils import time_split

FEATURE_COLS = [
    "pickup_zone",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_mean_168",
    "rolling_std_24",
    "avg_trip_distance",
    "avg_fare",
]

TARGET = "demand_count"

def evaluate(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5
    }

def prepare_data(df):
    train, test = time_split(df, "2024-01-25")
    X_train = train[FEATURE_COLS]
    y_train = train[TARGET]
    X_test = test[FEATURE_COLS]
    y_test = test[TARGET]
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    model = XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train):
    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    df = pd.read_parquet(DATA_PROCESSED / "model_data.parquet")
    X_train, X_test, y_train, y_test = prepare_data(df)

    models = {
        "random_forest": train_random_forest(X_train, y_train),
        "xgboost": train_xgboost(X_train, y_train),
        "lightgbm": train_lightgbm(X_train, y_train),
    }

    for name, model in models.items():
        preds = model.predict(X_test)
        metrics = evaluate(y_test, preds)
        print(name, metrics)
        joblib.dump(model, MODELS / f"{name}.pkl")