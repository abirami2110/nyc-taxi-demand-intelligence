import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.config import DATA_PROCESSED

def evaluate(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5
    }

def get_zone_series(df: pd.DataFrame, zone_id: int) -> pd.DataFrame:
    zone_df = (
        df[df["pickup_zone"] == zone_id]
        .sort_values("pickup_hour")[["pickup_hour", "demand_count"]]
        .copy()
    )
    return zone_df

def run_arima(zone_df: pd.DataFrame, train_size: float = 0.8):
    n = len(zone_df)
    split = int(n * train_size)

    train = zone_df.iloc[:split]
    test = zone_df.iloc[split:]

    model = ARIMA(train["demand_count"], order=(2, 1, 2))
    fitted = model.fit()

    forecast = fitted.forecast(steps=len(test))
    metrics = evaluate(test["demand_count"], forecast)

    return fitted, forecast, metrics, train, test

def run_prophet(zone_df: pd.DataFrame, train_size: float = 0.8):
    prophet_df = zone_df.rename(columns={"pickup_hour": "ds", "demand_count": "y"})
    n = len(prophet_df)
    split = int(n * train_size)

    train = prophet_df.iloc[:split]
    test = prophet_df.iloc[split:]

    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(train)

    future = test[["ds"]].copy()
    forecast = model.predict(future)

    metrics = evaluate(test["y"], forecast["yhat"])
    return model, forecast, metrics, train, test

if __name__ == "__main__":
    df = pd.read_parquet(DATA_PROCESSED / "zone_hour_demand.parquet")
    zone_df = get_zone_series(df, zone_id=132)

    _, _, arima_metrics, _, _ = run_arima(zone_df)
    _, _, prophet_metrics, _, _ = run_prophet(zone_df)

    print("ARIMA:", arima_metrics)
    print("Prophet:", prophet_metrics)