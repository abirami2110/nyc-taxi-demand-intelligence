import pandas as pd
from src.config import DATA_PROCESSED

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])

    df["hour"] = df["pickup_hour"].dt.hour
    df["day"] = df["pickup_hour"].dt.day
    df["day_of_week"] = df["pickup_hour"].dt.dayofweek
    df["month"] = df["pickup_hour"].dt.month
    df["weekofyear"] = df["pickup_hour"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df

def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["pickup_zone", "pickup_hour"]).copy()

    for lag in [1, 2, 24, 168]:
        df[f"lag_{lag}"] = df.groupby("pickup_zone")["demand_count"].shift(lag)

    df["rolling_mean_24"] = (
        df.groupby("pickup_zone")["demand_count"]
        .transform(lambda x: x.shift(1).rolling(24).mean())
    )
    df["rolling_mean_168"] = (
        df.groupby("pickup_zone")["demand_count"]
        .transform(lambda x: x.shift(1).rolling(168).mean())
    )
    df["rolling_std_24"] = (
        df.groupby("pickup_zone")["demand_count"]
        .transform(lambda x: x.shift(1).rolling(24).std())
    )

    return df

def build_features(input_file="zone_hour_demand.parquet", output_file="model_data.parquet"):
    df = pd.read_parquet(DATA_PROCESSED / input_file)
    df = create_time_features(df)
    df = create_lag_features(df)
    df = df.dropna().reset_index(drop=True)
    df.to_parquet(DATA_PROCESSED / output_file, index=False)
    return df

if __name__ == "__main__":
    df = build_features()
    print(df.head())