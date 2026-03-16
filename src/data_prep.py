import pandas as pd
from pathlib import Path
from src.config import DATA_RAW, DATA_PROCESSED

def load_trip_data(file_name: str) -> pd.DataFrame:
    file_path = DATA_RAW / file_name
    df = pd.read_parquet(file_path)
    return df

def clean_trip_data(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "tpep_pickup_datetime",
        "PULocationID",
        "trip_distance",
        "fare_amount",
        "total_amount"
    ]
    df = df[keep_cols].copy()

    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df = df.dropna(subset=["tpep_pickup_datetime", "PULocationID"])
    df = df[df["trip_distance"] >= 0]
    df = df[df["fare_amount"] >= 0]
    df = df[df["total_amount"] >= 0]

    df["pickup_zone"] = df["PULocationID"].astype(int)
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.floor("h")

    return df

def aggregate_zone_hour(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["pickup_zone", "pickup_hour"])
        .agg(
            demand_count=("pickup_zone", "size"),
            avg_trip_distance=("trip_distance", "mean"),
            avg_fare=("fare_amount", "mean"),
            avg_total_amount=("total_amount", "mean"),
        )
        .reset_index()
    )
    return agg

def save_processed(df: pd.DataFrame, file_name: str = "zone_hour_demand.parquet") -> None:
    output_path = DATA_PROCESSED / file_name
    df.to_parquet(output_path, index=False)

if __name__ == "__main__":
    df = load_trip_data("yellow_tripdata_2024-01.parquet")
    df = clean_trip_data(df)
    agg = aggregate_zone_hour(df)
    save_processed(agg)
    print(agg.head())