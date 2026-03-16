import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PROCESSED

def build_zone_features(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["hour"] = pd.to_datetime(temp["pickup_hour"]).dt.hour
    temp["is_peak"] = temp["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    temp["is_weekend"] = pd.to_datetime(temp["pickup_hour"]).dt.dayofweek.isin([5, 6]).astype(int)

    zone_features = (
        temp.groupby("pickup_zone")
        .agg(
            avg_demand=("demand_count", "mean"),
            avg_fare=("avg_fare", "mean"),
            avg_distance=("avg_trip_distance", "mean"),
            peak_share=("is_peak", "mean"),
            weekend_share=("is_weekend", "mean"),
        )
        .reset_index()
    )
    return zone_features

def run_kmeans(zone_features: pd.DataFrame, n_clusters: int = 4):
    feature_cols = ["avg_demand", "avg_fare", "avg_distance", "peak_share", "weekend_share"]

    scaler = StandardScaler()
    X = scaler.fit_transform(zone_features[feature_cols])

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    zone_features["cluster"] = model.fit_predict(X)

    return zone_features, model

def detect_hotspots(df: pd.DataFrame, top_n: int = 20):
    hotspot_df = (
        df.groupby("pickup_zone")["demand_count"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={"demand_count": "avg_hourly_demand"})
    )
    return hotspot_df

if __name__ == "__main__":
    df = pd.read_parquet(DATA_PROCESSED / "zone_hour_demand.parquet")
    zone_features = build_zone_features(df)
    clustered, _ = run_kmeans(zone_features)
    hotspots = detect_hotspots(df)

    print(clustered.head())
    print(hotspots.head())