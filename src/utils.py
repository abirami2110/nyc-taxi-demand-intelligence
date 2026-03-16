import pandas as pd

def time_split(df: pd.DataFrame, cutoff_date: str):
    df = df.copy()
    df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])

    train = df[df["pickup_hour"] < cutoff_date].copy()
    test = df[df["pickup_hour"] >= cutoff_date].copy()

    return train, test