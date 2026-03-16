import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.config import DATA_PROCESSED
from src.utils import time_split

FEATURE_COLS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "lag_1",
    "lag_24",
    "rolling_mean_24",
    "rolling_mean_168",
    "avg_trip_distance",
    "avg_fare",
]

def evaluate(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5
    }

def run_poisson(df: pd.DataFrame):
    train, test = time_split(df, "2024-01-25")

    formula = "demand_count ~ " + " + ".join(FEATURE_COLS)
    poisson_model = smf.glm(
        formula=formula,
        data=train,
        family=sm.families.Poisson()
    ).fit()

    preds = poisson_model.predict(test)
    metrics = evaluate(test["demand_count"], preds)

    return poisson_model, metrics

def run_negative_binomial(df: pd.DataFrame):
    train, test = time_split(df, "2024-01-25")

    formula = "demand_count ~ " + " + ".join(FEATURE_COLS)
    nb_model = smf.glm(
        formula=formula,
        data=train,
        family=sm.families.NegativeBinomial()
    ).fit()

    preds = nb_model.predict(test)
    metrics = evaluate(test["demand_count"], preds)

    return nb_model, metrics

if __name__ == "__main__":
    df = pd.read_parquet(DATA_PROCESSED / "model_data.parquet")

    poisson_model, poisson_metrics = run_poisson(df)
    nb_model, nb_metrics = run_negative_binomial(df)

    print("Poisson:", poisson_metrics)
    print("Negative Binomial:", nb_metrics)