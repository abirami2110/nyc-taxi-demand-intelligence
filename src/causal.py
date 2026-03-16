import pandas as pd
import statsmodels.formula.api as smf
from src.config import DATA_PROCESSED
import numpy as np
from scipy.optimize import minimize

def prepare_did_data(df: pd.DataFrame, treated_zones: list, intervention_date: str):
    did_df = df.copy()
    did_df["pickup_hour"] = pd.to_datetime(did_df["pickup_hour"])
    did_df["treated"] = did_df["pickup_zone"].isin(treated_zones).astype(int)
    did_df["post"] = (did_df["pickup_hour"] >= intervention_date).astype(int)
    did_df["treated_post"] = did_df["treated"] * did_df["post"]
    did_df["hour"] = did_df["pickup_hour"].dt.hour
    did_df["day_of_week"] = did_df["pickup_hour"].dt.dayofweek
    return did_df

def run_did(did_df: pd.DataFrame):
    model = smf.ols(
        "demand_count ~ treated + post + treated_post + C(hour) + C(day_of_week)",
        data=did_df
    ).fit(cov_type="HC3")
    return model

if __name__ == "__main__":
    df = pd.read_parquet(DATA_PROCESSED / "zone_hour_demand.parquet")

    treated_zones = [132, 138]
    did_df = prepare_did_data(df, treated_zones, "2024-01-15")

    model = run_did(did_df)
    print(model.summary())

def synthetic_control_weights(X_control, X_treated):
    n_controls = X_control.shape[1]

    def objective(w):
        synth = X_control @ w
        return np.sum((X_treated - synth) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n_controls)]
    w0 = np.ones(n_controls) / n_controls

    result = minimize(objective, w0, bounds=bounds, constraints=constraints)
    return result.x

def run_synthetic_control(df: pd.DataFrame, treated_zone: int, control_zones: list, intervention_date: str):
    df = df.copy()
    df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])

    pre = df[df["pickup_hour"] < intervention_date]
    post = df[df["pickup_hour"] >= intervention_date]

    treated_pre = (
        pre[pre["pickup_zone"] == treated_zone]
        .sort_values("pickup_hour")["demand_count"]
        .values
    )

    control_pre = []
    for zone in control_zones:
        series = (
            pre[pre["pickup_zone"] == zone]
            .sort_values("pickup_hour")["demand_count"]
            .values
        )
        control_pre.append(series)

    X_control = np.column_stack(control_pre)
    weights = synthetic_control_weights(X_control, treated_pre)

    treated_post = (
        post[post["pickup_zone"] == treated_zone]
        .sort_values("pickup_hour")[["pickup_hour", "demand_count"]]
        .reset_index(drop=True)
    )

    control_post = []
    for zone in control_zones:
        series = (
            post[post["pickup_zone"] == zone]
            .sort_values("pickup_hour")["demand_count"]
            .values
        )
        control_post.append(series)

    synth_post = np.column_stack(control_post) @ weights
    treated_post["synthetic_demand"] = synth_post
    treated_post["effect"] = treated_post["demand_count"] - treated_post["synthetic_demand"]

    return treated_post, weights