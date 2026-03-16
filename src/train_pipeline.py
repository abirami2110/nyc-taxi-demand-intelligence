import pandas as pd
from src.data_prep import load_trip_data, clean_trip_data, aggregate_zone_hour, save_processed
from src.features import build_features
from src.stat_models import run_poisson, run_negative_binomial
from src.ml_models import prepare_data, train_random_forest, train_xgboost, train_lightgbm, evaluate
from src.config import DATA_PROCESSED

def main():
    raw = load_trip_data("yellow_tripdata_2024-01.parquet")
    cleaned = clean_trip_data(raw)
    agg = aggregate_zone_hour(cleaned)
    save_processed(agg, "zone_hour_demand.parquet")

    model_df = build_features("zone_hour_demand.parquet", "model_data.parquet")

    poisson_model, poisson_metrics = run_poisson(model_df)
    nb_model, nb_metrics = run_negative_binomial(model_df)

    print("Poisson:", poisson_metrics)
    print("Negative Binomial:", nb_metrics)

    X_train, X_test, y_train, y_test = prepare_data(model_df)

    rf = train_random_forest(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)
    lgbm = train_lightgbm(X_train, y_train)

    for name, model in {"rf": rf, "xgb": xgb, "lgbm": lgbm}.items():
        preds = model.predict(X_test)
        print(name, evaluate(y_test, preds))

if __name__ == "__main__":
    main()