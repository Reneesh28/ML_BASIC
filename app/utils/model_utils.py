import xgboost as xgb
import pandas as pd
import numpy as np
from utils.preprocess import add_time_features

# Load model (Booster)
def load_model(path="../models/xgb_demand_model.json"):
    model = xgb.Booster()
    model.load_model(path)
    return model

# Prepare input features
def prepare_input(store, item, date):
    df = pd.DataFrame({
        "date": [pd.to_datetime(date)],
        "store": [store],
        "item": [item]
    })

    df = add_time_features(df)

    feature_cols = [
        "store", "item",
        "year", "month", "dayofweek",
        "month_sin", "month_cos"
    ]

    df = df[feature_cols].astype("float32")

    return df

# Predict demand
def predict_demand(model, store, item, date):
    df = prepare_input(store, item, date)
    dmatrix = xgb.DMatrix(df)
    pred = model.predict(dmatrix)[0]
    return round(float(pred), 2)
