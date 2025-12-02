# -------------------------
# Cell 1 — Imports + Load Model Utilities
# -------------------------
import sys
sys.path.append("..")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from utils.preprocess import add_time_features
from utils.model_utils import load_model, predict_demand

print("Notebook 03 ready — model & utilities imported successfully.")


# -------------------------
# Cell 2 — User Input + Generate Future Date Range
# -------------------------

def generate_future_dates(start_date, periods=30):
    """
    Generate a list of future dates starting from start_date.

    Parameters:
    - start_date: a starting date (string or datetime)
    - periods: number of future days to forecast
    """
    start_date = pd.to_datetime(start_date)
    future_dates = pd.date_range(start=start_date + pd.Timedelta(days=1), periods=periods)
    return future_dates


# -------- USER INPUT HERE --------
# Example user inputs (these will be replaced by Streamlit later)
user_start_date = input("Enter starting date (YYYY-MM-DD): ")
user_periods = int(input("Enter number of days to forecast (e.g., 30, 60, 90): "))

# Generate the forecast dates
future_dates = generate_future_dates(user_start_date, user_periods)

print("\nFuture Dates Generated:")
print(future_dates[:10])
print(f"\nTotal future days generated: {len(future_dates)}")


# -------------------------
# Cell 3 — Add Time Features to Future Dates
# -------------------------

def create_future_feature_dataframe(future_dates, store, item):
    """
    Convert future dates into the same feature structure used for XGBoost.
    Includes: store, item, year, month, dayofweek, month_sin, month_cos
    """
    
    df = pd.DataFrame({
        "date": future_dates,
        "store": store,   # constant store id
        "item": item      # constant item id
    })

    # Add time-based features
    df = add_time_features(df)

    # Keep only model-required columns
    feature_cols = [
        "store", "item",
        "year", "month", "dayofweek",
        "month_sin", "month_cos"
    ]

    df = df[feature_cols].astype("float32")
    return df


# -------- USER INPUT for store + item (will be replaced by Streamlit later) --------
user_store = int(input("Enter store id (e.g., 1–10): "))
user_item = int(input("Enter item id (e.g., 1–50): "))

# Create future dataframe with time features
future_df = create_future_feature_dataframe(future_dates, user_store, user_item)

print("\nFuture Feature DataFrame Preview:")
future_df.head()


# -------------------------
# Cell 4 — Generate Forecast Using XGBoost
# -------------------------

# Load the trained model
model = load_model("../models/xgb_demand_model.json")

# Convert future_df into DMatrix (Booster requires this)
dmatrix_future = xgb.DMatrix(future_df)

# Predict future demand
future_predictions = model.predict(dmatrix_future)

# Round predictions for readability
future_predictions = np.round(future_predictions, 2)

print("\nForecast Preview:")
future_predictions[:10]


# -------------------------
# Cell 5 — Build Forecast DataFrame
# -------------------------

forecast_df = pd.DataFrame({
    "date": future_dates,
    "store": user_store,
    "item": user_item,
    "predicted_sales": future_predictions
})

print("Forecast Table Preview:")
forecast_df.head()

# -------------------------
# Cell 6 — Plot Forecast
# -------------------------

plt.figure(figsize=(12, 6))
plt.plot(forecast_df["date"], forecast_df["predicted_sales"], marker="o")

plt.title(f"Forecasted Sales for Store {user_store}, Item {user_item}", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Predicted Sales", fontsize=12)
plt.grid(True)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# -------------------------
# Cell 7 — Export Forecast as CSV
# -------------------------

output_path = f"../datasets/forecast_store{user_store}_item{user_item}.csv"

forecast_df.to_csv(output_path, index=False)

print(f"Forecast CSV saved successfully at: {output_path}")

# -------------------------
# Cell 8 — Build Forecast Summary Table
# -------------------------

# Add day name for readability
forecast_df["day_name"] = forecast_df["date"].dt.day_name()

summary_df = forecast_df[[
    "date",
    "day_name",
    "store",
    "item",
    "predicted_sales"
]]

print("Forecast Summary Preview:")
summary_df.head(10)

# -------------------------
# Cell 9 — End-to-End Forecast Function
# -------------------------

def generate_forecast(store, item, start_date, periods=30):
    """
    Full forecasting pipeline:
    - Generate future dates
    - Create features
    - Predict using XGBoost
    - Return a clean forecast dataframe
    """

    # Step 1: Create future dates
    future_dates = generate_future_dates(start_date, periods)

    # Step 2: Create feature dataframe
    future_df = create_future_feature_dataframe(future_dates, store, item)

    # Step 3: Load model
    model = load_model("../models/xgb_demand_model.json")

    # Step 4: Convert to DMatrix + predict
    dmatrix_future = xgb.DMatrix(future_df)
    predictions = model.predict(dmatrix_future)

    # Step 5: Build final output dataframe
    forecast_df = pd.DataFrame({
        "date": future_dates,
        "store": store,
        "item": item,
        "predicted_sales": np.round(predictions, 2)
    })

    forecast_df["day_name"] = forecast_df["date"].dt.day_name()

    return forecast_df


# -------------------------
# Test the full pipeline
# -------------------------

test_forecast = generate_forecast(
    store=user_store,
    item=user_item,
    start_date=user_start_date,
    periods=user_periods
)

print("Full Forecast Pipeline Test:")
test_forecast.head()
