import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from utils.preprocess import add_time_features
from utils.model_utils import load_model

# ---------------------------------------------------
# Page title
# ---------------------------------------------------
st.title("📈 Future Sales Forecasting")

st.write("""
Generate future sales predictions using the XGBoost demand forecasting model.
""")

# ---------------------------------------------------
# Helper Functions (converted from Notebook 03)
# ---------------------------------------------------
def generate_future_dates(start_date, periods=30):
    start_date = pd.to_datetime(start_date)
    return pd.date_range(start=start_date + pd.Timedelta(days=1), periods=periods)


def create_future_feature_dataframe(future_dates, store, item):
    df = pd.DataFrame({
        "date": future_dates,
        "store": store,
        "item": item
    })

    df = add_time_features(df)

    feature_cols = [
        "store", "item",
        "year", "month", "dayofweek",
        "month_sin", "month_cos"
    ]

    return df[feature_cols].astype("float32")


def generate_forecast(store, item, start_date, periods=30):
    future_dates = generate_future_dates(start_date, periods)
    feature_df = create_future_feature_dataframe(future_dates, store, item)

    model = load_model("../models/xgb_demand_model.json")

    dmatrix = xgb.DMatrix(feature_df)
    predictions = model.predict(dmatrix)

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "store": store,
        "item": item,
        "predicted_sales": predictions
    })

    forecast_df["predicted_sales"] = forecast_df["predicted_sales"].round(2)
    forecast_df["day_name"] = forecast_df["date"].dt.day_name()

    return forecast_df


# ---------------------------------------------------
# UI Inputs
# ---------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    store = st.number_input("Store ID", min_value=1, max_value=10, value=1)

with col2:
    item = st.number_input("Item ID", min_value=1, max_value=50, value=1)

start_date = st.date_input("Forecast Start Date")
periods = st.slider("Forecast Duration (Days)", 7, 365, 30)


# ---------------------------------------------------
# Forecast Button
# ---------------------------------------------------
if st.button("Generate Forecast"):
    st.success("Forecast generated successfully!")

    forecast_df = generate_forecast(store, item, str(start_date), periods)

    # -------------------------
    # Display Forecast Table
    # -------------------------
    st.subheader("📄 Forecast Table")
    st.dataframe(forecast_df)

    # -------------------------
    # Plot Forecast
    # -------------------------
    st.subheader("📈 Forecast Plot")

    plt.figure(figsize=(12, 5))
    plt.plot(forecast_df["date"], forecast_df["predicted_sales"], marker="o")
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Predicted Sales")
    plt.title(f"Forecast: Store {store}, Item {item}")
    plt.grid(True)
    st.pyplot()

    # -------------------------
    # CSV Download Button
    # -------------------------
    csv_data = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv_data,
        file_name=f"forecast_store{store}_item{item}.csv",
        mime="text/csv"
    )
