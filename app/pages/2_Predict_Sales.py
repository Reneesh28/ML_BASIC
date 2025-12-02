import streamlit as st
import pandas as pd
import xgboost as xgb
from utils.model_utils import load_model, predict_demand

st.title("🔮 Single-Day Sales Prediction")

st.write("Enter store, item, and date to predict sales.")

store = st.number_input("Store ID", min_value=1, max_value=10, value=1)
item = st.number_input("Item ID", min_value=1, max_value=50, value=1)
date = st.date_input("Select Date")

model = load_model("../models/xgb_demand_model.json")

if st.button("Predict Sales"):
    pred = predict_demand(model, store, item, str(date))
    st.success(f"Predicted Sales for {date}: **{pred}**")
