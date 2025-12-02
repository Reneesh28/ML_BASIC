import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

st.title("📊 EDA Dashboard")

st.write("""
This dashboard provides insights into the Store Item Demand dataset.
""")

# Load data
df = pd.read_csv("../datasets/main/store_item_demand.csv", parse_dates=["date"])

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Sales Over Time (Sample)")
sample = df[df["store"] == 1].head(200)  # lighter for display

plt.figure(figsize=(12, 4))
plt.plot(sample["date"], sample["sales"])
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sample Sales Trend")
plt.tight_layout()
st.pyplot()

st.subheader("Correlation Heatmap")
plt.figure(figsize=(6, 4))
sns.heatmap(df[["store", "item", "sales"]].corr(), annot=True, cmap="coolwarm")
st.pyplot()