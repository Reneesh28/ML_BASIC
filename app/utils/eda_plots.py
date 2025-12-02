import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_sales_over_time(df):
    fig, ax = plt.subplots(figsize=(14,4))
    ax.plot(df['date'], df['sales'])
    ax.set_title("Sales Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    plt.tight_layout()
    return fig

def plot_sales_by_season(df):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(data=df, x='season', y='sales', ax=ax)
    ax.set_title("Sales by Season")
    plt.tight_layout()
    return fig

def plot_monthly_average(df):
    monthly_avg = df.groupby("month")["sales"].mean()
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(monthly_avg.index, monthly_avg.values, marker="o")
    ax.set_title("Average Sales per Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg Sales")
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_store_item_trend(df, store, item):
    subset = df[(df['store'] == store) & (df['item'] == item)]
    fig, ax = plt.subplots(figsize=(14,4))
    ax.plot(subset['date'], subset['sales'])
    ax.set_title(f"Sales Trend for Store {store}, Item {item}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    plt.tight_layout()
    return fig
