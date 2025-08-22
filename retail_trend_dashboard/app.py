import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Store Sales Trend Dashboard", layout="wide")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("sales_data.csv", parse_dates=["date"])
    df.columns = df.columns.str.strip()
    return df

# ---------- Detect Trend ----------
def detect_trend(df):
    trends = []
    for store in df['store_id'].unique():
        store_df = df[df['store_id'] == store].sort_values('date')
        X = (store_df['date'] - store_df['date'].min()).dt.days.values.reshape(-1, 1)
        y = store_df['sales'].values
        if len(X) >= 2:
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
            trends.append({'store_id': store, 'slope': slope})
    return pd.DataFrame(trends)

# ---------- Plot Trend (Short Style) ----------
def plot_store_trend(store_df, slope):
    store_df = store_df.sort_values('date')
    X = (store_df['date'] - store_df['date'].min()).dt.days.values.reshape(-1, 1)
    y = store_df['sales'].values

    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    fig, ax = plt.subplots(figsize=(6, 3))  # Smaller plot
    ax.plot(store_df['date'], y, marker='o', label='Actual Sales')
    ax.plot(store_df['date'], y_pred, linestyle='--', label='Trend Line')

    ax.set_title(f"Store {store_df['store_id'].iloc[0]} (Slope: {slope:.2f})", fontsize=12)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Sales", fontsize=10)
    ax.tick_params(axis='x', labelrotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(fontsize=8)
    plt.tight_layout()

    st.pyplot(fig)

# ---------- Main App ----------
st.title("📈 Store Sales Trend Monitoring")
st.markdown("Detect stores showing **increasing** sales trends using linear regression.")

df = load_data()
trend_df = detect_trend(df)
positive_trend_stores = trend_df[trend_df['slope'] > 0].sort_values(by='slope', ascending=False)

st.sidebar.header("🔍 Filter Stores")
store_selection = st.sidebar.selectbox("Select a Store", options=sorted(df['store_id'].unique()))

st.subheader("🟢 Stores with Upward Trends")
st.dataframe(positive_trend_stores)

store_df = df[df['store_id'] == store_selection]
store_slope = trend_df[trend_df['store_id'] == store_selection]['slope'].values[0]

st.subheader(f"📊 Sales Trend for Store {store_selection}")
plot_store_trend(store_df, store_slope)
