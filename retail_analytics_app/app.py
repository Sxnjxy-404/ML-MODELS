import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import random

# ---------- Load Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    df.columns = df.columns.str.strip()  # Ensure no extra spaces
    return df

# ---------- Recommendation ----------
def recommend_products(user_id, data):
    user_purchases = data[data['user_id'] == user_id]['product_id'].unique()
    all_products = set(data['product_id'].unique())
    recommended_ids = list(all_products - set(user_purchases))
    sampled_ids = random.sample(recommended_ids, min(5, len(recommended_ids)))

    # Safely get product names
    product_lookup = data[['product_id', 'product_name']].drop_duplicates()
    recommended = product_lookup[product_lookup['product_id'].isin(sampled_ids)]
    return recommended

# ---------- Demand Forecast ----------
def forecast_demand(data):
    demand = data.groupby(['product_id', 'product_name']).size().reset_index(name='demand')
    return demand.sort_values(by='demand', ascending=False)

# ---------- Customer Segmentation ----------
def segment_customers(data):
    pivot = data.pivot_table(index='user_id', columns='product_id', aggfunc='size', fill_value=0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(pivot)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled)
    return pd.DataFrame({'user_id': pivot.index, 'segment': clusters})

# ---------- Price Optimization ----------
def optimize_prices(data):
    prices = data.groupby(['product_id', 'product_name']).agg({'price': 'mean', 'quantity': 'sum'}).reset_index()
    prices['optimized_price'] = prices['price'] * (1 + 0.1 * np.random.randn(len(prices)))
    return prices[['product_id', 'product_name', 'optimized_price']]

# ---------- Streamlit UI ----------
st.title("🛒 Retail Analytics Dashboard")

user_id = st.number_input("Enter User ID:", min_value=1, step=1)

if st.button("Analyze"):
    try:
        data = load_data()

        if user_id not in data['user_id'].unique():
            st.warning("❌ User ID not found in data.")
        else:
            recommendations = recommend_products(user_id, data)
            demand = forecast_demand(data)
            segments = segment_customers(data)
            prices = optimize_prices(data)

            st.subheader("🔮 Product Recommendations")
            st.dataframe(recommendations)

            st.subheader("📊 Forecasted Product Demand")
            st.dataframe(demand)

            st.subheader("👥 Customer Segmentation")
            st.dataframe(segments)

            st.subheader("💰 Optimized Prices")
            st.dataframe(prices)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
