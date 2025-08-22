import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Return Risk Assessment",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.high-risk {
    background-color: #ffebee;
    border-left: 5px solid #f44336;
}
.medium-risk {
    background-color: #fff8e1;
    border-left: 5px solid #ff9800;
}
.low-risk {
    background-color: #e8f5e8;
    border-left: 5px solid #4caf50;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_components():
    """Load trained model components"""
    try:
        model = joblib.load('return_prediction_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        return model, scaler, label_encoder, feature_columns, True
    except:
        return None, None, None, None, False

def create_sample_model():
    """Create a simple mock model for demo purposes"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    # Create mock model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    
    # Fit with dummy data
    X_dummy = np.random.randn(100, 20)
    y_dummy = np.random.choice([0, 1], 100)
    model.fit(X_dummy, y_dummy)
    scaler.fit(X_dummy)
    label_encoder.fit(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'])
    
    feature_columns = [
        'customer_age', 'customer_loyalty_years', 'order_value', 'discount_amount',
        'shipping_cost', 'product_rating', 'num_reviews', 'is_new_product',
        'previous_orders', 'previous_returns', 'days_since_last_order',
        'order_month', 'is_weekend', 'product_category_encoded',
        'customer_return_rate', 'discount_percentage', 'price_per_review',
        'is_holiday_season', 'is_summer', 'customer_activity_score'
    ]
    
    return model, scaler, label_encoder, feature_columns

def preprocess_input(data, label_encoder):
    """Preprocess input data for prediction"""
    df = pd.DataFrame([data])
    
    # Feature engineering
    df['customer_return_rate'] = df['previous_returns'] / (df['previous_orders'] + 1)
    df['discount_percentage'] = df['discount_amount'] / df['order_value']
    df['price_per_review'] = df['order_value'] / (df['num_reviews'] + 1)
    df['is_holiday_season'] = ((df['order_month'] == 11) | (df['order_month'] == 12)).astype(int)
    df['is_summer'] = ((df['order_month'] >= 6) & (df['order_month'] <= 8)).astype(int)
    df['customer_activity_score'] = df['previous_orders'] / (df['days_since_last_order'] / 30 + 1)
    
    # Encode category
    try:
        df['product_category_encoded'] = label_encoder.transform([data['product_category']])[0]
    except:
        df['product_category_encoded'] = 0
    
    return df

def predict_return_risk(model, scaler, label_encoder, feature_columns, customer_data):
    """Predict return risk with mock calculation for demo"""
    
    # For demo purposes, calculate risk based on simple heuristics
    risk_score = 0.1  # Base risk
    
    # Add risk factors
    if customer_data['product_category'] == 'Clothing':
        risk_score += 0.15
    elif customer_data['product_category'] == 'Electronics':
        risk_score += 0.08
    
    if customer_data['order_value'] < 30:
        risk_score += 0.1
    
    if customer_data['product_rating'] < 3.5:
        risk_score += 0.2
    
    if customer_data['is_new_product']:
        risk_score += 0.08
    
    if customer_data['previous_orders'] > 0:
        return_rate = customer_data['previous_returns'] / customer_data['previous_orders']
        risk_score += return_rate * 0.3
    
    if customer_data['discount_amount'] / customer_data['order_value'] > 0.5:
        risk_score += 0.12
    
    if customer_data['customer_age'] < 25:
        risk_score += 0.05
    
    if customer_data['days_since_last_order'] > 180:
        risk_score += 0.07
    
    probability = min(risk_score, 0.8)
    
    if probability > 0.3:
        risk_level = 'High'
        color = '#f44336'
    elif probability > 0.15:
        risk_level = 'Medium'
        color = '#ff9800'
    else:
        risk_level = 'Low'
        color = '#4caf50'
    
    return probability, risk_level, color

def main():
    st.title("📦 Return Risk Assessment Tool")
    st.markdown("**Customer Service Dashboard for Predicting Product Return Likelihood**")
    
    # Load model
    model, scaler, label_encoder, feature_columns, model_loaded = load_model_components()
    
    if not model_loaded:
        st.warning("⚠️ Pre-trained model not found. Using demo mode with simplified predictions.")
        model, scaler, label_encoder, feature_columns = create_sample_model()
    
    # Sidebar for input
    st.sidebar.header("📋 Order Information")
    
    # Customer Information
    st.sidebar.subheader("Customer Details")
    customer_id = st.sidebar.text_input("Customer ID", value="CUST_12345")
    customer_age = st.sidebar.slider("Customer Age", 18, 80, 32)
    customer_loyalty_years = st.sidebar.slider("Loyalty Years", 0.0, 10.0, 2.5, 0.1)
    
    # Order Details
    st.sidebar.subheader("Order Details")
    product_category = st.sidebar.selectbox(
        "Product Category",
        ["Electronics", "Clothing", "Home", "Books", "Sports"]
    )
    order_value = st.sidebar.number_input("Order Value ($)", min_value=1.0, max_value=2000.0, value=89.99)
    discount_amount = st.sidebar.number_input("Discount Amount ($)", min_value=0.0, max_value=order_value, value=10.0)
    shipping_cost = st.sidebar.number_input("Shipping Cost ($)", min_value=0.0, max_value=50.0, value=5.99)
    
    # Product Details
    st.sidebar.subheader("Product Details")
    product_rating = st.sidebar.slider("Product Rating", 1.0, 5.0, 4.2, 0.1)
    num_reviews = st.sidebar.number_input("Number of Reviews", min_value=0, max_value=10000, value=156)
    is_new_product = st.sidebar.checkbox("New Product", value=False)
    
    # Customer History
    st.sidebar.subheader("Customer History")
    previous_orders = st.sidebar.number_input("Previous Orders", min_value=0, max_value=100, value=5)
    previous_returns = st.sidebar.number_input("Previous Returns", min_value=0, max_value=previous_orders, value=1)
    days_since_last_order = st.sidebar.number_input("Days Since Last Order", min_value=0, max_value=730, value=30)
    
    # Order Context
    st.sidebar.subheader("Order Context")
    order_month = st.sidebar.selectbox("Order Month", list(range(1, 13)), index=10)
    is_weekend = st.sidebar.checkbox("Weekend Order", value=False)
    
    # Prepare data for prediction
    customer_data = {
        'customer_id': customer_id,
        'customer_age': customer_age,
        'customer_loyalty_years': customer_loyalty_years,
        'product_category': product_category,
        'order_value': order_value,
        'discount_amount': discount_amount,
        'shipping_cost': shipping_cost,
        'product_rating': product_rating,
        'num_reviews': num_reviews,
        'is_new_product': int(is_new_product),
        'previous_orders': previous_orders,
        'previous_returns': previous_returns,
        'days_since_last_order': days_since_last_order,
        'order_month': order_month,
        'is_weekend': int(is_weekend)
    }
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Risk Assessment")
        
        # Predict return risk
        probability, risk_level, color = predict_return_risk(
            model, scaler, label_encoder, feature_columns, customer_data
        )
        
        # Display risk metrics
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric(
                label="Return Probability",
                value=f"{probability:.1%}",
                delta=f"{probability - 0.2:.1%}" if probability > 0.2 else f"{probability - 0.2:.1%}"
            )
        
        with col_b:
            st.metric(
                label="Risk Level",
                value=risk_level,
            )
        
        with col_c:
            financial_risk = order_value * probability
            st.metric(
                label="Financial Risk",
                value=f"${financial_risk:.2f}"
            )
        
        # Risk gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Return Risk Score (%)"},
            delta = {'reference': 20},
            gauge = {
                'axis': {'range': [None, 80]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 15], 'color': "lightgreen"},
                    {'range': [15, 30], 'color': "yellow"},
                    {'range': [30, 80], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 30
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Risk factors analysis
        st.subheader("🔍 Risk Factor Analysis")
        
        risk_factors = []
        
        if product_category == 'Clothing':
            risk_factors.append(("Product Category", "High return rate for clothing items", 0.15))
        elif product_category == 'Electronics':
            risk_factors.append(("Product Category", "Moderate return rate for electronics", 0.08))
        
        if order_value < 30:
            risk_factors.append(("Order Value", "Low-value orders have higher return rates", 0.10))
        
        if product_rating < 3.5:
            risk_factors.append(("Product Rating", "Poor product rating increases return risk", 0.20))
        
        if is_new_product:
            risk_factors.append(("New Product", "New products have higher uncertainty", 0.08))
        
        if previous_orders > 0 and (previous_returns / previous_orders) > 0.2:
            risk_factors.append(("Return History", "Customer has high return rate", 0.30))
        
        if discount_amount / order_value > 0.5:
            risk_factors.append(("High Discount", "Large discounts correlate with returns", 0.12))
        
        if customer_age < 25:
            risk_factors.append(("Young Customer", "Younger customers return more frequently", 0.05))
        
        if days_since_last_order > 180:
            risk_factors.append(("Inactive Customer", "Long time since last purchase", 0.07))
        
        if risk_factors:
            for factor, description, impact in risk_factors:
                st.write(f"⚠️ **{factor}**: {description} (+{impact:.1%} risk)")
        else:
            st.write("✅ No significant risk factors identified")
    
    with col2:
        st.header("📊 Customer Profile")
        
        # Customer summary
        return_rate = previous_returns / previous_orders if previous_orders > 0 else 0
        avg_order_value = order_value  # In real app, this would be calculated from history
        
        st.markdown(f"""
        **Customer ID:** {customer_id}
        
        **Demographics:**
        - Age: {customer_age} years
        - Loyalty: {customer_loyalty_years} years
        
        **Order History:**
        - Previous Orders: {previous_orders}
        - Previous Returns: {previous_returns}
        - Return Rate: {return_rate:.1%}
        - Days Since Last Order: {days_since_last_order}
        
        **Current Order:**
        - Category: {product_category}
        - Value: ${order_value:.2f}
        - Discount: ${discount_amount:.2f} ({discount_amount/order_value:.1%})
        - Product Rating: {product_rating}/5.0
        """)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        if probability > 0.3:
            st.error("**High Risk Order**")
            st.write("Consider:")
            st.write("- Additional order verification")
            st.write("- Proactive customer communication")
            st.write("- Enhanced packaging/shipping")
            st.write("- Follow-up after delivery")
        elif probability > 0.15:
            st.warning("**Medium Risk Order**")
            st.write("Consider:")
            st.write("- Standard verification process")
            st.write("- Quality packaging")
            st.write("- Delivery confirmation")
        else:
            st.success("**Low Risk Order**")
            st.write("✅ Standard processing recommended")
    
    # Additional insights
    st.header("📈 Historical Trends")
    
    # Create sample trend data
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    trend_data = pd.DataFrame({
        'Date': dates,
        'Daily_Returns': np.random.poisson(5, len(dates)),
        'Daily_Orders': np.random.poisson(50, len(dates))
    })
    trend_data['Return_Rate'] = trend_data['Daily_Returns'] / trend_data['Daily_Orders']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_trend = px.line(trend_data, x='Date', y='Return_Rate', 
                           title='Return Rate Trend (Last 30 Days)')
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        category_returns = pd.DataFrame({
            'Category': ['Electronics', 'Clothing', 'Home', 'Books', 'Sports'],
            'Return_Rate': [0.12, 0.25, 0.08, 0.05, 0.15]
        })
        fig_cat = px.bar(category_returns, x='Category', y='Return_Rate',
                        title='Return Rate by Category')
        st.plotly_chart(fig_cat, use_container_width=True)

if __name__ == "__main__":
    main()
