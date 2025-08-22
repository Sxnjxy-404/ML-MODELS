import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Generate synthetic dataset for demonstration
def generate_sample_data(n_samples=10000):
    """Generate synthetic e-commerce return data"""
    np.random.seed(42)
    
    # Customer demographics
    customer_ids = np.arange(1, n_samples + 1)
    customer_age = np.random.normal(35, 12, n_samples).clip(18, 80)
    customer_loyalty_years = np.random.exponential(2, n_samples).clip(0, 10)
    
    # Order details
    product_categories = np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], n_samples, 
                                        p=[0.25, 0.3, 0.2, 0.15, 0.1])
    order_values = np.random.lognormal(4, 0.8, n_samples).clip(10, 1000)
    discount_amounts = np.random.uniform(0, order_values * 0.3)
    shipping_costs = np.where(order_values > 50, 0, np.random.uniform(5, 15, n_samples))
    
    # Product details
    product_ratings = np.random.normal(4.2, 0.8, n_samples).clip(1, 5)
    num_reviews = np.random.poisson(50, n_samples)
    is_new_product = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    
    # Customer behavior
    previous_orders = np.random.poisson(5, n_samples)
    previous_returns = np.random.poisson(1, n_samples) * (previous_orders > 0)
    days_since_last_order = np.random.exponential(30, n_samples).clip(0, 365)
    
    # Seasonal factors
    order_month = np.random.randint(1, 13, n_samples)
    is_weekend = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    # Create return probability based on realistic factors
    return_prob = 0.1  # Base return rate
    
    # Adjust probability based on factors
    return_prob += (product_categories == 'Clothing') * 0.15  # Clothing has higher returns
    return_prob += (product_categories == 'Electronics') * 0.08
    return_prob += (order_values < 30) * 0.1  # Cheaper items more likely returned
    return_prob += (product_ratings < 3.5) * 0.2  # Poor ratings increase returns
    return_prob += (is_new_product == 1) * 0.08  # New products riskier
    return_prob += (previous_returns / (previous_orders + 1)) * 0.3  # Return history
    return_prob += (discount_amounts / order_values > 0.5) * 0.12  # High discounts
    return_prob += (customer_age < 25) * 0.05  # Younger customers return more
    return_prob += (days_since_last_order > 180) * 0.07  # Inactive customers
    
    # Generate actual returns
    returned = np.random.binomial(1, return_prob.clip(0, 0.8), n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'customer_id': customer_ids,
        'customer_age': customer_age.round(0),
        'customer_loyalty_years': customer_loyalty_years.round(1),
        'product_category': product_categories,
        'order_value': order_values.round(2),
        'discount_amount': discount_amounts.round(2),
        'shipping_cost': shipping_costs.round(2),
        'product_rating': product_ratings.round(1),
        'num_reviews': num_reviews,
        'is_new_product': is_new_product,
        'previous_orders': previous_orders,
        'previous_returns': previous_returns,
        'days_since_last_order': days_since_last_order.round(0),
        'order_month': order_month,
        'is_weekend': is_weekend,
        'returned': returned
    })
    
    return data

def preprocess_data(df):
    """Preprocess the data for model training"""
    # Create feature engineering
    df = df.copy()
    
    # Return rate for customer
    df['customer_return_rate'] = df['previous_returns'] / (df['previous_orders'] + 1)
    
    # Discount percentage
    df['discount_percentage'] = df['discount_amount'] / df['order_value']
    
    # Price per review (popularity indicator)
    df['price_per_review'] = df['order_value'] / (df['num_reviews'] + 1)
    
    # Seasonal indicators
    df['is_holiday_season'] = ((df['order_month'] == 11) | (df['order_month'] == 12)).astype(int)
    df['is_summer'] = ((df['order_month'] >= 6) & (df['order_month'] <= 8)).astype(int)
    
    # Customer activity level
    df['customer_activity_score'] = df['previous_orders'] / (df['days_since_last_order'] / 30 + 1)
    
    # Encode categorical variables
    le = LabelEncoder()
    df['product_category_encoded'] = le.fit_transform(df['product_category'])
    
    return df, le

def train_model(df, target_column='returned'):
    """Train the return prediction model"""
    
    # Preprocess data
    df_processed, label_encoder = preprocess_data(df)
    
    # Select features for training
    feature_columns = [
        'customer_age', 'customer_loyalty_years', 'order_value', 'discount_amount',
        'shipping_cost', 'product_rating', 'num_reviews', 'is_new_product',
        'previous_orders', 'previous_returns', 'days_since_last_order',
        'order_month', 'is_weekend', 'product_category_encoded',
        'customer_return_rate', 'discount_percentage', 'price_per_review',
        'is_holiday_season', 'is_summer', 'customer_activity_score'
    ]
    
    X = df_processed[feature_columns]
    y = df_processed[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate model
    print("Model Performance:")
    print("=" * 50)
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save model components
    joblib.dump(model, 'return_prediction_model.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    joblib.dump(feature_columns, 'feature_columns.pkl')
    
    return model, scaler, label_encoder, feature_columns, feature_importance

def predict_return_risk(model, scaler, label_encoder, feature_columns, customer_data):
    """Predict return risk for a single customer/order"""
    
    # Create DataFrame from input
    df = pd.DataFrame([customer_data])
    
    # Preprocess
    df_processed, _ = preprocess_data(df)
    
    # Ensure we have all required features
    for col in feature_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0
    
    # Select and scale features
    X = df_processed[feature_columns]
    X_scaled = scaler.transform(X)
    
    # Predict
    probability = model.predict_proba(X_scaled)[0][1]
    risk_level = 'High' if probability > 0.3 else 'Medium' if probability > 0.15 else 'Low'
    
    return probability, risk_level

if __name__ == "__main__":
    print("Generating sample e-commerce return data...")
    data = generate_sample_data(10000)
    
    print(f"\nDataset shape: {data.shape}")
    print(f"Return rate: {data['returned'].mean():.3f}")
    print("\nSample data:")
    print(data.head())
    
    print("\nTraining return prediction model...")
    model, scaler, le, features, importance = train_model(data)
    
    print("\nModel training completed!")
    print("Files saved: return_prediction_model.pkl, feature_scaler.pkl, label_encoder.pkl, feature_columns.pkl")
    
    # Example prediction
    sample_order = {
        'customer_id': 12345,
        'customer_age': 28,
        'customer_loyalty_years': 1.5,
        'product_category': 'Clothing',
        'order_value': 75.99,
        'discount_amount': 15.00,
        'shipping_cost': 0,
        'product_rating': 3.8,
        'num_reviews': 45,
        'is_new_product': 1,
        'previous_orders': 3,
        'previous_returns': 1,
        'days_since_last_order': 45,
        'order_month': 11,
        'is_weekend': 1
    }
    
    prob, risk = predict_return_risk(model, scaler, le, features, sample_order)
    print(f"\nExample Prediction:")
    print(f"Return Probability: {prob:.3f}")
    print(f"Risk Level: {risk}")
