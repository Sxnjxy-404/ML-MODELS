#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Sample Data
data = { 
    'Outlook':    ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 
                   'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 
                   'Overcast', 'Overcast', 'Rain'], 
    'Temperature':['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 
                   'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'], 
    'Humidity':   ['High', 'High', 'High', 'High', 'Normal', 'Normal', 
                   'Normal', 'High', 'Normal', 'Normal', 'Normal', 
                   'High', 'Normal', 'High'], 
    'Wind':       ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 
                   'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 
                   'Weak', 'Strong'], 
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 
                   'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No'] 
} 

# Load DataFrame
df = pd.DataFrame(data)

# Encode categorical columns
label_encoders = {}
df_encoded = df.copy()
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_encoded[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split features and target
X = df_encoded.drop('PlayTennis', axis=1)
y = df_encoded['PlayTennis']

# Train the Decision Tree Classifier
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

# Streamlit app
st.set_page_config(page_title="PlayTennis Predictor", layout="wide")
st.title("🎾 PlayTennis Prediction with ID3 Decision Tree")

# Sidebar for input
st.sidebar.header("🌦️ Input Weather Conditions")

def user_input():
    outlook = st.sidebar.selectbox("Outlook", df['Outlook'].unique())
    temp = st.sidebar.selectbox("Temperature", df['Temperature'].unique())
    humidity = st.sidebar.selectbox("Humidity", df['Humidity'].unique())
    wind = st.sidebar.selectbox("Wind", df['Wind'].unique())
    return pd.DataFrame([[outlook, temp, humidity, wind]],
                        columns=['Outlook', 'Temperature', 'Humidity', 'Wind'])

input_df = user_input()

# Encode user input
input_encoded = input_df.copy()
for col in input_encoded.columns:
    input_encoded[col] = label_encoders[col].transform(input_encoded[col])

# Predict
prediction = model.predict(input_encoded)[0]
prediction_label = label_encoders['PlayTennis'].inverse_transform([prediction])[0]

# Output
st.subheader("🎯 Prediction:")
st.success(f"The model predicts: **{prediction_label}**")

st.subheader("📥 Input Values:")
st.write(input_df)

# Optional: Show training data
if st.checkbox("📊 Show Training Data"):
    st.dataframe(df)

# Optional: Show decision tree plot
if st.checkbox("🌲 Show Decision Tree"):
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(model, filled=True, feature_names=X.columns, class_names=label_encoders['PlayTennis'].classes_, ax=ax)
    st.pyplot(fig)
