
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title='Customer Churn Prediction', layout='wide')
st.title("🔍 Customer Churn Prediction Dashboard")
st.markdown("This dashboard predicts whether a customer is likely to churn based on input features.")

# Load model and feature names
model = joblib.load("src/model.pkl")
feature_names = joblib.load("src/features.pkl") 

# Sidebar inputs
st.sidebar.header("Enter Customer Details")

def user_input_features():
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    MonthlyCharges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    TotalCharges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1000.0)
    InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    PaymentMethod = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    DeviceProtection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])

    # Manual feature engineering
    is_long_term_customer = 1 if tenure >= 12 else 0
    total_spent = tenure * MonthlyCharges

    df = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges],
        'InternetService': [InternetService],
        'PaymentMethod': [PaymentMethod],
        'Contract': [Contract],
        'gender': [gender],
        'PaperlessBilling': [PaperlessBilling],
        'Dependents': [Dependents],
        'DeviceProtection': [DeviceProtection],
        'is_long_term_customer': [is_long_term_customer],
        'total_spent': [total_spent]
    })
    return df

raw_input = user_input_features()

# One-hot encode and align
input_df = pd.get_dummies(raw_input)

# Add missing columns and reorder
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_names]  # Ensure correct order

# Show inputs
st.subheader("📥 Input Features")
st.write(raw_input)

# Predict
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

# Output results
st.subheader("📊 Prediction Result")
st.write("**Churn Prediction:**", "Yes" if prediction == 1 else "No")
# st.write("**Probability of Churn:**", f"{probability*100:.2f}%")

# Feature Importance Plot
st.subheader("📈 Top Feature Importances")
importances = model.feature_importances_
features = model.feature_names_in_
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
top10 = importance_df.sort_values(by='Importance', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=top10, ax=ax)
st.pyplot(fig)
