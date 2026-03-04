import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Churn Predictor", page_icon="📡", layout="centered")
st.title("📡 Telecom Customer Churn Predictor")
st.markdown("Enter customer details below to predict churn probability.")

st.sidebar.header("Customer Information")

# Input fields
gender        = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior        = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner       = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents    = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
tenure        = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
phone         = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
internet      = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract      = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless     = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
monthly       = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
total         = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly))

if st.button("🔍 Predict Churn"):
    # Simple rule-based estimate for demo (replace with loaded model in real use)
    risk_score = 0.0
    if contract == "Month-to-month":   risk_score += 0.35
    if internet == "Fiber optic":      risk_score += 0.15
    if tenure < 12:                    risk_score += 0.20
    if monthly > 80:                   risk_score += 0.15
    if paperless == "Yes":             risk_score += 0.05
    if partner == "No":                risk_score += 0.05
    if dependents == "No":             risk_score += 0.05
    risk_score = min(risk_score, 0.99)

    label = "🔴 Likely to CHURN" if risk_score > 0.5 else "🟢 Likely to STAY"
    st.subheader(f"Prediction: {label}")
    st.metric("Churn Probability", f"{risk_score:.0%}")
    st.progress(risk_score)

    if risk_score > 0.5:
        st.warning("⚠️ Recommended Action: Offer a retention discount or upgrade to an annual contract.")
    else:
        st.success("✅ Customer appears stable. Continue regular engagement.")

st.markdown("---")
st.caption("AI Mini Project — Telecom Churn Prediction | Group Project")