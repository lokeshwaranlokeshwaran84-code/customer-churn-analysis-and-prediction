import streamlit as st
import pandas as pd
import joblib

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
cols = joblib.load("columns.pkl")

st.set_page_config(page_title="Churn Predictor", page_icon="📉")
st.title("Customer Churn Predictor")
st.caption("Enter customer details to estimate churn risk (XGBoost model)")

c1, c2 = st.columns(2)
with c1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multi = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
with c2:
    device = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly Charges", 18.0, 120.0, 70.0)
    total = st.number_input("Total Charges", 0.0, 9000.0, float(monthly * tenure))

if st.button("Predict churn risk"):
    row = pd.DataFrame(0.0, index=[0], columns=cols)

    # numeric and binary columns
    row["SeniorCitizen"] = 1 if senior == "Yes" else 0
    row["gender"] = 1 if gender == "Male" else 0
    row["Partner"] = 1 if partner == "Yes" else 0
    row["Dependents"] = 1 if dependents == "Yes" else 0
    row["PhoneService"] = 1 if phone == "Yes" else 0
    row["PaperlessBilling"] = 1 if paperless == "Yes" else 0
    row["tenure"] = tenure
    row["MonthlyCharges"] = monthly
    row["TotalCharges"] = total

    # one-hot columns (baseline categories were dropped in training)
    cats = {
        "MultipleLines": multi, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": device, "TechSupport": tech,
        "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaymentMethod": payment,
    }
    for name, val in cats.items():
        key = f"{name}_{val}"
        if key in row.columns:
            row[key] = 1

    # scale numeric columns exactly as in training
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    row[num_cols] = scaler.transform(row[num_cols])

    prob = model.predict_proba(row)[0, 1]
    level = "High" if prob >= 0.7 else "Medium" if prob >= 0.4 else "Low"

    st.metric("Churn probability", f"{prob:.1%}")
    if level == "High":
        st.error("High risk: offer a contract upgrade or retention discount.")
    elif level == "Medium":
        st.warning("Medium risk: monitor and consider a loyalty offer.")
    else:
        st.success("Low risk: customer is likely to stay.")
