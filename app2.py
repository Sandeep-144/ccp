import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load trained model and scaler
model = load_model("churn_model.keras")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
X_columns = joblib.load("X_columns.pkl")  # to maintain column order

# Streamlit UI
st.title("📞 Customer Churn Prediction App")
st.write("Enter customer details to predict churn likelihood.")

# Input form
with st.form("customer_form"):
    call_failure = st.number_input("Call Failures", min_value=0, step=1)
    complains = st.selectbox("Complains", options=[0, 1])
    subscription_length = st.number_input("Subscription Length (months)", min_value=1)
    charge_amount = st.number_input("Charge Amount ($)", min_value=0.0)
    seconds_of_use = st.number_input("Seconds of Use", min_value=0)
    frequency_of_use = st.number_input("Frequency of Use", min_value=0)
    tariff_plan = st.selectbox("Tariff Plan", options=label_encoders["Tariff Plan"].classes_)
    status = st.selectbox("Status", options=label_encoders["Status"].classes_)
    age_group = st.selectbox("Age Group", options=label_encoders["Age Group"].classes_)
    frequency_of_sms = st.number_input("Frequency of SMS", min_value=0)
    customer_value = st.number_input("Customer Value", min_value=0.0)
    distinct_called = st.number_input("Distinct Called Numbers", min_value=0)

    submitted = st.form_submit_button("Predict Churn")

# Prediction logic
if submitted:
    input_dict = {
        "Call  Failure": call_failure,
        "Complains": complains,
        "Subscription  Length": subscription_length,
        "Charge  Amount": charge_amount,
        "Seconds of Use": seconds_of_use,
        "Frequency of use": frequency_of_use,
        "Tariff Plan": label_encoders["Tariff Plan"].transform([tariff_plan])[0],
        "Status": label_encoders["Status"].transform([status])[0],
        "Age Group": label_encoders["Age Group"].transform([age_group])[0],
        "Frequency of SMS": frequency_of_sms,
        "Customer Value": customer_value,
        "Distinct Called Numbers": distinct_called
    }

    new_df = pd.DataFrame([input_dict])
    new_df = new_df[X_columns]  # keep column order
    new_scaled = scaler.transform(new_df)
    prediction = model.predict(new_scaled)[0][0]
    result = "🔴 Churn" if prediction > 0.5 else "🟢 No Churn"

    st.subheader("📊 Prediction Result")
    st.write(f"**Churn Probability:** `{prediction:.2f}`")
    st.success(f"**Prediction:** {result}")
