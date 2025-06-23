import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# --- Load model and preprocessing objects ---
model = load_model("churn_model.keras")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# --- Streamlit UI setup ---
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("Customer Churn Prediction")
st.markdown("Enter customer details in the sidebar to predict the probability of churn.")

st.sidebar.header("Customer Information")

# --- Sidebar Inputs ---
tariff_plan = st.sidebar.selectbox("Tariff Plan", ["Standard", "Plus", "Premium"])
status = st.sidebar.selectbox("Status", ["Single", "Married", "Divorced"])
age_group = st.sidebar.selectbox("Age Group", ["18-25", "26-35", "36-50", "50+"])

call_failure = st.sidebar.slider("Call Failures", 0, 10, 1)
complains = st.sidebar.selectbox("Complains?", [0, 1])
subscription_length = st.sidebar.slider("Subscription Length (months)", 1, 60, 12)
charge_amount = st.sidebar.slider("Charge Amount", 10.0, 200.0, 50.0)
seconds_of_use = st.sidebar.slider("Seconds of Use", 100, 5000, 2000)
frequency_of_use = st.sidebar.slider("Frequency of Use", 1, 100, 15)
frequency_of_sms = st.sidebar.slider("Frequency of SMS", 0, 50, 5)
distinct_called = st.sidebar.slider("Distinct Called Numbers", 1, 100, 30)
customer_value = st.sidebar.slider("Customer Value", 0, 1000, 500)

# --- Prepare Input Data ---
input_data = {
    "Call  Failure": call_failure,
    "Complains": complains,
    "Subscription  Length": subscription_length,
    "Charge  Amount": charge_amount,
    "Seconds of Use": np.log1p(seconds_of_use),  # apply log1p if done during training
    "Frequency of use": frequency_of_use,
    "Frequency of SMS": frequency_of_sms,
    "Tariff Plan": tariff_plan,
    "Status": status,
    "Age Group": age_group,
    "Customer Value": customer_value,
    "Distinct Called Numbers": distinct_called
}

input_df = pd.DataFrame([input_data])

# --- Encode categorical features ---
for col in ["Tariff Plan", "Status", "Age Group"]:
    if input_df[col][0] in label_encoders[col].classes_:
        input_df[col] = label_encoders[col].transform([input_df[col][0]])
    else:
        input_df[col] = -1  # unknown class fallback

# --- Ensure column order ---
expected_columns = scaler.feature_names_in_
input_df = input_df[expected_columns]

# --- Scale input ---
scaled_input = scaler.transform(input_df)

# --- Debugging (optional) ---
# st.write("Model expects input shape:", model.input_shape)
# st.write("Input shape:", scaled_input.shape)
# st.write("Scaled Input:", scaled_input)

# --- Predict ---
churn_prob = model.predict(scaled_input)[0][0]
churn_label = "🔴 Churn" if churn_prob > 0.5 else "🟢 No Churn"

# --- Display Output ---
st.subheader("Prediction Result")
st.metric("Churn Probability", f"{churn_prob * 100:.2f}%")

if churn_prob > 0.5:
    st.error(churn_label)
else:
    st.success(churn_label)

st.progress(min(float(churn_prob), 1.0))

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & TensorFlow")
