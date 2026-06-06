
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and label encoder
try:
    model = joblib.load("fault_detection_model.pkl")
    label_encoder = joblib.load("fault_label_encoder.pkl")
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'fault_detection_model.pkl' and 'fault_label_encoder.pkl' are in the same directory.")
    st.stop()

st.title("Fault Prediction App")
st.write("Enter the details below to predict the fault type.")

# Input fields for features
voltage = st.number_input("Voltage (V)", value=228.0)
current = st.number_input("Current (A)", value=18.0)
power_load = st.number_input("Power Load (MW)", value=4100.0)
temperature = st.number_input("Temperature (°C)", value=42.0)
wind_speed = st.number_input("Wind Speed (km/h)", value=10.0)
weather_condition = st.selectbox(
    "Weather Condition",
    ["Clear", "Rainy", "Windstorm", "Snowy", "Foggy"]
)
maintenance_status = st.selectbox(
    "Maintenance Status",
    ["Scheduled", "Completed", "Pending"]
)
component_health = st.selectbox(
    "Component Health",
    ["Normal", "Faulty", "Overheated", "Degraded"]
)
fault_location = st.text_input(
    "Fault Location (Latitude, Longitude)",
    value="(17.3850, 78.4867)"
)
duration_of_fault = st.number_input("Duration of Fault (hrs)", value=45.0)
down_time = st.number_input("Down time (hrs)", value=30.0)


if st.button("Predict Fault"):
    # Create DataFrame from inputs
    new_sample = pd.DataFrame({
        "Voltage (V)": [voltage],
        "Current (A)": [current],
        "Power Load (MW)": [power_load],
        "Temperature (°C)": [temperature],
        "Wind Speed (km/h)": [wind_speed],
        "Weather Condition": [weather_condition],
        "Maintenance Status": [maintenance_status],
        "Component Health": [component_health],
        "Fault Location (Latitude, Longitude)": [fault_location],
        "Duration of Fault (hrs)": [duration_of_fault],
        "Down time (hrs)": [down_time]
    })

    # Re-apply feature engineering that was part of training
    # Voltage/Current ratio
    if ("Voltage (V)" in new_sample.columns) and ("Current (A)" in new_sample.columns):
        new_sample["V_I_Ratio"] = new_sample["Voltage (V)"] / (new_sample["Current (A)"] + 1e-6)

    # Temperature/Power Load Ratio
    if ("Temperature (°C)" in new_sample.columns) and ("Power Load (MW)" in new_sample.columns):
        new_sample["Temp_Load_Ratio"] = (
            new_sample["Temperature (°C)"] /
            (new_sample["Power Load (MW)"] + 1e-6)
        )

    # Power efficiency
    if (
        "Power Load (MW)" in new_sample.columns
        and "Voltage (V)" in new_sample.columns
        and "Current (A)" in new_sample.columns
    ):
        new_sample["Power_Factor_Approx"] = (
            new_sample["Power Load (MW)"] /
            ((new_sample["Voltage (V)"] * new_sample["Current (A)"]) + 1e-6)
        )
    # Fault duration impact
    if (
        "Duration of Fault (hrs)" in new_sample.columns
        and "Down time (hrs)" in new_sample.columns
    ):
        new_sample["Downtime_Ratio"] = (
            new_sample["Down time (hrs)"] /
            (new_sample["Duration of Fault (hrs)"] + 1e-6)
        )

    # Make prediction
    prediction = model.predict(new_sample)
    fault_type = label_encoder.inverse_transform(prediction)

    st.success(f"Predicted Fault: {fault_type[0]}")