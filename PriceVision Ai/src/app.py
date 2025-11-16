import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 🧠 Load model and scaler
model = joblib.load(r"C:\Users\LENOVO\Desktop\ai project\house_price_model2.pkl")
scaler = joblib.load(r"C:\Users\LENOVO\Desktop\ai project\scaler2.pkl")
model_columns = joblib.load(r"C:\Users\LENOVO\Desktop\ai project\model_columns2.pkl")

# 📘 Load dataset to get original landmark and furnishing options
df = pd.read_excel(r"C:\Users\LENOVO\Desktop\ai project\sample.xlsx")

# 🏠 App title
st.title("🏡w House Price Prediction App")

st.write("Enter the details below to estimate the house price:")

# Dropdowns from actual dataset
landmark_options = sorted(df['LANDMARK'].dropna().unique())
furnishing_options = sorted(df['FURNISHING_STATUS'].dropna().unique())

# User inputs
bhk = st.number_input("Number of BHK", min_value=1, max_value=10, value=2)
area_sqft = st.number_input("Area (sqft)", min_value=100, max_value=10000, value=1000)
furnishing = st.selectbox("Furnishing Status", furnishing_options)
landmark = st.selectbox("Landmark", landmark_options)
parking = st.selectbox("Parking Available?", ["Yes", "No"])
gas_pipeline = st.selectbox("Gas Pipeline Available?", ["Yes", "No"])
water_supply = st.selectbox("Water Supply Available?", ["Yes", "No"])

# Encode binary features
parking_val = 1 if parking == "Yes" else 0
gas_val = 1 if gas_pipeline == "Yes" else 0
water_val = 1 if water_supply == "Yes" else 0

# Encode furnishing & landmark
furnishing_encoded = pd.factorize(df['FURNISHING_STATUS'])[0][df['FURNISHING_STATUS'] == furnishing].mean()
landmark_encoded = pd.factorize(df['LANDMARK'])[0][df['LANDMARK'] == landmark].mean()

# Derived features (same as training)
room_area_ratio = area_sqft / bhk
bhk_area_combo = bhk * area_sqft
utilities_sum = parking_val + gas_val + water_val

# Create DataFrame for input
input_data = pd.DataFrame([{
    'BHK': bhk,
    'AREA_SQFT': area_sqft,
    'PARKING': parking_val,
    'GAS_PIPELINE': gas_val,
    'WATER_SUPPLY': water_val,
    'FURNISHING_STATUS': furnishing_encoded,
    'LANDMARK': landmark_encoded,
    'ROOM_AREA_RATIO': room_area_ratio,
    'BHK_AREA_COMBO': bhk_area_combo,
    'UTILITIES_SUM': utilities_sum,
    'Cluster': 0  # Placeholder if your model had a cluster column
}])

# Align columns with model training
input_data = input_data.reindex(columns=model_columns, fill_value=0)

# Predict button
if st.button("🔮 Predict Price"):
    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        st.success(f"💰 Estimated House Price: ₹{prediction:,.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
