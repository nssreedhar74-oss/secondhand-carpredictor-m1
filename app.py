import streamlit as st
import pandas as pd
import pickle
import os

# Page config
st.set_page_config(page_title="Car Price Predictor", layout="wide")

st.title("Car Price Prediction Model")

# Load model
if os.path.exists("model.pkl"):
    model = pickle.load(open("model.pkl", "rb"))
else:
    st.error("Model file not found! Please run train_model.py first.")
    st.stop()

st.sidebar.header("Enter Car Details")

# Inputs
vehicle_age = st.sidebar.slider("Vehicle Age", 0, 20, 5)
km_driven = st.sidebar.number_input("KM Driven", 0, 300000, 50000)

brand = st.sidebar.selectbox("Brand", ["Maruti", "Hyundai", "Honda", "Toyota", "Mahindra"])

seller_type = st.sidebar.selectbox("Seller Type", ["Individual", "Dealer"])
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])

transmission_type = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])

mileage = st.sidebar.number_input("Mileage (km/l)", 0.0, 40.0, 18.0)
engine = st.sidebar.number_input("Engine (CC)", 500, 5000, 1200)
max_power = st.sidebar.number_input("Max Power (bhp)", 20.0, 300.0, 80.0)
seats = st.sidebar.slider("Seats", 2, 10, 5)

# Create input DataFrame
input_data = pd.DataFrame({
    'vehicle_age': [vehicle_age],
    'km_driven': [km_driven],
    'brand': [brand],
    'seller_type': [seller_type],
    'fuel_type': [fuel_type],
    'transmission_type': [transmission_type],
    'mileage': [mileage],
    'engine': [engine],
    'max_power': [max_power],
    'seats': [seats]
})

# Prediction
if st.button("Predict Price", type="primary"):
    try:
        pred = model.predict(input_data)[0]
        st.success(f"Estimated Price: Rs. {pred:,.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")

# Display input data
with st.expander("View Input Data"):
    st.dataframe(input_data)
