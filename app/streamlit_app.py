import sys
import os

# Add the parent directory to sys.path to allow imports from the root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.predict import predict_rent

st.set_page_config(page_title="Rent Prediction App", layout="centered")

st.title("🏠 House Rent Prediction")
st.write("Enter property details to estimate monthly rent.")

# ---- User Inputs ----
bhk = st.number_input("BHK", min_value=1, max_value=6, value=2)
size = st.number_input("Size (sqft)", min_value=100, max_value=10000, value=900)
bathroom = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

current_floor = st.number_input("Current Floor", min_value=0, max_value=50, value=2)
total_floors = st.number_input("Total Floors", min_value=1, max_value=50, value=5)

area_type = st.selectbox(
    "Area Type",
    ["Super Area", "Carpet Area", "Built Area"]
)

locality = st.text_input(
    "Area Locality",
    value="Other",
    help="Unknown localities will be treated as 'Other'"
)

city = st.selectbox(
    "City",
    ["Bangalore", "Mumbai", "Chennai", "Hyderabad", "Delhi", "Kolkata"]
)

furnishing = st.selectbox(
    "Furnishing Status",
    ["Unfurnished", "Semi-Furnished", "Furnished"]
)

tenant = st.selectbox(
    "Tenant Preferred",
    ["Bachelors/Family", "Bachelors", "Family"]
)

# ---- Prediction ----
if st.button("Predict Rent"):
    input_data = {
        "BHK": bhk,
        "Size": size,
        "Bathroom": bathroom,
        "current_floor": current_floor,
        "total_floors": total_floors,
        "Area Type": area_type,
        "Area Locality": locality,
        "City": city,
        "Furnishing Status": furnishing,
        "Tenant Preferred": tenant
    }

    predicted_rent = predict_rent(input_data)

    st.success(f"💰 Estimated Monthly Rent: ₹ {predicted_rent:,}")
