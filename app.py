import streamlit as st
import joblib
import numpy as np
import pandas as pd

# streamlit run app.py

# Load model & features
model = joblib.load("house_price_model.pkl")
features = joblib.load("model_features.pkl")

st.title("🏠 House Price Prediction App")

# User input fields
def user_input():
    OverallQual = st.slider("Overall Quality (1–10)", 1, 10, 5)
    GrLivArea = st.number_input("Above Ground Living Area (sq ft)", 500, 5000, 1500)
    GarageCars = st.slider("Garage Capacity", 0, 4, 1)
    TotalBsmtSF = st.number_input("Basement Area (sq ft)", 0, 3000, 800)
    YearBuilt = st.number_input("Year Built", 1900, 2023, 2005)
    FirstFlr = st.number_input("1st Floor Area (sq ft)", 400, 3000, 900)
    FullBath = st.slider("Full Bathrooms", 0, 4, 2)
    TotalRooms = st.slider("Total Rooms Above Ground", 2, 14, 6)

    data = pd.DataFrame({
        "OverallQual": [OverallQual],
        "GrLivArea": [GrLivArea],
        "GarageCars": [GarageCars],
        "TotalBsmtSF": [TotalBsmtSF],
        "YearBuilt": [YearBuilt],
        "1stFlrSF": [FirstFlr],
        "FullBath": [FullBath],
        "TotRmsAbvGrd": [TotalRooms],
    })

    return data

input_data = user_input()

# Align columns
input_data = pd.get_dummies(input_data).reindex(columns=features, fill_value=0)

# Predict
prediction = model.predict(input_data)
price = np.expm1(prediction)[0]  # reverse log1p()
price_in_inr = price * 84.5

st.subheader(f"💰 Estimated House Price: **₹{price_in_inr:,.0f}**")
