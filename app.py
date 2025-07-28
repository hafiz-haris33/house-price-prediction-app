import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import urllib.request
import tempfile
from preprocessing import ClusterSimilarity, cluster_simil, default_num_pipeline, cat_pipeline, log_pipeline, ratio_pipeline, column_ratio, ratio_name, preprocessing

# Page configuration
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Load trained model

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# Sidebar
st.sidebar.title("📊 California Housing App")
st.sidebar.info("Enter housing details below to predict the median house value.")

# Main Title
st.title("🏡 California Housing Price Prediction")
st.markdown("Use the form below to predict the **Median House Value** for a district in California.")

# User Input Form
with st.form("prediction_form"):
    st.subheader("Enter District Information:")

    longitude = st.number_input("🧭 Longitude", -130.0, -110.0, step=0.1)
    latitude = st.number_input("📍 Latitude", 30.0, 45.0, step=0.1)
    housing_median_age = st.slider("🏗 Housing Median Age", 0, 60, 20)
    total_rooms = st.number_input("🏠 Total Rooms", 0, 40000, step=1)
    total_bedrooms = st.number_input("🛏 Total Bedrooms", 0, 7000, step=1)
    population = st.number_input("👥 Population", 0, 40000, step=1)
    households = st.number_input("🏘 Households", 1, 6500, step=1)
    median_income = st.number_input("💰 Median Income (10k USD)", 0.0, 20.0, step=0.1)

    ocean_proximity = st.selectbox("🌊 Ocean Proximity", [
        "<1H OCEAN",
        "INLAND",
        "NEAR OCEAN",
        "NEAR BAY",
        "ISLAND"
    ])

    submit_btn = st.form_submit_button("Predict 💡")

# When user submits form
if submit_btn:
    input_data = pd.DataFrame({
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income],
        "ocean_proximity": [ocean_proximity]
    })

    try:
        # Make sure the model is trained with encoder or pipeline that supports this column
        prediction = model.predict(input_data)
        st.success(f"🏷 **Predicted Median House Value:** ${prediction[0]:,.2f}")
        st.balloons()
    except Exception as e:
        st.error(f"Prediction failed due to: {e}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by **Hafiz Haris** | [LinkedIn](https://www.linkedin.com/in/hafiz-muhammad-haris-305211361)")
