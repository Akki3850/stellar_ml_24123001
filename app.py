import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Load the saved model (trained separately)
model = joblib.load("star_model.pkl")
full_pipline = joblib.load("full_pipeline.pkl")

# Streamlit app UI
st.set_page_config(page_title="Star Type Classifier")
st.title("Star Type Classifier")
st.markdown("Enter stellar parameters to predict the star type:")

# Star color dropdown (include all known + cleaned values)
color_options = [
    "Blue", "Blue White", "White", "Yellowish White", "Yellow",
    "Orange", "Red", "Blue white", "Blue-white", "Blue-White", "Pale yellow orange",
    "yellowish", "yellow-white", "Orange-Red", "White-Yellow", "white", "Whitish"
]
spectral_options = ['O', 'B', 'A', 'F', 'G', 'K', 'M']

# Input fields
temperature = st.number_input("Temperature (K)", min_value=1000.0, max_value=50000.0, format="%.4f")
luminosity = st.number_input("Luminosity (Lo)", min_value=0.0001, format="%.4f")
radius = st.number_input("Radius (Ro)", min_value=0.001, format="%.4f")
abs_magnitude = st.number_input("Absolute Magnitude", format="%.4f")
star_color = st.selectbox("Star Color", sorted(color_options))
spectral_class = st.selectbox("Spectral Class", spectral_options)

# Color label cleanup
color_mapping = {
    "Blue white": "Blue White",
    "Blue-white": "Blue White",
    "Blue-White": "Blue White",
    "Pale yellow orange": "Yellowish White",
    "yellowish": "Yellowish White",
    "yellow-white": "Yellowish White",
    "Orange-Red": "Orange",
    "White-Yellow": "White",
    "white": "White",
    "Whitish": "White"
}
star_color = color_mapping.get(star_color, star_color)

# Predict button
if st.button("Predict Star Type"):
    input_data = pd.DataFrame([{
        "Temperature_K": temperature,
        "Luminosity_Lo": luminosity,
        "Radius_Ro": radius,
        "Absolute_Magnitude": abs_magnitude,
        "Star_Color": star_color,
        "Spectral_Class": spectral_class,
        "S.No.": 0  # Dummy column (ignored in pipeline)
    }])

    # Remove unused column
    input_data = input_data.drop(columns=["S.No."])

    try:
        input_prepared = full_pipline.fit_transform(input_data)  # Only fit here if not saved
        prediction = model.predict(input_prepared)[0]
        st.success(f"🌟 Predicted Star Type: **{prediction}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
