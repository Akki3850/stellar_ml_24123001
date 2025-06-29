import numpy as np 
import pandas as pd 
import joblib
import streamlit as st 

#making background 
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.pexels.com/photos/1175136/pexels-photo-1175136.jpeg");
        background-size: cover;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)



#Loading saved model and pipleine 
model = joblib.load("star_model.pkl")
pipeline = joblib.load("full_pipeline.pkl")

#Title 
st.title("Star type Classifier")

st.write("Enter the stellar parameters below:")

#input 
temperature = st.number_input("Temperature (K)", min_value=1000.0, max_value=50000.0, value=5000.0,format="%.4f")
luminosity = st.number_input("Luminosity (Lo)", min_value=0.0001, value=0.005,format="%.4f")
radius = st.number_input("Radius (Ro)", min_value=0.001, value=1.0,format="%.4f")
abs_magnitude = st.number_input("Absolute Magnitude", value=1.0,format="%.4f")
star_color = st.selectbox("Star Color", [
    "Blue", "Blue White", "Blue white", "Blue-white", "Blue-White",
    "White", "White-Yellow", "Whitish", "white",
    "Yellow", "Yellowish White", "yellow-white", "yellowish", "Pale yellow orange",
    "Orange", "Orange-Red", 
    "Red"
])

spectral_class = st.selectbox("Spectral Class", ['O', 'B', 'A', 'F', 'G', 'K', 'M','D'])

if st.button("Predict Star Type"):
      # Fix inconsistent labels
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

    # Create a DataFrame for prediction
    data = pd.DataFrame([{
        "Temperature_K": temperature,
        "Luminosity_Lo": luminosity,
        "Radius_Ro": radius,
        "Absolute_Magnitude": abs_magnitude,
        "Star_Color": star_color,
        "Spectral_Class": spectral_class,
        "S.No.": 0  # Dummy value
    }])
        # Transform input
    data_prepared = pipeline.transform(data)
    prediction = model.predict(data_prepared)[0]

    st.success(f"Predicted Star Type: {prediction}")
else: 
    pass 
