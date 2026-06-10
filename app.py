import streamlit as st
import pickle
import pandas as pd

# Load the pre-trained NEO model
# NOTE: Replace this with your exact relative path or actual local pkl file location
with open(r"C:\Users\asus\OneDrive\Desktop\MachineLearning_Flow\model.pkl", "rb") as f:
    model = pickle.load(f)

# Page configuration
st.set_page_config(
    page_title="NEO Hazard Prediction",
    page_icon="☄️",
    layout="centered"
)

st.title("☄️ Near Earth Object (NEO) Hazard Prediction")
st.write("Enter the orbital and physical measurements of the asteroid below to assess its collision hazard risk:")

# Numeric Inputs for the 5 key features of the NEO dataset
est_diameter_min = st.number_input(
    "Minimum Estimated Diameter (km)",
    min_value=0.0,
    value=0.25,
    step=0.01
)

est_diameter_max = st.number_input(
    "Maximum Estimated Diameter (km)",
    min_value=0.0,
    value=0.55,
    step=0.01
)

relative_velocity = st.number_input(
    "Relative Velocity (km/h)",
    min_value=0.0,
    value=45000.0,
    step=500.0
)

miss_distance = st.number_input(
    "Miss Distance (km)",
    min_value=0.0,
    value=35000000.0,
    step=100000.0
)

absolute_magnitude = st.number_input(
    "Absolute Magnitude",
    min_value=0.0,
    value=22.0,
    step=0.1
)

# Trigger inference execution
if st.button("Predict Hazard Risk"):

    # Package user inputs into a DataFrame using identical feature names from the pipeline training process
    input_data = pd.DataFrame({
        "est_diameter_min": [est_diameter_min],
        "est_diameter_max": [est_diameter_max],
        "relative_velocity": [relative_velocity],
        "miss_distance": [miss_distance],
        "absolute_magnitude": [absolute_magnitude]
    })

    # Predict class (0 or 1) using the loaded model pipeline
    prediction = model.predict(input_data)[0]

    # Display prediction results safely using Streamlit alert boxes
    if prediction == 1:
        st.error("⚠️ CRITICAL ALERT: This Near Earth Object is classified as **HAZARDOUS**!")
    else:
        st.success("✅ CLEAR: This Near Earth Object is classified as **NON-HAZARDOUS**.")