import joblib
import os
import streamlit as st

# Streamlit file uploader to upload the model and model details files
uploaded_model_file = st.file_uploader("Upload a Stock Model (prediction_model.pkl)", type="pkl")
uploaded_model_details_file = st.file_uploader("Upload Model Details (model_details.pkl)", type="pkl")

# Function to load model from uploaded file
def load_uploaded_model(model_file):
    if model_file is not None:
        return joblib.load(model_file)
    return None

# Load the uploaded model and model details
if uploaded_model_file and uploaded_model_details_file:
    model = load_uploaded_model(uploaded_model_file)
    model_details = load_uploaded_model(uploaded_model_details_file)
    
    if model and model_details:
        # Display model details
        st.write(f"### Model Details from Uploaded File")
        st.write(f"**Model Type:** {model_details['model']}")
        st.write(f"**Accuracy (R-squared %):** {model_details['accuracy']:.2f}%")
        st.write(f"**R-squared:** {model_details['r2_score']:.2f}")
        st.write(f"**Mean Squared Error:** {model_details['mean_squared_error']:.4f}")
        st.write(f"**Intercept:** {model_details['intercept']:.4f}")
        st.write(f"**Coefficients:**")
        st.write(f"  - Inflation Rate: {model_details['coefficients']['Inflation Rate']:.4f}")
        st.write(f"  - Interest Rate: {model_details['coefficients']['Interest Rate']:.4f}")
        st.write(f"  - VIX: {model_details['coefficients']['VIX']:.4f}")
        
        # Feature input to make predictions (just an example with sample data)
        st.write("### Sample Model Prediction")
        sample_input = [0.03, 0.05, 20]  # Example values for Inflation Rate, Interest Rate, and VIX
        st.write(f"Predicting for input data: {sample_input}")
        prediction = model.predict([sample_input])
        st.write(f"Predicted value: {prediction[0]}")
    else:
        st.error("Failed to load the model or model details. Please upload the correct files.")
else:
    st.info("Please upload both the model and model details files.")
