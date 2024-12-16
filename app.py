import joblib
import streamlit as st

# Streamlit file uploader to upload a combined model and model details file
uploaded_file = st.file_uploader("Upload a Combined Stock Model (model_combined.pkl)", type="pkl")

# Function to load model from uploaded file
def load_uploaded_model_and_details(uploaded_file):
    if uploaded_file is not None:
        return joblib.load(uploaded_file)
    return None

# Load the uploaded model and its details
if uploaded_file:
    combined_data = load_uploaded_model_and_details(uploaded_file)
    
    if combined_data:
        model = combined_data['model']
        model_details = combined_data['model_details']
        
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
        st.error("Failed to load the model or model details. Please upload the correct file.")
else:
    st.info("Please upload the combined model file.")
