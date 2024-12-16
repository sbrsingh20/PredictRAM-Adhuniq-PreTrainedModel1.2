import joblib
import os
import streamlit as st

# Path to the models folder
models_folder = 'models'

# List of model files
model_files = [f for f in os.listdir(models_folder) if f.endswith('_prediction_model.pkl')]

# Streamlit Sidebar to select a model
selected_model_file = st.sidebar.selectbox('Select a stock model', model_files)

# Load the selected model and its details
if selected_model_file:
    # Load the model and its details
    model_path = os.path.join(models_folder, selected_model_file)
    model_details_path = model_path.replace('_prediction_model.pkl', '_model_details.pkl')
    
    model = joblib.load(model_path)
    model_details = joblib.load(model_details_path)
    
    # Display model details
    st.write(f"### {selected_model_file} Model Details")
    st.write(f"**Model Type:** {model_details['model']}")
    st.write(f"**Accuracy (R-squared %):** {model_details['accuracy']:.2f}%")
    st.write(f"**R-squared:** {model_details['r2_score']:.2f}")
    st.write(f"**Mean Squared Error:** {model_details['mean_squared_error']:.4f}")
    st.write(f"**Intercept:** {model_details['intercept']:.4f}")
    st.write(f"**Coefficients:**")
    st.write(f"  - Inflation Rate: {model_details['coefficients']['Inflation Rate']:.4f}")
    st.write(f"  - Interest Rate: {model_details['coefficients']['Interest Rate']:.4f}")
    st.write(f"  - VIX: {model_details['coefficients']['VIX']:.4f}")
    
    # You can also add a feature to input new data and make predictions
    # For simplicity, let's show the first few coefficients and predictions from the model
    st.write("### Sample Model Prediction")
    st.write(model.predict([[0.03, 0.05, 20]]))  # You can adjust this sample input
