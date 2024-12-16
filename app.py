import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt

# Function to load models from .pkl files
def load_model(model_filename):
    return joblib.load(model_filename)

def load_model_details(model_details_filename):
    return joblib.load(model_details_filename)

# Function to plot stock data
def plot_stock_data(stock_data):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(stock_data.index, stock_data['Adj Close'], label="Adjusted Close", color='b')
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close Price")
    ax.set_title("Historical Stock Data")
    ax.legend()
    st.pyplot(fig)

# Streamlit app interface
st.title("Stock Prediction and Model Evaluation")

# Upload a trained model file
model_file = st.file_uploader("Upload a pre-trained model (.pkl)", type="pkl")

if model_file is not None:
    # Load the model and its details
    model = joblib.load(model_file)
    model_details_filename = model_file.name.replace('prediction_model', 'model_details')
    
    if os.path.exists(model_details_filename):
        model_details = load_model_details(model_details_filename)
        st.subheader("Model Details")
        st.write(f"Model Type: {model_details['model']}")
        st.write(f"Accuracy: {model_details['accuracy']:.2f}%")
        st.write(f"R2 Score: {model_details['r2_score']:.2f}")
        st.write(f"Mean Squared Error: {model_details['mean_squared_error']:.2f}")
        st.write("Model Coefficients:")
        st.write(f"Inflation Rate: {model_details['coefficients']['Inflation Rate']}")
        st.write(f"Interest Rate: {model_details['coefficients']['Interest Rate']}")
        st.write(f"VIX: {model_details['coefficients']['VIX']}")
        st.write(f"Intercept: {model_details['intercept']}")
    else:
        st.write("Model details file not found!")

    # Input macroeconomic parameters
    st.subheader("Enter Macroeconomic Parameters for Prediction")
    inflation = st.number_input("Inflation Rate (%)", min_value=-100.0, max_value=100.0, value=0.0, step=0.1)
    interest_rate = st.number_input("Interest Rate (%)", min_value=-100.0, max_value=100.0, value=0.0, step=0.1)
    vix = st.number_input("VIX Index", min_value=0.0, max_value=200.0, value=20.0, step=0.1)

    # Prepare input features for prediction
input_data = np.array([[inflation, interest_rate, vix]])  # This already ensures it's a 2D array

# Make prediction if model is loaded
if st.button("Predict Stock Returns"):
    # Prediction (returns based on macroeconomic inputs)
    predicted_return = model.predict(input_data)[0]  # This extracts the first (and only) prediction
    st.write(f"Predicted Stock Return: {predicted_return * 100:.2f}%")

    # Display historical stock data and make prediction
    st.subheader("Select Stock for Historical Data and Prediction")
    stock_file_name = st.selectbox("Select Stock File", model_details['model'].keys())
    
    stock_data = pd.read_excel(stock_file_name, parse_dates=['Date'], engine='openpyxl')
    plot_stock_data(stock_data)
