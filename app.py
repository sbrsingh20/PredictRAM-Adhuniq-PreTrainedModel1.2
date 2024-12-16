import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to load the machine learning model from a .pkl file
def load_trained_model(model_filename):
    return joblib.load(model_filename)

# Function to load model details (coefficients, accuracy, etc.)
def load_model_details(model_details_filename):
    return joblib.load(model_details_filename)

# Function to plot historical stock data
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

# Upload a trained model (.pkl)
model_file = st.file_uploader("Upload a pre-trained model (.pkl)", type="pkl")

model = None
model_details = None

if model_file is not None:
    try:
        # Load the trained machine learning model
        model = load_trained_model(model_file)
        
        # Try loading the model details
        model_details_filename = model_file.name.replace('prediction_model', 'model_details')

        # Check if the model details file exists
        if os.path.exists(model_details_filename):
            model_details = load_model_details(model_details_filename)
        else:
            st.warning("Model details file not found.")
    except Exception as e:
        st.error(f"Error loading model: {e}")

# Ensure the model is loaded before proceeding
if model is not None:
    # Allow the user to select multiple stocks
    stock_files = [f for f in os.listdir('stockdata') if f.endswith('.xlsx')]
    selected_stocks = st.multiselect("Select Stocks for Historical Data and Prediction", stock_files)

    if selected_stocks:
        # Input macroeconomic parameters (outside the loop to avoid duplication issues)
        inflation = st.number_input("Inflation Rate (%)", min_value=-100.0, max_value=100.0, value=0.0, step=0.1)
        interest_rate = st.number_input("Interest Rate (%)", min_value=-100.0, max_value=100.0, value=0.0, step=0.1)
        vix = st.number_input("VIX Index", min_value=0.0, max_value=200.0, value=20.0, step=0.1)

        # Prepare input features for prediction (ensure it's a 2D array)
        input_data = np.array([[inflation, interest_rate, vix]])

        # Prediction Button
        if st.button("Predict Stock Returns"):
            for stock_file_name in selected_stocks:
                stock_file_path = os.path.join('stockdata', stock_file_name)

                # Load stock data
                stock_data = pd.read_excel(stock_file_path, parse_dates=['Date'], engine='openpyxl')
                
                # Display stock data and plot
                st.subheader(f"Historical Data for {stock_file_name}")
                plot_stock_data(stock_data)

                try:
                    # Prediction (returns based on macroeconomic inputs)
                    prediction = model.predict(input_data)
                    
                    # Check if the prediction is 1D or 2D array
                    predicted_return = prediction[0] if prediction.ndim == 1 else prediction[0, 0]
                    
                    st.write(f"Predicted Stock Return for {stock_file_name}: {predicted_return * 100:.2f}%")
                except Exception as e:
                    st.error(f"Error during prediction for {stock_file_name}: {e}")

        # If model details are available, display them
        if model_details:
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
