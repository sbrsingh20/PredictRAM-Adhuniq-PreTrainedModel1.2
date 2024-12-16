import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Set the app title and description
st.title("Stock Return Prediction App")
st.write("""
    This app allows you to:
    - Upload a pre-trained machine learning model (.pkl file).
    - View model details such as parameters and accuracy.
    - Select multiple stocks for prediction.
    - Input macroeconomic parameters like inflation, interest rate, and VIX scenario.
    - Predict stock returns based on the input parameters.
    - Visualize historical stock data and predicted returns.
""")

# Step 1: Upload a pre-trained model
model_file = st.file_uploader("Upload a Pre-Trained Model (.pkl)", type=["pkl"])

if model_file is not None:
    try:
        # Load the model and display its details
        model_details = joblib.load(model_file)

        # Print the available keys to debug and check the structure of the model
        st.write("Loaded model details:")
        st.write(model_details.keys())

        # Step 2: Ensure the required keys exist before trying to access them
        if 'model' in model_details and 'accuracy' in model_details:
            # Display the model details
            st.subheader("Model Details")
            st.write(f"**Model Type:** {model_details['model']}")
            st.write(f"**Accuracy (R²):** {model_details['accuracy']}%")
            st.write(f"**R² Score:** {model_details['r2_score']}")
            st.write(f"**Mean Squared Error:** {model_details['mean_squared_error']}")
            st.write(f"**Intercept:** {model_details['intercept']}")
            st.write(f"**Coefficients:**")
            st.write(model_details['coefficients'])
        else:
            st.error("Model details do not contain expected keys ('model', 'accuracy', etc.). Please check the uploaded file.")

    except Exception as e:
        st.error(f"Error loading model: {e}")

    # Step 3: Select Stocks for Prediction
    stock_options = st.multiselect("Select Stocks for Prediction", options=["Stock1", "Stock2", "Stock3"])  # List of available stocks
    if stock_options:
        for stock in stock_options:
            st.write(f"Prediction for {stock}: (Stock data and prediction will be displayed here)")

    # Step 4: Input Macroeconomic Parameters
    st.subheader("Input Macroeconomic Parameters")
    
    inflation = st.number_input("Inflation Rate (%)", value=2.5, min_value=-100.0, max_value=100.0, step=0.1)
    interest_rate = st.number_input("Interest Rate (%)", value=3.0, min_value=-100.0, max_value=100.0, step=0.1)
    vix = st.number_input("VIX Value", value=20.0, min_value=0.0, max_value=100.0, step=0.1)

    # Prepare input data for prediction
    macroeconomic_data = np.array([[inflation, interest_rate, vix]])

    # Step 5: Make Prediction for Selected Stocks
    if st.button("Predict Stock Returns"):
        if model_file:
            # Predict the stock returns using the uploaded model
            X = macroeconomic_data
            y_pred = model_details['model'].predict(X)

            # Display predicted returns
            st.subheader("Predicted Stock Returns")
            st.write(f"Predicted returns: {y_pred[0]:.4f}")

            # Visualization of Historical Data (Example for 1 stock)
            st.subheader("Historical Stock Data")
            try:
                # Load historical stock data (this is a placeholder, replace with your actual data source)
                stock_data = pd.read_csv("stock_data.csv")  # Load historical stock data from CSV or API
                st.line_chart(stock_data['Adj Close'])
            except FileNotFoundError:
                st.error("Historical stock data file not found. Please ensure you have the correct file.")

            # Visualizing Predictions vs Historical Returns
            st.subheader("Predicted vs Historical Returns")
            fig, ax = plt.subplots()
            ax.plot(stock_data['Date'], stock_data['Adj Close'], label="Historical Data", color='blue')
            ax.plot(stock_data['Date'], np.repeat(y_pred[0], len(stock_data)), label="Predicted Return", color='red', linestyle='--')
            ax.set_xlabel('Date')
            ax.set_ylabel('Stock Price')
            ax.legend()
            st.pyplot(fig)

# Run the app with the command below:
# streamlit run app.py
