%%writefile app.py
import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('model.joblib') # Make sure 'model.joblib' is in the same directory

# Create a function to predict the status of a child
def predict_status(age, weight, height, gender, other_feature): # Replace 'other_feature' with the actual feature name
    input_data = np.asarray([age, weight, height, gender, other_feature])
    input_data_reshaped = input_data.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return prediction[0]

# Streamlit app
st.title("Child Status Prediction")

# Get user input
age = st.number_input("Enter the child's age in months:", min_value=0)
weight = st.number_input("Enter the child's weight in kilograms:", min_value=0.0)
height = st.number_input("Enter the child's height in centimeters:", min_value=0.0)
gender = st.selectbox("Enter the child's gender:", ["Female", "Male"])
gender = 0 if gender == "Female" else 1
other_feature = st.number_input("Enter the value for [other feature name]:") # Replace '[other feature name]'

# Make prediction
if st.button("Predict"):
    predicted_status = predict_status(age, weight, height, gender, other_feature)
    st.success(f"The predicted status of the child is: {predicted_status}")

