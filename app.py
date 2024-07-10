import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Load the trained SVM model and MinMaxScaler
model = joblib.load('svm_model.joblib')
minmax = joblib.load('minmax_scaler.joblib')

# Define the application layout
st.title('Stunting Prediction App')

# Get user input
age = st.number_input('Age (Month)', min_value=0, max_value=72)
gender = st.selectbox('Gender', ['Female', 'Male'])
body_height = st.number_input('Body Height (cm)', min_value=0, max_value=120)
body_weight = st.number_input('Body Weight (kg)', min_value=0, max_value=30)

# Preprocess user input
gender_code = 0 if gender == 'Female' else 1
user_input = np.array([[age, gender_code, body_height, body_weight]])

# Check if user_input needs normalization and normalize body_weight
if minmax is not None:
    user_input[:, 3] = minmax.transform(user_input[:, 3].reshape(-1, 1)).flatten()

# Check the shape of user_input before predicting
if user_input.shape[1] != 4:
    st.warning(f'Unexpected number of features ({user_input.shape[1]}) in user input.')

# Predict the status
prediction = model.predict(user_input)[0]

# Display the prediction
if st.button('Predict'):
    if prediction == 'stunted':
        st.write('The child is predicted to be stunted.')
    elif prediction == 'tall':
        st.write('The child is predicted to be tall.')
    elif prediction == 'normal':
        st.write('The child is predicted to be normal.')
    elif prediction == 'severe_stunting':
        st.write('The child is predicted to be severely stunted.')
