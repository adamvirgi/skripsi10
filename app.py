
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
user_input = np.array([[age, gender, body_height, body_weight]])
user_input[:, 1] = user_input[:, 1].replace({'Female': 0, 'Male': 1})
user_input[:, 3] = minmax.transform(user_input[:, 3].reshape(-1, 1))

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
    else:
        st.write('The child is predicted to be severely stunted.')
