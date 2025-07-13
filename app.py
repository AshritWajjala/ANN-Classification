import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import dill

# Load the trained model
model = tf.keras.models.load_model('model.keras')

# Load the preprocessor
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = dill.load(f)

# Streamlit app
st.title("Customer Churn Prediction")

# User input
geography_options = preprocessor.named_transformers_['cat'].categories_[0]
gender_options = preprocessor.named_transformers_['cat'].categories_[1]
geography = st.selectbox('Geography', geography_options)
gender = st.selectbox('Gender', gender_options)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

input_data_scaled = preprocessor.transform(input_data)

# making prediction
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

# Display result
st.subheader("Prediction Result:")
if prediction_probability >= 0.5:
    st.error(f"The customer is **likely to churn**. 🔴\nProbability: {prediction_probability:.2f}")
else:
    st.success(f"The customer is **unlikely to churn**. 🟢\nProbability: {prediction_probability:.2f}")
