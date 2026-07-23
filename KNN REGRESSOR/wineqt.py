import streamlit as st
import numpy as np
import pickle

with open('knn_model.pkl', 'rb') as file:
    knn = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title("Wine Quality Prediction App")

fixed_acidity = st.number_input("Fixed Acidity", value=7.4)
volatile_acidity = st.number_input("Volatile Acidity", value=0.70)
citric_acid = st.number_input("Citric Acid", value=0.00)
residual_sugar = st.number_input("Residual Sugar", value=1.9)
chlorides = st.number_input("Chlorides", value=0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0)
density = st.number_input("Density", value=0.9978)
pH = st.number_input("pH", value=3.51)
sulphates = st.number_input("Sulphates", value=0.56)
alcohol = st.number_input("Alcohol", value=9.4)

if st.button("Predict Quality"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                            chlorides, free_sulfur_dioxide, total_sulfur_dioxide, 
                            density, pH, sulphates, alcohol]])
    
    input_scaled = scaler.transform(input_data)
    prediction = knn.predict(input_scaled)
    st.success(f"Predicted Wine Quality: {prediction[0]:.2f}")