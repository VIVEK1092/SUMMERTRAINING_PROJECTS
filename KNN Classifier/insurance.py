import streamlit as st
import numpy as np
import pickle

# 1. मॉडल और स्केलर लोड करना
with open('knn_classifier.pkl', 'rb') as file:
    knn = pickle.load(file)

with open('scaler_classifier.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title("Smoker Prediction App")
st.write("Enter details to predict if the person is a smoker or not based on insurance data.")

# 2. यूजर इनपुट
age = st.number_input("Age", min_value=1, max_value=100, value=25)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x==0 else "Male")
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
region = st.selectbox("Region", options=[0, 1, 2, 3], 
                      format_func=lambda x: ["Northeast", "Northwest", "Southeast", "Southwest"][x])
charges = st.number_input("Insurance Charges ($)", min_value=100.0, value=5000.0, step=100.0)

# 3. प्रेडिक्शन बटन
if st.button("Predict Smoker Status"):
    # फीचर्स का एरे बनाना
    input_data = np.array([[age, sex, bmi, children, region, charges]])
    
    # स्केल और प्रेडिक्ट करना
    input_scaled = scaler.transform(input_data)
    prediction = knn.predict(input_scaled)
    
    # रिजल्ट दिखाना
    if prediction[0] == 'yes':
        st.error("🚬 The person is likely a smoker.")
    else:
        st.success("🚭 The person is likely a non-smoker.")