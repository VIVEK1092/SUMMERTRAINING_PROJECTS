import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="centered")

@st.cache_resource
def load_and_train():
    try:
        df = pd.read_csv("WineQT.csv")
    except FileNotFoundError:
        return None, None
        
    df = df.dropna()
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
        
    x = df.drop(columns=['quality'])
    y = df['quality']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(x_train, y_train)
    
    y_pred = dt.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("--- Decision Tree Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    
    return dt, x.columns

dt, feature_columns = load_and_train()

if dt is None:
    st.error("Error: 'WineQT.csv' file nahi mili. Check karein ki file sahi folder me hai.")
    st.stop()

st.title("🍷 Wine Quality Predictor")
st.write("Predict the quality score of wine based on its chemical properties using a Decision Tree Classifier.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity:", min_value=0.0, max_value=20.0, value=8.3, step=0.1)
    volatile_acidity = st.number_input("Volatile Acidity:", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
    citric_acid = st.number_input("Citric Acid:", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    residual_sugar = st.number_input("Residual Sugar:", min_value=0.0, max_value=20.0, value=2.2, step=0.1)
    chlorides = st.number_input("Chlorides:", min_value=0.0, max_value=1.0, value=0.08, step=0.001, format="%.3f")
    free_sulfur = st.number_input("Free Sulfur Dioxide:", min_value=0.0, max_value=100.0, value=15.0, step=1.0)

with col2:
    total_sulfur = st.number_input("Total Sulfur Dioxide:", min_value=0.0, max_value=400.0, value=46.0, step=1.0)
    density = st.number_input("Density:", min_value=0.9, max_value=1.1, value=0.996, step=0.001, format="%.4f")
    ph = st.number_input("pH Level:", min_value=0.0, max_value=14.0, value=3.3, step=0.01)
    sulphates = st.number_input("Sulphates:", min_value=0.0, max_value=2.0, value=0.6, step=0.01)
    alcohol = st.number_input("Alcohol Percentage:", min_value=0.0, max_value=20.0, value=10.5, step=0.1)

if st.button("Predict Quality", type="primary"):
    input_data = pd.DataFrame([{
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur,
        'total sulfur dioxide': total_sulfur,
        'density': density,
        'pH': ph,
        'sulphates': sulphates,
        'alcohol': alcohol
    }])
    
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)
    prediction = dt.predict(input_data)
    
    st.divider()
    st.success("Prediction Successful!")
    st.metric(label="Predicted Wine Quality Rating (3 - 8):", value=int(prediction[0]))