import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

@st.cache_resource
def load_and_train():
    try:
        df = pd.read_csv("heart.csv")
    except FileNotFoundError:
        return None, None
        
    df = df.dropna()
    
    if len(df) <= 5:
        return "INSUFFICIENT_DATA", None
        
    main_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach']
    
    for col in main_features:
        if col not in df.columns:
            return "MISSING_COLUMNS", None

    x = df[main_features]
    y = df['target']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(x_train, y_train)
    
    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("--- Random Forest Classifier Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    
    return rf, x.columns

rf, feature_columns = load_and_train()

if rf is None:
    st.error("Error: 'heart.csv' file nahi mili. Sahi folder me check karein.")
    st.stop()
elif rf == "INSUFFICIENT_DATA":
    st.error("Error: Dataset me rows bohot kam hain (n_samples <= 5). Sahi CSV file upload karein.")
    st.stop()
elif rf == "MISSING_COLUMNS":
    st.error("Error: Dataset me required medical columns missing hain.")
    st.stop()

st.title("❤️ Heart Disease Predictor")
st.write("Predict the presence of heart disease based on key medical attributes using a Random Forest Classifier.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age:", min_value=1, max_value=120, value=54)
    sex = st.selectbox("Sex (0 = Female, 1 = Male):", options=[1, 0])
    cp = st.selectbox("Chest Pain Type (0-3):", options=[0, 1, 2, 3])

with col2:
    trestbps = st.number_input("Resting Blood Pressure (mm Hg):", min_value=50, max_value=250, value=130)
    chol = st.number_input("Serum Cholestoral (mg/dl):", min_value=100, max_value=600, value=240)
    thalach = st.number_input("Maximum Heart Rate Achieved:", min_value=60, max_value=250, value=150)

if st.button("Predict Health Status", type="primary"):
    input_data = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'thalach': thalach
    }])
    
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)
    prediction = rf.predict(input_data)
    
    st.divider()
    if prediction[0] == 1:
        st.error("Warning: High risk of heart disease detected.")
    else:
        st.success("Good news: Low risk of heart disease detected.")