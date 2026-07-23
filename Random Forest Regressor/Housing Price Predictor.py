import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

st.set_page_config(page_title="Housing Price Predictor", page_icon="🏠", layout="centered")

@st.cache_resource
def load_and_train():
    try:
        df = pd.read_csv("HousingData.csv")
    except FileNotFoundError:
        return None, None
        
    main_features = ['RM', 'LSTAT', 'DIS', 'CRIM']
    for col in main_features:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    x = df[main_features]
    y = df['MEDV']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_split=2, random_state=42)
    rf.fit(x_train, y_train)
    
    y_pred = rf.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("--- Optimized Random Forest Regressor Metrics ---")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    return rf, x.columns

rf, feature_columns = load_and_train()

if rf is None:
    st.error("Error: 'HousingData.csv' file nahi mili. Check karein ki file sahi folder me hai.")
    st.stop()

st.title("🏠 Housing Price Predictor")
st.write("Predict estimated property values using an optimized Random Forest Regressor model.")
st.divider()

rm = st.number_input("Average Rooms per Dwelling (RM):", min_value=1.0, max_value=10.0, value=6.2, step=0.1)
lstat = st.number_input("Percentage of Lower Status Population (LSTAT):", min_value=0.0, max_value=50.0, value=12.5, step=0.5)
dis = st.number_input("Weighted Distances to Employment Centers (DIS):", min_value=0.0, max_value=20.0, value=3.8, step=0.1)
crim = st.number_input("Crime Rate (CRIM):", min_value=0.0, max_value=100.0, value=0.1, step=0.01, format="%.2f")

if st.button("Predict Price", type="primary"):
    input_data = pd.DataFrame([{
        'RM': rm, 
        'LSTAT': lstat, 
        'DIS': dis, 
        'CRIM': crim
    }])
    
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)
    prediction = rf.predict(input_data)
    
    st.divider()
    st.success("Prediction Successful!")
    st.metric(label="Estimated House Value (MEDV in $1000s):", value=f"${prediction[0]:.2f}k")