import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

df = pd.read_csv("SATvsGPA.csv")
df = df.dropna(subset=['SAT', 'GPA'])

x = df[['SAT']]
y = df['GPA']

x_train, x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("--- Model Evaluation Metrics ---")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2 Score): {r2:.4f}")
print(f"Intercept (c): {lr.intercept_:.4f}")
print(f"Slope/Coefficient (m): {lr.coef_[0]:.4f}")

st.set_page_config(page_title="SAT vs GPA Predictor", page_icon="🎓", layout="centered")

st.title("SAT vs GPA Predictor")
st.write("Check your predicted GPA using a Linear Regression model.")
st.divider()

sat_input = st.number_input(
    label="Enter SAT Score (400 - 1600):",
    min_value=400,
    max_value=1600,
    value=1200, 
    step=10
)

if st.button("Predict GPA", type="primary"):
    prediction = lr.predict([[sat_input]])
    predicted_gpa = np.clip(prediction[0], 1.0, 4.0)
    
    st.success("Prediction Successful!")
    st.metric(label="Predicted GPA", value=f"{predicted_gpa:.2f}")
    
    if predicted_gpa >= 3.5:
        st.balloons() 
        st.write("Good Score ... ")
    elif predicted_gpa < 2.5:
        st.write("Need Improvement !!!")