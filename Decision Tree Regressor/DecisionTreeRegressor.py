import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

st.set_page_config(page_title="Data Science Salary Predictor", page_icon="💰", layout="centered")

@st.cache_resource
def load_and_train():
    np.random.seed(42)
    experience = np.random.uniform(1, 15, 200)
    test_score = np.random.uniform(50, 100, 200)
    salary = (experience * 8000) + (test_score * 300) + np.random.normal(0, 5000, 200)
    
    df = pd.DataFrame({'Experience': experience, 'TestScore': test_score, 'Salary': salary})
    
    x = df[['Experience', 'TestScore']]
    y = df['Salary']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    dtr = DecisionTreeRegressor(max_depth=5, random_state=42)
    dtr.fit(x_train, y_train)
    
    y_pred = dtr.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("--- Decision Tree Regressor Metrics ---")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    return dtr, x.columns

dtr, feature_columns = load_and_train()

st.title("💰 Data Science Salary Predictor")
st.write("Predict estimated salary based on years of experience and technical test scores using a Decision Tree Regressor.")
st.divider()

experience_input = st.number_input("Years of Experience:", min_value=0.0, max_value=20.0, value=5.0, step=0.5)
test_score_input = st.number_input("Technical Test Score (0-100):", min_value=0.0, max_value=100.0, value=75.0, step=1.0)

if st.button("Predict Salary", type="primary"):
    input_data = pd.DataFrame([{
        'Experience': experience_input,
        'TestScore': test_score_input
    }])
    
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)
    prediction = dtr.predict(input_data)
    
    st.divider()
    st.success("Prediction Successful!")
    st.metric(label="Estimated Salary ($):", value=f"{prediction[0]:,.2f}")