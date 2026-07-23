import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

@st.cache_resource
def load_and_train():
    try:
        df = pd.read_csv("titanic.csv")
    except FileNotFoundError:
        return None, None
        
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    
    x = df.drop(columns=['Survived'])
    y = df['Survived']
    
    x_train, x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    lr = LogisticRegression(max_iter=1000, solver='lbfgs')
    lr.fit(x_train, y_train)
    
    y_pred = lr.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("--- Logistic Regression Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return lr, x.columns

lr, feature_columns = load_and_train()

if lr is None:
    st.error("Error: 'titanic.csv' file nahi mili. Check karein ki file aapki Python script ke sath sahi folder me rakhi hai.")
    st.stop()

st.title("🚢 Titanic Survival Predictor")
st.write("Predict whether a passenger would survive the Titanic disaster based on demographic and ticketing data.")
st.divider()

pclass = st.selectbox(label="Ticket Class (1st, 2nd, 3rd):", options=[1, 2, 3])
age = st.number_input(label="Age (in years):", min_value=1, max_value=100, value=30)
sibsp = st.number_input(label="Number of Siblings/Spouses Aboard:", min_value=0, max_value=10, value=0)
parch = st.number_input(label="Number of Parents/Children Aboard:", min_value=0, max_value=10, value=0)
fare = st.number_input(label="Ticket Fare (in £):", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
sex = st.selectbox(label="Sex:", options=["male", "female"])
embarked = st.selectbox(label="Port of Embarkation:", options=["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])

if st.button("Predict Survival", type="primary"):
    embarked_code = "S" if "Southampton" in embarked else "C" if "Cherbourg" in embarked else "Q"
    
    input_data = pd.DataFrame([{
        'Pclass': pclass,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Sex': sex,
        'Embarked': embarked_code
    }])
    
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    
    prediction = lr.predict(input_encoded)
    
    st.divider()
    if prediction[0] == 1:
        st.balloons()
        st.success("The passenger is predicted to SURVIVE.")
    else:
        st.error("The passenger is predicted to NOT SURVIVE.")