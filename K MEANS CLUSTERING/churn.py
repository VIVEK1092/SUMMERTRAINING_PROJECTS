import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="centered"
)

@st.cache_resource
def load_models():
    with open('churn.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

try:
    kmeans, scaler = load_models()
except FileNotFoundError:
    st.error("Required model files ('churn.pkl' or 'scaler.pkl') are missing.")
    st.stop()

st.title("📊 Telecom Customer Churn Segmentation")
st.write("Analyze customer subscription metrics to determine risk profiles and strategic actions.")
st.write("---")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (Months)", min_value=1, max_value=72, value=12, step=1)

with col2:
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=50.0, step=0.5)

if st.button("Analyze Profile", type="primary", use_container_width=True):
    user_data = np.array([[tenure, monthly_charges]])
    user_scaled = scaler.transform(user_data)
    predicted_cluster = kmeans.predict(user_scaled)[0]
    
    st.subheader("Analysis Results")
    
    m1, m2 = st.columns(2)
    m1.metric(label="Assigned Segment", value=f"Cluster {predicted_cluster}")
    
    if predicted_cluster == 0:
        m2.metric(label="Risk Status", value="HIGH RISK", delta="- Critical", delta_color="inverse")
        st.error("**Segment Profile:** New Account & High Revenue Output")
        
    elif predicted_cluster == 1:
        m2.metric(label="Risk Status", value="MODERATE RISK", delta="- Watchlist", delta_color="off")
        st.warning("**Segment Profile:** Established Account & High Revenue Output")
        
    else:
        m2.metric(label="Risk Status", value="LOW RISK", delta="+ Stable", delta_color="normal")
        st.success("**Segment Profile:** Standard Account & Baseline Revenue Output")
        