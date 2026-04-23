import streamlit as st

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Flower Predictor")

st.write("""
Welcome to an interactive machine learning application built around the Iris dataset.

Use the sidebar to navigate through:
- 📊 Dataset overview  
- 🔍 Exploratory Data Analysis  
- 🌸 Prediction tool  
""")



st.markdown("### 🚀 What this project demonstrates")

st.write("""
- End-to-end ML workflow  
- Data exploration & visualization  
- Model inference interface  
- Clean multipage Streamlit architecture  
""")