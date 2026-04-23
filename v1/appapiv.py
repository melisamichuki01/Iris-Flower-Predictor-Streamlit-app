import streamlit as st
import numpy as np
import requests

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌸",
    layout="centered"
)

# ── API URL ────────────────────────────────────────────────
API_URL = "https://iris-flower-predictor-streamlit-app.onrender.com/v1/predict"

# ── UI ─────────────────────────────────────────────────────
st.title("🌸 Iris Flower Predictor (API Powered)")
st.write("This app sends input to a FastAPI backend and returns predictions.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
    sepal_width  = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)

with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
    petal_width  = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

st.divider()

# ── Predict Button ─────────────────────────────────────────
if st.button("🔍 Predict Species", use_container_width=True):

    # Prepare payload for API
    payload = {
        "features": [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    }

    try:
        # Call FastAPI
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()  # raises error for bad responses

        result = response.json()

        # ── Display Results ────────────────────────────────
        st.success(f"Predicted Species: {result['class_name']}")
        st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")

        st.info(f"Raw prediction index: {result['prediction']}")

    except requests.exceptions.ConnectionError:
        st.error(" Cannot connect to API. Is FastAPI running?")
    except requests.exceptions.HTTPError as e:
        st.error(f" API Error: {e.response.text}")