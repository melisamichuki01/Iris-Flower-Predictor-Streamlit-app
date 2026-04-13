import streamlit as st
import pickle
import numpy as np
# ── Page Config ────────────────────────────────────────────

st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌸",
    layout="centered"
)
# ── Load the Saved Model ───────────────────────────────────

@st.cache_resource   # Caches the model so it loads only once
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model
model = load_model()
# ── App UI ─────────────────────────────────────────────────
st.title("🌸Iris Flower Predictor")
st.write("Enter the flower measurements below and click **Predict** to find out the species!")
st.divider()
# Input sliders
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
    sepal_width  = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.4)
with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
    petal_width  = st.slider("Petal Width (cm)",  0.1, 2.5, 0.2)
st.divider()
# Predict button
if st.button("🔍Predict Species", use_container_width=True):
# Prepare input for the model
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    # Map number to flower name
    flower_names = ["Setosa ", "Versicolor ", "Virginica "]

    predicted_flower = flower_names[prediction]
    confidence = probability[prediction] * 100
    # Display results
    st.success(f"Predicted Species: {predicted_flower}")
    st.metric(label="Confidence", value=f"{confidence:.1f}%")
    