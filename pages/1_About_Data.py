import streamlit as st
from utils import load_data

st.title("📊 About the Iris Dataset")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.sample(10, random_state=42))

st.subheader("Feature Explanation")

st.markdown("""
- **Sepal Length**: Length of outer flower part  
- **Sepal Width**: Width of outer flower part  
- **Petal Length**: Length of inner petal  
- **Petal Width**: Width of inner petal  
""")

st.subheader("Class Distribution")
st.write(df["species"].value_counts())

st.info("Dataset loaded from scikit-learn for reproducibility.")