import streamlit as st
import plotly.express as px
from utils import load_data

st.title("🔍 Exploratory Data Analysis")

df = load_data()

# -------------------------
# Feature distribution
# -------------------------
feature = st.selectbox("Select Feature", df.columns[:-1])

fig = px.histogram(
    df,
    x=feature,
    color="species",
    barmode="overlay",
    title=f"{feature} Distribution by Species"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Scatter plot
# -------------------------
st.subheader("Feature Relationships")

x_axis = st.selectbox("X-axis", df.columns[:-1], key="x")
y_axis = st.selectbox("Y-axis", df.columns[:-1], key="y")

fig2 = px.scatter(
    df,
    x=x_axis,
    y=y_axis,
    color="species",
    title=f"{x_axis} vs {y_axis}"
)

st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# Insights
# -------------------------
st.subheader("🧠 Key Insights")

st.markdown("""
- Setosa is clearly separable from other classes  
- Petal features are the strongest predictors  
- Versicolor and Virginica slightly overlap  
""")