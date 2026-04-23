import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris

@st.cache_data
def load_data():
    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    df["species"] = iris.target
    df["species"] = df["species"].map({
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    })

    return df