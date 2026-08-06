import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Hospital Readmission Prediction",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Hospital Readmission Prediction Dashboard")

st.markdown("""
Predict the risk of patient hospital readmission using healthcare data.

### Features
- 📊 Dataset Preview
- 📈 Exploratory Data Analysis
- 🤖 Machine Learning Prediction
- 📉 Visual Analytics
""")
import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    return pd.read_csv(
        "https://raw.githubusercontent.com/malavneha/Hospital-Readmission-Prediction/main/diabetic_data.csv"
    )

df = load_data()

st.success("✅ Dataset loaded successfully!")

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())
    st.subheader("Dataset Shape")

    col1, col2 = st.columns(2)

    col1.metric("Rows", df.shape[0])

    col2.metric("Columns", df.shape[1])

    st.subheader("Missing Values")

    st.write(df.isnull().sum())
