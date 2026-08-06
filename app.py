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
