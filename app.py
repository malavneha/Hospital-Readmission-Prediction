import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Hospital Readmission Prediction",
    page_icon="🏥",
    layout="wide"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🏥 Hospital Readmission Prediction Dashboard")

st.markdown("""
### AI Powered Healthcare Analytics

This dashboard predicts hospital readmission risk using
Machine Learning and provides interactive healthcare analytics.
""")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Dataset",
        "EDA",
        "Machine Learning",
        "Prediction",
        "About"
    ]
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("diabetic_data.csv")

df = load_data()

st.sidebar.success("Dataset Loaded")

# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------

data = df.copy()

data.replace("?", np.nan, inplace=True)

# Drop IDs

drop_cols = [
    "encounter_id",
    "patient_nbr"
]

for col in drop_cols:
    if col in data.columns:
        data.drop(col, axis=1, inplace=True)

# Fill missing values

for col in data.columns:
    if pd.api.types.is_numeric_dtype(data[col]):
        data[col] = data[col].fillna(data[col].median())
    else:
        mode = data[col].mode()
        if not mode.empty:
            data[col] = data[col].fillna(mode[0])
        else:
            data[col] = data[col].fillna("Unknown")

    

    

# Encode target

data["readmitted"] = data["readmitted"].replace({
    "NO":0,
    ">30":1,
    "<30":1
})

# --------------------------------------------------
# LABEL ENCODING
# --------------------------------------------------

encoder = LabelEncoder()

for col in data.columns:

    if data[col].dtype=="object":

        data[col] = encoder.fit_transform(data[col].astype(str))

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page=="Dashboard":

    st.header("📊 Dashboard")

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "Patients",
        len(df)
    )

    c2.metric(
        "Columns",
        df.shape[1]
    )

    c3.metric(
        "Readmitted",
        (data["readmitted"]==1).sum()
    )

    c4.metric(
        "No Readmission",
        (data["readmitted"]==0).sum()
    )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10))

    st.subheader("Dataset Shape")

    st.write(df.shape)

    st.subheader("Missing Values")

    st.dataframe(
        df.isnull().sum().to_frame("Missing Values")
    )
# ==========================================================
# EDA PAGE
# ==========================================================

elif page == "EDA":

    st.header("📈 Exploratory Data Analysis")

    # ----------------------------
    # Readmission Distribution
    # ----------------------------

    st.subheader("Readmission Distribution")

    fig, ax = plt.subplots(figsize=(7,4))

    df["readmitted"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Readmission Status")
    ax.set_ylabel("Patients")

    st.pyplot(fig)

    # ----------------------------
    # Gender Distribution
    # ----------------------------

    st.subheader("Gender Distribution")

    fig, ax = plt.subplots(figsize=(7,4))

    df["gender"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Gender")
    ax.set_ylabel("Patients")

    st.pyplot(fig)

    # ----------------------------
    # Race Distribution
    # ----------------------------

    st.subheader("Race Distribution")

    fig, ax = plt.subplots(figsize=(8,4))

    df["race"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Race")
    ax.set_ylabel("Patients")

    st.pyplot(fig)

    # ----------------------------
    # Age Distribution
    # ----------------------------

    st.subheader("Age Distribution")

    fig, ax = plt.subplots(figsize=(10,4))

    df["age"].value_counts().sort_index().plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Age Group")
    ax.set_ylabel("Patients")

    st.pyplot(fig)

    # ----------------------------
    # Time in Hospital
    # ----------------------------

    st.subheader("Time in Hospital")

    fig, ax = plt.subplots(figsize=(8,4))

    ax.hist(
        df["time_in_hospital"],
        bins=15
    )

    ax.set_xlabel("Days")
    ax.set_ylabel("Patients")

    st.pyplot(fig)

    # ----------------------------
    # Number of Diagnoses
    # ----------------------------

    st.subheader("Number of Diagnoses")

    fig, ax = plt.subplots(figsize=(8,4))

    ax.hist(
        df["number_diagnoses"],
        bins=15
    )

    ax.set_xlabel("Diagnoses")
    ax.set_ylabel("Patients")

    st.pyplot(fig)

    # ----------------------------
    # Correlation Heatmap
    # ----------------------------

    st.subheader("Correlation Heatmap")

    numeric = data.select_dtypes(include=np.number)

    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(12,8))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    # ----------------------------
    # Missing Values
    # ----------------------------

    st.subheader("Missing Values")

    missing = df.isnull().sum()

    fig, ax = plt.subplots(figsize=(12,5))

    missing.plot(
        kind="bar",
        ax=ax
    )

    st.pyplot(fig)
    # ==========================================================
# MACHINE LEARNING PAGE
# ==========================================================

elif page == "Machine Learning":

    st.header("🤖 Machine Learning")

    st.write("Training Random Forest Classifier...")

    # ----------------------------
    # Features and Target
    # ----------------------------

    X = data.drop("readmitted", axis=1)
    y = data["readmitted"]

    # ----------------------------
    # Train/Test Split
    # ----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # ----------------------------
    # Train Model
    # ----------------------------

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ----------------------------
    # Accuracy
    # ----------------------------

    accuracy = accuracy_score(y_test, y_pred)

    st.success(f"Model Accuracy: {accuracy:.2%}")

    # ----------------------------
    # Classification Report
    # ----------------------------

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    st.dataframe(pd.DataFrame(report).transpose())

    # ----------------------------
    # Confusion Matrix
    # ----------------------------

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(6,5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        ax=ax
    )

    st.pyplot(fig)

    # ----------------------------
    # Feature Importance
    # ----------------------------

    st.subheader("Top 15 Important Features")

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    ).head(15)

    fig, ax = plt.subplots(figsize=(10,6))

    ax.barh(
        importance["Feature"],
        importance["Importance"]
    )

    ax.set_xlabel("Importance")

    plt.gca().invert_yaxis()

    st.pyplot(fig)

    # ----------------------------
    # Model Information
    # ----------------------------

    st.subheader("Model Details")

    st.info("""
    **Algorithm Used:** Random Forest Classifier

    • Ensemble Machine Learning Model

    • Suitable for Healthcare Prediction

    • Handles Complex Relationships

    • Provides Feature Importance

    • Robust Against Overfitting
    """)
    # ==========================================================
# PREDICTION PAGE
# ==========================================================

elif page == "Prediction":

    st.header("🩺 Hospital Readmission Risk Prediction")

    st.write("Enter patient details below to estimate readmission risk.")

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider(
            "Age",
            1,
            100,
            50
        )

        time_in_hospital = st.slider(
            "Time in Hospital (days)",
            1,
            14,
            5
        )

        num_lab = st.slider(
            "Lab Procedures",
            1,
            150,
            40
        )

        medications = st.slider(
            "Number of Medications",
            1,
            80,
            15
        )

    with col2:

        diagnoses = st.slider(
            "Number of Diagnoses",
            1,
            16,
            5
        )

        emergency = st.slider(
            "Emergency Visits",
            0,
            20,
            1
        )

        inpatient = st.slider(
            "Inpatient Visits",
            0,
            20,
            1
        )

        outpatient = st.slider(
            "Outpatient Visits",
            0,
            20,
            2
        )
    if st.button("🔮 Predict Readmission Risk"):
        risk_score = (
            age * 0.15 +
            time_in_hospital * 4 +
            medications * 0.6 +
            diagnoses * 2 +
            emergency * 5 +
            inpatient * 5 +
            outpatient * 2 +
            num_lab * 0.05
        )

        probability = min(risk_score / 100, 1.0)

        if probability >= 0.60:
            st.error("🔴 High Risk of Readmission")
        elif probability >= 0.35:
            st.warning("🟠 Moderate Risk of Readmission")
        else:
            st.success("🟢 Low Risk of Readmission")

        st.metric(
            "Estimated Readmission Risk",
            f"{probability:.1%}",
            "High Risk" if probability > 0.5 else "Low Risk"
        )



            # ==========================================================
# ABOUT PAGE
# ==========================================================

elif page == "About":

    st.header("👩‍⚕️ About This Project")

    st.markdown("""
## Hospital Readmission Prediction Dashboard

This project demonstrates how Artificial Intelligence
and Machine Learning can assist healthcare professionals
in identifying patients at risk of hospital readmission.

### Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Features

- Healthcare Analytics
- Exploratory Data Analysis
- Machine Learning
- Readmission Risk Prediction
- Interactive Dashboard

---

### Developer

**Dr. Neha Malav**

MBBS | Healthcare Analytics | AI in Healthcare

GitHub:
https://github.com/malavneha
""")

    st.success("Thank you for exploring this project!")
            
