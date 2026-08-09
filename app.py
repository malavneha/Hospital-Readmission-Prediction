8import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "diabetic_data.csv"
MODEL_PATH = ROOT / "models" / "readmission_model.joblib"
RESULTS_PATH = ROOT / "results" / "final_test_results.json"
CV_PATH = ROOT / "results" / "cross_validation_results.csv"

st.set_page_config(
    page_title="Hospital Readmission Prediction",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Hospital Readmission Prediction")
st.caption("Research prototype • UCI Diabetes 130-US Hospitals dataset")

st.warning(
    "Research-use prototype only. This application is not a clinically validated "
    "decision-support system and should not be used for patient-care decisions."
)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    if not
MODEL_PATH.exists():
        return None
    artifact = 
joblib.load(MODEL_PATH)
    return artifact["model"] 
if isinstance(artifact, dict) 
else artifact 

df = load_data()
model = load_model()

df_clean = df.copy()
df_clean.replace("?", np.nan, inplace=True)

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Dataset", "EDA", "Model Performance", "Prediction", "About"]
)

if page == "Dashboard":
    st.header("Dashboard")
    a, b, c, d = st.columns(4)
    a.metric("Encounters", f"{len(df):,}")
    b.metric("Variables", f"{df.shape[1]:,}")
    c.metric("Readmitted", f"{(df['readmitted'].isin(['>30','<30'])).sum():,}")
    d.metric("No readmission", f"{(df['readmitted'] == 'NO').sum():,}")

    st.subheader("Dataset preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Missing values")
    missing = df_clean.isna().sum().sort_values(ascending=False)
    st.dataframe(missing.to_frame("Missing values"), use_container_width=True)

elif page == "Dataset":
    st.header("Dataset")
    st.write(
        "The dataset contains hospital encounters from the UCI Diabetes 130-US Hospitals "
        "benchmark. The target is transformed to binary readmission: NO = 0; >30 or <30 = 1."
    )
    st.dataframe(df, use_container_width=True, height=500)

elif page == "EDA":
    st.header("Exploratory Data Analysis")

    st.subheader("Binary readmission distribution")
    target = df["readmitted"].map({"NO": "No readmission", ">30": "Readmitted", "<30": "Readmitted"})
    fig, ax = plt.subplots(figsize=(7, 4))
    target.value_counts().plot(kind="bar", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Encounters")
    st.pyplot(fig, clear_figure=True)

    st.subheader("Age distribution")
    fig, ax = plt.subplots(figsize=(9, 4))
    df["age"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_xlabel("Age group")
    ax.set_ylabel("Encounters")
    st.pyplot(fig, clear_figure=True)

    st.subheader("Time in hospital")
    fig, ax = plt.subplots(figsize=(8, 4))
    df["time_in_hospital"].plot(kind="hist", bins=15, ax=ax)
    ax.set_xlabel("Days")
    ax.set_ylabel("Encounters")
    st.pyplot(fig, clear_figure=True)

    st.subheader("Missingness")
    missing = df_clean.isna().sum().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 5))
    missing.sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel("Missing values")
    st.pyplot(fig, clear_figure=True)

elif page == "Model Performance":
    st.header("Model Performance")

    if not RESULTS_PATH.exists() or not CV_PATH.exists():
        st.info(
            "The research model has not been trained in this repository yet. "
            "Run train_model.py once to create the model and results files."
        )
    else:
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        cv = pd.read_csv(CV_PATH)

        st.success(f"Selected model: {results['selected_model']}")

        m = results["metrics"]
        cols = st.columns(6)
        labels = [
            ("Accuracy", m["accuracy"]),
            ("Precision", m["precision"]),
            ("Recall", m["recall"]),
            ("F1", m["f1"]),
            ("ROC-AUC", m["roc_auc"]),
            ("PR-AUC", m["pr_auc"]),
        ]
        for col, (label, value) in zip(cols, labels):
            col.metric(label, f"{value:.3f}")

        st.subheader("Cross-validation model comparison")
        st.dataframe(cv, use_container_width=True)

        st.subheader("Confusion matrix")
        cm = np.array(results["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No readmission", "Readmitted"],
        ).plot(ax=ax, values_format="d", colorbar=False)
        st.pyplot(fig, clear_figure=True)

        st.subheader("Evaluation design")
        st.write(
            f"Patient-level holdout: {results['test_rows']:,} test encounters across "
            f"{results['test_unique_patients']:,} unique patients. "
            f"Patient overlap between training and test: {results['patient_overlap']}."
        )

elif page == "Prediction":
    st.header("🩺 Readmission Prediction")

    if model is None:
        st.info(
            "The trained research model is not available yet. Run train_model.py and "
            "place models/readmission_model.joblib in the repository."
        )
    else:
        st.write(
            "Enter encounter-level information. The form is intended to demonstrate "
            "the trained model interface; it is not a validated clinical calculator."
        )

        # Use a subset of common model inputs. Unspecified fields are filled by the
        # fitted pipeline's imputers. This is a transparent interface, not a full EHR.
        age = st.selectbox(
            "Age group",
            sorted(df["age"].dropna().unique().tolist()),
            index=min(7, len(sorted(df["age"].dropna().unique().tolist())) - 1),
        )
        gender = st.selectbox("Gender", sorted(df["gender"].dropna().unique().tolist()))
        time_in_hospital = st.slider("Time in hospital (days)", 1, 14, 4)
        num_lab_procedures = st.slider("Number of laboratory procedures", 0, 150, 40)
        num_medications = st.slider("Number of medications", 0, 100, 15)
        number_diagnoses = st.slider("Number of diagnoses", 1, 16, 8)
        number_inpatient = st.slider("Previous inpatient visits", 0, 20, 0)
        number_emergency = st.slider("Previous emergency visits", 0, 20, 0)
        number_outpatient = st.slider("Previous outpatient visits", 0, 40, 0)

        if st.button("Predict readmission risk", type="primary"):
            # Build a full row with dataset-compatible columns.
            row = {c: np.nan for c in df.columns if c not in ["readmitted", "patient_nbr", "encounter_id"]}
            row.update({
                "age": age,
                "gender": gender,
                "time_in_hospital": time_in_hospital,
                "num_lab_procedures": num_lab_procedures,
                "num_medications": num_medications,
                "number_diagnoses": number_diagnoses,
                "number_inpatient": number_inpatient,
                "number_emergency": number_emergency,
                "number_outpatient": number_outpatient,
            })
            input_df = pd.DataFrame([row])

            probability = float(model.predict_proba(input_df)[0, 1])
            prediction = int(probability >= 0.5)

            st.metric("Model-estimated probability", f"{probability:.1%}")
            if prediction:
                st.error("Model classification: higher predicted readmission risk")
            else:
                st.success("Model classification: lower predicted readmission risk")

            st.caption(
                "This probability comes from the trained research model. It is not a "
                "clinically calibrated risk score."
            )

elif page == "About":
    st.header("About this project")
    st.write(
        "This project demonstrates an end-to-end machine-learning workflow for "
        "hospital readmission prediction using the UCI Diabetes 130-US Hospitals dataset."
    )
    st.subheader("Research design")
    st.markdown(
        """
        - Patient-level train/test separation
        - Leakage-safe preprocessing with a scikit-learn Pipeline
        - One-hot encoding for categorical variables
        - Grouped cross-validation for model comparison
        - Primary model-selection metric: PR-AUC
        - Final evaluation on an untouched patient-level test set
        """
    )
    st.subheader("Responsible use")
    st.write(
        "The project is a research and educational prototype. It has not been prospectively "
        "validated, externally validated, calibrated for clinical deployment, or evaluated "
        "for clinical impact or fairness."
    )
