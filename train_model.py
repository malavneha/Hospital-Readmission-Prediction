"""
Research-grade training and evaluation pipeline for the Hospital Readmission project.

Run:
    python train_model.py

The script:
1. Loads the UCI Diabetes 130-US Hospitals CSV.
2. Creates the binary target used by the current project.
3. Uses patient_nbr only for grouping; it is never used as a model feature.
4. Splits patients into train/test groups.
5. Fits all preprocessing steps on the training data only.
6. Compares Logistic Regression, Random Forest, and Extra Trees.
7. Reports accuracy, precision, recall, F1, ROC-AUC and PR-AUC.
8. Selects the model by cross-validated positive-class F1 on the training set.
9. Refits the selected pipeline on the training partition.
10. Saves a complete joblib artifact for the Streamlit application.
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = "diabetic_data.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "readmission_model.joblib")
RANDOM_STATE = 42
TEST_SIZE = 0.20

def load_dataset():
    df = pd.read_csv(DATA_PATH)
    if "readmitted" not in df.columns:
        raise ValueError("Expected a 'readmitted' target column.")
    if "patient_nbr" not in df.columns:
        raise ValueError("Expected 'patient_nbr' for patient-level evaluation.")
    return df

def make_target(df):
    y = df["readmitted"].map({"NO": 0, ">30": 1, "<30": 1})
    if y.isna().any():
        raise ValueError("Unexpected values found in readmitted target.")
    return y.astype(int)

def make_features(df):
    # Patient ID is used for grouping only and is never a predictive feature.
    groups = df["patient_nbr"].copy()
    X = df.drop(columns=["readmitted", "encounter_id", "patient_nbr"], errors="ignore").copy()
    X = X.replace("?", np.nan)
    return X, groups

def build_preprocessor(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    ), numeric, categorical

def models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", solver="liblinear",
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1,
            random_state=RANDOM_STATE, min_samples_leaf=2
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1,
            random_state=RANDOM_STATE, min_samples_leaf=2
        ),
    }

def evaluate(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }

def make_schema(X):
    schema = []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            default = float(X[c].median()) if not X[c].dropna().empty else 0.0
            schema.append({"name": c, "kind": "numeric", "default": default})
        else:
            vals = X[c].dropna().astype(str).value_counts().index.tolist()
            if not vals:
                vals = ["Unknown"]
            vals = vals[:100]
            schema.append({
                "name": c, "kind": "categorical",
                "default": vals[0], "choices": vals
            })
    return schema

def main():
    df = load_dataset()
    y = make_target(df)
    X, groups = make_features(df)

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    g_train = groups.iloc[train_idx]

    preprocessor, numeric, categorical = build_preprocessor(X_train)
    model_defs = models()

    comparison = []
    cv = GroupKFold(n_splits=3)

    print("\nCross-validated model comparison")
    print("-" * 70)

    selected_name = None
    best_cv_f1 = -np.inf

    for name, estimator in model_defs.items():
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", estimator),
        ])
        scores = cross_val_score(
            pipe, X_train, y_train,
            groups=g_train,
            cv=cv,
            scoring="f1",
            n_jobs=1,
        )
        mean_f1 = float(scores.mean())
        sd_f1 = float(scores.std())
        print(f"{name}: CV F1={mean_f1:.4f} ± {sd_f1:.4f}")

        if mean_f1 > best_cv_f1:
            best_cv_f1 = mean_f1
            selected_name = name

    print(f"\nSelected by training-set group CV F1: {selected_name}")

    final_pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model_defs[selected_name]),
    ])
    final_pipe.fit(X_train, y_train)

    y_prob = final_pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)
    test_metrics = evaluate(y_test, y_pred, y_prob)

    print("\nHeld-out patient-group test performance")
    print("-" * 70)
    print(json.dumps(test_metrics, indent=2))
    print("\nClassification report")
    print(classification_report(y_test, y_pred, digits=4))
    print("\nConfusion matrix")
    print(confusion_matrix(y_test, y_pred))

    # Evaluate all candidates on the held-out test set for transparent comparison.
    for name, estimator in model_defs.items():
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.50).astype(int)
        m = evaluate(y_test, pred, prob)
        comparison.append({"model": name, **m})

    # Extract global importance where the selected estimator exposes it.
    feature_importance = []
    fitted_model = final_pipe.named_steps["model"]
    fitted_pre = final_pipe.named_steps["preprocess"]
    try:
        names = fitted_pre.get_feature_names_out()
        if hasattr(fitted_model, "feature_importances_"):
            vals = fitted_model.feature_importances_
            order = np.argsort(vals)[::-1][:30]
            feature_importance = [
                (str(names[i]), float(vals[i])) for i in order
            ]
    except Exception:
        pass

    os.makedirs(MODEL_DIR, exist_ok=True)
    artifact = {
        "model": final_pipe,
        "model_name": selected_name,
        "test_metrics": test_metrics,
        "comparison": comparison,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_importance": feature_importance,
        "input_schema": make_schema(X_train),
        "threshold": 0.50,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "evaluation": "patient-group held-out test split; preprocessing fitted on training data only",
        "dataset_rows": int(len(df)),
        "dataset_columns": int(df.shape[1]),
    }
    joblib.dump(artifact, MODEL_PATH)

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_model": selected_name,
                "test_metrics": test_metrics,
                "comparison": comparison,
            },
            f,
            indent=2,
        )

    print(f"\nSaved model artifact to {MODEL_PATH}")

if __name__ == "__main__":
    main()
