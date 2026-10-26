import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
if not DATA_DIR.exists():
    DATA_DIR = PROJECT_ROOT / "model" / "data"
DATA_FILE = DATA_DIR / "Train.csv"
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("credit_frontend")

st.set_page_config(page_title="Credit Score Demo", layout="wide")
st.title("Bank Credit Score Classification")
data = pd.read_csv(DATA_FILE)

with st.expander("Data overview", expanded=True):
    st.write("Credit score classes")
    st.bar_chart(data["Credit_Score"].value_counts())
    left, right = st.columns(2)
    left.write("Missing values")
    left.dataframe(data.isna().sum().rename("Missing").to_frame())
    right.write("Numeric feature correlations")
    numeric = data.select_dtypes(include="number").drop(columns=["ID"], errors="ignore")
    right.dataframe(numeric.corr().round(2))
    selected = st.selectbox("Numeric feature distribution", list(numeric.columns))
    values = numeric[selected].dropna().to_numpy()
    counts, edges = np.histogram(values, bins=10)

    def short_value(value):
        if abs(value) >= 1000:
            return f"{value / 1000:.1f}k"
        return f"{value:.1f}"

    ranges = [f"{short_value(edges[i])}–{short_value(edges[i + 1])}" for i in range(len(counts))]
    histogram = pd.DataFrame({"Rows": counts}, index=ranges)
    st.bar_chart(histogram)

st.header("Model results")
try:
    result = requests.get(f"{API_URL}/metrics", timeout=5)
    result.raise_for_status()
    metrics = result.json()
    st.write(f"Model selected using validation weighted F1: **{metrics['selected_model']}**")
    rows = [{"Model": name, "Validation accuracy": item["validation"]["accuracy"],
             "Validation weighted F1": item["validation"]["weighted_f1"],
             "Test accuracy": item["test"]["accuracy"], "Test weighted F1": item["test"]["weighted_f1"]}
            for name, item in metrics["models"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.subheader("Per-class error analysis")
    analysis_model = st.selectbox("Model to analyze", list(metrics["models"]), key="analysis_model")
    analysis_split = st.selectbox("Evaluation data", ["test", "validation"], key="analysis_split")
    analysis = metrics["models"][analysis_model][analysis_split]
    detail_rows = []
    for label in analysis["class_labels"]:
        error = analysis["per_class_errors"][label]
        class_metrics = analysis["report"][label]
        detail_rows.append({
            "Class": label,
            "Examples": error["support"],
            "Correct": error["correct"],
            "Misclassified": error["misclassified"],
            "Error rate": error["error_rate"],
            "Precision": class_metrics["precision"],
            "Recall": class_metrics["recall"],
            "F1": class_metrics["f1-score"],
            "Most often predicted as": error["most_common_misclassification"] or "None",
        })
    st.dataframe(pd.DataFrame(detail_rows), hide_index=True)
    st.caption("Rows are actual classes; columns are predicted classes. Off-diagonal values are errors.")
    labels = analysis["class_labels"]
    confusion = pd.DataFrame(analysis["confusion_matrix"],
                             index=[f"Actual {label}" for label in labels],
                             columns=[f"Predicted {label}" for label in labels])
    st.dataframe(confusion)
except requests.RequestException:
    st.info("Model metrics will appear after the backend finishes training.")

st.header("Try a prediction")
st.caption("Values below are example inputs. The result is a model estimate, not a lending decision.")
with st.form("prediction"):
    a, b, c = st.columns(3)
    features = {
        "Age": a.number_input("Age", 14.0, 100.0, 33.0),
        "Annual_Income": a.number_input("Annual income", 0.0, 1_000_000.0, 50000.0),
        "Monthly_Inhand_Salary": a.number_input("Monthly salary", 0.0, 100_000.0, 4000.0),
        "Interest_Rate": b.number_input("Interest rate", 0.0, 100.0, 14.0),
        "Credit_Mix": b.selectbox("Credit mix", ["Good", "Standard", "Bad"]),
        "Outstanding_Debt": b.number_input("Outstanding debt", 0.0, 100_000.0, 1400.0),
        "Credit_Utilization_Ratio": c.number_input("Credit utilization ratio", 0.0, 100.0, 32.0),
        "Payment_of_Min_Amount": c.selectbox("Paid only minimum", ["No", "Yes", "NM"]),
        "Total_EMI_per_month": c.number_input("Monthly EMI", 0.0, 100_000.0, 100.0),
        "Monthly_Balance": c.number_input("Monthly balance", 0.0, 100_000.0, 400.0),
    }
    model = st.selectbox("Model", ["logistic_regression", "decision_tree", "random_forest"])
    submit = st.form_submit_button("Predict")

if submit:
    try:
        log.info("Requesting prediction from %s", model)
        response = requests.post(f"{API_URL}/predict", json={"model": model, "features": features}, timeout=20)
        response.raise_for_status()
        st.json(response.json())
    except requests.RequestException as exc:
        st.error(f"Could not reach the prediction API: {exc}")
