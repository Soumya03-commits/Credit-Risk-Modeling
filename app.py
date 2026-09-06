from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model_bundle.joblib"
BACKEND_PATH = ROOT / "trained_features.csv"


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model_bundle.joblib is missing. Run `python build_model_artifacts.py` first."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_backend_data():
    if not BACKEND_PATH.exists():
        raise FileNotFoundError(
            "trained_features.csv is missing. Run `python build_model_artifacts.py` first."
        )
    return pd.read_csv(BACKEND_PATH)


def set_manual_values(row: pd.DataFrame, marital_status: str, age: float,
                      education: str, income: float, bundle: dict) -> pd.DataFrame:
    row = row.copy()
    if "AGE" in row.columns:
        row["AGE"] = age
    if "NETMONTHLYINCOME" in row.columns:
        row["NETMONTHLYINCOME"] = income

    education_value = bundle["education_mapping"].get(education)
    category_values = {
        "MARITALSTATUS": marital_status,
        "EDUCATION": education_value,
    }
    for source_column, selected_value in category_values.items():
        prefix = f"{source_column}_"
        encoded_columns = [
            column for column in row.columns if column.startswith(prefix)
        ]
        for column in encoded_columns:
            row.at[row.index[0], column] = int(
                column == f"{prefix}{selected_value}"
            )

    return row


def main() -> None:
    st.set_page_config(page_title="Credit Risk Predictor", page_icon="CR", layout="wide")
    st.title("Credit Risk Predictor")
    st.caption("Look up an existing customer by Prospect ID, update key details, and predict the approval class.")

    try:
        bundle = load_model_bundle()
        backend = load_backend_data()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    if "PROSPECTID" not in backend.columns:
        st.error("The backend CSV must contain a PROSPECTID column.")
        st.stop()

    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            prospect_id = st.number_input("Prospect ID", min_value=0, step=1)
            marital_status = st.selectbox("Marital status", ["Married", "Single"])
            age = st.number_input("Age", min_value=18.0, max_value=100.0, value=35.0)
        with right:
            education = st.selectbox(
                "Education",
                list(bundle["education_mapping"].keys()),
            )
            income = st.number_input(
                "Net monthly income",
                min_value=0.0,
                value=30000.0,
                step=1000.0,
            )
        submitted = st.form_submit_button("Predict", type="primary")

    if not submitted:
        return

    matches = backend.loc[backend["PROSPECTID"] == prospect_id]
    if matches.empty:
        st.error("No customer was found for that Prospect ID.")
        return

    features = matches.drop(columns=["PROSPECTID"], errors="ignore")
    features = features.reindex(columns=bundle["trained_columns"], fill_value=0)
    features = set_manual_values(
        features,
        marital_status,
        age,
        education,
        income,
        bundle,
    )

    prediction_code = int(bundle["model"].predict(features)[0])
    prediction = bundle["label_encoder"].inverse_transform([prediction_code])[0]

    st.success(f"Predicted Approved_Flag: {prediction}")
    with st.expander("Prediction details"):
        st.write({
            "Prospect ID": prospect_id,
            "Features used": len(bundle["trained_columns"]),
            "Backend row": "Found",
        })


if __name__ == "__main__":
    main()