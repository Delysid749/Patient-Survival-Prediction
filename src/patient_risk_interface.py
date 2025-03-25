import sys
from pathlib import Path

# Set project root path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
import cloudpickle
from io import StringIO
from src.model.Ensemble import predict_new_patients
from src.config.path_config import MODEL_DIR
from src.common.variable_array import final_numerical_arr, categorical_array

st.set_page_config(page_title="Patient Mortality Prediction", layout="wide")
st.title("Inpatient Mortality Risk Prediction System")

# Load models
with open(MODEL_DIR / "base_models.pkl", "rb") as f:
    base_models = cloudpickle.load(f)

with open(MODEL_DIR / "meta_model.pkl", "rb") as f:
    meta_model = cloudpickle.load(f)

# Important fields
TOP_N_NUMERIC_FIELDS = [
    "apache_4a_hospital_death_prob",
    "apache_4a_icu_death_prob",
    "ventilated_apache",
    "intubated_apache",
    "d1_heartrate_max"
]
TOP_CATEGORICAL_FIELDS = [
    "apache_3j_bodysystem", "icu_admit_source", "apache_2_bodysystem",
    "icu_type", "icu_stay_type", "ethnicity", "gender"
]
field_description = {
    "apache_4a_hospital_death_prob": "Predicted hospital mortality probability (APACHE IVa).",
    "apache_4a_icu_death_prob": "Predicted ICU mortality probability (APACHE IVa).",
    "ventilated_apache": "Whether patient was under mechanical ventilation.",
    "intubated_apache": "Whether the patient was intubated.",
    "d1_heartrate_max": "Maximum heart rate in the first 24h.",
    "apache_3j_bodysystem": "APACHE III admission body system group.",
    "icu_admit_source": "Where the patient was admitted from (e.g. emergency).",
    "apache_2_bodysystem": "APACHE II admission body system group.",
    "icu_type": "ICU type (e.g., Med-Surg ICU).",
    "icu_stay_type": "Type of ICU stay (admit/transfer).",
    "ethnicity": "Patient ethnicity group.",
    "gender": "Biological gender of patient."
}

category_options = {
    "ethnicity": ["missing", "Caucasian", "African American", "Other"],
    "gender": ["missing", "M", "F", "Other"],
    "icu_admit_source": ["missing", "Accident & Emergency", "Operating Room / Recovery", "Other"],
    "icu_stay_type": ["missing", "admit", "transfer", "Other"],
    "icu_type": ["missing", "Med-Surg ICU", "MICU", "Other"],
    "apache_2_bodysystem": ["missing","Cardiovascular","Neurologic", "Other"],
    "apache_3j_bodysystem": ["missing","Cardiovascular","Neurologic", "Other"]
}




DEFAULT_NUMERIC_VALUES = {
    col: 0.0 for col in final_numerical_arr if col not in TOP_N_NUMERIC_FIELDS
}
DEFAULT_CATEGORICAL_VALUES = {
    col: "missing" for col in categorical_array if col not in TOP_CATEGORICAL_FIELDS
}

st.header("Select Data Input Mode")
input_mode = st.radio("Input Mode", ["Manual Input (Single Patient)", "Upload CSV File (Batch Prediction)"])

if input_mode == "Manual Input (Single Patient)":
    st.subheader("Enter Patient Feature Information")

    with st.form("manual_input_form"):
        input_data = {}

        for field in TOP_N_NUMERIC_FIELDS:
            en_desc = {
                "apache_4a_hospital_death_prob": "Predicted hospital mortality probability (APACHE IVa).",
                "apache_4a_icu_death_prob": "Predicted ICU mortality probability (APACHE IVa).",
                "ventilated_apache": "Whether patient was under mechanical ventilation.",
                "intubated_apache": "Whether the patient was intubated.",
                "d1_heartrate_max": "Maximum heart rate in the first 24h."
            }.get(field, "")
            st.markdown(f"**{field}**: {en_desc}")
            input_data[field] = st.number_input(f"Enter value for {field}", value=0.0)


        for field in TOP_CATEGORICAL_FIELDS:
            options = category_options.get(field, ["missing"])
            en_desc = {
                "apache_3j_bodysystem": "APACHE III admission body system group.",
                "icu_admit_source": "Where the patient was admitted from (e.g. emergency).",
                "apache_2_bodysystem": "APACHE II admission body system group.",
                "icu_type": "ICU type (e.g., Med-Surg ICU).",
                "icu_stay_type": "Type of ICU stay (admit/transfer).",
                "ethnicity": "Patient ethnicity group.",
                "gender": "Biological gender of patient."
            }.get(field, "")
            st.markdown(f"**{field}**: {en_desc}")
            input_data[field] = st.selectbox(f"Select value for {field}", options=options)

        # 填充剩余字段默认值
        input_data.update(DEFAULT_NUMERIC_VALUES)
        input_data.update(DEFAULT_CATEGORICAL_VALUES)

        submitted = st.form_submit_button("Predict Patient Mortality Risk")
        if submitted:
            df = pd.DataFrame([input_data])
            pred, prob = predict_new_patients(df, base_models, meta_model)
            st.success("Prediction Result: " + ("Death" if pred[0] == 1 else "Survival"))
            st.info(f"Death Probability: {prob[0]:.4f}")


else:
    st.subheader("Upload CSV File Containing Patient Data (All Required Fields)")
    file = st.file_uploader("Choose CSV File", type=["csv"])
    if file:
        try:
            df = pd.read_csv(file, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file, encoding="gbk")
            except Exception as e:
                st.error(f"Failed to read file: {e}")
                st.stop()

        required_columns = final_numerical_arr + categorical_array
        missing_cols = [col for col in required_columns if col not in df.columns]

        if missing_cols:
            st.error(f"The following required fields are missing from the CSV file:\n{missing_cols}")
        else:
            st.success("Field check passed. Starting prediction...")
            pred, prob = predict_new_patients(df, base_models, meta_model)
            df["Prediction Result"] = ["Death" if p == 1 else "Survival" for p in pred]
            df["Death Probability"] = prob
            st.subheader("Prediction Results")
            st.dataframe(df)

            # Export button
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Prediction Results (CSV)",
                data=csv_buffer.getvalue(),
                file_name="prediction_result.csv",
                mime="text/csv"
            )
