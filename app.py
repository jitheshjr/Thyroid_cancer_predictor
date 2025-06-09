import streamlit as st
import numpy as np
import pandas as pd
import joblib

# --- Load shared components ---
risk_scaler = joblib.load('scaler_risk.pkl')
risk_bin_edges = joblib.load('binning_edges_risk.pkl')
risk_final_columns = joblib.load('feature_columns_risk.pkl')

diagnosis_scaler = joblib.load('scaler_diagnosis.pkl')
diagnosis_bin_edges = joblib.load('binning_edges_diagnosis.pkl')
diagnosis_final_columns = joblib.load('feature_columns_diagnosis.pkl')


# --- Styling ---
st.markdown("""
    <style>
        body {
            background-color: #f5f3ff;
        }
        .main {
            background-color: #f5f3ff;
        }
        .stButton>button {
            background-color: #6a1b9a;
            color: white;
            font-weight: bold;
        }
        .result-card {
            background-color: #e1bee7;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Model selection next to input form ---
st.title("🧬 Thyroid Cancer Prediction App")
st.header("📋 Enter Patient Information")

col0, col1, col2 = st.columns([1.5, 1, 1])

with col0:
    model_choice = st.selectbox("🧠 Select Prediction Target", ["Thyroid Cancer Risk", "Diagnosis (Benign/Malignant)"])

        # --- Load selected model and components ---
    if model_choice == "Thyroid Cancer Risk":
        model = joblib.load("risk_predictor.pkl")
        label_map = {0: "Low", 1: "Medium", 2: "High"}
        result_label = "Risk Level"
        scaler = risk_scaler
        bin_edges = risk_bin_edges
        final_columns = risk_final_columns
    else:
        model = joblib.load("diagnosis_predictor.pkl")
        label_map = {0: "Benign", 1: "Malignant"}
        result_label = "Diagnosis"
        scaler = diagnosis_scaler
        bin_edges = diagnosis_bin_edges
        final_columns = diagnosis_final_columns

with col1:
    family_history = st.selectbox("Family History", ['No', 'Yes'])
    radiation_exposure = st.selectbox("Radiation Exposure", ['No', 'Yes'])
    iodine_deficiency = st.selectbox("Iodine Deficiency", ['No', 'Yes'])
    country = st.selectbox("Country", ['China', 'Germany', 'India', 'Japan', 'Nigeria', 'Russia', 'South Korea', 'UK', 'USA'])

with col2:
    ethnicity = st.selectbox("Ethnicity", ['Asian', 'Caucasian', 'Hispanic', 'Middle Eastern'])
    age = st.number_input("Age (years)", min_value=0, max_value=120, step=1)
    tsh_level = st.number_input("TSH Level (µIU/mL)", min_value=0.01, max_value=100.0, step=0.1)
    t3_level = st.number_input("T3 Level (ng/dL)", min_value=40.0, max_value=600.0, step=1.0)
    t4_level = st.number_input("T4 Level (µg/dL)", min_value=2.0, max_value=25.0, step=0.1)
    nodule_size = st.number_input("Nodule Size (cm)", min_value=0.1, max_value=10.0, step=0.1)


# --- Predict Button ---
if st.button("🔍 Predict"):
    # Original features
    raw_input = {
        'Family_History': 1 if family_history == 'Yes' else 0,
        'Radiation_Exposure': 1 if radiation_exposure == 'Yes' else 0,
        'Iodine_Deficiency': 1 if iodine_deficiency == 'Yes' else 0,
        'Country': country,
        'Ethnicity': ethnicity,
        'Age': age,
        'TSH_Level': tsh_level,
        'T3_Level': t3_level,
        'T4_Level': t4_level,
        'Nodule_Size': nodule_size
    }

    df_input = pd.DataFrame([raw_input])

    # Binning based on selected model's bin_edges
    df_input['TSH_Binned'] = pd.cut(df_input['TSH_Level'], bins=bin_edges['TSH_Level'],
                                    labels=["Low", "Medium", "High", "Very High"], include_lowest=True)
    df_input['T3_Binned'] = pd.cut(df_input['T3_Level'], bins=bin_edges['T3_Level'],
                                   labels=["Low", "Medium", "High", "Very High"], include_lowest=True)
    df_input['T4_Binned'] = pd.cut(df_input['T4_Level'], bins=bin_edges['T4_Level'],
                                   labels=["Low", "Medium", "High", "Very High"], include_lowest=True)
    df_input['Age_Binned'] = pd.cut(df_input['Age'], bins=bin_edges['Age'],
                                    labels=["Young", "Middle-aged", "Elderly", "Very Elderly"], include_lowest=True)
    df_input['Nodule_Size_Binned'] = pd.cut(df_input['Nodule_Size'], bins=bin_edges['Nodule_Size'],
                                            labels=["Small", "Medium", "Large", "Very Large"], include_lowest=True)

    # One-hot encode binned and categorical
    one_hot_cols = ['Country', 'Ethnicity', 'TSH_Binned', 'T3_Binned', 'T4_Binned',
                    'Age_Binned', 'Nodule_Size_Binned']
    df_input = pd.get_dummies(df_input, columns=one_hot_cols, drop_first=True)

    # Align with expected input columns
    df_input = df_input.reindex(columns=final_columns, fill_value=0)

    # Scale numeric features
    X_scaled = scaler.transform(df_input)

    # Predict
    prediction = model.predict(X_scaled)[0]
    probas = model.predict_proba(X_scaled)[0]

    # Display results
    st.markdown(f"""
    <div class='result-card'>
        <h2>🩺 Predicted {result_label}: <span style='color: #4a148c'>{label_map[prediction]}</span></h2>
        <h3>📊 Probability Distribution:</h3>
        <ul>
    """, unsafe_allow_html=True)

    for i in range(len(probas)):
        st.markdown(f"<li><strong>{label_map[i]}:</strong> {probas[i]*100:.2f}%</li>", unsafe_allow_html=True)

    st.markdown("</ul></div>", unsafe_allow_html=True)

