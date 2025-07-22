import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

# Load model
model = joblib.load("salary_model.joblib")

st.title("💼 Employee Salary Predictor")

with st.form("prediction_form"):
    st.subheader("📝 Employee Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 80, 40)
        fnlwgt = st.number_input("Final Weight (fnlwgt)", 10000, 1000000, 250000)
        education = st.selectbox("Education", list(range(16)))
        marital_status = st.selectbox("Marital Status", list(range(7)))
        occupation = st.selectbox("Occupation", list(range(14)))
        relationship = st.selectbox("Relationship", list(range(6)))
        gender = st.selectbox("Gender", [0, 1])  # 0 = Female, 1 = Male

    with col2:
        workclass = st.selectbox("Workclass", list(range(8)))
        educational_num = st.slider("Educational Num", 1, 16, 10)
        race = st.selectbox("Race", list(range(5)))
        capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
        capital_loss = st.number_input("Capital Loss", 0, 100000, 0)
        hours_per_week = st.slider("Hours per Week", 1, 100, 40)
        native_country = st.selectbox("Native Country", list(range(41)))

    submitted = st.form_submit_button("🔍 Predict Salary Class")

if submitted:
    input_data = pd.DataFrame([[
        age, workclass, fnlwgt, education, educational_num, marital_status,
        occupation, relationship, race, gender, capital_gain,
        capital_loss, hours_per_week, native_country
    ]], columns=[
        'age', 'workclass', 'fnlwgt', 'education', 'educational_num',
        'marital_status', 'occupation', 'relationship', 'race', 'gender',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country'
    ])

    prediction = model.predict(input_data)[0]
    result = ">50K" if prediction == 1 else "<=50K"
    st.success(f"✅ Predicted Salary Class: `{result}`")

st.markdown("---")
st.markdown(
    "<small>Developed by DIVIJ SHARAN B | Internship Project - AICTE Edunet Foundation & IBM SkillsBuild</small>",
    unsafe_allow_html=True
)
