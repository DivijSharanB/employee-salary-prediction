import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Employee Salary Predictor", layout="centered", page_icon="💼")

model = joblib.load("salary_model.pkl")

st.sidebar.title("About")
st.sidebar.info(
    """
    👨‍💻 **Employee Salary Predictor**  
    This app predicts whether a person earns **>50K or <=50K** per year.

    💡 Built with:
    - Streamlit
    - Scikit-learn
    - Pandas

    📊 Dataset: UCI Adult Income Dataset
    """
)
st.sidebar.write("Developed by DIVIJ SHARAN B | AICTE-IBM Internship Project")

st.title("💼 Employee Salary Predictor")
st.markdown("Enter the details below to predict the salary class.")

with st.form("prediction_form"):
    age = st.number_input("Age", 18, 90, 30)
    workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Federal-gov', 'Self-emp-inc', 'Without-pay'])
    fnlwgt = st.number_input("Final Weight (fnlwgt)", 10000, 1000000, 50000)
    education = st.selectbox("Education", ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', 'Assoc-voc', 'Doctorate', 'Prof-school'])
    educational_num = st.slider("Educational Number", 1, 16, 10)
    marital_status = st.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent'])
    occupation = st.selectbox("Occupation", ['Exec-managerial', 'Handlers-cleaners', 'Adm-clerical', 'Sales', 'Craft-repair', 'Transport-moving', 'Prof-specialty', 'Tech-support', 'Other-service', 'Machine-op-inspct'])
    relationship = st.selectbox("Relationship", ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'])
    race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
    gender = st.radio("Gender", ['Male', 'Female'])
    capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
    capital_loss = st.number_input("Capital Loss", 0, 100000, 0)
    hours_per_week = st.slider("Hours per Week", 1, 100, 40)
    native_country = st.selectbox("Native Country", ['United-States', 'India', 'Mexico', 'Philippines', 'Germany', 'Canada', 'England', 'Italy', 'China', 'Iran'])

    submitted = st.form_submit_button("🔍 Predict Salary Class")

if submitted:
    try:
        gender = 1 if gender == "Male" else 0

        workclass_map = {k: i for i, k in enumerate(['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Federal-gov', 'Self-emp-inc', 'Without-pay'])}
        education_map = {k: i for i, k in enumerate(['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', 'Assoc-voc', 'Doctorate', 'Prof-school'])}
        marital_map = {k: i for i, k in enumerate(['Never-married', 'Married-civ-spouse', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent'])}
        occupation_map = {k: i for i, k in enumerate(['Exec-managerial', 'Handlers-cleaners', 'Adm-clerical', 'Sales', 'Craft-repair', 'Transport-moving', 'Prof-specialty', 'Tech-support', 'Other-service', 'Machine-op-inspct'])}
        relationship_map = {k: i for i, k in enumerate(['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'])}
        race_map = {k: i for i, k in enumerate(['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])}
        country_map = {k: i for i, k in enumerate(['United-States', 'India', 'Mexico', 'Philippines', 'Germany', 'Canada', 'England', 'Italy', 'China', 'Iran'])}

        input_data = pd.DataFrame([[
            age,
            workclass_map[workclass],
            fnlwgt,
            education_map[education],
            educational_num,
            marital_map[marital_status],
            occupation_map[occupation],
            relationship_map[relationship],
            race_map[race],
            gender,
            capital_gain,
            capital_loss,
            hours_per_week,
            country_map[native_country]
        ]], columns=[
            'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
            'marital-status', 'occupation', 'relationship', 'race',
            'gender', 'capital-gain', 'capital-loss',
            'hours-per-week', 'native-country'
        ])

        prediction = model.predict(input_data)[0]
        prediction_text = ">50K" if prediction == 1 else "<=50K"

        st.success(f"✅ Predicted Salary Class: `{prediction_text}`")

    except Exception as e:
        st.error(f"Something went wrong: {e}")

st.markdown("---")
st.markdown(
    "<small>Developed by DIVIJ SHARAN B | Internship Project - AICTE Edunet Foundation & IBM SkillsBuild</small>",
    unsafe_allow_html=True
)
