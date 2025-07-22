import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Employee Salary Predictor", layout="centered", page_icon="💼")

with open("salary_model.pkl", "rb") as f:
    model = joblib.load(f)

st.title("💼 Employee Salary Predictor")
st.markdown("Predict whether an employee earns >50K or <=50K per year.")

st.sidebar.title("About")
st.sidebar.info(
    """
    👨‍💻 Predicts salary class based on UCI Adult dataset.
    - Model: RandomForestClassifier
    - Features: Demographics, Work Info
    """
)

with st.form("prediction_form"):
    st.subheader("Enter Employee Details")

    age = st.number_input("Age", min_value=18, max_value=90, value=30)
    workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                                           'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
    fnlwgt = st.number_input("Fnlwgt (Final Weight)", value=150000)
    education = st.selectbox("Education", ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
                                           'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters',
                                           '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
    educational_num = st.slider("Educational-Num", 1, 16, 10)
    marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married',
                                                     'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
    occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                                             'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                                             'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                                             'Transport-moving', 'Priv-house-serv', 'Protective-serv',
                                             'Armed-Forces'])
    relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family',
                                                 'Other-relative', 'Unmarried'])
    race = st.selectbox("Race", ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
    gender = st.radio("Gender", ['Male', 'Female'])
    capital_gain = st.number_input("Capital Gain", value=0)
    capital_loss = st.number_input("Capital Loss", value=0)
    hours_per_week = st.slider("Hours per Week", 1, 100, 40)
    native_country = st.selectbox("Native Country", ['United-States', 'Cambodia', 'England', 'Puerto-Rico',
                                                     'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India',
                                                     'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras',
                                                     'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam',
                                                     'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic',
                                                     'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary',
                                                     'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia',
                                                     'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'])

    submitted = st.form_submit_button("🔍 Predict Salary Class")

    if submitted:
        try:
            input_data = pd.DataFrame([[age, workclass, fnlwgt, education, educational_num,
                                        marital_status, occupation, relationship, race, gender,
                                        capital_gain, capital_loss, hours_per_week, native_country]],
                                      columns=['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
                                               'marital-status', 'occupation', 'relationship', 'race', 'gender',
                                               'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'])

            prediction = model.predict(input_data)[0]
            result = ">50K" if prediction == 1 else "<=50K"
            st.success(f"✅ Predicted Salary Class: `{result}`")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

st.markdown("---")
st.markdown(
    "<small>Developed by DIVIJ SHARAN B | Internship Project - AICTE Edunet Foundation & IBM SkillsBuild</small>",
    unsafe_allow_html=True
)
