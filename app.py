import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Employee Salary Predictor", layout="centered", page_icon="💼")

with open("salary_model.pkl", "rb") as f:
    model = joblib.load(f)

st.title("💼 Employee Salary Predictor")
st.markdown("Predict whether an employee earns >50K or <=50K per year.")

# Sidebar
st.sidebar.title("About")
st.sidebar.info("This app uses AI/ML to predict employee salary class based on the UCI dataset.")

# Manual encoding dictionaries (you must use the same from training)
gender_map = {'Male': 1, 'Female': 0}
workclass_map = {'Private': 0, 'Self-emp-not-inc': 1, 'Self-emp-inc': 2, 'Federal-gov': 3,
                 'Local-gov': 4, 'State-gov': 5, 'Without-pay': 6, 'Never-worked': 7}
education_map = {'Bachelors': 0, 'Some-college': 1, '11th': 2, 'HS-grad': 3, 'Prof-school': 4,
                 'Assoc-acdm': 5, 'Assoc-voc': 6, '9th': 7, '7th-8th': 8, '12th': 9, 'Masters': 10,
                 '1st-4th': 11, '10th': 12, 'Doctorate': 13, '5th-6th': 14, 'Preschool': 15}
marital_map = {'Married-civ-spouse': 0, 'Divorced': 1, 'Never-married': 2,
               'Separated': 3, 'Widowed': 4, 'Married-spouse-absent': 5, 'Married-AF-spouse': 6}
occupation_map = {'Tech-support': 0, 'Craft-repair': 1, 'Other-service': 2, 'Sales': 3,
                  'Exec-managerial': 4, 'Prof-specialty': 5, 'Handlers-cleaners': 6,
                  'Machine-op-inspct': 7, 'Adm-clerical': 8, 'Farming-fishing': 9,
                  'Transport-moving': 10, 'Priv-house-serv': 11, 'Protective-serv': 12,
                  'Armed-Forces': 13}
relationship_map = {'Wife': 0, 'Own-child': 1, 'Husband': 2, 'Not-in-family': 3,
                    'Other-relative': 4, 'Unmarried': 5}
race_map = {'White': 0, 'Asian-Pac-Islander': 1, 'Amer-Indian-Eskimo': 2, 'Other': 3, 'Black': 4}
country_map = {'United-States': 0, 'India': 1, 'Canada': 2, 'Mexico': 3, 'Philippines': 4,
               'Germany': 5, 'China': 6, 'Iran': 7, 'England': 8, 'France': 9}

with st.form("prediction_form"):
    age = st.number_input("Age", 18, 90, 30)
    workclass = st.selectbox("Workclass", list(workclass_map.keys()))
    fnlwgt = st.number_input("Fnlwgt", 10000, 1000000, 150000)
    education = st.selectbox("Education", list(education_map.keys()))
    educational_num = st.slider("Educational-Num", 1, 16, 10)
    marital_status = st.selectbox("Marital Status", list(marital_map.keys()))
    occupation = st.selectbox("Occupation", list(occupation_map.keys()))
    relationship = st.selectbox("Relationship", list(relationship_map.keys()))
    race = st.selectbox("Race", list(race_map.keys()))
    gender = st.radio("Gender", list(gender_map.keys()))
    capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
    capital_loss = st.number_input("Capital Loss", 0, 100000, 0)
    hours_per_week = st.slider("Hours per Week", 1, 100, 40)
    native_country = st.selectbox("Native Country", list(country_map.keys()))

    submitted = st.form_submit_button("🔍 Predict Salary Class")

    if submitted:
        try:
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
                gender_map[gender],
                capital_gain,
                capital_loss,
                hours_per_week,
                country_map[native_country]
            ]], columns=['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
                         'marital-status', 'occupation', 'relationship', 'race', 'gender',
                         'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'])

            pred = model.predict(input_data)[0]
            result = ">50K" if pred == 1 else "<=50K"
            st.success(f"✅ Predicted Salary Class: `{result}`")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

st.markdown("---")
st.markdown("<small>Developed by DIVIJ SHARAN B</small>", unsafe_allow_html=True)
