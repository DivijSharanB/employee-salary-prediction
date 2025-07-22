import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Employee Salary Predictor", layout="centered", page_icon="💼")

model = joblib.load("salary_model.pkl")

st.sidebar.title("About")
st.sidebar.info(
    """
    👨‍💻 **Employee Salary Class Predictor**  
    Predicts whether a person earns **>50K or <=50K** per year.

    💡 Built using:
    - Streamlit
    - Scikit-learn
    - Pandas

    📊 Dataset: UCI Adult Income Dataset
    """
)
st.sidebar.write("Developed by DIVIJ SHARAN B - AICTE Edunet Foundation & IBM SkillsBuild")

st.title("💼 Employee Salary Predictor")
st.markdown("Enter the employee details to predict income class.")

with st.form("prediction_form"):
    st.subheader("📝 Employee Details")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 80, 30)
        education_num = st.slider("Education Level (1-16)", 1, 16, 10)
        hours_per_week = st.slider("Hours per Week", 1, 100, 40)
        gender = st.radio("Gender", ['Male', 'Female'])
        race = st.selectbox("Race", [
            'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'
        ])

    with col2:
        occupation = st.selectbox("Occupation", [
            'Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
            'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair',
            'Transport-moving', 'Machine-op-inspct', 'Tech-support',
            'Protective-serv', 'Armed-Forces'
        ])
        education = st.selectbox("Education", [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', '5th-6th',
            '10th', '1st-4th', 'Preschool', '12th'
        ])
        relationship = st.selectbox("Relationship", [
            'Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'
        ])

    submitted = st.form_submit_button("🔍 Predict Salary Class")

if submitted:
    try:
        workclass = 'Private'
        marital_status = 'Never-married'
        native_country = 'United-States'
        fnlwgt = 150000
        capital_gain = 0
        capital_loss = 0

        input_data = pd.DataFrame([[
            capital_gain,
            capital_loss,
            education,
            education_num,
            fnlwgt,
            marital_status,
            native_country,
            occupation,
            race,
            relationship,
            gender,
            workclass,
            age,
            hours_per_week
        ]], columns=[
            'capital-gain', 'capital-loss', 'education', 'educational-num',
            'fnlwgt', 'marital-status', 'native-country', 'occupation',
            'race', 'relationship', 'gender', 'workclass', 'age', 'hours-per-week'
        ])

        prediction = model.predict(input_data)[0]
        prediction_text = ">50K" if prediction == 1 else "<=50K"

        st.success(f"✅ **Predicted Salary Class:** `{prediction_text}`")

    except Exception as e:
        st.error(f"Something went wrong: {e}")

st.markdown("---")
st.markdown(
    "<small>Developed by DIVIJ SHARAN B | Internship Project - AICTE Edunet Foundation & IBM SkillsBuild</small>",
    unsafe_allow_html=True
)
