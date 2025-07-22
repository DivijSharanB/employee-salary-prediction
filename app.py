import streamlit as st
import pandas as pd
import numpy as np
import pickle


st.set_page_config(page_title="Employee Salary Predictor", layout="centered", page_icon="💼")


with open("salary_model.pkl", "rb") as f:
    model = pickle.load(f)


st.sidebar.title("About")
st.sidebar.info(
    """
    👨‍💻 **Employee Salary Class Predictor**  
    This app predicts whether a person earns **>50K or <=50K** per year based on features like age, education, occupation, and working hours.

    💡 Built using:
    - Streamlit
    - Scikit-learn
    - Pandas

    📊 Dataset: UCI Adult Income Dataset
    """
)
st.sidebar.write("Developed as part of an AI Internship project.")


st.title("💼 Employee Salary Predictor")
st.markdown(
    """
    Enter the employee details below to predict whether their income is likely to be **more than 50K** per year or not.
    """
)


with st.form("prediction_form"):
    st.subheader("📝 Employee Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 80, 30)
        education_num = st.slider("Education Level (numeric)", 1, 16, 10)

    with col2:
        hours_per_week = st.slider("Hours per Week", 1, 100, 40)
        occupation = st.selectbox("Occupation", [
            'Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
            'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair',
            'Transport-moving', 'Machine-op-inspct', 'Tech-support',
            'Protective-serv', 'Armed-Forces'
        ])
        sex = st.radio("Sex", ['Male', 'Female'])

    submitted = st.form_submit_button("🔍 Predict Salary Class")

if submitted:
    try:
        
        sex_encoded = 1 if sex == 'Male' else 0

       
        occupation_mapping = {
            'Adm-clerical': 0, 'Exec-managerial': 1, 'Handlers-cleaners': 2, 'Prof-specialty': 3,
            'Other-service': 4, 'Sales': 5, 'Craft-repair': 6, 'Transport-moving': 7,
            'Machine-op-inspct': 8, 'Tech-support': 9, 'Protective-serv': 10, 'Armed-Forces': 11
        }
        occupation_encoded = occupation_mapping.get(occupation, -1)

        if occupation_encoded == -1:
            st.error("Invalid occupation selected.")
        else:
            
            input_data = pd.DataFrame([[age, education_num, hours_per_week, occupation_encoded, sex_encoded]],
                                      columns=["age", "education-num", "hours-per-week", "occupation", "sex"])


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
