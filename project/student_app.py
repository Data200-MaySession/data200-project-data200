import streamlit as st
import pandas as pd
import joblib

# Load model and column names
model = joblib.load('project/student_success_model.pkl')
input_columns = joblib.load('project/input_columns.pkl')

# Predefined course names from dataset
course_options = [
    "Agronomy", "Design", "Education", "Nursing", "Journalism",
    "Management", "Social Service", "Technologies"
]

# Predefined marital statuses
marital_status_options = ["Single", "Married", "Divorced", "Widowed"]

# Streamlit UI
st.set_page_config(page_title="Student Success Predictor", layout="centered")
st.title("🎓 Student Dropout & Success Predictor")

st.markdown("Fill in the student’s admission and demographic information to predict the academic outcome.")

with st.form("student_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age at enrollment", 17, 60, 20)
    marital_status = st.selectbox("Marital Status", marital_status_options)
    scholarship = st.selectbox("Scholarship holder", ["No", "Yes"])
    tuition_fees = st.selectbox("Tuition fees up to date", ["No", "Yes"])
    displaced = st.selectbox("Displaced", ["No", "Yes"])
    admission_grade = st.slider("Admission grade", 90.0, 200.0, 150.0)
    prev_qualification_grade = st.slider("Previous qualification grade", 90.0, 200.0, 150.0)
    course = st.selectbox("Course", course_options)
    app_order = st.slider("Application order", 1, 20, 1)
    attendance = st.selectbox("Daytime or Evening Attendance", ['Daytime', 'Evening'])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Create base input dict
    input_data = {
        'Gender': 1 if gender == "Female" else 0,
        'Age at enrollment': age,
        'Scholarship holder': 1 if scholarship == "Yes" else 0,
        'Tuition fees up to date': 1 if tuition_fees == "Yes" else 0,
        'Displaced': 1 if displaced == "Yes" else 0,
        'Admission grade': admission_grade,
        'Previous qualification (grade)': prev_qualification_grade,
        'Application order': app_order,
    }

    # Encode one-hot columns for categorical variables
    for status in marital_status_options:
        input_data[f"Marital Status_{status}"] = 1 if marital_status == status else 0

    for c in course_options:
        input_data[f"Course_{c}"] = 1 if course == c else 0

    input_data["Daytime/evening attendance_Evening"] = 1 if attendance == "Evening" else 0

    # Add any missing columns as 0
    for col in input_columns:
        input_data.setdefault(col, 0)

    # Reorder to match model
    input_df = pd.DataFrame([input_data])[input_columns]

    prediction = model.predict(input_df)[0]
    st.success(f"🎯 Predicted Outcome: **{prediction}**")

    # Prediction confidence
    probs = model.predict_proba(input_df)[0]
    st.subheader("📊 Prediction Confidence")
    st.bar_chart(pd.Series(probs, index=model.classes_))

    # Feature importance
    importances = model.feature_importances_
    feat_series = pd.Series(importances, index=input_columns).sort_values(ascending=False)[:10]

    st.subheader("💡 Top 10 Important Features")
    st.bar_chart(feat_series)