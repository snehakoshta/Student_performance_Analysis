import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Apply custom styles
st.markdown(
    """
    <style>
    .stApp {
        background-color: lightblue;
        font-family: 'Arial', sans-serif;
        color: #232323; 
        font-size: 16px;
        line-height: 1.5;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        margin: 20px; 
        background-image: url('https://media.istockphoto.com/photos/graduate-students-tossing-up-hats-over-blue-sky-picture-id487673748?k=6&m=487673748&s=612x612&w=0&h=gyZVFnmcEqFtAeWRLsGV4g7ZOilpoiRDuyook8dMsSU='); /* Replace with your image URL */
        background-size: cover;
        }
    .title {
        font-size: 36px;
        color: #232323;
        text-align: center;
    }
    .instructions {
        font-size: 18px;
        color: #566573;
        text-align: center;
    }
    .input-box {
        background-color: #F7F9F9;
        border: 7px solid #AAB7B8;
        border-radius: 5px;
        padding: 5px;
        margin-bottom: 10px;
        font-size: 16px;
        font-family: 'Arial', sans-serif;
        font-weight: bold;
    }
    .button {
        background-color: yellowgreen;
        color: pink;
        font-size: 18px;
        padding: 10px 15px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .button:hover {
        background-color: pinkgreen;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the app
st.markdown('<h1 class="title">Student Exam Performance & Analysis Prediction</h1>', unsafe_allow_html=True)

# Load the saved models
dt_model_path = 'E:/performance/decision_tree_model.pkl'  # Update with your file path
rf_model_path = 'E:/performance/random_forest_model.pkl'  # Update with your file path

with open(dt_model_path, 'rb') as dt_file:
    dt_model = pickle.load(dt_file)

with open(rf_model_path, 'rb') as rf_file:
    rf_model = pickle.load(rf_file)

# Display instructions with styling
st.markdown('<p class="instructions" style="color:pink;">Analyze exam results to uncover insights that drive improvement, inspire achievement, and set students on the path to success!</p>', unsafe_allow_html=True)
# Add custom styles to input boxes
st.markdown(
    """
    <style>
    .stTextInput > div > input > div > input > div > div {
        font-size: 30px !important;
        font-weight: bold !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Create input boxes with added styles
student_id = st.text_input("Student ID", key="student_id")
age = st.number_input("Age", min_value=0, max_value=150, step=1, key="age")
gender = st.selectbox("Gender", options=["Male", "Female"], key="gender")
study_hours_per_day = st.number_input("Study Hours per Day", min_value=0.0, max_value=24.0, step=0.1, key="study_hours")
social_media_hours = st.number_input("Social Media Hours per Day", min_value=0.0, max_value=24.0, step=0.1, key="social_media_hours")
netflix_hours = st.number_input("Netflix Hours per Day", min_value=0.0, max_value=24.0, step=0.1, key="netflix_hours")
part_time_job = st.selectbox("Part-Time Job", options=["Yes", "No"], key="part_time_job")
attendance_percentage = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, step=0.1, key="attendance_percentage")
sleep_hours = st.number_input("Sleep Hours per Day", min_value=0.0, max_value=24.0, step=0.1, key="sleep_hours")
diet_quality = st.selectbox("Diet Quality", options=["Poor", "Fair", "Good"], key="diet_quality")
exercise_frequency = st.number_input("Exercise Frequency per Week", min_value=0, max_value=7, step=1, key="exercise_frequency")
parental_education_level = st.selectbox("Parental Education Level", options=["High School", "Bachelor", "Master", "PhD"], key="education_level")
internet_quality = st.selectbox("Internet Quality", options=["Poor", "Average", "Good"], key="internet_quality")
mental_health_rating = st.number_input("Mental Health Rating (1-10)", min_value=1, max_value=10, step=1, key="mental_health_rating")
extracurricular_participation = st.selectbox("Extracurricular Participation", options=["Yes", "No"], key="extracurricular_participation")

# Mapping categorical inputs to numerical values
gender_mapping = {'Male': 0, 'Female': 1}
part_time_job_mapping = {'No': 0, 'Yes': 1}
diet_quality_mapping = {'Poor': 0, 'Fair': 1, 'Good': 2}
parental_education_mapping = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
internet_quality_mapping = {'Poor': 0, 'Average': 1, 'Good': 2}
extracurricular_mapping = {'No': 0, 'Yes': 1}

# Convert inputs to numerical values for the model
input_data = [
    age,
    gender_mapping[gender],
    study_hours_per_day,
    social_media_hours,
    netflix_hours,
    part_time_job_mapping[part_time_job],
    attendance_percentage,
    sleep_hours,
    diet_quality_mapping[diet_quality],
    exercise_frequency,
    parental_education_mapping[parental_education_level],
    internet_quality_mapping[internet_quality],
    mental_health_rating,
    extracurricular_mapping[extracurricular_participation]
]


if st.button("Predict Exam Score", key="predict_button"):
    # Predict using both models
    dt_prediction = dt_model.predict([input_data])
    rf_prediction = rf_model.predict([input_data])

    # Display predictions with custom font size and bold styling
    st.markdown(
        f"<p style='font-size:20px; font-weight:bold;'>Prediction of your Exam Score: {dt_prediction[0]:.2f}</p>",
        unsafe_allow_html=True
    )
    if dt_prediction[0] > 90 or rf_prediction[0] > 90:
        st.markdown(
            "<p style='font-size:20px; font-weight:bold;color:green;'>Excellent! Keep up the great work and maintain your performance.</p>",
            unsafe_allow_html=True
        )
    elif dt_prediction[0] > 75 or rf_prediction[0] > 75:
        st.markdown(
            "<p style='font-size:20px; font-weight:bold; color:green;'>Good job! You're doing well, but there's room for improvement.</p>",
            unsafe_allow_html=True
        )
    elif dt_prediction[0] > 50 or rf_prediction[0] > 50:
        st.markdown(
            "<p style='font-size:20px; font-weight:bold; color:red;'>Fair performance. Consider dedicating more time to studies. Focus on improving your study habits and reducing distractions.</p>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<p style='font-size:20px; font-weight:bold;color:red; '>Poor performance. Seek guidance and work harder to improve.</p>",
            unsafe_allow_html=True
        )
    
