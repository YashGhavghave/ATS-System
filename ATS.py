import streamlit as st
import numpy as np
import torch
import chardet
import pandas as pd
from stable_baselines3 import PPO
import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyB2pMezOsgVcvUaT9QY4m6XLCSiTQ-ACD8")

MODEL_PATH = "resume_rl_model"
FEEDBACK_CSV = "feedback_data.csv"
if os.path.exists(MODEL_PATH + ".zip"):
    model = PPO.load(MODEL_PATH)
else:
    model = PPO("MlpPolicy", gym.make("CartPole-v1"), verbose=1) 
    model.save(MODEL_PATH)

RULEBOOK = {
    "Rejected": {
        "min_experience": 0,
        "min_education_score": 0,
        "required_skills": []
    },
    "Shortlisted": {
        "min_experience": 2,
        "min_education_score": 0.6,
        "required_skills": ["Python", "Machine Learning"]
    },
    "Accepted": {
        "min_experience": 5,
        "min_education_score": 0.8,
        "required_skills": ["Python", "Machine Learning", "Deep Learning"]
    }
}

def extract_resume_features(resume_text):
    """Uses Gemini API to extract structured data from resumes."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f"Extract skills, experience, education score, file type, and owner details from: {resume_text}")
    return response.text if response else "Error processing resume"

def classify_resume(extracted_data):
    """Classifies resume based on rulebook."""
    experience = 3  
    education_score = 0.7  
    skills = ["Python", "Machine Learning"]  
    
    for category, rules in RULEBOOK.items():
        if experience >= rules["min_experience"] and education_score >= rules["min_education_score"] and all(skill in skills for skill in rules["required_skills"]):
            return category
    return "Rejected"

st.title("AI-Powered Resume Screening")
st.write("Upload a resume to analyze and classify it.")

uploaded_file = st.file_uploader("Upload Resume (TXT or PDF)", type=["txt", "pdf"])

if uploaded_file is not None:
    raw_data = uploaded_file.read()
    detected_encoding = chardet.detect(raw_data)["encoding"]
    if detected_encoding is None:
        detected_encoding = "utf-8" 
    resume_text = raw_data.decode(detected_encoding, errors="ignore")
    
    st.write("### Extracted Resume Data:")
    extracted_data = extract_resume_features(resume_text)
    
    st.markdown("**Extracted Information:**")
    for line in extracted_data.split("\n"):
        if line.strip():
            st.markdown(f"- {line}")
    
    classification = classify_resume(extracted_data)
    st.write(f"### Classification: {classification}")
    
    feedback = st.radio("Was the classification correct?", ("Yes", "No"))
    if st.button("Submit Feedback"):
        if feedback == "No":
            correct_action = st.selectbox("Select the correct classification", ["Rejected", "Shortlisted", "Accepted"])
            
            feedback_data = pd.DataFrame([[resume_text, classification, correct_action]], columns=["Resume Text", "Predicted Classification", "Correct Classification"])
            if os.path.exists(FEEDBACK_CSV):
                feedback_data.to_csv(FEEDBACK_CSV, mode='a', header=False, index=False)
            else:
                feedback_data.to_csv(FEEDBACK_CSV, mode='w', header=True, index=False)
            st.success("Feedback recorded successfully!")
