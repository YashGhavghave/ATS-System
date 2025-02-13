import gym
import numpy as np
import pandas as pd
import streamlit as st
import os
from stable_baselines3 import PPO
from gym import spaces
import google.generativeai as genai

genai.configure(api_key="API_HERE")

def extract_resume_features(resume_text):
    """Uses Gemini 1.5 Flash to extract structured data from resumes."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f"Extract skills, experience, and education score from: {resume_text}")
    return response.text if response else "Error processing resume"

class ResumeScreeningEnv(gym.Env):
    def __init__(self, resume_data):
        super(ResumeScreeningEnv, self).__init__()
        
        self.data = resume_data
        self.current_index = 0
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.data.shape[1]-1,), dtype=np.float32)
        
        self.action_space = spaces.Discrete(3)
        
    def reset(self):
        self.current_index = 0
        return self.data.iloc[self.current_index, :-1].values.astype(np.float32)
    
    def step(self, action):
        reward = self.calculate_reward(action, self.data.iloc[self.current_index, -1])
        
        self.current_index += 1
        done = self.current_index >= len(self.data)
        
        if done:
            obs = np.zeros(self.observation_space.shape)
        else:
            obs = self.data.iloc[self.current_index, :-1].values.astype(np.float32)
        
        return obs, reward, done, {}
    
    def calculate_reward(self, action, actual_label):
        """ Reward system: Encourage correct classifications """
        return 1 if action == actual_label else -1

data = {
    'skills_match': [0.9, 0.4, 0.8, 0.3, 0.7],
    'experience_score': [0.8, 0.5, 0.6, 0.2, 0.9],
    'education_score': [0.7, 0.4, 0.9, 0.3, 0.6],
    'label': [2, 0, 1, 0, 2]  
}
df = pd.DataFrame(data)

env = ResumeScreeningEnv(df)

model_path = "resume_rl_model.zip"
if os.path.exists(model_path):
    model = PPO.load(model_path, env)
    print("Loaded trained RL model")
else:
    print("No saved model found. Training a new one...")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save(model_path)


st.title("AI-Powered Resume Screening")
st.write("Upload a resume to analyze and classify it.")

uploaded_file = st.file_uploader("Upload Resume (TXT or PDF)", type=["txt", "pdf"])

if uploaded_file is not None:
    resume_text = uploaded_file.read().decode("utf-8")
    st.write("### Extracted Resume Data:")
    extracted_data = extract_resume_features(resume_text)
    st.write(extracted_data)
    
    resume_features = np.array([0.8, 0.7, 0.9], dtype=np.float32)  
    action, _ = model.predict(resume_features)
    
    classification = ["Rejected", "Shortlisted", "Accepted"][action]
    st.write(f"### Classification: {classification}")
