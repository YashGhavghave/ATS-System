# ATS System

## Overview
The ATS System is a reinforcement learning-based Applicant Tracking System (ATS) designed to evaluate resumes efficiently. It utilizes an RL model that continuously learns and improves based on feedback, enabling more accurate and adaptive resume scoring. The system is integrated with Gemini API to enhance decision-making.

## Features
- Reinforcement learning-based ATS model
- Resume scanning and scoring
- Continuous learning through feedback-based retraining
- Streamlit-based frontend for easy interaction
- Integration with Gemini API for improved analysis

## Installation
Follow these steps to set up and run the ATS System locally:

### Prerequisites
Ensure you have Python installed on your system. You can download it from [python.org](https://www.python.org/).

### Steps to Install and Run
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/ATS-System.git
   ```
2. Navigate to the project directory:
   ```sh
   cd ATS-System
   ```
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```sh
   streamlit run App.py
   ```

## Usage
- Upload a resume file (PDF or DOCX).
- The system scans and evaluates the resume based on predefined parameters.
- The RL model assigns an ATS score, which improves over time with learning.
- Users can provide feedback to enhance the model’s accuracy.

## Technologies Used
- **Python** (Machine Learning, Deep Learning, Data Processing)
- **Reinforcement Learning** (Model Training & Improvement)
- **Streamlit** (Frontend)
- **Gemini API** (Enhanced Resume Analysis)
- **MongoDB** (Data Storage)

## Future Enhancements
- Implement a more sophisticated feedback loop for better RL tuning.
- Expand scoring criteria to improve resume assessment.
- Develop a dashboard for visualizing candidate evaluations.
- Enhance integration with other AI models for improved predictions.

## Contributors
- **Yash Ghavghave** (AIML Student at FET, DMIHER College)
- Open for contributions! Feel free to submit PRs or raise issues.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---
For any queries, contact [your email/contact info].

