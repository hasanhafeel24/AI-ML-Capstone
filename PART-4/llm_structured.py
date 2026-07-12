import os
import json
import joblib
import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------
# Load environment variables
# ---------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

# ---------------------------------------------------
# Load trained model
# ---------------------------------------------------

model = joblib.load("../PART-3/best_model.pkl")

# ---------------------------------------------------
# Sample student
# ---------------------------------------------------

student = {
    "Hours_Studied": 8,
    "Attendance": 90,
    "Sleep_Hours": 7,
    "Previous_Scores": 82,
    "Tutoring_Sessions": 2,
    "Physical_Activity": 3,
    "Parental_Involvement": "High",
    "Access_to_Resources": "High",
    "Extracurricular_Activities": "Yes",
    "Motivation_Level": "High",
    "Internet_Access": "Yes",
    "Family_Income": "Medium",
    "Teacher_Quality": "Good",
    "School_Type": "Public",
    "Peer_Influence": "Positive",
    "Learning_Disabilities": "No",
    "Parental_Education_Level": "College",
    "Distance_from_Home": "Near",
    "Gender": "Male"
}

student_df = pd.DataFrame([student])

prediction = int(model.predict(student_df)[0])

result = "High Performer" if prediction == 1 else "Needs Improvement"

# ---------------------------------------------------
# Prompt
# ---------------------------------------------------

prompt = f"""
You are an educational advisor.

Student prediction: {result}

Student data:

{json.dumps(student, indent=2)}

Respond ONLY in JSON.

Required format:

{{
  "prediction": "",
  "summary": "",
  "recommendations": [
      "",
      "",
      ""
  ]
}}
"""

# ---------------------------------------------------
# Gemini API
# ---------------------------------------------------

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

payload = {
    "contents": [
        {
            "parts": [
                {
                    "text": prompt
                }
            ]
        }
    ]
}

response = requests.post(url, json=payload)

if response.status_code != 200:
    print("Status Code:", response.status_code)
    print(response.text)
    exit()

response.raise_for_status()

data = response.json()

text = data["candidates"][0]["content"]["parts"][0]["text"]

print("=" * 60)
print("MODEL PREDICTION")
print("=" * 60)
print(result)

print("\n" + "=" * 60)
print("LLM RESPONSE")
print("=" * 60)
print(text)