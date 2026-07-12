import os
import requests
import joblib
import pandas as pd
from dotenv import load_dotenv

# ============================================
# Load Environment Variables
# ============================================

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")

URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL = "openai/gpt-4.1-mini"

# ============================================
# Reusable LLM Function
# ============================================

def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(
        URL,
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        print("Status Code:", response.status_code)
        print(response.text)
        return None

    return response.json()["choices"][0]["message"]["content"]


# ============================================
# Load Trained Model
# ============================================

model = joblib.load("best_model.pkl")

print("✅ Model loaded successfully.")

# ============================================
# Sample Student
# ============================================

student = pd.DataFrame([
    {
        "Hours_Studied": 8,
        "Attendance": 92,
        "Parental_Involvement": "High",
        "Access_to_Resources": "High",
        "Extracurricular_Activities": "Yes",
        "Sleep_Hours": 7,
        "Previous_Scores": 85,
        "Motivation_Level": "High",
        "Internet_Access": "Yes",
        "Tutoring_Sessions": 2,
        "Family_Income": "Medium",
        "Teacher_Quality": "Good",
        "School_Type": "Public",
        "Peer_Influence": "Positive",
        "Physical_Activity": 3,
        "Learning_Disabilities": "No",
        "Parental_Education_Level": "College",
        "Distance_from_Home": "Near",
        "Gender": "Male"
    }
])

# ============================================
# Predict
# ============================================

prediction = model.predict(student)[0]
probability = model.predict_proba(student)[0]

print("\nPrediction :", prediction)
print("Probability:", probability)

# ============================================
# Ask LLM for Explanation
# ============================================

system_prompt = """
You are an educational AI assistant.

Return ONLY valid JSON.

Do not write markdown.

Return exactly this structure:

{
  "prediction": "",
  "confidence": "",
  "explanation": "",
  "study_tips": [
      "",
      "",
      ""
  ]
}
"""

user_prompt = f"""
Prediction: {"Pass" if prediction==1 else "Fail"}

Confidence:
Pass = {probability[1]:.2%}

Generate the JSON only.
"""

reply = call_llm(system_prompt, user_prompt)

print("\n==============================")
print("AI EXPLANATION")
print("==============================\n")

import json

response = json.loads(reply)

print("\nPrediction :", response["prediction"])
print("Confidence :", response["confidence"])
print("\nExplanation:")
print(response["explanation"])

print("\nStudy Tips")

for i, tip in enumerate(response["study_tips"], 1):
    print(f"{i}. {tip}")