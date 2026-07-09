import joblib
import pandas as pd

# Load trained model once
model = joblib.load("../PART-2/models/student_performance_model.pkl")


def predict_score(form_data):
    """
    Takes form data and returns predicted exam score.
    """

    data = {
        "Hours_Studied": [int(form_data["hours"])],
        "Attendance": [int(form_data["attendance"])],
        "Parental_Involvement": ["Medium"],
        "Access_to_Resources": ["Medium"],
        "Extracurricular_Activities": ["Yes"],
        "Sleep_Hours": [7],
        "Previous_Scores": [75],
        "Motivation_Level": ["Medium"],
        "Internet_Access": ["Yes"],
        "Tutoring_Sessions": [2],
        "Family_Income": ["Medium"],
        "Teacher_Quality": ["Medium"],
        "School_Type": ["Public"],
        "Peer_Influence": ["Positive"],
        "Physical_Activity": [3],
        "Learning_Disabilities": ["No"],
        "Parental_Education_Level": ["College"],
        "Distance_from_Home": ["Near"],
        "Gender": ["Male"]
    }

    df = pd.DataFrame(data)

    prediction = model.predict(df)

    return round(float(prediction[0]), 2)