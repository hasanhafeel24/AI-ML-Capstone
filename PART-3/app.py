from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("../PART-2/models/student_performance_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    hours = int(request.form["hours"])
    attendance = int(request.form["attendance"])

    # Default values for remaining features
    data = {
        "Hours_Studied": [hours],
        "Attendance": [attendance],
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

    return f"<h2>Predicted Exam Score: {prediction[0]:.2f}</h2>"


if __name__ == "__main__":
    app.run(debug=True)