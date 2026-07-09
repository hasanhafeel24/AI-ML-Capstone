from flask import Flask, render_template, request

from predictor import predict_score
from ai_insights import generate_insights

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    score = predict_score(request.form)

    insights = generate_insights(score)

    return render_template(
        "index.html",
        prediction=score,
        insights=insights
    )


if __name__ == "__main__":
    app.run(debug=True)