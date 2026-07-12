# Part 4 – Explainable AI with LLM Integration

## Objective

The objective of Part 4 was to combine the trained machine learning model with a Large Language Model (Google Gemini) to produce human-readable explanations of predictions.

The application loads the trained classifier, predicts whether a student is likely to pass, calculates prediction confidence, and uses an LLM to generate an explanation and personalized study recommendations.

---

# Components

The application consists of:

- Trained Random Forest Pipeline (`best_model.pkl`)
- Google Gemini API
- Prompt Engineering
- JSON Output Validation
- Explainable AI Layer

---

# Workflow

1. Load the trained model.
2. Accept student input features.
3. Predict Pass or Fail.
4. Calculate prediction probabilities.
5. Send prediction details to Gemini.
6. Receive a structured explanation.
7. Display:
   - Prediction
   - Confidence
   - Explanation
   - Study Tips

---

# Machine Learning Prediction

Example Output

Prediction

Pass

Confidence

76%

---

# LLM Generated Explanation

Example

Prediction: Pass

Confidence: 76%

Explanation:

The prediction indicates a pass with a confidence level of 76%, suggesting that the student has a strong likelihood of performing successfully based on the provided academic information.

---

# AI Study Tips

Example

1. Review key concepts regularly.
2. Practice previous exam questions.
3. Maintain a consistent study schedule.

---

# Prompt Engineering

The system prompt instructs Gemini to act as an educational assistant.

The user prompt includes:

- Prediction
- Confidence
- Required JSON format

This ensures the response remains structured and easy to parse.

---

# JSON Validation

The application validates the returned JSON before displaying the explanation.

If invalid JSON is received, the request is automatically retried.

This improves robustness and reliability.

---

# Model Serialization

The machine learning model is loaded using Joblib.

```python
model = joblib.load("best_model.pkl")
```

The serialized model allows prediction without retraining.

---

# Files Included

- app.py
- best_model.pkl
- README.md
- requirements.txt
- .env.example

---

# Conclusion

Part 4 successfully integrates a trained machine learning model with Google Gemini to create an Explainable AI system.

The final application predicts student performance, estimates confidence, and generates easy-to-understand explanations with personalized study recommendations, making the model more interpretable for end users.