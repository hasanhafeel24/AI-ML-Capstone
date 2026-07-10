# PART-4: AI Application Guardrails

## Objective

Build a production-ready AI application by implementing essential guardrails around the Gemini API.

---

## Features Implemented

### 1. Gemini API Integration

- Google Gemini 3 Flash Preview
- Environment variable based API Key
- Secure authentication

---

### 2. Input Validation

The application validates user input before sending it to Gemini.

Validation Rules:

- Input cannot be empty
- Maximum input length: 500 characters
- Removes unnecessary whitespace

---

### 3. Retry Mechanism

Implemented using the Tenacity library.

Configuration:

- Maximum retries: 3
- Wait time: 2 seconds

This improves reliability during temporary API failures.

---

### 4. Logging

Application logs are stored in:

logs/app.log

The logs include:

- User Prompt
- Gemini Response Status
- Errors
- Usage Statistics

---

### 5. Usage Tracking

The application records:

- Prompt character count
- Response character count

This helps estimate application usage.

---

## Project Structure

PART-4/

```
PART-4/
│
├── logs/
│ └── app.log
│
├── .env
├── .env.example
├── app.py
├── guardrails.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run

```bash
python app.py
```

---

## Example

Input

```
Explain Machine Learning
```

Output

```
Machine Learning is a branch of Artificial Intelligence...
```

---

## Technologies Used

- Python
- Google Gemini API
- Tenacity
- python-dotenv
- Logging

---

## Learning Outcomes

- Secure API Key Management
- Input Validation
- Retry Handling
- Logging
- Usage Tracking
- AI Application Reliability