import os
from dotenv import load_dotenv
from google import genai

from schema import StudyResponse
from utils import validate_json

# Load environment variables
load_dotenv()

# Create Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Prompt
prompt = """
Return ONLY valid JSON.

{
    "topic":"",
    "difficulty":"",
    "summary":""
}

Explain Machine Learning.
"""

# Retry configuration
MAX_RETRIES = 3
data = None

for attempt in range(MAX_RETRIES):

    response = client.models.generate_content(
        model="models/gemini-3-flash-preview",
        contents=prompt
    )

    data = validate_json(response.text)

    if data:
        print(f"\n✅ Valid JSON received on attempt {attempt + 1}")
        break

    print(f"❌ Invalid JSON. Retrying... ({attempt + 1}/{MAX_RETRIES})")

# If all retries fail
if not data:
    raise Exception("Failed to generate valid JSON after 3 attempts.")

# Validate JSON using Pydantic
study = StudyResponse(**data)

# Display Output
print("\n========== VALIDATED OUTPUT ==========\n")
print(study)