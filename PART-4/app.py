import os

from guardrails import validate_input, generate_response, log_usage

from dotenv import load_dotenv
from google import genai

from guardrails import validate_input, generate_response

# Load environment variables
load_dotenv()

# Create Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

try:

    user_prompt = input("Ask Gemini: ")

    user_prompt = validate_input(user_prompt)

    response = generate_response(client, user_prompt)

    log_usage(user_prompt, response)

    print("\n========== GEMINI RESPONSE ==========\n")
    print(response.text)

except Exception as e:
    import logging

    logging.error(str(e))

    print("\nError:", e)