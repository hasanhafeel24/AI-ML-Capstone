import logging

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from tenacity import retry, stop_after_attempt, wait_fixed


def validate_input(user_input):
    """
    Validate and sanitize user input.
    """

    user_input = user_input.strip()

    if not user_input:
        raise ValueError("Input cannot be empty.")

    if len(user_input) > 500:
        raise ValueError("Input is too long (maximum 500 characters).")

    return user_input


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2)
)
def generate_response(client, prompt):
    """
    Retry Gemini request up to 3 times.
    """
    print("➡️ Contacting Gemini...")
    logging.info(f"User Prompt: {prompt}")

    response = client.models.generate_content(
        model="models/gemini-3-flash-preview",
        contents=prompt
    )
    print("✅ Response received!")
    logging.info("Gemini response generated successfully.")

    return response

def log_usage(prompt, response):
    """
    Track basic request usage.
    """

    prompt_length = len(prompt)
    response_length = len(response.text)

    logging.info(
        f"Usage | Prompt Characters: {prompt_length} | Response Characters: {response_length}"
    )