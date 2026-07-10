import json


def validate_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None