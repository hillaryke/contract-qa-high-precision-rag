from dotenv import load_dotenv
import os

def load_env(key_name = None):
    load_dotenv()  # take environment variables from .env.

    # Check if key_name is not provided and return early if not provided.
    if key_name is None:
        return

    api_key = os.getenv(key_name)

    if api_key is None:
        raise Exception(f"API key {key_name} not found in environment variables.")
    return api_key