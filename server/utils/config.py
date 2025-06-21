import os

from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "None")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "None")
PARSE_SERVER_BASE_URL = os.environ.get("PARSE_SERVER_BASE_URL", "http://localhost:8888")
CHAT_CRITIQUE_REVISE_ITERATIONS = int(os.environ.get("NUM_ITERATIONS_CHAT", 1))
