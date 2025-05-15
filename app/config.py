import os

# Default password if not set in environment
DEFAULT_PASSWORD = "123456"

# Get password from environment variable or use default
API_KEY = os.environ.get("API_KEY", DEFAULT_PASSWORD)

# Directory for service account credential files
CREDENTIALS_DIR = os.environ.get("CREDENTIALS_DIR", "/app/credentials")

# JSON string for service account credentials (can be one or multiple comma-separated)
GOOGLE_CREDENTIALS_JSON_STR = os.environ.get("GOOGLE_CREDENTIALS_JSON")

# API Key for Vertex Express Mode
raw_vertex_keys = os.environ.get("VERTEX_EXPRESS_API_KEY")
if raw_vertex_keys:
    VERTEX_EXPRESS_API_KEY_VAL = [key.strip() for key in raw_vertex_keys.split(',') if key.strip()]
else:
    VERTEX_EXPRESS_API_KEY_VAL = []

# Fake streaming settings for debugging/testing
FAKE_STREAMING_ENABLED = os.environ.get("FAKE_STREAMING", "false").lower() == "true"
FAKE_STREAMING_INTERVAL_SECONDS = float(os.environ.get("FAKE_STREAMING_INTERVAL", "1.0"))

# URL for the remote JSON file containing model lists
MODELS_CONFIG_URL = os.environ.get("MODELS_CONFIG_URL", "https://gist.githubusercontent.com/gzzhongqi/e0b684f319437a859bcf5bd6203fd1f6/raw")

# Validation logic moved to app/auth.py