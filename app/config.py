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
VERTEX_EXPRESS_API_KEY_VAL = os.environ.get("VERTEX_EXPRESS_API_KEY")

# Fake streaming settings for debugging/testing
FAKE_STREAMING_ENABLED = os.environ.get("FAKE_STREAMING", "false").lower() == "true"
FAKE_STREAMING_INTERVAL_SECONDS = float(os.environ.get("FAKE_STREAMING_INTERVAL", "1.0"))

# Function to validate API key
def validate_api_key(api_key: str) -> bool:
    """
    Validate the provided API key against the configured key
    
    Args:
        api_key: The API key to validate
        
    Returns:
        bool: True if the key is valid, False otherwise
    """
    if not API_KEY:
        # If no API key is configured, authentication is disabled
        return True
    
    return api_key == API_KEY