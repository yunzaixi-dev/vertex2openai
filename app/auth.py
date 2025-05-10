from fastapi import HTTPException, Header, Depends
from fastapi.security import APIKeyHeader
from typing import Optional
from config import API_KEY # Import API_KEY directly for use in local validation

# Function to validate API key (moved from config.py)
def validate_api_key(api_key_to_validate: str) -> bool:
    """
    Validate the provided API key against the configured key.
    """
    if not API_KEY: # API_KEY is imported from config
        # If no API key is configured, authentication is disabled (or treat as invalid)
        # Depending on desired behavior, for now, let's assume if API_KEY is not set, all keys are invalid unless it's an empty string match
        return False # Or True if you want to disable auth when API_KEY is not set
    return api_key_to_validate == API_KEY

# API Key security scheme
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# Dependency for API key validation
async def get_api_key(authorization: Optional[str] = Header(None)):
    if authorization is None:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Please include 'Authorization: Bearer YOUR_API_KEY' header."
        )
    
    # Check if the header starts with "Bearer "
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key format. Use 'Authorization: Bearer YOUR_API_KEY'"
        )
    
    # Extract the API key
    api_key = authorization.replace("Bearer ", "")
    
    # Validate the API key
    if not validate_api_key(api_key): # Call local validate_api_key
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return api_key