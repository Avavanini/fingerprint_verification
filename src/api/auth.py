"""
auth.py — Simple API Key authentication.
"""

import os
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

# In a real scenario, this would be in a .env file and loaded via python-dotenv.
# For this demo, we'll use a hardcoded fallback if the environment variable isn't set.
API_KEY = os.environ.get("FINGERPRINT_API_KEY", "secret-demo-key-123")
API_KEY_NAME = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials"
    )
