"""
schemas.py — Pydantic models for API request and response validation.
"""

from pydantic import BaseModel

class EnrollResponse(BaseModel):
    user_id: str
    status: str
    message: str

class VerifyResponse(BaseModel):
    user_id: str
    match: bool
    score: float
    threshold: float
    message: str

class ErrorResponse(BaseModel):
    detail: str
