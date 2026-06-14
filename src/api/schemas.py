"""
schemas.py — Pydantic models for API request and response validation.
"""

from pydantic import BaseModel
from typing import Optional

class EnrollResponse(BaseModel):
    user_id: str
    status: str
    message: str

class VerifyResponse(BaseModel):
    user_id: str
    match: bool
    score: float
    threshold: float
    embedding_score: Optional[float] = None
    minutiae_score: Optional[float] = None
    message: str

class ErrorResponse(BaseModel):
    detail: str
