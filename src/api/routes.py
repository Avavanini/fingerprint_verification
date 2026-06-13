"""
routes.py — API endpoints for enrollment and verification.
"""

import cv2
import json
import numpy as np
from fastapi import APIRouter, Depends, UploadFile, File, Form, Request, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import ValidationError
from typing import List

from .database import get_db
from .models import UserTemplate
from .schemas import EnrollResponse, VerifyResponse, ErrorResponse
from .auth import verify_api_key

from src.matching.scorer import compute_embedding_score
from src.matching.decision import make_decision

router = APIRouter(dependencies=[Depends(verify_api_key)])

# From Phase 5 evaluation, our optimal threshold is 0.9548
OPERATING_THRESHOLD = 0.9548

def process_image(request: Request, file_bytes: bytes) -> np.ndarray:
    """Helper to process image bytes into a 128-D embedding."""
    # Convert bytes to numpy array
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if img_cv is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
        
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
    
    # Apply transform
    transform = request.app.state.transform
    tensor = transform(img_rgb)
    
    # Extract embedding
    model = request.app.state.model
    device = request.app.state.device
    
    # model is in eval mode from main.py
    emb = request.app.state.extract_embedding(model, tensor, device).cpu().numpy()[0]
    return emb


@router.post("/enroll", response_model=EnrollResponse, responses={400: {"model": ErrorResponse}})
async def enroll(
    request: Request,
    user_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Enroll a new user by processing their fingerprint image and storing the deep embedding.
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id cannot be empty")
        
    # Check if user already exists
    existing_user = db.query(UserTemplate).filter(UserTemplate.user_id == user_id).first()
    if existing_user:
        raise HTTPException(status_code=400, detail=f"User {user_id} is already enrolled.")
        
    file_bytes = await file.read()
    
    try:
        emb = process_image(request, file_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
        
    # Convert numpy array to JSON list for storage
    emb_list = emb.tolist()
    emb_str = json.dumps(emb_list)
    
    new_template = UserTemplate(user_id=user_id, embedding_str=emb_str)
    db.add(new_template)
    db.commit()
    
    return EnrollResponse(
        user_id=user_id,
        status="enrolled",
        message="Successfully enrolled fingerprint template."
    )


@router.post("/verify", response_model=VerifyResponse, responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}})
async def verify(
    request: Request,
    user_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Verify a fingerprint against a specific enrolled user's template.
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id cannot be empty")
        
    # Retrieve user template
    user = db.query(UserTemplate).filter(UserTemplate.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found. Please enroll first.")
        
    # Process probe image
    file_bytes = await file.read()
    try:
        probe_emb = process_image(request, file_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
        
    # Reconstruct gallery embedding
    gallery_emb_list = json.loads(user.embedding_str)
    gallery_emb = np.array(gallery_emb_list, dtype=np.float32)
    
    # Score
    score = compute_embedding_score(probe_emb, gallery_emb)
    is_match = make_decision(score, OPERATING_THRESHOLD)
    
    return VerifyResponse(
        user_id=user_id,
        match=is_match,
        score=score,
        threshold=OPERATING_THRESHOLD,
        message="Match successful." if is_match else "Match failed."
    )
