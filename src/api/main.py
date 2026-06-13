"""
main.py — FastAPI application initialization and startup events.
"""

import os
import torch
from fastapi import FastAPI
from contextlib import asynccontextmanager
from torchvision import transforms
from pathlib import Path

from .database import engine, Base
from .routes import router
from src.feature_extraction.embedding import SiameseNetwork, extract_embedding

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'siamese_model.pth'

# Create DB tables
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the deep learning model into app state
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = SiameseNetwork(embedding_dim=128, pretrained=False).to(device)
    
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded Siamese Network weights from {MODEL_PATH}")
    else:
        print(f"WARNING: Model weights not found at {MODEL_PATH}. Using untrained model.")
        
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Store in app state
    app.state.model = model
    app.state.device = device
    app.state.transform = transform
    app.state.extract_embedding = extract_embedding
    
    yield
    
    # Shutdown logic (if any)
    pass

app = FastAPI(
    title="Fingerprint Verification API",
    description="API for enrolling and verifying fingerprints using Deep Embeddings.",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fingerprint Verification API. See /docs for usage."}
