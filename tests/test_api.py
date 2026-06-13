"""
test_api.py — Unit tests for the FastAPI application.
"""

import os
import sys
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.api.main import app, lifespan
from src.api.database import Base, get_db
from src.api.auth import API_KEY, API_KEY_NAME

from sqlalchemy.pool import StaticPool

# Setup test database (in-memory)
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

# Ensure tables are created in the in-memory db
Base.metadata.create_all(bind=engine)

# Create a test client
# We need to manually trigger the lifespan since TestClient handles it in newer Starlette versions
# if used with `with TestClient(app) as client:`
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture
def dummy_image():
    # Create a 224x224 grayscale image (like a fingerprint)
    img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    _, encoded = cv2.imencode('.png', img)
    return encoded.tobytes()

@pytest.fixture
def auth_headers():
    return {API_KEY_NAME: API_KEY}

def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_missing_api_key(client, dummy_image):
    response = client.post(
        "/api/v1/enroll",
        data={"user_id": "test_user"},
        files={"file": ("dummy.png", dummy_image, "image/png")}
    )
    assert response.status_code == 403

def test_enroll_success(client, dummy_image, auth_headers):
    response = client.post(
        "/api/v1/enroll",
        headers=auth_headers,
        data={"user_id": "test_user_1"},
        files={"file": ("dummy.png", dummy_image, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "test_user_1"
    assert data["status"] == "enrolled"

def test_enroll_duplicate_user(client, dummy_image, auth_headers):
    # Already enrolled in the previous test? 
    # Let's ensure it by enrolling test_user_2
    client.post(
        "/api/v1/enroll",
        headers=auth_headers,
        data={"user_id": "test_user_2"},
        files={"file": ("dummy.png", dummy_image, "image/png")}
    )
    
    # Try again
    response = client.post(
        "/api/v1/enroll",
        headers=auth_headers,
        data={"user_id": "test_user_2"},
        files={"file": ("dummy.png", dummy_image, "image/png")}
    )
    assert response.status_code == 400
    assert "already enrolled" in response.json()["detail"]

def test_verify_success(client, dummy_image, auth_headers):
    # Enroll test_user_3
    client.post(
        "/api/v1/enroll",
        headers=auth_headers,
        data={"user_id": "test_user_3"},
        files={"file": ("dummy.png", dummy_image, "image/png")}
    )
    
    # Verify with the exact same image (should be a perfect match, cosine sim ~ 1.0)
    response = client.post(
        "/api/v1/verify",
        headers=auth_headers,
        data={"user_id": "test_user_3"},
        files={"file": ("dummy.png", dummy_image, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "test_user_3"
    assert data["match"] is True
    assert data["score"] > 0.95 # Should be very high

def test_verify_unregistered_user(client, dummy_image, auth_headers):
    response = client.post(
        "/api/v1/verify",
        headers=auth_headers,
        data={"user_id": "unknown_user"},
        files={"file": ("dummy.png", dummy_image, "image/png")}
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
