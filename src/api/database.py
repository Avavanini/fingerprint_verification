"""
database.py — Database configuration and session management.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from pathlib import Path

# Create data directory if it doesn't exist
DATA_DIR = Path(os.path.abspath(__file__)).parent.parent.parent / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Use SQLite for local development
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DATA_DIR}/fingerprints.db"

# Setting check_same_thread=False is needed for SQLite in FastAPI
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
