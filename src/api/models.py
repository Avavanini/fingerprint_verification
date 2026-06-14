"""
models.py — SQLAlchemy database models.
"""

from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy.sql import func
from .database import Base

class UserTemplate(Base):
    __tablename__ = "user_templates"

    user_id = Column(String, primary_key=True, index=True)
    
    # We will store the 128-D embedding as a comma-separated string or JSON string
    # For a production database, a Vector or Array type (like in PostgreSQL/pgvector) is preferred.
    embedding_str = Column(String, nullable=False)
    
    # Serialized minutiae template (JSON) for hybrid matching.
    # Nullable for backward compatibility with enrollments made before hybrid matching.
    minutiae_template_str = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
