"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogpost" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime

# Core domain schemas for the AI Ops dashboard

class Model(BaseModel):
    name: str = Field(..., description="Model name")
    version: str = Field("v1", description="Semantic version")
    status: Literal["ready", "training", "stale", "needs-training"] = Field(
        "ready", description="Operational status"
    )
    task: Literal["classification", "generation", "embedding", "rl", "other"] = "generation"
    last_trained_at: Optional[datetime] = Field(None, description="Last training timestamp")
    accuracy: Optional[float] = Field(None, ge=0, le=1, description="Validation accuracy (0-1)")
    owner: Optional[str] = Field(None, description="Owner or team")
    # Artifact metadata for uploaded model files (kept for quick access / active artifact)
    artifact_filename: Optional[str] = Field(None, description="Stored filename of the active model artifact")
    artifact_path: Optional[str] = Field(None, description="Server-side path to the active artifact")
    artifact_size: Optional[int] = Field(None, description="Size in bytes of the active artifact")
    artifact_content_type: Optional[str] = Field(None, description="MIME type of the active artifact")
    active_version: Optional[str] = Field(None, description="Active artifact version for this model")
    artifact_count: int = Field(0, description="How many artifacts exist for this model")

class Artifact(BaseModel):
    model_id: str = Field(..., description="Associated model id")
    version: str = Field(..., description="Version label, e.g., v1, v2")
    filename: str
    path: str
    size: int
    content_type: Optional[str] = None
    checksum_sha256: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    sb3_valid: Optional[bool] = Field(None, description="Whether the artifact appears to be a valid Stable-Baselines3 .zip")

class Workflow(BaseModel):
    name: str
    description: Optional[str] = None
    model_ids: List[str] = Field(default_factory=list, description="Associated models")
    schedule: Optional[str] = Field(None, description="CRON or cadence descriptor")

class TrainingRun(BaseModel):
    model_id: str
    workflow_id: Optional[str] = None
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    epochs: int = 1
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    notes: Optional[str] = None

class SimulationRun(BaseModel):
    model_id: str
    scenario: str = "default"
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    score: Optional[float] = None
    tokens: Optional[int] = None
    latency_ms: Optional[float] = None
