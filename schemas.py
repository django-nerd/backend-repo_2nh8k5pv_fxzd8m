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
