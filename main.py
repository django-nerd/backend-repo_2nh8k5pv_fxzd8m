import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import db, create_document, get_documents
from schemas import (
    Model as ModelSchema,
    TrainingRun as TrainingRunSchema,
    SimulationRun as SimulationRunSchema,
    Workflow as WorkflowSchema,
)

app = FastAPI(title="AI Ops Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helpers

def serialize_doc(doc: dict) -> dict:
    if not doc:
        return doc
    d = dict(doc)
    if "_id" in d:
        d["id"] = str(d.pop("_id"))
    # Convert datetimes to isoformat
    for k, v in list(d.items()):
        if hasattr(v, "isoformat"):
            d[k] = v.isoformat()
    return d


def list_collection(name: str, limit: Optional[int] = None, filter_dict: Optional[dict] = None):
    docs = get_documents(name, filter_dict or {}, limit)
    return [serialize_doc(x) for x in docs]


# Root and health
@app.get("/")
def read_root():
    return {"message": "AI Ops Dashboard API running"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = getattr(db, "name", "✅ Connected")
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    return response


# Schemas for request bodies
class CreateModel(BaseModel):
    name: str
    version: Optional[str] = "v1"
    task: Optional[str] = "generation"
    owner: Optional[str] = None


class UpdateModelStatus(BaseModel):
    status: str


class CreateTrainingRequest(BaseModel):
    epochs: int = 1
    workflow_id: Optional[str] = None
    notes: Optional[str] = None


class CreateSimulationRequest(BaseModel):
    scenario: str = "default"


# Models endpoints
@app.get("/api/models")
def get_models():
    return list_collection("model")


@app.post("/api/models")
def create_model(body: CreateModel):
    data = ModelSchema(
        name=body.name,
        version=body.version or "v1",
        task=body.task or "generation",
        owner=body.owner,
        status="ready",
        last_trained_at=None,
    )
    new_id = create_document("model", data)
    return {"id": new_id}


@app.post("/api/models/{model_id}/needs-training")
def mark_needs_training(model_id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    try:
        from bson import ObjectId
        res = db["model"].update_one(
            {"_id": ObjectId(model_id)},
            {"$set": {"status": "needs-training", "updated_at": datetime.now(timezone.utc)}},
        )
        if res.matched_count == 0:
            raise HTTPException(status_code=404, detail="Model not found")
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Training runs
@app.get("/api/training-runs")
def get_training_runs():
    return list_collection("trainingrun", limit=100)


@app.post("/api/models/{model_id}/train")
def trigger_training(model_id: str, body: CreateTrainingRequest):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    # Ensure model exists
    try:
        from bson import ObjectId
        m = db["model"].find_one({"_id": ObjectId(model_id)})
        if not m:
            raise HTTPException(status_code=404, detail="Model not found")
        run = TrainingRunSchema(
            model_id=model_id,
            workflow_id=body.workflow_id,
            status="queued",
            epochs=body.epochs,
            notes=body.notes,
        )
        run_id = create_document("trainingrun", run)
        # Update model status
        db["model"].update_one(
            {"_id": ObjectId(model_id)},
            {"$set": {"status": "training", "updated_at": datetime.now(timezone.utc)}},
        )
        return {"id": run_id, "status": "queued"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Simulations
@app.get("/api/simulations")
def get_simulations():
    return list_collection("simulationrun", limit=100)


@app.post("/api/models/{model_id}/simulate")
def trigger_simulation(model_id: str, body: CreateSimulationRequest):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    try:
        from bson import ObjectId
        m = db["model"].find_one({"_id": ObjectId(model_id)})
        if not m:
            raise HTTPException(status_code=404, detail="Model not found")
        sim = SimulationRunSchema(model_id=model_id, scenario=body.scenario, status="queued")
        sim_id = create_document("simulationrun", sim)
        return {"id": sim_id, "status": "queued"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Workflows (basic list/create)
@app.get("/api/workflows")
def get_workflows():
    return list_collection("workflow")


class CreateWorkflow(BaseModel):
    name: str
    description: Optional[str] = None


@app.post("/api/workflows")
def create_workflow(body: CreateWorkflow):
    wf = WorkflowSchema(name=body.name, description=body.description, model_ids=[])
    wf_id = create_document("workflow", wf)
    return {"id": wf_id}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
