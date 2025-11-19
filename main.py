import os
import hashlib
import zipfile
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from database import db, create_document, get_documents
from schemas import (
    Model as ModelSchema,
    TrainingRun as TrainingRunSchema,
    SimulationRun as SimulationRunSchema,
    Workflow as WorkflowSchema,
    Artifact as ArtifactSchema,
)

app = FastAPI(title="AI Ops Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Storage configuration
MODELS_DIR = os.getenv("MODELS_DIR", "uploads")
os.makedirs(MODELS_DIR, exist_ok=True)


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


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def is_probably_sb3_zip(path: str) -> bool:
    if not path.lower().endswith('.zip'):
        return False
    try:
        with zipfile.ZipFile(path, 'r') as z:
            names = {n.lower() for n in z.namelist()}
            # Common SB3 contents
            expected = {"policy.pth", "policy.pkl", "data.pkl", "parameters.pkl", "params.json"}
            return any(e in names or any(n.endswith(e) for n in names) for e in expected)
    except Exception:
        return False


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


class PromoteArtifactRequest(BaseModel):
    artifact_id: str


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
        artifact_count=0,
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


# Upload artifact (versioned), with optional promotion to active
@app.post("/api/models/{model_id}/artifact")
def upload_model_artifact(
    model_id: str,
    file: UploadFile = File(...),
    version: Optional[str] = Form(None),
    promote: Optional[bool] = Form(False),
):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    try:
        from bson import ObjectId
        m = db["model"].find_one({"_id": ObjectId(model_id)})
        if not m:
            raise HTTPException(status_code=404, detail="Model not found")

        # Directory for this model
        model_dir = os.path.join(MODELS_DIR, model_id)
        os.makedirs(model_dir, exist_ok=True)

        # Preserve original filename; avoid path traversal
        original_name = os.path.basename(file.filename or "artifact.bin")
        # If version provided, prefix to avoid collisions
        stored_name = f"{version}_{original_name}" if version else original_name
        stored_path = os.path.join(model_dir, stored_name)

        # Save file to disk
        with open(stored_path, "wb") as out:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)

        size = os.path.getsize(stored_path)
        content_type = file.content_type or "application/octet-stream"
        checksum = sha256_file(stored_path)
        sb3_valid = is_probably_sb3_zip(stored_path)

        art = ArtifactSchema(
            model_id=model_id,
            version=version or datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S"),
            filename=stored_name,
            path=stored_path,
            size=size,
            content_type=content_type,
            checksum_sha256=checksum,
            sb3_valid=sb3_valid,
        )
        artifact_id = create_document("artifact", art)

        # bump artifact_count
        db["model"].update_one({"_id": ObjectId(model_id)}, {"$inc": {"artifact_count": 1}})

        # Optionally promote to active
        if promote or not m.get("artifact_filename"):
            db["model"].update_one(
                {"_id": ObjectId(model_id)},
                {"$set": {
                    "artifact_filename": stored_name,
                    "artifact_path": stored_path,
                    "artifact_size": size,
                    "artifact_content_type": content_type,
                    "active_version": art.version,
                    "updated_at": datetime.now(timezone.utc),
                }}
            )

        return {"ok": True, "artifact_id": artifact_id, "filename": stored_name, "size": size, "sb3_valid": sb3_valid}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# List artifacts for a model
@app.get("/api/models/{model_id}/artifacts")
def list_model_artifacts(model_id: str) -> List[dict]:
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    try:
        from bson import ObjectId
        # verify model exists
        m = db["model"].find_one({"_id": ObjectId(model_id)})
        if not m:
            raise HTTPException(status_code=404, detail="Model not found")
        docs = get_documents("artifact", {"model_id": model_id}, limit=200)
        return [serialize_doc(d) for d in docs]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Promote an artifact to be active for the model
@app.post("/api/models/{model_id}/artifacts/{artifact_id}/promote")
def promote_artifact(model_id: str, artifact_id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    try:
        from bson import ObjectId
        art = db["artifact"].find_one({"_id": ObjectId(artifact_id), "model_id": model_id})
        if not art:
            raise HTTPException(status_code=404, detail="Artifact not found")
        db["model"].update_one(
            {"_id": ObjectId(model_id)},
            {"$set": {
                "artifact_filename": art.get("filename"),
                "artifact_path": art.get("path"),
                "artifact_size": art.get("size"),
                "artifact_content_type": art.get("content_type"),
                "active_version": art.get("version"),
                "updated_at": datetime.now(timezone.utc)
            }}
        )
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Delete an artifact
@app.delete("/api/models/{model_id}/artifacts/{artifact_id}")
def delete_artifact(model_id: str, artifact_id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    try:
        from bson import ObjectId
        art = db["artifact"].find_one({"_id": ObjectId(artifact_id), "model_id": model_id})
        if not art:
            raise HTTPException(status_code=404, detail="Artifact not found")
        # remove file on disk (best-effort)
        try:
            if art.get("path") and os.path.exists(art["path"]):
                os.remove(art["path"])
        except Exception:
            pass
        db["artifact"].delete_one({"_id": ObjectId(artifact_id)})
        # decrement artifact_count
        db["model"].update_one({"_id": ObjectId(model_id)}, {"$inc": {"artifact_count": -1}})
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Download a specific artifact by id
@app.get("/api/artifacts/{artifact_id}/download")
def download_artifact(artifact_id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    try:
        from bson import ObjectId
        art = db["artifact"].find_one({"_id": ObjectId(artifact_id)})
        if not art:
            raise HTTPException(status_code=404, detail="Artifact not found")
        path = art.get("path")
        filename = art.get("filename") or "artifact.bin"
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Artifact file missing from disk")
        return FileResponse(path, media_type=art.get("content_type") or "application/octet-stream", filename=filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Existing active-artifact info route (kept for convenience)
@app.get("/api/models/{model_id}/artifact")
def get_model_artifact_info(model_id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    try:
        from bson import ObjectId
        m = db["model"].find_one({"_id": ObjectId(model_id)})
        if not m:
            raise HTTPException(status_code=404, detail="Model not found")
        info = {k: m.get(k) for k in [
            "artifact_filename",
            "artifact_path",
            "artifact_size",
            "artifact_content_type",
            "active_version",
            "artifact_count",
        ]}
        return serialize_doc(info)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/models/{model_id}/artifact/download")
def download_model_artifact(model_id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    try:
        from bson import ObjectId
        m = db["model"].find_one({"_id": ObjectId(model_id)})
        if not m:
            raise HTTPException(status_code=404, detail="Model not found")
        path = m.get("artifact_path")
        filename = m.get("artifact_filename") or "artifact.bin"
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(path, media_type=m.get("artifact_content_type") or "application/octet-stream", filename=filename)
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
