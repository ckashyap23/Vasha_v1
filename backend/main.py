"""
FastAPI backend for the BN<->EN translation/scoring UI.

Endpoints:
  POST /api/translate   — upload an xlsx (ben, ref_en?, source?), pick a model,
                           kicks off a background Kaggle job. Returns job_id.
  POST /api/score       — upload an xlsx (ben, ref_en?, source?, translated_en),
                           kicks off a background Kaggle scoring job. Returns job_id.
  GET  /api/jobs/{id}   — poll job status: queued | running | done | failed.
  GET  /api/jobs/{id}/download — download the resulting xlsx once done.

Jobs run in a background thread since a single Kaggle run can take several
minutes (GPU cold start + model load + inference/scoring). The frontend
polls /api/jobs/{id} until status is "done" or "failed".
"""

import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import kaggle_orchestrator as kg

app = FastAPI(title="BN-EN Translation UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual frontend origin once deployed
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/tmp/bn_translate_ui/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job store. Fine for a single-instance deployment; swap for Redis
# or a DB if you ever run more than one backend replica.
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

SUPPORTED_MODELS = ["qwen_base"]  # add "qwen_finetuned_v1" etc. here later


class JobStatus(BaseModel):
    job_id: str
    kind: Literal["translate", "score"]
    status: Literal["queued", "running", "done", "failed"]
    error: Optional[str] = None
    created_at: str
    result_filename: Optional[str] = None


def _set_job(job_id: str, **fields):
    with JOBS_LOCK:
        JOBS[job_id].update(fields)


def _run_translate_job(job_id: str, input_path: Path, run_label: str, model_choice: str):
    _set_job(job_id, status="running")
    try:
        result_path = kg.run_translate_job(input_path, run_label, model_choice, job_id)
        _set_job(job_id, status="done", result_path=str(result_path), result_filename=result_path.name)
    except Exception as e:
        _set_job(job_id, status="failed", error=str(e))


def _run_score_job(job_id: str, input_path: Path, run_label: str):
    _set_job(job_id, status="running")
    try:
        result_path = kg.run_score_job(input_path, run_label, job_id)
        _set_job(job_id, status="done", result_path=str(result_path), result_filename=result_path.name)
    except Exception as e:
        _set_job(job_id, status="failed", error=str(e))


@app.get("/api/models")
def list_models():
    return {"models": SUPPORTED_MODELS}


def _validate_excel(path: Path) -> None:
    """Raise HTTPException 400 if the file cannot be read as Excel."""
    for engine in ("openpyxl", "xlrd"):
        try:
            pd.read_excel(path, engine=engine, nrows=0)
            return  # readable — done
        except Exception:
            pass
    path.unlink(missing_ok=True)
    raise HTTPException(
        400,
        "The uploaded file could not be read as Excel. "
        "Please open it in Excel/LibreOffice, choose File → Save As, "
        "select 'Excel Workbook (*.xlsx)' format, and re-upload."
    )


@app.post("/api/translate")
async def translate(
    file: UploadFile = File(...),
    model: str = Form(...),
    run_label: str = Form(default=""),
):
    if model not in SUPPORTED_MODELS:
        raise HTTPException(400, f"Unknown model '{model}'. Supported: {SUPPORTED_MODELS}")
    if not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(400, "Please upload an .xlsx or .xls file.")

    job_id = str(uuid.uuid4())[:12]
    input_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    input_path.write_bytes(await file.read())
    _validate_excel(input_path)

    label = run_label.strip() or model
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "kind": "translate",
            "status": "queued",
            "created_at": datetime.utcnow().isoformat(),
            "error": None,
            "result_filename": None,
        }

    thread = threading.Thread(
        target=_run_translate_job, args=(job_id, input_path, label, model), daemon=True
    )
    thread.start()

    return {"job_id": job_id}


@app.post("/api/score")
async def score(
    file: UploadFile = File(...),
    run_label: str = Form(default=""),
):
    if not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(400, "Please upload an .xlsx or .xls file.")

    job_id = str(uuid.uuid4())[:12]
    input_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    input_path.write_bytes(await file.read())
    _validate_excel(input_path)

    label = run_label.strip() or "scoring_run"
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "kind": "score",
            "status": "queued",
            "created_at": datetime.utcnow().isoformat(),
            "error": None,
            "result_filename": None,
        }

    thread = threading.Thread(
        target=_run_score_job, args=(job_id, input_path, label), daemon=True
    )
    thread.start()

    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    return JobStatus(**{k: v for k, v in job.items() if k in JobStatus.model_fields})


@app.get("/api/jobs/{job_id}/download")
def download_job_result(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    if job["status"] != "done":
        raise HTTPException(409, f"Job is not finished yet (status: {job['status']}).")
    result_path = Path(job["result_path"])
    if not result_path.exists():
        raise HTTPException(410, "Result file no longer available.")
    return FileResponse(
        result_path,
        filename=job["result_filename"],
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# Serve the simple frontend as static files
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
