"""
Integrated Multi-Agent Pipeline Server
=======================================
Unified FastAPI server that orchestrates all 6 agents in the Service Issue Resolution pipeline.

Agents:
  1. Complaint Classifier
  2. Evidence Collector
  3. Remedy Planner
  4. Credit Trigger
  5. Customer Communicator
  6. Audit Logger
"""

import os
import sys
import re
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure project root is on the Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

app = FastAPI(
    title="Service Issue Resolution Agent",
    description="Manage Customer Service Issue Resolution Agent — 6 AI agents pipeline",
    version="1.0.0"
)

# CORS - allow local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================== HEALTH & STATUS =====================

@app.get("/health")
async def health_check():
    """Basic health check - confirms server is running."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Service Issue Resolution Agent",
        "port": int(os.getenv("PIPELINE_PORT", 8080))
    }


@app.get("/api/status")
async def pipeline_status():
    """Shows which agents are integrated and ready."""
    return {
        "pipeline": "Service Issue Resolution Processing",
        "agents": {
            "agent1_classifier": {"status": "ready", "name": "Complaint Classifier", "endpoint": "/api/agent1/classify"},
            "agent2_evidence": {"status": "ready", "name": "Evidence Collector", "endpoint": "/api/agent2/collect"},
            "agent3_remedy": {"status": "ready", "name": "Remedy Planner", "endpoint": "/api/agent3/plan"},
            "agent4_credit": {"status": "ready", "name": "Credit Trigger", "endpoint": "/api/agent4/credit"},
            "agent5_communicator": {"status": "ready", "name": "Customer Communicator", "endpoint": "/api/agent5/communicate"},
            "agent6_audit": {"status": "ready", "name": "Audit Logger", "endpoint": "/api/agent6/audit"},
        },
        "autogen_version": "0.4.x",
        "llm_configured": bool(os.getenv("AZURE_OPENAI_API_KEY")),
        "timestamp": datetime.now().isoformat()
    }


# ===================== AGENT ENDPOINTS =====================
# These will be added one by one as we integrate each agent.
# Phase 1: /api/agent1/classify
# Phase 2: /api/agent2/collect
# Phase 3: /api/agent3/plan
# Phase 4: /api/agent4/credit
# Phase 5: /api/agent5/communicate
# Phase 6: /api/pipeline/run (full chain)

from agents.agent1_classifier.classifier import ComplaintClassifierAgent
from agents.agent2_evidence.evidence_collector import EvidenceCollectorAgent
from agents.agent3_remedy.remedy_planner import RemedyPlannerAgent
from agents.agent4_credit.credit_trigger import CreditTriggerAgent
from agents.agent5_communicator.communicator import CustomerCommunicatorAgent
from agents.agent6_audit.audit_logger import AuditLoggerAgent
from orchestrator import PipelineOrchestrator

import json

INPUTS_DIR = Path(PROJECT_ROOT) / "inputs" / "samples"


@app.post("/api/agent1/classify")
async def classify_complaint(sample: str = "sample_01"):
    """Run Agent 1 - Complaint Classifier on a sample folder.

    The sample folder must contain:
      - complaint_text.txt
      - customer_profile.json
      - interaction_history.json
    """
    sample_dir = INPUTS_DIR / sample
    if not sample_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Sample folder not found: {sample}")

    complaint_path = sample_dir / "complaint_text.txt"
    profile_path = sample_dir / "customer_profile.json"
    history_path = sample_dir / "interaction_history.json"

    for p in [complaint_path, profile_path, history_path]:
        if not p.exists():
            raise HTTPException(status_code=400, detail=f"Missing input file: {p.name}")

    agent = ComplaintClassifierAgent()
    result = await agent.process_case(
        complaint_path=complaint_path,
        customer_profile_path=profile_path,
        interaction_history_path=history_path,
    )
    return result


@app.post("/api/agent2/collect")
async def collect_evidence(sample: str = "sample_01", complaint_id: str | None = None):
    """Run Agent 2 - Evidence Collector.

    Reads Agent 1 output from outputs/complaint_category/{complaint_id}.json,
    plus customer profile from inputs/samples/{sample}/customer_profile.json.
    If complaint_id is not provided, uses the most recent file in the output folder.
    """
    # Load Agent 1 output (classification)
    category_dir = Path(PROJECT_ROOT) / "outputs" / "complaint_category"
    if complaint_id:
        classification_path = category_dir / f"{complaint_id}.json"
    else:
        # Find the most recent file
        files = sorted(category_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            raise HTTPException(status_code=400, detail="Run Agent 1 first — no classification output found.")
        classification_path = files[0]
    if not classification_path.exists():
        raise HTTPException(status_code=400, detail=f"Classification output not found: {classification_path.name}")
    with open(classification_path, "r", encoding="utf-8") as f:
        complaint_data = json.load(f)

    # Load customer profile
    profile_path = INPUTS_DIR / sample / "customer_profile.json"
    if not profile_path.exists():
        raise HTTPException(status_code=400, detail="Missing customer_profile.json in sample folder.")
    with open(profile_path, "r", encoding="utf-8") as f:
        customer_profile = json.load(f)

    agent = EvidenceCollectorAgent()
    result = await agent.process(complaint_data, customer_profile)
    return result


@app.post("/api/agent3/plan")
async def plan_remedy(sample: str = "sample_01", complaint_id: str | None = None):
    """Run Agent 3 - Remedy Planner.

    Reads Agent 1 output (classification) and Agent 2 output (evidence packet).
    """
    category_dir = Path(PROJECT_ROOT) / "outputs" / "complaint_category"
    evidence_dir = Path(PROJECT_ROOT) / "outputs" / "evidence_packet"

    # Resolve complaint_id from latest file if not provided
    if complaint_id:
        class_path = category_dir / f"{complaint_id}.json"
        evidence_path = evidence_dir / f"{complaint_id}.json"
    else:
        files = sorted(category_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            raise HTTPException(status_code=400, detail="Run Agent 1 first.")
        class_path = files[0]
        cid = class_path.stem
        evidence_path = evidence_dir / f"{cid}.json"

    if not class_path.exists():
        raise HTTPException(status_code=400, detail=f"Classification not found: {class_path.name}")
    if not evidence_path.exists():
        raise HTTPException(status_code=400, detail=f"Evidence not found. Run Agent 2 first.")

    with open(class_path, "r", encoding="utf-8") as f:
        classification = json.load(f)
    with open(evidence_path, "r", encoding="utf-8") as f:
        evidence = json.load(f)

    agent = RemedyPlannerAgent()
    result = await agent.process(classification, evidence)
    return result


# Phase 4: /api/agent4/credit

@app.post("/api/agent4/credit")
async def trigger_credit(sample: str = "sample_01", complaint_id: str | None = None):
    """Run Agent 4 - Credit Trigger.

    Reads Agent 3 output (resolution plan) and customer profile.
    """
    plan_dir = Path(PROJECT_ROOT) / "outputs" / "resolution_plan"

    # Resolve complaint_id from latest file if not provided
    if complaint_id:
        plan_path = plan_dir / f"{complaint_id}.json"
    else:
        files = sorted(plan_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            raise HTTPException(status_code=400, detail="Run Agent 3 first.")
        plan_path = files[0]
        complaint_id = plan_path.stem

    if not plan_path.exists():
        raise HTTPException(status_code=400, detail=f"Resolution plan not found: {plan_path.name}")

    # Load customer profile from sample inputs
    sample_dir = INPUTS_DIR / sample
    profile_path = sample_dir / "customer_profile.json"
    if not profile_path.exists():
        raise HTTPException(status_code=400, detail=f"Customer profile not found: {profile_path}")

    with open(plan_path, "r", encoding="utf-8") as f:
        resolution_plan = json.load(f)
    with open(profile_path, "r", encoding="utf-8") as f:
        customer_account = json.load(f)

    agent = CreditTriggerAgent()
    result = await agent.process(resolution_plan, customer_account)
    return result


# Phase 5: /api/agent5/communicate

@app.post("/api/agent5/communicate")
async def communicate(sample: str = "sample_01", complaint_id: str | None = None):
    """Run Agent 5 - Customer Communicator.

    Reads Agent 3 output (resolution plan), Agent 4 output (credit confirmation),
    and customer profile to generate a personalized resolution message.
    """
    plan_dir = Path(PROJECT_ROOT) / "outputs" / "resolution_plan"
    credit_dir = Path(PROJECT_ROOT) / "outputs" / "credit_confirmation"

    # Resolve complaint_id from latest file if not provided
    if complaint_id:
        plan_path = plan_dir / f"{complaint_id}.json"
        credit_path = credit_dir / f"{complaint_id}.json"
    else:
        files = sorted(plan_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            raise HTTPException(status_code=400, detail="Run Agent 3 first.")
        plan_path = files[0]
        cid = plan_path.stem
        credit_path = credit_dir / f"{cid}.json"

    if not plan_path.exists():
        raise HTTPException(status_code=400, detail=f"Resolution plan not found: {plan_path.name}")
    if not credit_path.exists():
        raise HTTPException(status_code=400, detail="Credit confirmation not found. Run Agent 4 first.")

    # Load customer profile from sample inputs
    profile_path = INPUTS_DIR / sample / "customer_profile.json"
    if not profile_path.exists():
        raise HTTPException(status_code=400, detail="Customer profile not found.")

    with open(plan_path, "r", encoding="utf-8") as f:
        resolution_plan = json.load(f)
    with open(credit_path, "r", encoding="utf-8") as f:
        credit_confirmation = json.load(f)
    with open(profile_path, "r", encoding="utf-8") as f:
        customer_profile = json.load(f)

    agent = CustomerCommunicatorAgent()
    result = await agent.process(resolution_plan, credit_confirmation, customer_profile)
    return result


# ===================== AGENT 6: AUDIT LOGGER =====================

@app.post("/api/agent6/audit")
async def audit_log(sample: str = "sample_01"):
    """Run Agent 6 - Audit Logger.

    Runs the full pipeline first, then processes the results through the audit agent.
    """
    sample_dir = INPUTS_DIR / sample
    if not sample_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Sample folder not found: {sample}")

    complaint_path = sample_dir / "complaint_text.txt"
    profile_path = sample_dir / "customer_profile.json"
    history_path = sample_dir / "interaction_history.json"

    for p in [complaint_path, profile_path, history_path]:
        if not p.exists():
            raise HTTPException(status_code=400, detail=f"Missing input file: {p.name}")

    orchestrator = PipelineOrchestrator()
    pipeline_result = await orchestrator.run(
        complaint_path=complaint_path,
        customer_profile_path=profile_path,
        interaction_history_path=history_path,
    )

    agent = AuditLoggerAgent()
    audit_result = await agent.process(pipeline_result)
    return audit_result


# ===================== FULL PIPELINE =====================

@app.post("/api/pipeline/run")
async def run_pipeline(sample: str = "sample_01"):
    """Run the full 5-agent pipeline end-to-end on a sample folder."""
    sample_dir = INPUTS_DIR / sample
    if not sample_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Sample folder not found: {sample}")

    complaint_path = sample_dir / "complaint_text.txt"
    profile_path = sample_dir / "customer_profile.json"
    history_path = sample_dir / "interaction_history.json"

    for p in [complaint_path, profile_path, history_path]:
        if not p.exists():
            raise HTTPException(status_code=400, detail=f"Missing input file: {p.name}")

    orchestrator = PipelineOrchestrator()
    result = await orchestrator.run(
        complaint_path=complaint_path,
        customer_profile_path=profile_path,
        interaction_history_path=history_path,
    )
    return result


# ===================== FRONTEND =====================

# Serve static assets (CSS, JS)
FRONTEND_DIR = Path(PROJECT_ROOT) / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def serve_index():
    """Serve the dashboard HTML."""
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/api/inputs/{sample}")
async def get_inputs(sample: str):
    """Return sample input files for preview in the UI."""
    if not re.match(r'^[a-zA-Z0-9_-]+$', sample):
        raise HTTPException(status_code=400, detail="Invalid sample name")
    sample_dir = INPUTS_DIR / sample
    if not sample_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Sample not found: {sample}")

    result = {}
    complaint_path = sample_dir / "complaint_text.txt"
    if complaint_path.exists():
        result["complaint_text"] = complaint_path.read_text(encoding="utf-8")

    profile_path = sample_dir / "customer_profile.json"
    if profile_path.exists():
        with open(profile_path, "r", encoding="utf-8") as f:
            result["customer_profile"] = json.load(f)

    history_path = sample_dir / "interaction_history.json"
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            result["interaction_history"] = json.load(f)

    return result


# ===================== UPLOAD & SAMPLES =====================

@app.get("/api/samples")
async def list_samples():
    """List all available sample folders."""
    samples = sorted(
        d.name for d in INPUTS_DIR.iterdir()
        if d.is_dir() and (d / "complaint_text.txt").exists()
    )
    return {"samples": samples}


@app.post("/api/upload")
async def upload_inputs(
    complaint_text: UploadFile = File(...),
    customer_profile: UploadFile = File(...),
    interaction_history: UploadFile = File(...),
):
    """Upload 3 input files to create a new sample folder.

    Files required:
      - complaint_text (.txt)
      - customer_profile (.json)
      - interaction_history (.json)

    Returns the sample folder name.
    """
    # Generate unique folder name
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_name = f"upload_{ts}"
    sample_dir = INPUTS_DIR / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Save files with fixed names
    try:
        content = await complaint_text.read()
        (sample_dir / "complaint_text.txt").write_bytes(content)

        content = await customer_profile.read()
        # Validate JSON
        json.loads(content)
        (sample_dir / "customer_profile.json").write_bytes(content)

        content = await interaction_history.read()
        json.loads(content)
        (sample_dir / "interaction_history.json").write_bytes(content)
    except json.JSONDecodeError as e:
        # Clean up on failure
        import shutil
        shutil.rmtree(sample_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {e}")
    except Exception as e:
        import shutil
        shutil.rmtree(sample_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Upload failed: {e}")

    return {"sample": sample_name, "files": ["complaint_text.txt", "customer_profile.json", "interaction_history.json"]}


@app.get("/api/audit-csv/{complaint_id}")
async def download_audit_csv(complaint_id: str):
    """Download the audit trail CSV for a complaint."""
    if not re.match(r'^CMP-[A-Za-z0-9]+$', complaint_id):
        raise HTTPException(status_code=400, detail="Invalid complaint ID format")
    csv_path = Path(PROJECT_ROOT) / "outputs" / "audit_trail" / f"{complaint_id}_index.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"Audit CSV not found for {complaint_id}")
    return FileResponse(str(csv_path), media_type="text/csv", filename=f"{complaint_id}_audit.csv")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PIPELINE_PORT", 8080))
    print(f"\n{'='*60}")
    print(f"  Service Issue Resolution Agent")
    print(f"  Starting on http://localhost:{port}")
    print(f"  Health check: http://localhost:{port}/health")
    print(f"  Pipeline status: http://localhost:{port}/api/status")
    print(f"{'='*60}\n")
    uvicorn.run(app, host="127.0.0.1", port=port)
