"""
Agent 6 — Audit Logger & Compliance Agent
==========================================
Aggregates results from all 5 upstream agents, validates the audit trail,
checks GDPR/SOX compliance, detects anomalies, and generates a comprehensive
audit report with JSON + CSV outputs.

Autogen 0.4.x path: uses AssistantAgent for LLM-enhanced compliance analysis.
Fallback: rule-based validation, compliance checking, and report generation.
"""

import csv
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "audit_trail"

# ── Agent output → workflow step & agent-id mapping ────────

AGENT_MAP = {
    "agent1_classifier": {"workflow_step": "Complaint_Classifier", "agent_id": "CLA-01"},
    "agent2_evidence":   {"workflow_step": "Evidence_Collector",   "agent_id": "ECA-01"},
    "agent3_remedy":     {"workflow_step": "Remedy_Planner",       "agent_id": "RPA-01"},
    "agent4_credit":     {"workflow_step": "Credit_Trigger",       "agent_id": "CTA-01"},
    "agent5_communicator": {"workflow_step": "Customer_Communicator", "agent_id": "COM-01"},
}

REQUIRED_STEPS = {
    "Complaint_Classifier",
    "Evidence_Collector",
    "Remedy_Planner",
    "Credit_Trigger",
    "Customer_Communicator",
}


# ── Data structures ────────────────────────────────────────

@dataclass
class AuditLogEntry:
    workflow_step: str
    agent_id: str
    timestamp: str
    status: str
    complaint_id: str
    customer_id: str
    details: Optional[Dict[str, Any]] = None
    validation_status: Optional[str] = None
    compliance_flags: Optional[List[str]] = None


# ── Validators ─────────────────────────────────────────────

VALID_STATUSES = {
    "classified", "evidence_ready", "plan_ready", "credit_approved",
    "message_sent", "completed", "failed", "clarification_requested",
    "exception_handling",
}
VALID_STEPS = REQUIRED_STEPS | {"Audit_Logger"}
VALID_PREFIXES = ("CLA", "ECA", "RPA", "CTA", "COM", "ALA")


def _validate_entry(entry: AuditLogEntry) -> Dict[str, Any]:
    errors: list[str] = []
    for field in ("workflow_step", "agent_id", "timestamp", "status", "complaint_id", "customer_id"):
        val = getattr(entry, field, None)
        if not val or (isinstance(val, str) and not val.strip()):
            errors.append(f"Missing required field: {field}")
    if entry.workflow_step and entry.workflow_step not in VALID_STEPS:
        errors.append(f"Invalid workflow_step: {entry.workflow_step}")
    if entry.agent_id and not any(entry.agent_id.startswith(p) for p in VALID_PREFIXES):
        errors.append(f"Invalid agent_id format: {entry.agent_id}")
    if entry.complaint_id and not entry.complaint_id.startswith("CMP-"):
        errors.append(f"Invalid complaint_id format: {entry.complaint_id}")
    if entry.timestamp:
        try:
            datetime.fromisoformat(entry.timestamp.rstrip("Z"))
        except (ValueError, TypeError):
            errors.append(f"Invalid timestamp: {entry.timestamp}")
    if entry.status and entry.status not in VALID_STATUSES:
        errors.append(f"Invalid status: {entry.status}")
    return {"valid": len(errors) == 0, "errors": errors}


# ── Compliance checker ─────────────────────────────────────

def _check_compliance(entry: AuditLogEntry) -> Dict[str, Any]:
    flags: list[str] = []
    if entry.validation_status not in ("pass", "failed"):
        flags.append("MISSING_VALIDATION_STATUS")
    if not entry.timestamp:
        flags.append("MISSING_TIMESTAMP")
    if not entry.details:
        flags.append("MISSING_ENTRY_DETAILS")
    if not (entry.timestamp and entry.agent_id and entry.validation_status and entry.details):
        flags.append("SOX_COMPLIANCE_ISSUE")
    # GDPR check for communicator
    if entry.workflow_step == "Customer_Communicator" and entry.details:
        comp = entry.details.get("compliance")
        if isinstance(comp, dict) and not comp.get("gdpr", False):
            flags.append("GDPR_COMMUNICATION_VIOLATION")
    return {"compliant": len(flags) == 0, "flags": flags}


def _check_workflow_compliance(audit_trail: Dict[str, Any]) -> Dict[str, Any]:
    issues: list[dict] = []
    checks_passed = 0
    checks_total = 0

    summary = audit_trail.get("summary", {})
    entries = audit_trail.get("audit_entries", [])

    # 1. All steps present?
    checks_total += 1
    if summary.get("all_steps_completed"):
        checks_passed += 1
    else:
        issues.append({"check": "workflow_completeness", "status": "failed",
                        "details": f"Missing steps: {summary.get('missing_steps', [])}"})

    # 2. No exceptions?
    checks_total += 1
    exc = audit_trail.get("exceptions", [])
    if not exc:
        checks_passed += 1
    else:
        issues.append({"check": "exception_handling", "status": "warning",
                        "count": len(exc), "details": "Exceptions detected"})

    # 3. All validated?
    checks_total += 1
    invalid = [e for e in entries if e.get("validation_status") != "pass"]
    if not invalid:
        checks_passed += 1
    else:
        issues.append({"check": "entry_validation", "status": "warning",
                        "count": len(invalid), "details": f"{len(invalid)} entries with issues"})

    # 4. All required agents present?
    checks_total += 1
    agents_present = {e.get("agent_id") for e in entries}
    required_agents = {"CLA-01", "ECA-01", "RPA-01", "CTA-01", "COM-01"}
    if required_agents.issubset(agents_present):
        checks_passed += 1
    else:
        issues.append({"check": "agent_presence", "status": "failed",
                        "missing_agents": list(required_agents - agents_present)})

    status = "pass" if not issues else ("warning" if checks_passed >= 3 else "review_required")
    return {"status": status, "checks_passed": checks_passed, "checks_total": checks_total, "details": issues}


# ══════════════════════════════════════════════════════════
#  Main Agent Class
# ══════════════════════════════════════════════════════════

class AuditLoggerAgent:
    """
    Processes the combined output of all 5 upstream agents and produces:
      - audit_trail.json   (per-entry validated, compliance-checked)
      - audit_index.csv    (summary table)
      - compliance report  (GDPR + SOX)
      - anomaly report
    """

    AGENT_ID = "ALA-01"

    def __init__(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self._load_config()

    def _load_config(self):
        cfg_path = CONFIG_DIR / "audit_config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {
                "compliance_rules": {"require_all_steps": True, "require_validation": True},
                "workflow_steps": list(VALID_STEPS),
            }

    # ── public entry-point ─────────────────────────────────

    async def process(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accept the full pipeline_results dict (same shape returned by
        PipelineOrchestrator.run()) and generate audit trail.

        Returns a dict suitable for JSON response / UI rendering.
        """
        print("\n" + "─" * 50)
        print("Audit Logger Agent  (ALA-01)")
        print("─" * 50)

        results_block = pipeline_results.get("results", pipeline_results)
        pipeline_run = pipeline_results.get("pipeline_run", {})
        complaint_id = pipeline_run.get("complaint_id") or self._find_complaint_id(results_block)
        customer_id = self._find_customer_id(results_block)

        # ── Build audit entries for each upstream agent ────
        audit_entries: list[AuditLogEntry] = []
        exceptions: list[dict] = []

        for agent_key, meta in AGENT_MAP.items():
            agent_output = results_block.get(agent_key)
            if agent_output is None or "error" in agent_output:
                continue

            status_map = {
                "Complaint_Classifier": "classified",
                "Evidence_Collector": "evidence_ready",
                "Remedy_Planner": "plan_ready",
                "Credit_Trigger": "credit_approved",
                "Customer_Communicator": "message_sent",
            }

            entry = AuditLogEntry(
                workflow_step=meta["workflow_step"],
                agent_id=meta["agent_id"],
                timestamp=datetime.now(timezone.utc).isoformat(),
                status=status_map.get(meta["workflow_step"], "completed"),
                complaint_id=complaint_id or "UNKNOWN",
                customer_id=customer_id or "UNKNOWN",
                details=agent_output,
                validation_status=agent_output.get("validation_status", "unknown"),
            )

            # Validate
            val = _validate_entry(entry)
            if not val["valid"]:
                entry.validation_status = "failed"
                exceptions.append({
                    "complaint_id": complaint_id,
                    "workflow_step": meta["workflow_step"],
                    "agent_id": meta["agent_id"],
                    "errors": val["errors"],
                })
            else:
                entry.validation_status = "pass"

            # Compliance
            comp = _check_compliance(entry)
            entry.compliance_flags = comp["flags"]

            audit_entries.append(entry)

        # Add self (Audit_Logger step)
        self_entry = AuditLogEntry(
            workflow_step="Audit_Logger",
            agent_id=self.AGENT_ID,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status="completed",
            complaint_id=complaint_id or "UNKNOWN",
            customer_id=customer_id or "UNKNOWN",
            details={"agents_audited": len(audit_entries), "exceptions_found": len(exceptions)},
            validation_status="pass",
            compliance_flags=[],
        )
        audit_entries.append(self_entry)

        # ── Build aggregated trail ─────────────────────────
        entries_dicts = [asdict(e) for e in audit_entries]
        steps_present = {e.workflow_step for e in audit_entries}
        missing_steps = list(REQUIRED_STEPS - steps_present)

        timestamps = [e.timestamp for e in audit_entries]
        audit_trail = {
            "complaint_id": complaint_id,
            "customer_id": customer_id,
            "audit_entries": entries_dicts,
            "summary": {
                "total_steps": len(audit_entries),
                "steps_completed": list(steps_present),
                "missing_steps": missing_steps,
                "all_steps_completed": len(missing_steps) == 0,
                "start_timestamp": min(timestamps) if timestamps else None,
                "end_timestamp": max(timestamps) if timestamps else None,
            },
            "exceptions": exceptions,
        }

        # ── Workflow compliance ────────────────────────────
        compliance = _check_workflow_compliance(audit_trail)
        audit_trail["compliance_status"] = compliance["status"]
        audit_trail["compliance_checks"] = compliance

        # ── LLM enhancement (try Autogen 0.4.x) ───────────
        llm_analysis = await self._try_llm_analysis(audit_trail)
        mode = "llm_enhanced" if llm_analysis else "rule_based"
        if llm_analysis:
            audit_trail["llm_analysis"] = llm_analysis

        # ── Save outputs ───────────────────────────────────
        files_saved = self._save_outputs(complaint_id, audit_trail)

        # ── Build report metrics ───────────────────────────
        total = len(audit_entries)
        valid_count = sum(1 for e in audit_entries if e.validation_status == "pass")
        compliant_count = sum(1 for e in audit_entries if not e.compliance_flags)

        result = {
            "complaint_id": complaint_id,
            "customer_id": customer_id,
            "agent_id": self.AGENT_ID,
            "mode": mode,
            "audit_summary": {
                "total_entries": total,
                "valid_entries": valid_count,
                "compliant_entries": compliant_count,
                "validation_rate": round(valid_count / total, 2) if total else 0,
                "compliance_rate": round(compliant_count / total, 2) if total else 0,
                "steps_completed": list(steps_present),
                "missing_steps": missing_steps,
                "all_steps_completed": len(missing_steps) == 0,
            },
            "audit_trail_entries": [
                {
                    "workflow_step": e.workflow_step,
                    "agent_id": e.agent_id,
                    "timestamp": e.timestamp,
                    "status": e.status,
                    "validation_status": e.validation_status,
                    "compliance_flags": e.compliance_flags or [],
                }
                for e in audit_entries
            ],
            "compliance": compliance,
            "exceptions": exceptions,
            "files_saved": files_saved,
            "validation_status": compliance["status"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        print(f"  Mode        →  {mode}")
        print(f"  Entries     →  {total} ({valid_count} valid, {compliant_count} compliant)")
        print(f"  Compliance  →  {compliance['status']} ({compliance['checks_passed']}/{compliance['checks_total']} checks)")
        print(f"  Exceptions  →  {len(exceptions)}")
        if files_saved:
            print(f"  Saved       →  {', '.join(files_saved.values())}")

        return result

    # ── LLM path (Autogen 0.4.x) ──────────────────────────

    async def _try_llm_analysis(self, audit_trail: Dict[str, Any]) -> Optional[str]:
        try:
            from autogen_agentchat.agents import AssistantAgent
            from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

            client = AzureOpenAIChatCompletionClient(
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            )

            agent = AssistantAgent(
                name="AuditAnalyst",
                model_client=client,
                system_message=(
                    "You are an Audit & Compliance Analyst for a Service Issue Resolution pipeline. "
                    "Analyze the provided audit trail and produce a brief compliance assessment covering "
                    "GDPR, SOX, workflow completeness, and any anomalies. Be concise (3-5 sentences)."
                ),
            )

            print("  Invoking Autogen 0.4.x AuditAnalyst for LLM compliance review...")

            summary_for_llm = json.dumps({
                "complaint_id": audit_trail.get("complaint_id"),
                "summary": audit_trail.get("summary"),
                "compliance_status": audit_trail.get("compliance_status"),
                "compliance_checks": audit_trail.get("compliance_checks"),
                "exceptions": audit_trail.get("exceptions"),
            }, indent=2)

            from autogen_agentchat.messages import TextMessage
            from autogen_core import CancellationToken

            response = await agent.on_messages(
                [TextMessage(content=f"Analyze this audit trail:\n{summary_for_llm}", source="user")],
                cancellation_token=CancellationToken(),
            )
            await client.close()

            analysis = response.chat_message.content
            print(f"  LLM analysis →  received ({len(analysis)} chars)")
            return analysis

        except Exception as exc:
            print(f"  LLM analysis failed ({exc}), falling back to rule-based")
            return None

    # ── File outputs ───────────────────────────────────────

    def _save_outputs(self, complaint_id: str, audit_trail: Dict[str, Any]) -> Dict[str, str]:
        try:
            cid = complaint_id or "UNKNOWN"

            # JSON
            json_path = OUTPUT_DIR / f"{cid}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(audit_trail, f, indent=2)

            # CSV index
            csv_path = OUTPUT_DIR / f"{cid}_index.csv"
            entries = audit_trail.get("audit_entries", [])
            if entries:
                fieldnames = ["workflow_step", "agent_id", "timestamp", "status",
                              "complaint_id", "customer_id", "validation_status"]
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(entries)

            return {"json_file": str(json_path), "csv_file": str(csv_path)}
        except Exception as exc:
            logger.error("Error saving audit outputs: %s", exc)
            return {}

    # ── helpers ────────────────────────────────────────────

    @staticmethod
    def _find_complaint_id(results: dict) -> Optional[str]:
        for key in ("agent1_classifier", "agent2_evidence", "agent3_remedy", "agent4_credit", "agent5_communicator"):
            block = results.get(key, {})
            if isinstance(block, dict):
                cid = block.get("complaint_id") or block.get("complaint_category", {}).get("complaint_id")
                if cid:
                    return cid
        return None

    @staticmethod
    def _find_customer_id(results: dict) -> Optional[str]:
        for key in ("agent2_evidence", "agent1_classifier", "agent5_communicator"):
            block = results.get(key, {})
            if isinstance(block, dict):
                cid = block.get("customer_id")
                if cid:
                    return cid
        return None
