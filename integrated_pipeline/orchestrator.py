"""
Pipeline Orchestrator
=====================
Chains all 5 agents sequentially: Classify → Collect → Plan → Credit → Communicate.
Each agent's output feeds into the next agent's input.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from agents.agent1_classifier.classifier import ComplaintClassifierAgent
from agents.agent2_evidence.evidence_collector import EvidenceCollectorAgent
from agents.agent3_remedy.remedy_planner import RemedyPlannerAgent
from agents.agent4_credit.credit_trigger import CreditTriggerAgent
from agents.agent5_communicator.communicator import CustomerCommunicatorAgent
from agents.agent6_audit.audit_logger import AuditLoggerAgent

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent


class PipelineOrchestrator:
    """Runs the full 6-agent Service Issue Resolution pipeline end-to-end."""

    def __init__(self):
        self.agent1 = ComplaintClassifierAgent()
        self.agent2 = EvidenceCollectorAgent()
        self.agent3 = RemedyPlannerAgent()
        self.agent4 = CreditTriggerAgent()
        self.agent5 = CustomerCommunicatorAgent()
        self.agent6 = AuditLoggerAgent()

    async def run(
        self,
        complaint_path: Path,
        customer_profile_path: Path,
        interaction_history_path: Path,
    ) -> dict:
        """Execute the full pipeline on a single complaint.

        Returns a dict with per-agent results and overall summary.
        """
        started = datetime.now(timezone.utc)
        results = {}
        failed_at = None

        # Load customer profile once (needed by agents 2, 4, 5)
        with open(customer_profile_path, "r", encoding="utf-8") as f:
            customer_profile = json.load(f)

        # ── Agent 1: Classify ──────────────────────────────────
        try:
            logger.info("Pipeline: running Agent 1 — Complaint Classifier")
            a1 = await self.agent1.process_case(
                complaint_path=complaint_path,
                customer_profile_path=customer_profile_path,
                interaction_history_path=interaction_history_path,
            )
            results["agent1_classifier"] = a1
            classification = a1.get("complaint_category", a1)
        except Exception as exc:
            logger.error("Agent 1 failed: %s", exc)
            results["agent1_classifier"] = {"error": str(exc)}
            failed_at = "agent1_classifier"
            return self._summary(results, started, failed_at)

        # ── Agent 2: Evidence ──────────────────────────────────
        try:
            logger.info("Pipeline: running Agent 2 — Evidence Collector")
            a2 = await self.agent2.process(classification, customer_profile)
            results["agent2_evidence"] = a2
        except Exception as exc:
            logger.error("Agent 2 failed: %s", exc)
            results["agent2_evidence"] = {"error": str(exc)}
            failed_at = "agent2_evidence"
            return self._summary(results, started, failed_at)

        # ── Agent 3: Remedy ────────────────────────────────────
        try:
            logger.info("Pipeline: running Agent 3 — Remedy Planner")
            a3 = await self.agent3.process(classification, a2)
            results["agent3_remedy"] = a3
        except Exception as exc:
            logger.error("Agent 3 failed: %s", exc)
            results["agent3_remedy"] = {"error": str(exc)}
            failed_at = "agent3_remedy"
            return self._summary(results, started, failed_at)

        # ── Agent 4: Credit ────────────────────────────────────
        try:
            logger.info("Pipeline: running Agent 4 — Credit Trigger")
            a4 = await self.agent4.process(a3, customer_profile)
            results["agent4_credit"] = a4
        except Exception as exc:
            logger.error("Agent 4 failed: %s", exc)
            results["agent4_credit"] = {"error": str(exc)}
            failed_at = "agent4_credit"
            return self._summary(results, started, failed_at)

        # ── Agent 5: Communicate ───────────────────────────────
        try:
            logger.info("Pipeline: running Agent 5 — Customer Communicator")
            a5 = await self.agent5.process(a3, a4, customer_profile)
            results["agent5_communicator"] = a5
        except Exception as exc:
            logger.error("Agent 5 failed: %s", exc)
            results["agent5_communicator"] = {"error": str(exc)}
            failed_at = "agent5_communicator"
            return self._summary(results, started, failed_at)

        # ── Agent 6: Audit ─────────────────────────────────
        try:
            logger.info("Pipeline: running Agent 6 — Audit Logger")
            # Build intermediate summary for Agent 6
            interim = self._summary(results, started, None)
            a6 = await self.agent6.process(interim)
            results["agent6_audit"] = a6
        except Exception as exc:
            logger.error("Agent 6 failed: %s", exc)
            results["agent6_audit"] = {"error": str(exc)}
            failed_at = "agent6_audit"
            return self._summary(results, started, failed_at)

        return self._summary(results, started, failed_at)

    # ── helpers ────────────────────────────────────────────────

    def _summary(self, results: dict, started: datetime, failed_at: str | None) -> dict:
        finished = datetime.now(timezone.utc)
        elapsed = (finished - started).total_seconds()

        complaint_id = None
        for key in ("agent1_classifier", "agent2_evidence"):
            block = results.get(key, {})
            if isinstance(block, dict):
                complaint_id = (
                    block.get("complaint_id")
                    or block.get("complaint_category", {}).get("complaint_id")
                )
                if complaint_id:
                    break

        summary = {
            "pipeline_run": {
                "complaint_id": complaint_id,
                "started": started.isoformat(),
                "finished": finished.isoformat(),
                "elapsed_seconds": round(elapsed, 2),
                "status": "completed" if failed_at is None else "failed",
                "failed_at": failed_at,
                "agents_executed": [k for k in results if "error" not in results[k]],
            },
            "results": results,
        }

        # Persist full pipeline result
        out_dir = PROJECT_ROOT / "outputs" / "pipeline_run"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = complaint_id or f"run-{finished.strftime('%Y%m%d-%H%M%S')}"
        out_path = out_dir / f"{fname}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Pipeline result saved to %s", out_path)

        return summary
