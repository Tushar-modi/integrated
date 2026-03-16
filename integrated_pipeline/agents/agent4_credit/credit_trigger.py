"""Credit Trigger Agent — Autogen 0.4.x

Ported from credit-trigger-agent/credit_trigger_agent.py.
Receives resolution plans from Agent 3, validates, checks customer eligibility,
approves/rejects credits, and generates credit confirmation for SAP FI posting.

Business Rules:
- Max credit per transaction: 50,000 INR
- Customer account must be active (KNA1 present, not blocked)
- Plan must be policy-compliant and validation_status == "pass"
- Supports GoodwillCredit (discretionary) and CreditMemo (corrective) actions
- Returns not_applicable when resolution plan has no financial credit actions
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MAX_CREDIT_AMOUNT = 50_000.0
CREDIT_ACTION_TYPES = {"GoodwillCredit", "CreditMemo"}


# ─── Validation helpers ───

def _validate_resolution_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Validate resolution plan structure, compliance, and credit actions."""
    required = ["complaint_id", "customer_id", "actions", "policy_compliance"]
    for f in required:
        if f not in plan:
            return {"valid": False, "reason": f"Missing required field: {f}", "has_credit": False}

    if not plan.get("policy_compliance"):
        return {"valid": False, "reason": "Resolution plan not policy compliant", "has_credit": False}

    if plan.get("validation_status") != "pass":
        return {"valid": False, "reason": "Resolution plan validation failed", "has_credit": False}

    # Check for any financial credit actions
    credit_actions = [a for a in plan.get("actions", []) if a.get("type") in CREDIT_ACTION_TYPES]
    if not credit_actions:
        # Valid plan but no credit needed (e.g., replacement-only)
        return {"valid": True, "reason": "", "has_credit": False}

    # Validate each credit action has amount details
    for action in credit_actions:
        details = action.get("details", {})
        if "estimated_amount" not in details:
            return {"valid": False, "reason": f"{action['type']} action missing amount details", "has_credit": True}

    return {"valid": True, "reason": "", "has_credit": True}


def _is_credit_eligible(customer_account: Dict[str, Any]) -> bool:
    """Check if customer qualifies for credit (account active, not blocked)."""
    if "KNA1" not in customer_account:
        return False
    if not customer_account["KNA1"].get("KUNNR"):
        return False
    # CASSD flag = customer blocked for sales
    if "KNVV" in customer_account:
        if customer_account["KNVV"].get("CASSD"):
            return False
    return True


def _extract_credit_amount(plan: Dict[str, Any]) -> float:
    """Sum all credit actions (GoodwillCredit + CreditMemo), capped at MAX_CREDIT_AMOUNT."""
    total = 0.0
    for action in plan.get("actions", []):
        if action.get("type") in CREDIT_ACTION_TYPES:
            total += float(action.get("details", {}).get("estimated_amount", 0.0))
    return min(total, MAX_CREDIT_AMOUNT)


def _build_confirmation(plan: Dict[str, Any],
                        customer: Dict[str, Any],
                        credit_amount: float) -> Dict[str, Any]:
    """Generate approved credit confirmation for SAP FI posting."""
    ts = datetime.now(timezone.utc)
    credit_doc = f"CR-{ts.strftime('%Y-%m-%d-%H%M%S')}"
    currency = customer.get("KNVV", {}).get("WAERS", "INR")

    # Determine credit type breakdown for SAP posting text
    credit_types = [a.get("type") for a in plan.get("actions", []) if a.get("type") in CREDIT_ACTION_TYPES]
    if "CreditMemo" in credit_types and "GoodwillCredit" in credit_types:
        sgtxt = f"Corrective credit + goodwill for {plan.get('category', 'complaint')}"
    elif "CreditMemo" in credit_types:
        sgtxt = f"Corrective credit memo for {plan.get('category', 'complaint')}"
    else:
        sgtxt = f"Goodwill credit for {plan.get('category', 'complaint')}"

    return {
        "complaint_id": plan.get("complaint_id"),
        "customer_id": customer["KNA1"]["KUNNR"],
        "approval": {
            "status": "approved",
            "amount": credit_amount,
            "currency": currency,
            "credit_doc": credit_doc,
            "credit_types": credit_types,
            "conditions": [
                "single_use",
                "visible_in_next_statement",
                "expires_in_90_days",
            ],
        },
        "customer_details": {
            "name": customer["KNA1"].get("NAME1"),
            "email": customer["KNA1"].get("SMTP_ADDR"),
        },
        "sap_fi_posting": {
            "BKPF": {
                "BELNR": credit_doc,
                "BUDAT": ts.strftime("%Y-%m-%d"),
                "WAERS": currency,
            },
            "BSEG": {
                "KUNNR": customer["KNA1"]["KUNNR"],
                "WRBTR": credit_amount,
                "WAERS": currency,
                "SGTXT": sgtxt,
            },
        },
        "audit": {
            "timestamp": ts.isoformat(),
            "agent_id": "CTA-01",
            "agent_name": "Credit Trigger Agent",
        },
        "validation_status": "pass",
    }


def _build_rejection(plan: Dict[str, Any],
                     customer: Dict[str, Any],
                     reason: str) -> Dict[str, Any]:
    """Generate rejection document."""
    return {
        "complaint_id": plan.get("complaint_id"),
        "customer_id": customer.get("KNA1", {}).get("KUNNR"),
        "approval": {
            "status": "rejected",
            "reason": reason,
        },
        "audit": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": "CTA-01",
            "agent_name": "Credit Trigger Agent",
        },
        "validation_status": "fail",
    }


def _build_not_applicable(plan: Dict[str, Any],
                          customer: Dict[str, Any]) -> Dict[str, Any]:
    """No financial credit required — resolution is non-monetary (e.g., replacement only)."""
    action_types = [a.get("type", "?") for a in plan.get("actions", [])]
    return {
        "complaint_id": plan.get("complaint_id"),
        "customer_id": customer.get("KNA1", {}).get("KUNNR"),
        "approval": {
            "status": "not_applicable",
            "reason": f"No financial credit required — resolution via {', '.join(action_types)}",
            "amount": 0,
            "currency": customer.get("KNVV", {}).get("WAERS", "INR"),
        },
        "audit": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": "CTA-01",
            "agent_name": "Credit Trigger Agent",
        },
        "validation_status": "pass",
    }


# ─── Agent class ───

class CreditTriggerAgent:
    """Processes resolution plans → approve/reject goodwill credits."""

    def __init__(self, agent_id: str = "CTA-01") -> None:
        self.agent_id = agent_id
        self.audit_log: List[Dict[str, Any]] = []
        self.model_client = self._build_model_client() if AUTOGEN_AVAILABLE else None

    def _build_model_client(self):
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_BASE")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        if not all([endpoint, api_key, deployment, api_version]):
            return None
        return AzureOpenAIChatCompletionClient(
            azure_deployment=deployment,
            model=deployment,
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )

    def _log(self, message: str) -> None:
        self.audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": self.agent_id,
            "message": message,
        })

    def _rule_based_process(self, resolution_plan: Dict[str, Any],
                            customer_account: Dict[str, Any]) -> Dict[str, Any]:
        """Pure rule-based credit approval/rejection."""
        # Step 1: Validate plan
        val = _validate_resolution_plan(resolution_plan)
        if not val["valid"]:
            self._log(f"Rejected: {val['reason']}")
            return _build_rejection(resolution_plan, customer_account, val["reason"])

        # Step 1b: If valid but no credit actions, return not_applicable
        if not val.get("has_credit"):
            self._log("No financial credit required — non-monetary resolution")
            result = _build_not_applicable(resolution_plan, customer_account)
            result["mode"] = "rule_based"
            return result

        # Step 2: Check eligibility
        if not _is_credit_eligible(customer_account):
            reason = "Customer not eligible for credit"
            self._log(f"Rejected: {reason}")
            return _build_rejection(resolution_plan, customer_account, reason)

        # Step 3: Extract credit amount (sums all credit actions)
        credit_amount = _extract_credit_amount(resolution_plan)
        if credit_amount <= 0:
            reason = "No valid credit amount found"
            self._log(f"Rejected: {reason}")
            return _build_rejection(resolution_plan, customer_account, reason)

        # Step 4: Generate confirmation
        confirmation = _build_confirmation(resolution_plan, customer_account, credit_amount)
        self._log(
            f"Approved credit of {credit_amount} {confirmation['approval']['currency']} "
            f"for customer {customer_account['KNA1']['KUNNR']}"
        )
        confirmation["mode"] = "rule_based"
        return confirmation

    async def _llm_enrich(self, resolution_plan: Dict[str, Any],
                          customer_account: Dict[str, Any],
                          base_result: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt LLM enrichment via Autogen 0.4.x — adds explanation & reasoning."""
        if not AUTOGEN_AVAILABLE or not self.model_client:
            return base_result

        try:
            print("Invoking Autogen 0.4.x Credit Trigger for LLM enrichment...")
            assistant = AssistantAgent(
                name="CreditTriggerAssistant",
                model_client=self.model_client,
                system_message=(
                    "You are a Credit Trigger Agent in a Service Issue Resolution workflow. "
                    "Analyze the credit decision and provide a concise explanation. "
                    "Respond with compact JSON: {\"explanation\": \"...\", \"risk_notes\": \"...\"}"
                ),
            )

            summary = json.dumps({
                "decision": base_result.get("approval", {}),
                "complaint_id": base_result.get("complaint_id"),
                "customer": customer_account.get("KNA1", {}).get("NAME1"),
                "category": resolution_plan.get("category"),
            }, indent=2)

            result = await assistant.run(
                task=f"Explain this credit decision:\n{summary}"
            )

            content = ""
            for msg in reversed(result.messages):
                c = getattr(msg, "content", "")
                if isinstance(c, str) and c.strip():
                    content = c
                    break

            if content:
                text = content.strip()
                if text.startswith("```"):
                    text = text.strip("`")
                    if "\n" in text:
                        text = text.split("\n", 1)[1]
                llm_data = json.loads(text)
                base_result["llm_explanation"] = llm_data.get("explanation", "")
                base_result["risk_notes"] = llm_data.get("risk_notes", "")
                base_result["mode"] = "llm_enriched"

        except Exception as exc:
            print(f"LLM enrichment failed: {exc}")

        return base_result

    async def process(self, resolution_plan: Dict[str, Any],
                      customer_account: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry — rule-based decision, then optional LLM enrichment."""
        base_result = self._rule_based_process(resolution_plan, customer_account)

        # Only try LLM enrichment for approved credits
        if base_result.get("approval", {}).get("status") == "approved":
            base_result = await self._llm_enrich(resolution_plan, customer_account, base_result)

        # Persist
        cid = base_result.get("complaint_id", "UNKNOWN")
        out_dir = OUTPUT_DIR / "credit_confirmation"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{cid}.json").write_text(
            json.dumps(base_result, indent=2), encoding="utf-8"
        )

        return base_result


__all__ = ["CreditTriggerAgent"]
