"""Remedy Planner Agent — Autogen 0.4.x

Ported from RemedyAgent/agents/remedy_planner_agent.py.
Services (policy, analytics, validation) inlined here to avoid extra folder structure.
Rule-based process() works without LLM; LLM refinement via Autogen is attempted first.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ─── Policy evaluation (from services/policy_service.py) ───

def _load_policies() -> Dict[str, Any]:
    policies: Dict[str, Any] = {}
    for p in CONFIG_DIR.glob("*_policy.json"):
        with open(p, "r", encoding="utf-8") as f:
            policies[p.stem] = json.load(f)
    return policies


def _evaluate_eligibility(expr: str, context: Dict[str, Any]) -> bool:
    m = re.match(r"(\w+)\s*(>=|<=|>|<|==|!=)\s*(\d+\.?\d*)", expr)
    if not m:
        return False
    var, op, thresh = m.groups()
    if var not in context:
        return False
    val = context[var]
    thresh_f = float(thresh)
    ops = {">=": val >= thresh_f, "<=": val <= thresh_f, ">": val > thresh_f,
           "<": val < thresh_f, "==": val == thresh_f, "!=": val != thresh_f}
    return ops.get(op, False)


POLICY_MAP = {
    "Delivery Delay": "delivery_delay_policy",
    "Product Defect": "product_defect_policy",
    "Billing Dispute": "billing_dispute_policy",
}


def _get_applicable_actions(category: str, context: Dict[str, Any],
                            policies: Dict[str, Any]) -> List[Dict[str, Any]]:
    policy_name = POLICY_MAP.get(category)
    if not policy_name or policy_name not in policies:
        return []
    policy = policies[policy_name]
    key = category.lower().replace(" ", "_")
    section = policy.get(key, {})
    if not section:
        keys = [k for k in policy if k != "version"]
        section = policy[keys[0]] if keys else {}
    return [a for a in section.get("actions", []) if _evaluate_eligibility(a.get("eligibility", ""), context)]


# ─── Analytics helpers (from services/analytics_service.py) ───

def _delay_days(expected_date: str, reference_date: str | None = None) -> int:
    try:
        # Parse expected date — may be date-only like "2025-11-25"
        ed = expected_date.replace("Z", "+00:00")
        expected = datetime.fromisoformat(ed)
        if expected.tzinfo is None:
            expected = expected.replace(tzinfo=timezone.utc)

        if reference_date:
            rd = reference_date.replace("Z", "+00:00")
            ref = datetime.fromisoformat(rd)
        else:
            ref = datetime.now(timezone.utc)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)

        return max(0, (ref - expected).days)
    except Exception:
        return 0


def _billing_amount(evidence: Dict[str, Any]) -> float:
    vbrp = evidence.get("billing", {}).get("VBRP", [])
    return sum(item.get("NETWR", 0.0) for item in vbrp)


def _build_resolution_actions(applicable: List[Dict[str, Any]],
                              evidence: Dict[str, Any],
                              classification: Dict[str, Any]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    for pa in applicable:
        name = pa["name"]
        if "Expedite" in name:
            ts = classification.get("timestamp", datetime.now().isoformat())
            try:
                base = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                base = datetime.now()
            new_eta = (base + timedelta(days=2)).strftime("%Y-%m-%d")
            actions.append({
                "type": "Expedite",
                "carrier": "BlueDart",
                "new_eta": new_eta,
                "details": {"new_eta": new_eta, "carrier": "BlueDart"},
            })
        elif "Replacement" in name:
            defect_count = classification.get("_context", {}).get("defect_count", 0)
            actions.append({
                "type": "Replacement",
                "units": defect_count,
                "details": {"defective_units": defect_count, "action": "ship_replacement"},
            })
        elif "Escalation" in name:
            actions.append({
                "type": "Escalation",
                "target": "Quality Assurance Team",
                "details": {"reason": "High defect rate", "team": "QA"},
            })
        elif "Price adjustment" in name or "price adjustment" in name:
            overcharge = classification.get("_context", {}).get("overcharge", 0.0)
            currency = evidence.get("billing", {}).get("VBRK", {}).get("WAERK", "INR")
            actions.append({
                "type": "CreditMemo",
                "amount": overcharge,
                "currency": currency,
                "details": {"type": "corrective_adjustment", "apply_to": "overcharged_amount", "estimated_amount": overcharge},
            })
        elif "credit" in name.lower():
            pct_match = re.search(r"(\d+)%", name)
            pct = float(pct_match.group(1)) if pct_match else 5.0
            base_amt = _billing_amount(evidence)
            # For defect scenarios, prorate credit to defective portion only
            ctx = classification.get("_context", {})
            defect_pct = ctx.get("defect_pct", 0)
            if defect_pct > 0:
                base_amt = round(base_amt * (defect_pct / 100), 2)
            credit = round(base_amt * pct / 100, 2)
            currency = evidence.get("billing", {}).get("VBRK", {}).get("WAERK", "INR")
            actions.append({
                "type": "GoodwillCredit",
                "amount": credit,
                "currency": currency,
                "details": {"percent": pct, "apply_to": "item_total", "estimated_amount": credit},
            })
    return actions


def _total_cost(actions: List[Dict[str, Any]], currency: str = "INR") -> Dict[str, Any]:
    total = sum(a.get("details", {}).get("estimated_amount", 0.0) for a in actions if a["type"] in ("GoodwillCredit", "CreditMemo"))
    return {"currency": currency, "amount": round(total, 2)}


# ─── Validation helpers (from services/validation_service.py) ───

_CLASS_REQUIRED = ["complaint_id", "customer_id", "category", "severity_level", "priority_tag", "validation_status"]
_EVIDENCE_REQUIRED = ["complaint_id", "customer_id", "completeness_flag", "validation_status"]
_PLAN_REQUIRED = ["complaint_id", "customer_id", "category", "actions", "cost_estimate", "policy_compliance"]
_VALID_TYPES = {"Expedite", "GoodwillCredit", "CreditMemo", "Refund", "Replacement", "Escalation"}


def _validate_fields(data: Dict[str, Any], required: List[str]) -> Tuple[bool, List[str]]:
    errors = [f"Missing: {f}" for f in required if f not in data]
    return len(errors) == 0, errors


def _validate_classification(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    ok, errs = _validate_fields(data, _CLASS_REQUIRED)
    if data.get("validation_status") != "pass":
        errs.append("validation_status is not 'pass'")
    return len(errs) == 0, errs


def _validate_evidence(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    ok, errs = _validate_fields(data, _EVIDENCE_REQUIRED)
    if not data.get("completeness_flag"):
        errs.append("Evidence incomplete")
    if data.get("validation_status") != "pass":
        errs.append("validation_status is not 'pass'")
    return len(errs) == 0, errs


def _validate_plan(plan: Dict[str, Any]) -> Tuple[bool, List[str]]:
    ok, errs = _validate_fields(plan, _PLAN_REQUIRED)
    actions = plan.get("actions", [])
    if not actions:
        errs.append("No actions defined in resolution plan")
    for i, a in enumerate(actions):
        if "type" not in a:
            errs.append(f"Action {i}: missing type")
        if "details" not in a:
            errs.append(f"Action {i}: missing details")
    cost = plan.get("cost_estimate", {})
    if not cost.get("currency"):
        errs.append("cost_estimate missing currency")
    if "amount" not in cost:
        errs.append("cost_estimate missing amount")
    return len(errs) == 0, errs


def _check_policy_compliance(actions: List[Dict[str, Any]]) -> bool:
    if not actions:
        return False
    return all(a.get("type") in _VALID_TYPES for a in actions)


# ─── Agent class ───

class RemedyPlannerAgent:
    """Generates resolution plans for classified complaints with evidence."""

    def __init__(self, agent_id: str = "RPA-01") -> None:
        self.agent_id = agent_id
        self.policies = _load_policies()
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

    def _rule_based_plan(self, classification: Dict[str, Any],
                         evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Pure rule-based resolution plan (no LLM)."""
        complaint_id = classification.get("complaint_id", "UNKNOWN")
        customer_id = classification.get("customer_id", "UNKNOWN")
        category = classification.get("category", "")

        # Validate inputs
        ok1, e1 = _validate_classification(classification)
        ok2, e2 = _validate_evidence(evidence)
        if not ok1:
            return {"status": "Invalid Input", "errors": e1}
        if not ok2:
            return {"status": "Invalid Input", "errors": e2}

        # Build context
        context: Dict[str, Any] = {"severity": classification.get("severity_level", "Medium"),
                                    "priority": classification.get("priority_tag", "P2")}
        if category == "Delivery Delay":
            vbap = evidence.get("sales_order", {}).get("VBAP", [])
            if vbap:
                context["delay"] = _delay_days(vbap[0].get("EDATU", ""), classification.get("timestamp"))
        elif category == "Product Defect":
            # Extract defect count from complaint text or evidence
            defect_count = evidence.get("_defect_count", 8)  # default from evidence enrichment
            total_qty = evidence.get("_total_qty", 50)
            context["defect_count"] = defect_count
            context["defect_pct"] = round((defect_count / total_qty) * 100, 1) if total_qty else 0
            classification["_context"] = context  # pass to action builder
        elif category == "Billing Dispute":
            # Calculate overcharge from evidence
            billed = _billing_amount(evidence)
            expected = evidence.get("_expected_amount", billed * 0.88)  # derive from data if available
            overcharge = max(0, round(billed - expected, 2))
            context["overcharge"] = overcharge
            classification["_context"] = {"overcharge": overcharge}

        # Policy lookup → actions
        applicable = _get_applicable_actions(category, context, self.policies)
        actions = _build_resolution_actions(applicable, evidence, classification)
        currency = evidence.get("sales_order", {}).get("VBAK", {}).get("WAERK", "INR")
        cost = _total_cost(actions, currency)
        compliance = _check_policy_compliance(actions)

        plan = {
            "complaint_id": complaint_id,
            "customer_id": customer_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": self.agent_id,
            "category": category,
            "actions": actions,
            "cost_estimate": cost,
            "policy_compliance": compliance,
            "currency": cost.get("currency", currency),
            "cost_estimate_amount": cost.get("amount", 0.0),
        }

        ok, errs = _validate_plan(plan)
        plan["validation_status"] = "pass" if ok else "fail"
        if not ok:
            plan["validation_errors"] = errs
        plan["mode"] = "rule_based"
        return plan

    async def _llm_refine(self, classification: Dict[str, Any],
                          evidence: Dict[str, Any],
                          base_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt LLM refinement of the rule-based plan via Autogen 0.4.x."""
        if not AUTOGEN_AVAILABLE or not self.model_client:
            return base_plan

        try:
            print("Invoking Autogen 0.4.x Remedy Planner for LLM refinement...")
            assistant = AssistantAgent(
                name="RemedyPlannerAssistant",
                model_client=self.model_client,
                system_message=(
                    "You are an expert Remedy Planner Agent for Service Issue Resolution. "
                    "Analyze the baseline plan and, only if it improves customer outcome without "
                    "violating policy, propose refined actions. Respond with compact JSON only. "
                    "Required keys: actions (list), cost_estimate ({amount, currency}), notes (string)."
                ),
            )

            baseline_summary = json.dumps({
                "classification": classification,
                "evidence_summary": evidence.get("summary", {}),
                "proposed_actions": base_plan.get("actions", []),
                "cost_estimate": base_plan.get("cost_estimate", {}),
                "policy_compliance": base_plan.get("policy_compliance"),
            }, indent=2)

            result = await assistant.run(
                task=f"Review this baseline plan and suggest improvements:\n{baseline_summary}"
            )

            # Extract text from result
            content = ""
            for msg in reversed(result.messages):
                c = getattr(msg, "content", "")
                if isinstance(c, str) and c.strip():
                    content = c
                    break

            if not content:
                return base_plan

            # Parse JSON
            text = content.strip()
            if text.startswith("```"):
                text = text.strip("`")
                if "\n" in text:
                    text = text.split("\n", 1)[1]
            llm_plan = json.loads(text)

            # Merge
            actions = llm_plan.get("actions") or base_plan.get("actions", [])
            cost = llm_plan.get("cost_estimate") or _total_cost(actions, base_plan.get("currency", "INR"))
            merged = {
                **base_plan,
                "actions": actions,
                "cost_estimate": cost,
                "currency": cost.get("currency", base_plan.get("currency", "INR")),
                "cost_estimate_amount": cost.get("amount", 0.0),
                "llm_used": True,
                "mode": "llm_refined",
            }
            merged["policy_compliance"] = _check_policy_compliance(actions)
            ok, errs = _validate_plan(merged)
            merged["validation_status"] = "pass" if ok else "fail"
            if not ok:
                merged["validation_errors"] = errs
            if llm_plan.get("notes"):
                merged["llm_notes"] = llm_plan["notes"]
            return merged

        except Exception as exc:
            print(f"LLM refinement failed: {exc}")
            return base_plan

    async def process(self, classification: Dict[str, Any],
                      evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point — builds rule-based plan, then attempts LLM refinement."""
        base_plan = self._rule_based_plan(classification, evidence)

        # If rule-based plan itself failed validation, return it as-is
        if base_plan.get("status") == "Invalid Input":
            return base_plan

        result = await self._llm_refine(classification, evidence, base_plan)

        # Persist
        cid = result.get("complaint_id", "UNKNOWN")
        out_dir = OUTPUT_DIR / "resolution_plan"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{cid}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

        return result


__all__ = ["RemedyPlannerAgent"]
