"""Customer Communicator Agent — Autogen 0.4.x

Ported from customer_communication_agent/Customer_Communicator_Agent/customer_communicator_agent.py.
Receives resolution plan (Agent 3), credit confirmation (Agent 4), and customer profile,
then generates a personalized, compliance-checked resolution message.

Features:
- Template-based message rendering (rule-based fallback)
- GDPR + brand compliance validation
- Dispatch channel auto-detection (email / sms / portal)
- Autogen 0.4.x LLM refinement when available
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import TextMessage
    from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
    from autogen_core import CancellationToken
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CONFIG_DIR = PROJECT_ROOT / "config"


# ─── Template helpers ──────────────────────────────────────

CATEGORY_TEMPLATE_MAP = {
    "Delivery Delay": "delay_resolution",
    "Quality Issue": "quality_resolution",
    "Billing Error": "billing_resolution",
    "Service Issue": "service_resolution",
}


def _load_templates() -> Dict[str, str]:
    """Load communication templates from config/communication_templates.json."""
    path = CONFIG_DIR / "communication_templates.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "delay_resolution": "Hello {{name}}, we apologise for the delay. A credit of {{amount}} {{currency}} has been approved.",
        "signature": "Regards,\nCustomer Care",
    }


def _fill_template(template: str, context: Dict[str, Any]) -> str:
    """Replace {{key}} placeholders in *template* with values from *context*."""
    result = template
    for key, value in context.items():
        result = result.replace(f"{{{{{key}}}}}", str(value))
    return result


# ─── Recipient / context extraction ───────────────────────

def _extract_recipient(customer_profile: Dict[str, Any]) -> Dict[str, str]:
    kna1 = customer_profile.get("KNA1", {})
    full_name = kna1.get("NAME1", "Valued Customer")
    short_name = full_name.split()[0] if full_name else "Customer"
    return {
        "name": short_name,
        "full_name": full_name,
        "email": kna1.get("SMTP_ADDR", ""),
        "phone": kna1.get("TELF1", ""),
    }


def _build_context(
    resolution_plan: Dict[str, Any],
    credit_confirmation: Dict[str, Any],
    customer_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge data from upstream agents into a flat rendering context."""
    recipient = _extract_recipient(customer_profile)
    actions = resolution_plan.get("actions", [])
    expedite = next((a for a in actions if a.get("type") == "Expedite"), None)
    credit_info = credit_confirmation.get("approval", {})

    return {
        "name": recipient["name"],
        "full_name": recipient["full_name"],
        "complaint_id": resolution_plan.get("complaint_id", ""),
        "category": resolution_plan.get("category", ""),
        "eta": expedite["details"]["new_eta"] if expedite else "",
        "carrier": expedite["details"].get("carrier", "") if expedite else "",
        "amount": credit_info.get("amount", 0),
        "currency": customer_profile.get("KNVV", {}).get("WAERS", "INR"),
        "credit_doc": credit_info.get("credit_doc", ""),
        "conditions": credit_info.get("conditions", []),
    }


def _dispatch_channel(customer_profile: Dict[str, Any]) -> str:
    kna1 = customer_profile.get("KNA1", {})
    if kna1.get("SMTP_ADDR"):
        return "email"
    if kna1.get("TELF1"):
        return "sms"
    return "portal"


# ─── Compliance checks ────────────────────────────────────

SENSITIVE_PATTERNS = ["password", "card number", "ssn", "date of birth"]
EMPATHY_WORDS = ["apologise", "apologize", "sorry", "understand", "help", "assist"]
DEMAND_WORDS = ["demand", "must comply", "you are required"]


def _check_gdpr(message: str) -> bool:
    lower = message.lower()
    return not any(p in lower for p in SENSITIVE_PATTERNS)


def _check_brand(message: str) -> bool:
    lower = message.lower()
    has_empathy = any(w in lower for w in EMPATHY_WORDS)
    has_demand = any(w in lower for w in DEMAND_WORDS)
    return has_empathy and not has_demand


def _validate_compliance(message: str) -> Dict[str, Any]:
    gdpr = _check_gdpr(message)
    brand = _check_brand(message)
    return {
        "gdpr": gdpr,
        "brand": brand,
        "compliance_status": "pass" if (gdpr and brand) else "fail",
    }


# ═══════════════════════════════════════════════════════════
# Main Agent class
# ═══════════════════════════════════════════════════════════

class CustomerCommunicatorAgent:
    AGENT_ID = "COM-01"

    def __init__(self):
        self.templates = _load_templates()
        self.model_client = self._build_model_client() if AUTOGEN_AVAILABLE else None

    # ── Azure OpenAI client (same pattern as agents 1-4) ──

    def _build_model_client(self):
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        if not all([endpoint, api_key, deployment, api_version]):
            return None
        return AzureOpenAIChatCompletionClient(
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            model=deployment,
            api_version=api_version,
            api_key=api_key,
        )

    # ── Public entry point ─────────────────────────────────

    async def process(
        self,
        resolution_plan: Dict[str, Any],
        credit_confirmation: Dict[str, Any],
        customer_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate, validate, and return a resolution message."""
        print(f"\n{'─'*50}")
        print("Customer Communicator Agent  (COM-01)")
        print(f"{'─'*50}")

        context = _build_context(resolution_plan, credit_confirmation, customer_profile)
        recipient = _extract_recipient(customer_profile)

        # 1. Generate message body (LLM → fallback template)
        body = await self._generate_message(context)

        # 2. Compliance check
        compliance = _validate_compliance(body)
        print(f"  Compliance  →  GDPR={compliance['gdpr']}  Brand={compliance['brand']}  Status={compliance['compliance_status']}")

        # 3. Dispatch channel
        channel = _dispatch_channel(customer_profile)
        print(f"  Channel     →  {channel}")

        now = datetime.now(timezone.utc).isoformat()

        output = {
            "complaint_id": resolution_plan.get("complaint_id"),
            "customer_id": resolution_plan.get("customer_id"),
            "to": {
                "name": recipient["full_name"],
                "email": recipient["email"],
                "phone": recipient["phone"],
            },
            "body": body,
            "dispatch_channel": channel,
            "tone": "empathetic",
            "compliance": {
                "gdpr": compliance["gdpr"],
                "brand": compliance["brand"],
            },
            "validation_status": compliance["compliance_status"],
            "timestamp": now,
            "agent_id": self.AGENT_ID,
            "mode": "rule_based",
        }

        # Save output
        self._save(output)
        return output

    # ── Message generation ─────────────────────────────────

    async def _generate_message(self, context: Dict[str, Any]) -> str:
        """Try LLM generation first, fall back to template rendering."""
        # Attempt LLM
        llm_body = await self._llm_generate(context)
        if llm_body:
            return llm_body

        # Fallback: template
        print("  Mode        →  rule_based (template)")
        return self._render_template(context)

    def _render_template(self, context: Dict[str, Any]) -> str:
        """Select template by category, fill placeholders, append signature."""
        key = CATEGORY_TEMPLATE_MAP.get(context.get("category", ""), "delay_resolution")
        template = self.templates.get(key, self.templates.get("delay_resolution", ""))
        body = _fill_template(template, context)
        signature = self.templates.get("signature", "Regards,\nCustomer Care")
        return f"{body}\n\n{signature}"

    async def _llm_generate(self, context: Dict[str, Any]) -> str | None:
        """Attempt Autogen 0.4.x AssistantAgent message generation."""
        if not AUTOGEN_AVAILABLE or not self.model_client:
            return None
        try:
            print("  Invoking Autogen 0.4.x MessageGenerator for LLM drafting...")
            agent = AssistantAgent(
                name="MessageGenerator",
                model_client=self.model_client,
                system_message=(
                    "You are an expert customer communication specialist. "
                    "Generate a personalized, empathetic resolution message. "
                    "Output ONLY the message body — no JSON, no markdown fences."
                ),
            )
            prompt = (
                f"Customer: {context.get('name')}\n"
                f"Complaint: {context.get('complaint_id')} — {context.get('category')}\n"
                f"New ETA: {context.get('eta')}\n"
                f"Credit: {context.get('amount')} {context.get('currency')}\n"
                f"Carrier: {context.get('carrier')}\n"
                "Write a concise, empathetic resolution email (under 200 words)."
            )
            response = await agent.on_messages(
                [TextMessage(content=prompt, source="user")],
                CancellationToken(),
            )
            text = response.chat_message.content.strip()
            if text:
                print("  Mode        →  llm (Autogen 0.4.x)")
                return text
        except Exception as e:
            print(f"  LLM generation failed ({e}), falling back to template")
        return None

    # ── Persistence ────────────────────────────────────────

    def _save(self, output: Dict[str, Any]) -> None:
        out_dir = OUTPUT_DIR / "customer_message"
        out_dir.mkdir(parents=True, exist_ok=True)
        cid = output.get("complaint_id", "unknown")
        path = out_dir / f"{cid}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved       →  {path.relative_to(PROJECT_ROOT)}")
