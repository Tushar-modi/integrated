"""Autogen 0.4.x Complaint Classifier Agent.

Adapted from capstone/agents/complaint_classifier_agent.py.
Uses autogen-agentchat 0.4.x API (AssistantAgent + AzureOpenAIChatCompletionClient)
instead of legacy pyautogen (UserProxyAgent + initiate_chat).
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import TextMessage
    from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
    from autogen_core import CancellationToken
    AUTOGEN_AVAILABLE = True
except ImportError:
    AssistantAgent = None  # type: ignore
    AUTOGEN_AVAILABLE = False

from textblob import TextBlob

from agents.agent1_classifier.exception_manager import ExceptionManager
from agents.agent1_classifier.utils import (
    CONFIG_DIR, PROJECT_ROOT, SeverityPriority, enrich_priority,
    load_json, validation_checks,
)

CATEGORY_ALIASES = {
    "delivery issue": "Delivery Delay",
    "delivery delay": "Delivery Delay",
    "shipping delay": "Delivery Delay",
    "billing issue": "Billing Dispute",
    "billing dispute": "Billing Dispute",
}

OUTPUT_DIR = PROJECT_ROOT / "outputs"


class ComplaintClassifierAgent:
    """Coordinates LLM classification, scoring, and exception handling."""

    def __init__(self, agent_id: str = "CLA-01") -> None:
        self.agent_id = agent_id
        self.exception_mgr = ExceptionManager()
        self.routing_rules = load_json(CONFIG_DIR / "routing_rules.json")
        self.sla_map = load_json(CONFIG_DIR / "sla_policies.json")
        self.categories = load_json(CONFIG_DIR / "categories.json")
        self.model_client = self._build_model_client() if AUTOGEN_AVAILABLE else None
        self.assistant: Optional[Any] = None

    def _build_model_client(self):
        """Build Azure OpenAI model client for autogen 0.4.x."""
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
            temperature=0.1,
        )

    def _ensure_agent(self) -> None:
        """Create the AssistantAgent if not already created."""
        if not AUTOGEN_AVAILABLE or not self.model_client:
            return
        if self.assistant:
            return
        self.assistant = AssistantAgent(
            name="classifier_llm",
            model_client=self.model_client,
            system_message="You are an enterprise complaint triage specialist.",
        )

    async def _llm_classify(self, complaint_text: str, customer_profile: Dict[str, Any],
                            interaction_history: Dict[str, Any]) -> Dict[str, Any]:
        """Use Autogen 0.4.x AssistantAgent for LLM classification."""
        if not AUTOGEN_AVAILABLE or not self.model_client:
            return self._rule_based_classification(complaint_text)
        try:
            print("Invoking Autogen 0.4.x AssistantAgent for complaint classification...")
            self._ensure_agent()
            prompt = self._classification_prompt(complaint_text, customer_profile, interaction_history)

            response = await self.assistant.on_messages(
                [TextMessage(content=prompt, source="user")],
                cancellation_token=CancellationToken(),
            )

            message = response.chat_message.content
            print(f"LLM response: {message}")

            payload = json.loads(message)
            return payload
        except Exception as exc:
            self.exception_mgr.add_event("llm_failure", {"error": str(exc)})
            return self._rule_based_classification(complaint_text)

    def _classification_prompt(self, complaint_text: str, customer_profile: Dict[str, Any],
                               interaction_history: Dict[str, Any]) -> str:
        return (
            "You will receive a complaint, customer profile, and interaction history."
            " Return ONLY valid JSON with keys: category, severity_level (High/Medium/Low),"
            " priority_hint (P1/P2/P3), confidence (float 0-1), and rationale (string)."
            " Valid categories: Delivery Delay, Billing Dispute, Product Defect, Service Outage, Account Access."
            f"\n\nComplaint:\n{complaint_text}\n"
            f"\nCustomer Profile:\n{json.dumps(customer_profile)}\n"
            f"\nInteraction History:\n{json.dumps(interaction_history)}"
        )

    def _normalize_category(self, category: Optional[str]) -> str:
        if not category:
            return "General"
        key = category.strip().lower()
        return CATEGORY_ALIASES.get(key, category)

    def _rule_based_classification(self, complaint_text: str) -> Dict[str, Any]:
        """Fallback when LLM is unavailable."""
        text = complaint_text.lower()
        category = "General"
        for entry in self.categories:
            if any(keyword in text for keyword in entry["keywords"]):
                category = entry["name"]
                break
        category = self._normalize_category(category)
        severity = "Medium"
        if any(term in text for term in ["outage", "offline", "down", "escalat"]):
            severity = "High"
        elif any(term in text for term in ["delay", "late", "waiting"]):
            severity = "Medium"
        elif any(term in text for term in ["question", "inquiry", "clarify"]):
            severity = "Low"
        priority_hint = next(
            (entry["default_priority"] for entry in self.categories if entry["name"] == category), "P2"
        )
        return {
            "category": category,
            "severity_level": severity,
            "priority_hint": priority_hint,
            "rationale": "Rule-based fallback",
            "confidence": 0.72,
        }

    async def process_case(
        self,
        complaint_path: Path,
        customer_profile_path: Path,
        interaction_history_path: Path,
        complaint_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Main entry point — reads files, classifies, returns structured output."""
        complaint_text = complaint_path.read_text(encoding="utf-8")
        customer_profile = json.loads(customer_profile_path.read_text(encoding="utf-8"))
        interaction_history = json.loads(interaction_history_path.read_text(encoding="utf-8"))

        if not complaint_text.strip():
            self.exception_mgr.add_event("missing_evidence", {"field": "complaint_text", "file": str(complaint_path)})
        if "KNA1" not in customer_profile:
            self.exception_mgr.add_event("missing_evidence", {"field": "customer_profile.KNA1", "file": str(customer_profile_path)})
        if not interaction_history.get("events"):
            self.exception_mgr.add_event("missing_evidence", {"field": "interaction_history.events", "file": str(interaction_history_path)})

        customer_id = customer_profile.get("KNA1", {}).get("KUNNR", "unknown")
        computed_id = complaint_id or f"CMP-{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.now(timezone.utc).isoformat()

        if self._is_invalid_actor(complaint_text):
            self.exception_mgr.add_event("invalid_actor", {
                "reason": "Complaint appears to be from internal staff/manager",
                "file": str(complaint_path),
            })
            summary = self.exception_mgr.summary()
            return self._build_incomplete_result(
                complaint_id=computed_id,
                customer_id=customer_id,
                timestamp=timestamp,
                summary=summary,
            )

        if self._has_blocking_missing_data():
            summary = self.exception_mgr.summary()
            return self._build_incomplete_result(
                complaint_id=computed_id,
                customer_id=customer_id,
                timestamp=timestamp,
                summary=summary,
            )

        sentiment = TextBlob(complaint_text).sentiment.polarity

        classification = await self._llm_classify(complaint_text, customer_profile, interaction_history)
        category = self._normalize_category(classification.get("category"))
        severity_label = classification.get("severity_level", "Medium")
        priority_hint = classification.get("priority_hint", "P2")
        confidence = classification.get("confidence", 0.91)

        tier = "Standard"
        knvv = customer_profile.get("KNVV", {})
        if knvv.get("VKORG") == "1000":
            tier = "Gold"

        severity_priority = enrich_priority(category, severity_label, tier, sentiment, self.sla_map)
        valid, validation_status = validation_checks(category, severity_priority, self.sla_map)
        if not valid:
            self.exception_mgr.add_event(validation_status, {
                "category": category,
                "severity": severity_priority.severity_label,
                "priority": severity_priority.priority_label,
                "customer_id": customer_id,
            })

        routing = self.routing_rules.get(category)
        if not routing:
            self.exception_mgr.add_event("unclear_classification", {"category": category, "complaint_id": complaint_id})
            routing = {"queue": "EvidenceCollector-General", "target_agent": "Evidence_Collector_Agent"}

        complaint_payload = {
            "complaint_id": computed_id,
            "customer_id": customer_id,
            "category": category,
            "severity_level": severity_priority.severity_label,
            "priority_tag": severity_priority.priority_label,
            "routing_queue": routing["queue"],
            "accuracy_score": round(confidence, 2),
            "validation_status": validation_status,
            "timestamp": timestamp,
            "agent_id": self.agent_id,
            "workflow_step": "Complaint_Classifier",
        }

        routing_payload = {
            "complaint_id": computed_id,
            "target_agent": routing["target_agent"],
            "queue": routing["queue"],
            "timestamp": timestamp,
            "agent_id": self.agent_id,
            "validation_status": validation_status,
        }

        severity_payload = {
            "complaint_id": computed_id,
            "severity": {
                "score": severity_priority.severity_score,
                "label": severity_priority.severity_label,
            },
            "priority": {
                "score": severity_priority.priority_score,
                "label": severity_priority.priority_label,
            },
            "sla": {
                "target_hours": severity_priority.sla_target_hours,
                "breach_risk": severity_priority.breach_risk,
            },
            "validation_status": validation_status,
            "timestamp": timestamp,
            "agent_id": self.agent_id,
        }

        self._persist_output("complaint_category", computed_id, complaint_payload)
        self._persist_output("routing_metadata", computed_id, routing_payload)
        self._persist_output("severity_priority", computed_id, severity_payload)

        return {
            "complaint_category": complaint_payload,
            "routing_metadata": routing_payload,
            "severity_priority": severity_payload,
            "exceptions": self.exception_mgr.summary(),
        }

    def _has_blocking_missing_data(self) -> bool:
        return any(event.exception_type == "missing_evidence" for event in self.exception_mgr.events)

    def _build_incomplete_result(
        self,
        complaint_id: str,
        customer_id: str,
        timestamp: str,
        summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        missing_messages = summary.get("messages") or []
        message = ("Missing required evidence. " + "; ".join(missing_messages)) if missing_messages else "Missing required evidence."
        base_payload = {
            "complaint_id": complaint_id,
            "customer_id": customer_id,
            "status": "incomplete",
            "message": message,
            "missing_details": missing_messages,
            "timestamp": timestamp,
            "agent_id": self.agent_id,
            "workflow_step": "Complaint_Classifier",
        }
        routing_stub = {
            "complaint_id": complaint_id,
            "status": "incomplete",
            "message": message,
            "missing_details": missing_messages,
            "timestamp": timestamp,
            "agent_id": self.agent_id,
        }
        severity_stub = {
            "complaint_id": complaint_id,
            "status": "incomplete",
            "message": message,
            "missing_details": missing_messages,
            "timestamp": timestamp,
            "agent_id": self.agent_id,
        }
        return {
            "complaint_category": base_payload,
            "routing_metadata": routing_stub,
            "severity_priority": severity_stub,
            "exceptions": summary,
        }

    def _is_invalid_actor(self, complaint_text: str) -> bool:
        text = complaint_text.lower()
        invalid_cues = [
            "i am the manager",
            "as the manager",
            "internal note",
            "per sop",
            "per our sop",
            "as per sop",
            "escalation template",
            "case manager",
            "csr note",
            "internal escalation",
        ]
        return any(cue in text for cue in invalid_cues)

    def _persist_output(self, folder_name: str, complaint_id: str, payload: Dict[str, Any]) -> None:
        folder = OUTPUT_DIR / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        filename = folder / f"{complaint_id}.json"
        filename.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = ["ComplaintClassifierAgent"]
