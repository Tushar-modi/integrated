"""Evidence Collector Agent — Autogen 0.4.x

Ported from Evidence_Collector_agent-main/Capstone/agents/evidence_collector.py.
Already used autogen-agentchat 0.4.x API — minimal changes needed.
Added rule-based fallback for when LLM is unavailable.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import TextMessage
    from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
    from autogen_core import CancellationToken
    AUTOGEN_AVAILABLE = True
except ImportError:
    AssistantAgent = None  # type: ignore
    AUTOGEN_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_SOURCE_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ─── Data-fetching tools (same as original) ───

def _read_json_file(filename: str) -> Dict[str, Any]:
    try:
        filepath = DATA_SOURCE_DIR / filename
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        return {"error": str(e)}


def _read_all_records(filename: str) -> List[Dict[str, Any]]:
    """Read a JSON file that may contain a single record or a list of records."""
    data = _read_json_file(filename)
    if not data:
        return []
    if isinstance(data, list):
        return data
    return [data]


async def fetch_sales_orders(customer_id: str) -> List[Dict[str, Any]]:
    """Fetch sales orders for a given customer ID."""
    records = _read_all_records("sap_sales_order_vbak_vbap.json")
    return [r for r in records if r.get("VBAK", {}).get("KUNNR") == customer_id]


async def fetch_deliveries(customer_id: str) -> List[Dict[str, Any]]:
    """Fetch deliveries for a given customer ID."""
    records = _read_all_records("sap_delivery_likp_lip.json")
    return [r for r in records if r.get("LIKP", {}).get("KUNNR") == customer_id]


async def fetch_billing_docs(customer_id: str) -> List[Dict[str, Any]]:
    """Fetch billing documents for a given customer ID."""
    records = _read_all_records("sap_billing_vbrk_vbrp.json")
    return [r for r in records if r.get("VBRK", {}).get("KUNNR") == customer_id]


async def fetch_warehouse_logs(delivery_id: str) -> List[Dict[str, Any]]:
    """Fetch warehouse logs for a given delivery ID."""
    records = _read_all_records("warehouse_log_sample_01.json")
    return [r for r in records if r.get("delivery") == delivery_id]


class EvidenceCollectorAgent:
    """Gathers SAP evidence data for a classified complaint."""

    def __init__(self, agent_id: str = "ECA-01") -> None:
        self.agent_id = agent_id
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

    async def _llm_collect(self, complaint_data: Dict[str, Any],
                           customer_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Use Autogen AssistantAgent with tool-calling to gather evidence."""
        if not AUTOGEN_AVAILABLE or not self.model_client:
            return await self._rule_based_collect(complaint_data, customer_profile)
        try:
            print("Invoking Autogen 0.4.x Evidence Collector with tool-calling...")
            agent = AssistantAgent(
                name="Evidence_Collector_Agent",
                model_client=self.model_client,
                tools=[fetch_sales_orders, fetch_deliveries, fetch_billing_docs, fetch_warehouse_logs],
                system_message=(
                    "You are an expert Evidence Collector Agent in an Order-to-Cash process. "
                    "Your goal is to gather all supporting data for a given complaint. "
                    "Validate inputs first, then fetch sales orders, deliveries, billing docs, "
                    "and warehouse logs. Return a complete evidence packet as valid JSON."
                ),
            )

            message = (
                f"Please collect evidence for the following complaint.\n\n"
                f"Complaint Details:\n{json.dumps(complaint_data, indent=2)}\n\n"
                f"Customer Profile:\n{json.dumps(customer_profile, indent=2)}\n\n"
                f"Produce the final evidence packet output."
            )

            result = await agent.run(task=message)

            # Extract the last text message as the evidence packet
            last_content = ""
            for msg in reversed(result.messages):
                if isinstance(msg, TextMessage) and msg.content.strip():
                    last_content = msg.content
                    break
                elif hasattr(msg, "content") and str(msg.content).strip():
                    last_content = str(msg.content)
                    break

            print(f"LLM Evidence response length: {len(last_content)} chars")

            # Try to parse JSON from the response
            try:
                return json.loads(last_content)
            except json.JSONDecodeError:
                # LLM returned non-JSON text — try to extract JSON block
                import re
                json_match = re.search(r'\{[\s\S]*\}', last_content)
                if json_match:
                    return json.loads(json_match.group())
                # Fall back to wrapping the text
                return {"llm_summary": last_content, "mode": "llm_text"}

        except Exception as exc:
            print(f"LLM evidence collection failed: {exc}")
            return await self._rule_based_collect(complaint_data, customer_profile)

    async def _rule_based_collect(self, complaint_data: Dict[str, Any],
                                  customer_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback: directly fetch all SAP data without LLM."""
        customer_id = complaint_data.get("customer_id") or customer_profile.get("KNA1", {}).get("KUNNR", "")

        # Validate inputs
        profile_customer_id = customer_profile.get("KNA1", {}).get("KUNNR", "")
        if customer_id and profile_customer_id and customer_id != profile_customer_id:
            return {
                "status": "Invalid Input",
                "reason": f"Mismatched Customer IDs: Complaint has {customer_id}, Profile has {profile_customer_id}",
            }

        sales_orders = await fetch_sales_orders(customer_id)
        deliveries = await fetch_deliveries(customer_id)
        billing_docs = await fetch_billing_docs(customer_id)

        # Fetch warehouse logs for each delivery
        warehouse_logs = []
        for deliv in deliveries:
            delivery_id = deliv.get("LIKP", {}).get("VBELN", "")
            if delivery_id:
                logs = await fetch_warehouse_logs(delivery_id)
                warehouse_logs.extend(logs)

        # Build summary
        last_event = None
        current_status = "UNKNOWN"
        if deliveries:
            current_status = deliveries[0].get("LIKP", {}).get("STATUS", "UNKNOWN")
        if warehouse_logs and warehouse_logs[0].get("events"):
            events = warehouse_logs[0]["events"]
            last_event = events[-1] if events else None

        # Detect requested delivery date from sales order
        issue_hint = ""
        if sales_orders:
            edatu = sales_orders[0].get("VBAP", [{}])[0].get("EDATU", "")
            if edatu:
                issue_hint = f"Shipment delay vs requested date (EDATU {edatu})"

        complaint_id = complaint_data.get("complaint_id", "UNKNOWN")

        evidence_packet = {
            "complaint_id": complaint_id,
            "customer_id": customer_id,
            "sales_order": sales_orders[0] if sales_orders else None,
            "delivery": deliveries[0] if deliveries else None,
            "billing": billing_docs[0] if billing_docs else None,
            "warehouse_logs": warehouse_logs[0] if warehouse_logs else None,
            "summary": {
                "issue": issue_hint or "Evidence collected via rule-based fallback",
                "current_status": current_status,
                "last_event": last_event,
            },
            "completeness_flag": bool(sales_orders and deliveries and billing_docs),
            "validation_status": "pass" if sales_orders else "incomplete",
            "accuracy_score": 0.85,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": self.agent_id,
            "mode": "rule_based_fallback",
        }
        return evidence_packet

    async def process(self, complaint_data: Dict[str, Any],
                      customer_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point — collects evidence for a classified complaint."""
        result = await self._llm_collect(complaint_data, customer_profile)

        # Persist output
        complaint_id = result.get("complaint_id") or complaint_data.get("complaint_id", "UNKNOWN")
        output_dir = OUTPUT_DIR / "evidence_packet"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{complaint_id}.json"
        output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

        return result


__all__ = ["EvidenceCollectorAgent"]
