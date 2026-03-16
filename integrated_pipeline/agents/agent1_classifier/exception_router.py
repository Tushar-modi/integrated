"""Centralized exception routing utilities for Agent 1."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Paths relative to integrated_pipeline root (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "exception_policies.json"
AUDIT_LOG = PROJECT_ROOT / "logs" / "audit" / "exceptions.jsonl"


def load_policies() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def record_exception(exception_type: str, payload: Dict[str, Any]) -> None:
    policies = load_policies()
    policy = policies.get(exception_type, {"severity": "unknown"})
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "exception_type": exception_type,
        "policy": policy,
        "payload": payload,
    }
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
    resolution = policy.get("resolution", "n/a")
    print(f"Logged exception: {exception_type} -> {resolution}")


def needs_clarification(exception_type: str) -> bool:
    policies = load_policies()
    policy = policies.get(exception_type, {})
    return policy.get("next_agent") == "Clarification_Composer_Agent"
