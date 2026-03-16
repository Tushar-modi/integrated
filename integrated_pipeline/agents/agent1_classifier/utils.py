"""Shared utilities for Agent 1 - Complaint Classifier."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

# Paths relative to integrated_pipeline root (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


@dataclass
class SeverityPriority:
    severity_label: str
    severity_score: float
    priority_label: str
    priority_score: float
    sla_target_hours: int
    breach_risk: float


CATEGORY_WEIGHTS = {
    "Delivery Delay": 0.6,
    "Billing Dispute": 0.5,
    "Product Defect": 0.7,
    "Service Outage": 0.9,
    "Account Access": 0.6,
}

SEVERITY_SCALE = {"Low": 0.3, "Medium": 0.65, "High": 0.9}
PRIORITY_SCALE = {"P3": 0.3, "P2": 0.65, "P1": 0.9}


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def estimate_breach_risk(priority: str, severity: str) -> float:
    """Lightweight heuristic when no historical ticket logs are loaded."""
    base = {"P1": 0.55, "P2": 0.35, "P3": 0.15}.get(priority, 0.3)
    severity_bonus = {"High": 0.3, "Medium": 0.15, "Low": 0.0}.get(severity, 0.1)
    risk = min(0.95, base + severity_bonus)
    return round(risk, 2)


def enrich_priority(category: str, severity: str, customer_tier: str, sentiment: float, sla_map: Dict) -> SeverityPriority:
    severity_score = SEVERITY_SCALE.get(severity, 0.5)
    base_priority = "P1" if severity == "High" else "P2" if severity == "Medium" else "P3"
    if customer_tier in {"Gold", "Platinum"} and severity != "Low":
        base_priority = "P1" if severity == "High" else "P2"
    if sentiment < -0.4:
        base_priority = "P1"
    priority_score = PRIORITY_SCALE.get(base_priority, 0.5)
    sla_target = sla_map.get(category, {}).get(base_priority, {}).get("target_hours", 24)
    breach_risk = estimate_breach_risk(base_priority, severity)
    return SeverityPriority(
        severity_label=severity,
        severity_score=round(severity_score, 2),
        priority_label=base_priority,
        priority_score=round(priority_score, 2),
        sla_target_hours=sla_target,
        breach_risk=breach_risk,
    )


def validation_checks(category: str, severity_priority: SeverityPriority, sla_map: Dict) -> Tuple[bool, str]:
    sla_target = sla_map.get(category, {}).get(severity_priority.priority_label, {}).get("target_hours")
    if sla_target is None:
        return False, "sla_lookup_missing"
    if severity_priority.severity_label == "High" and severity_priority.priority_label == "P3":
        return False, "priority_mismatch"
    if severity_priority.severity_label == "Low" and severity_priority.priority_label == "P1":
        return False, "priority_mismatch"
    return True, "pass"
