"""Exception handling orchestration for the Complaint Classifier Agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from agents.agent1_classifier.exception_router import needs_clarification, record_exception


@dataclass
class ExceptionEvent:
    exception_type: str
    details: Dict[str, Any]


class ExceptionManager:
    """Routes validation errors and classification gaps."""

    def __init__(self) -> None:
        self.events: List[ExceptionEvent] = []

    def add_event(self, exception_type: str, details: Dict[str, Any]) -> None:
        event = ExceptionEvent(exception_type=exception_type, details=details)
        self.events.append(event)
        record_exception(exception_type, details)

    def unresolved(self) -> List[ExceptionEvent]:
        return self.events

    def summary(self) -> Dict[str, Any]:
        return {
            "count": len(self.events),
            "requires_clarification": any(needs_clarification(e.exception_type) for e in self.events),
            "events": [e.__dict__ for e in self.events],
            "messages": [self._describe_event(e) for e in self.events],
        }

    @staticmethod
    def _describe_event(event: ExceptionEvent) -> str:
        details = event.details or {}
        target = details.get("field") or details.get("file") or details.get("category") or details.get("complaint_id")
        if target:
            return f"{event.exception_type}: {target}"
        return event.exception_type
