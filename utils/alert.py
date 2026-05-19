from __future__ import annotations

from typing import Any


class AlertManager:
    """Keeps alert state simple and frontend-friendly."""

    def __init__(self) -> None:
        self.last_level = "LOW"

    def clear(self) -> None:
        self.last_level = "LOW"

    def update(self, risk: dict[str, Any]) -> dict[str, Any]:
        level = risk["level"]
        self.last_level = level
        return {
            "active": level in {"MEDIUM", "HIGH"},
            "alarm": level == "HIGH",
            "message": self._message(level),
        }

    @staticmethod
    def _message(level: str) -> str:
        if level == "HIGH":
            return "High risk detected. Check infant immediately."
        if level == "MEDIUM":
            return "Warning: infant needs attention."
        return "Monitoring normally."
