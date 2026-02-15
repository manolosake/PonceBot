from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskResult:
    status: str
    summary: str
    artifacts: list[str] = field(default_factory=list)
    logs: str = ""
    next_action: str | None = None
    structured_digest: dict[str, Any] = field(default_factory=dict)
