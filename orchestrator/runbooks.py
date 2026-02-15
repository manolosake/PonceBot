from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time

from .schemas.task import Task


@dataclass(frozen=True)
class Runbook:
    runbook_id: str
    role: str
    interval_seconds: int
    prompt: str
    mode_hint: str = "ro"
    priority: int = 2
    enabled: bool = True


def load_runbooks(path: Path) -> list[Runbook]:
    if not path.exists():
        return []
    data = _parse_yaml_like(path.read_text(encoding="utf-8", errors="replace"))
    out: list[Runbook] = []
    for item in data:
        rid = str(item.get("id", "")).strip()
        role = str(item.get("role", "")).strip().lower()
        prompt = str(item.get("prompt", "")).strip()
        if not rid or not role or not prompt:
            continue
        try:
            interval = int(item.get("interval_seconds", 3600))
        except Exception:
            interval = 3600
        interval = max(60, interval)
        mode_hint = str(item.get("mode_hint", "ro")).strip().lower() or "ro"
        if mode_hint not in ("ro", "rw", "full"):
            mode_hint = "ro"
        try:
            priority = int(item.get("priority", 2))
        except Exception:
            priority = 2
        if priority < 1:
            priority = 1
        if priority > 3:
            priority = 3
        enabled = bool(item.get("enabled", True))
        out.append(
            Runbook(
                runbook_id=rid,
                role=role,
                interval_seconds=interval,
                prompt=prompt,
                mode_hint=mode_hint,
                priority=priority,
                enabled=enabled,
            )
        )
    return out


def due(runbook: Runbook, *, last_run_at: float, now: float | None = None) -> bool:
    n = time.time() if now is None else float(now)
    return runbook.enabled and (n - float(last_run_at or 0.0)) >= float(runbook.interval_seconds)


def to_task(runbook: Runbook, *, chat_id: int) -> Task:
    return Task.new(
        source="telegram",
        role=runbook.role,
        input_text=runbook.prompt,
        request_type="maintenance",
        priority=int(runbook.priority),
        model="",
        effort="medium",
        mode_hint=runbook.mode_hint,
        requires_approval=(runbook.mode_hint == "full"),
        max_cost_window_usd=8.0,
        chat_id=int(chat_id),
        is_autonomous=True,
        owner="scheduler",
        labels={"runbook": runbook.runbook_id},
        trace={"runbook_id": runbook.runbook_id},
    )


def _parse_yaml_like(content: str) -> list[dict[str, object]]:
    # Same minimal YAML subset used in agents.py.
    items: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    current_list_key: str | None = None

    for raw_line in content.splitlines():
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue

        indent = len(line) - len(line.lstrip())
        text = line.strip()

        if text.startswith("-") and indent == 0:
            if current is not None:
                items.append(current)
            current = {}
            current_list_key = None
            text = text[1:].strip()
            if text and ":" in text:
                k, v = _split_kv(text)
                current[k] = _coerce(v)
            continue

        if current is None:
            continue

        if indent >= 2 and ":" in text:
            key, value = _split_kv(text)
            if value is not None:
                current[key] = _coerce(value)
                current_list_key = None
            elif value == "":
                current_list_key = key
                current.setdefault(key, [])
            continue

        if indent >= 4 and text.startswith("-") and current_list_key:
            item = text[1:].strip()
            if item:
                current.setdefault(current_list_key, [])
                lst = current.get(current_list_key)
                if isinstance(lst, list):
                    lst.append(_coerce(item))
            continue

    if current is not None:
        items.append(current)
    return items


def _split_kv(text: str) -> tuple[str, str | None]:
    k, v = text.split(":", 1)
    k = k.strip()
    v = v.strip()
    if v == "":
        return k, None
    return k, v


def _coerce(value: str | None) -> object:
    if value is None:
        return None
    v = value.strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    if v.lower() in ("true", "yes", "on"):
        return True
    if v.lower() in ("false", "no", "off"):
        return False
    try:
        if "." in v:
            return float(v)
        return int(v)
    except Exception:
        return v

