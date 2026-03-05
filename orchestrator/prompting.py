from __future__ import annotations

from typing import Any

from .schemas.task import Task


def build_agent_prompt(task: Task, *, profile: dict[str, Any] | None = None) -> str:
    """
    Build a single prompt string that turns a role profile into "agent behavior".

    This is intentionally plain-text so it works with `codex exec` and resume sessions.
    """
    p = profile or {}
    role = (task.role or "backend").strip().lower()
    system_prompt = str(p.get("system_prompt") or "").strip()
    allowed_tools = p.get("allowed_tools") or []
    if not isinstance(allowed_tools, list):
        allowed_tools = []
    tools_csv = ", ".join(str(x) for x in allowed_tools if str(x).strip())

    # Keep this short; Codex already has its own system instructions.
    trace = dict(task.trace or {})
    order_branch = str(trace.get("order_branch") or "").strip()
    project_path = str(trace.get("project_path") or "").strip()

    header_lines: list[str] = [
        f"ROLE: {role}",
        f"REQUEST_TYPE: {task.request_type}",
        f"MODE_HINT: {task.mode_hint}",
        f"ARTIFACTS_DIR: {task.artifacts_dir}",
    ]
    if order_branch:
        header_lines.append(f"ORDER_BRANCH: {order_branch}")
    if project_path:
        header_lines.append(f"PROJECT_PATH: {project_path}")
    if tools_csv:
        header_lines.append(f"ALLOWED_TOOLS: {tools_csv}")
    if system_prompt:
        header_lines.append("")
        header_lines.append("ROLE_SYSTEM_PROMPT:")
        header_lines.append(system_prompt)

    header = "\n".join(header_lines).strip()
    user_text = (task.input_text or "").strip()

    # Output contract: human summary + structured JSON.
    # Jarvis may be autonomous/child (autopilot) but still allowed to delegate when explicitly enabled.
    allow_delegation = bool((task.trace or {}).get("allow_delegation", False))
    req_type = str(task.request_type or "task").strip().lower() or "task"
    can_delegate = req_type in ("task", "maintenance", "review")
    if role in ("jarvis", "orchestrator", "ceo") and can_delegate and ((not task.parent_job_id and not task.is_autonomous) or allow_delegation):
        schema_hint = (
            '{\n'
            '  "summary": "string",\n'
            '  "subtasks": [\n'
            "    {\n"
            '      "key": "short_unique_id",\n'
            '      "role": "frontend|backend|qa|sre|product_ops|security|research|release_mgr|jarvis",\n'
            '      "text": "task instruction",\n'
            '      "mode_hint": "ro|rw|full (optional; omit to use role defaults)",\n'
            '      "priority": 1,\n'
            '      "depends_on": ["other_key"],\n'
            '      "requires_approval": false,\n'
            '      "acceptance_criteria": ["measurable checks"],\n'
            '      "definition_of_done": ["what must be true to close"],\n'
            '      "eta_minutes": 120,\n'
            '      "sla_tier": "normal|high|urgent"\n'
            "    }\n"
            "  ],\n"
            '  "next_action": null\n'
            "}\n"
        )
    else:
        if role == "frontend":
            schema_hint = (
                '{\n'
                '  "summary": "string",\n'
                '  "artifacts": ["optional_paths"],\n'
                '  "snapshot_url": null,\n'
                '  "next_action": null\n'
                "}\n"
            )
        else:
            schema_hint = (
                '{\n'
                '  "summary": "string",\n'
                '  "artifacts": ["optional_paths"],\n'
                '  "next_action": null\n'
                "}\n"
            )

    android_guardrails = (
        "- ANDROID RULES: if the request touches Android/mobile app, use existing repo `/home/aponce/OmniCrewApp.android` and the order branch context.\n"
        "- ANDROID RULES: native UI only (Jetpack Compose/Material 3). Do NOT use WebView unless CEO explicitly asks for WebView.\n"
    )

    return (
        header
        + "\n\n"
        + "USER_REQUEST:\n"
        + user_text
        + "\n\n"
        + "OUTPUT_FORMAT:\n"
        + "1) Provide a short human-readable response.\n"
        + "2) Provide a single JSON object in a fenced block:\n"
        + "```json\n"
        + schema_hint
        + "```\n"
        + "\n"
        + "IMPORTANT:\n"
        + "- Keep the JSON valid (double quotes, no trailing commas).\n"
        + "- If you cannot do something safely, explain and set next_action.\n"
        + "- If you delegate subtasks, include acceptance_criteria + definition_of_done + eta_minutes + sla_tier for each subtask.\n"
        + android_guardrails
        + "- FRONTEND ONLY: visual evidence is mandatory before completion.\n"
        + "  Create `.codexbot_preview/preview.html` in the workspace so the bot can capture it.\n"
        + "  If live preview is not possible, include multiple screenshots (mobile/tablet/desktop).\n"
    )
