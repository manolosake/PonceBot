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
    header_lines: list[str] = [
        f"ROLE: {role}",
        f"REQUEST_TYPE: {task.request_type}",
        f"MODE_HINT: {task.mode_hint}",
        f"ARTIFACTS_DIR: {task.artifacts_dir}",
    ]
    if tools_csv:
        header_lines.append(f"ALLOWED_TOOLS: {tools_csv}")
    if system_prompt:
        header_lines.append("")
        header_lines.append("ROLE_SYSTEM_PROMPT:")
        header_lines.append(system_prompt)

    header = "\n".join(header_lines).strip()
    user_text = (task.input_text or "").strip()

    # Output contract: human summary + structured JSON.
    if role in ("jarvis", "orchestrator", "ceo") and not task.parent_job_id and not task.is_autonomous:
        schema_hint = (
            '{\n'
            '  "summary": "string",\n'
            '  "subtasks": [\n'
            "    {\n"
            '      "key": "short_unique_id",\n'
            '      "role": "frontend|backend|qa|sre|jarvis",\n'
            '      "text": "task instruction",\n'
            '      "mode_hint": "ro|rw|full (optional; omit to use role defaults)",\n'
            '      "priority": 1,\n'
            '      "depends_on": ["other_key"],\n'
            '      "requires_approval": false\n'
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
        + "- FRONTEND ONLY: if you can provide visual evidence, create a self-contained HTML preview at\n"
        + "  `.codexbot_preview/preview.html` in the workspace. The bot will automatically screenshot it.\n"
    )
