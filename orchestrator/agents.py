from __future__ import annotations

from pathlib import Path


def load_agent_profiles(path: Path) -> dict[str, dict[str, str | int | bool | list[str]]]:
    if not path.exists():
        return {}

    profiles: dict[str, dict[str, str | int | bool | list[str]]] = {}
    data = _parse_yaml_like(path.read_text(encoding="utf-8", errors="replace"))
    for item in data:
        role = str(item.get("role", "")).strip()
        if not role:
            continue
        if role in profiles:
            continue
        profiles[role] = item
    return profiles


def _parse_yaml_like(content: str) -> list[dict[str, object]]:
    # Lightweight parser for this repo-specific YAML shape.
    # Supports blocks like:
    # - key: value
    #   list:
    #     - item
    #   key2: value2
    items: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    current_list_key: str | None = None
    multiline_key: str | None = None
    multiline_indent: int = 0
    multiline_lines: list[str] = []

    for raw_line in content.splitlines():
        line = raw_line.rstrip("\n")
        # Multiline block scalar support: key: | (literal) or key: > (folded-ish, treated literally here).
        if multiline_key is not None and current is not None:
            if not line.strip():
                multiline_lines.append("")
                continue
            indent = len(line) - len(line.lstrip())
            if indent > multiline_indent:
                cut = min(len(raw_line), multiline_indent + 2)
                multiline_lines.append(raw_line[cut:].rstrip("\n"))
                continue
            current[multiline_key] = "\n".join(multiline_lines).rstrip("\n")
            multiline_key = None
            multiline_lines = []

        if not line.strip():
            continue
        # Treat comments as comments only outside of multiline blocks.
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
            if text:
                # form: - role: ceo
                if ":" in text:
                    k, v = _split_kv(text)
                    current[k] = _coerce(v)
            continue

        if current is None:
            continue

        if indent >= 2 and ":" in text:
            key, value = _split_kv(text)
            if value is None:
                current_list_key = key
                current.setdefault(key, [])
                continue
            if value in ("|", ">"):
                multiline_key = key
                multiline_indent = indent
                multiline_lines = []
                current[key] = ""
                current_list_key = None
                continue
            current[key] = _coerce(value)
            current_list_key = None
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
        if multiline_key is not None:
            current[multiline_key] = "\n".join(multiline_lines).rstrip("\n")
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
