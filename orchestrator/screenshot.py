from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Viewport:
    width: int = 1280
    height: int = 720


def capture(url: str, out_path: Path, *, viewport: Viewport | None = None, timeout_ms: int = 30_000) -> Path:
    """
    Capture a screenshot of a URL using Playwright (if installed).
    """
    u = (url or "").strip()
    if not u:
        raise ValueError("Empty url")
    if "://" not in u:
        u = "https://" + u
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vp = viewport or Viewport()

    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]
    except Exception as e:
        raise RuntimeError("Playwright not available. Install: pip install playwright && python -m playwright install chromium") from e

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": int(vp.width), "height": int(vp.height)})
            page.goto(u, wait_until="networkidle", timeout=int(timeout_ms))
            page.screenshot(path=str(out_path), full_page=True)
        finally:
            browser.close()

    return out_path

