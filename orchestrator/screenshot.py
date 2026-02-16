from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from pathlib import Path
from typing import Any
import socket
import urllib.parse


@dataclass(frozen=True)
class Viewport:
    width: int = 1280
    height: int = 720


@dataclass(frozen=True)
class ScreenshotUrlCheck:
    ok: bool
    normalized_url: str
    reason: str = ""
    # If true: caller may choose to proceed only after an explicit approval.
    overrideable: bool = False


def validate_screenshot_url(
    url: str,
    *,
    allowed_hosts: set[str] | frozenset[str] | None = None,
    allow_private: bool = False,
    resolver: Any | None = None,
) -> ScreenshotUrlCheck:
    """
    Best-effort anti-SSRF URL validation for screenshots.

    Grounded goal: block non-http(s) schemes and private/reserved destinations by default.

    Notes:
    - `allow_private=True` is meant to be gated by an explicit operator approval.
    - `allowed_hosts` (if non-empty) acts as an allowlist (exact host match, case-insensitive).
    - `resolver` is injectable for tests; default uses `socket.getaddrinfo`.
    """
    raw = (url or "").strip()
    if not raw:
        return ScreenshotUrlCheck(ok=False, normalized_url="", reason="empty url", overrideable=False)

    u = raw if "://" in raw else "https://" + raw
    try:
        parsed = urllib.parse.urlsplit(u)
    except Exception:
        return ScreenshotUrlCheck(ok=False, normalized_url=u, reason="invalid url", overrideable=False)

    scheme = (parsed.scheme or "").strip().lower()
    if scheme not in ("http", "https"):
        return ScreenshotUrlCheck(ok=False, normalized_url=u, reason=f"blocked scheme: {scheme}", overrideable=False)

    if parsed.username or parsed.password:
        return ScreenshotUrlCheck(ok=False, normalized_url=u, reason="blocked userinfo in url", overrideable=False)

    host = (parsed.hostname or "").strip().rstrip(".").lower()
    if not host:
        return ScreenshotUrlCheck(ok=False, normalized_url=u, reason="missing host", overrideable=False)

    allow = set(h.strip().rstrip(".").lower() for h in (allowed_hosts or set()) if str(h).strip())
    if allow and host not in allow and not allow_private:
        return ScreenshotUrlCheck(
            ok=False,
            normalized_url=u,
            reason=f"host not in allowlist: {host}",
            overrideable=True,
        )

    # Literal IPs: block private/reserved by default.
    try:
        ip = ipaddress.ip_address(host)
        if (not allow_private) and (not ip.is_global):
            return ScreenshotUrlCheck(
                ok=False,
                normalized_url=u,
                reason=f"blocked ip: {host}",
                overrideable=True,
            )
        return ScreenshotUrlCheck(ok=True, normalized_url=u)
    except ValueError:
        pass

    # Hostnames: resolve and block if any target IP is non-global.
    port = parsed.port
    if port is None:
        port = 443 if scheme == "https" else 80
    getaddrinfo = resolver if resolver is not None else socket.getaddrinfo
    try:
        infos = getaddrinfo(host, int(port))
    except Exception:
        return ScreenshotUrlCheck(
            ok=False,
            normalized_url=u,
            reason=f"failed to resolve host: {host}",
            overrideable=True,
        )

    addrs: set[str] = set()
    try:
        for it in infos or []:
            addr = it[4][0] if isinstance(it, (list, tuple)) and len(it) > 4 else None
            if isinstance(addr, str) and addr:
                addrs.add(addr)
    except Exception:
        addrs = set()

    for addr in sorted(addrs):
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if (not allow_private) and (not ip.is_global):
            return ScreenshotUrlCheck(
                ok=False,
                normalized_url=u,
                reason=f"host resolves to blocked ip: {host} -> {addr}",
                overrideable=True,
            )

    return ScreenshotUrlCheck(ok=True, normalized_url=u)


def capture(
    url: str,
    out_path: Path,
    *,
    viewport: Viewport | None = None,
    timeout_ms: int = 30_000,
    allowed_hosts: set[str] | frozenset[str] | None = None,
    allow_private: bool = False,
) -> Path:
    """
    Capture a screenshot of a URL using Playwright (if installed).
    """
    check = validate_screenshot_url(url, allowed_hosts=allowed_hosts, allow_private=allow_private)
    if not check.ok:
        raise ValueError(check.reason)
    u = check.normalized_url
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
            host_cache: dict[str, bool] = {}

            def _route_guard(route: Any, request: Any) -> None:
                """
                Best-effort SSRF guardrails: block requests to non-global/private targets unless allow_private=True.
                """
                try:
                    req_url = str(getattr(request, "url", "") or "")
                    parsed_req = urllib.parse.urlsplit(req_url)
                    host = (parsed_req.hostname or "").strip().rstrip(".").lower()
                    if not host:
                        route.abort()
                        return
                    ok = host_cache.get(host)
                    if ok is None:
                        chk = validate_screenshot_url(
                            req_url,
                            allowed_hosts=allowed_hosts,
                            allow_private=allow_private,
                        )
                        ok = bool(chk.ok)
                        host_cache[host] = ok
                    if ok:
                        route.continue_()
                        return
                    route.abort()
                except Exception:
                    try:
                        route.abort()
                    except Exception:
                        pass

            try:
                page.route("**/*", _route_guard)
            except Exception:
                # If routing isn't available, fall back to single upfront validation only.
                pass
            page.goto(u, wait_until="networkidle", timeout=int(timeout_ms))
            page.screenshot(path=str(out_path), full_page=True)
        finally:
            browser.close()

    return out_path
