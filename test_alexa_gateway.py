from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import alexa_gateway


REPO_ROOT = Path(__file__).resolve().parent


def _ask_request(query: str) -> dict:
    return {
        "version": "1.0",
        "session": {
            "new": True,
            "sessionId": "SessionId.test",
            "application": {"applicationId": "amzn1.ask.skill.test"},
            "user": {"userId": "amzn1.ask.account.test"},
        },
        "context": {
            "System": {
                "application": {"applicationId": "amzn1.ask.skill.test"},
                "user": {"userId": "amzn1.ask.account.test"},
            }
        },
        "request": {
            "type": "IntentRequest",
            "requestId": "EdwRequestId.test",
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "intent": {
                "name": "AskPonceIntent",
                "slots": {"query": {"name": "query", "value": query}},
            },
        },
    }


def test_extract_query_from_search_slot() -> None:
    intent = _ask_request("que tengo pendiente hoy")["request"]["intent"]
    assert alexa_gateway._extract_query(intent) == "que tengo pendiente hoy"


def test_launch_response_keeps_session_open() -> None:
    out = alexa_gateway._alexa_response("Aqui Ponce Bot.", reprompt="Que necesitas?", should_end_session=False)
    assert out["version"] == "1.0"
    assert out["response"]["shouldEndSession"] is False
    assert out["response"]["outputSpeech"]["type"] == "SSML"
    assert "reprompt" in out["response"]


def test_application_id_prefers_session() -> None:
    assert alexa_gateway._application_id(_ask_request("hola")) == "amzn1.ask.skill.test"


def test_speech_text_strips_markdown_and_urls() -> None:
    out = alexa_gateway._speech_text("**Hola** [link](https://example.com) `codigo`", max_chars=200)
    assert "https://" not in out
    assert "*" not in out
    assert "link" in out
    assert "codigo" in out


def test_gateway_submits_query_and_returns_ticket(monkeypatch) -> None:
    calls = {}

    class FakeQueue:
        def update_trace(self, job_id, **metadata):
            calls["trace_job_id"] = job_id
            calls["trace"] = metadata
            return True

        def append_audit_event(self, **metadata):
            calls["audit"] = metadata

        def get_job(self, job_id):
            calls["polled_job_id"] = job_id
            return None

    def fake_submit(**kwargs):
        calls["submit"] = kwargs
        return True, "abcdef123456"

    monkeypatch.setattr(alexa_gateway.poncebot, "_submit_orchestrator_task", fake_submit)
    gateway = alexa_gateway.AlexaGateway(
        cfg=object(),
        orch_q=FakeQueue(),
        profiles={},
        gateway_cfg=alexa_gateway.AlexaGatewayConfig(
            host="127.0.0.1",
            port=0,
            endpoint_path="/alexa",
            path_secret="",
            verify_signature=False,
            skill_id="",
            allowed_skill_ids=(),
            timestamp_tolerance_seconds=150,
            chat_id=123,
            user_id=456,
            wait_seconds=0,
            max_speech_chars=500,
        ),
    )

    out = gateway.handle(_ask_request("que tengo pendiente hoy"))
    assert calls["submit"]["job"].user_text == "que tengo pendiente hoy"
    assert calls["submit"]["user_id"] == 456
    assert calls["trace_job_id"] == "abcdef123456"
    assert "Ticket abcdef12" in out["response"]["outputSpeech"]["ssml"]


def test_parse_skill_ids_deduplicates_legacy_and_allowed() -> None:
    assert alexa_gateway._parse_skill_ids(
        "amzn1.ask.skill.one",
        "amzn1.ask.skill.two, amzn1.ask.skill.one",
        "amzn1.ask.skill.three;amzn1.ask.skill.two",
    ) == (
        "amzn1.ask.skill.one",
        "amzn1.ask.skill.two",
        "amzn1.ask.skill.three",
    )


def test_application_id_accepts_configured_allowed_id() -> None:
    class FakeServer:
        gateway_cfg = alexa_gateway.AlexaGatewayConfig(
            host="127.0.0.1",
            port=0,
            endpoint_path="/alexa",
            path_secret="",
            verify_signature=False,
            skill_id="amzn1.ask.skill.primary",
            allowed_skill_ids=("amzn1.ask.skill.primary", "amzn1.ask.skill.secondary"),
            timestamp_tolerance_seconds=150,
            chat_id=123,
            user_id=456,
            wait_seconds=0,
            max_speech_chars=500,
        )

    handler = object.__new__(alexa_gateway.AlexaHandler)
    handler.server = FakeServer()
    req = _ask_request("hola")
    req["session"]["application"]["applicationId"] = "amzn1.ask.skill.secondary"
    req["context"]["System"]["application"]["applicationId"] = "amzn1.ask.skill.secondary"

    handler._check_application_id(req)


def test_alexa_skill_package_has_manifest_and_locales() -> None:
    manifest_path = REPO_ROOT / "alexa" / "skill-package" / "skill.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    custom = manifest["manifest"]["apis"]["custom"]
    uri = custom["endpoint"]["uri"]

    assert uri.startswith("https://")
    assert "REPLACE_WITH_PUBLIC_HTTPS_HOST" in uri
    locales = manifest["manifest"]["publishingInformation"]["locales"]
    assert set(locales) >= {"es-MX", "es-US"}
    assert locales["es-MX"]["name"] == "Ponce Bot"


def test_alexa_interaction_models_route_freeform_query() -> None:
    for locale in ("es-MX", "es-US"):
        model_path = (
            REPO_ROOT
            / "alexa"
            / "skill-package"
            / "interactionModels"
            / "custom"
            / f"{locale}.json"
        )
        model = json.loads(model_path.read_text(encoding="utf-8"))
        language_model = model["interactionModel"]["languageModel"]
        ask_intent = next(
            intent
            for intent in language_model["intents"]
            if intent["name"] == "AskPonceIntent"
        )

        assert language_model["invocationName"] == "ponce bot"
        assert {"name": "query", "type": "AMAZON.SearchQuery"} in ask_intent["slots"]
        assert any("{query}" in sample for sample in ask_intent["samples"])
