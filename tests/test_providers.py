"""Provider-adapter and HTTP-layer tests.

These do NOT call a real AI provider. They check that each adapter builds the request in the
documented format and parses the documented response shape, that rate limits / transient errors
are retried, and (via a local HTTP server) that the full pipeline works over a real HTTP stack.
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from skillleak import llm, pipeline, clo  # noqa: E402
from skillleak.extract import load_sample_documents  # noqa: E402
from fake_llm import fake_call_json  # noqa: E402


class FakeResp:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code, self._p, self.headers = status, payload or {}, headers or {}
        self.text = json.dumps(self._p)

    def json(self):
        return self._p


@pytest.fixture
def capture(monkeypatch):
    calls = []
    replies = []

    def fake_post(url, **kw):
        calls.append((url, kw))
        return replies.pop(0)
    monkeypatch.setattr(llm.requests, "post", fake_post)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    return calls, replies


def test_gemini_adapter(capture):
    calls, replies = capture
    replies.append(FakeResp(200, {"candidates": [{"content": {"parts": [{"text": '{"ok": 1}'}]}}]}))
    cfg = llm.LLMConfig("Google Gemini", "KEY", "gemini-x")
    assert llm.call_json(cfg, "sys", "user") == {"ok": 1}
    url, kw = calls[0]
    assert url.endswith("/models/gemini-x:generateContent")
    assert kw["headers"]["x-goog-api-key"] == "KEY" and "KEY" not in url
    assert kw["json"]["generationConfig"]["responseMimeType"] == "application/json"
    assert kw["json"]["systemInstruction"]["parts"][0]["text"].startswith("sys")


def test_gemini_blocked_response_is_clear_error(capture):
    calls, replies = capture
    replies.append(FakeResp(200, {"promptFeedback": {"blockReason": "SAFETY"}}))
    with pytest.raises(llm.LLMError, match="Unexpected Gemini response"):
        llm.call_json(llm.LLMConfig("Google Gemini", "KEY", "g"), "s", "u")


def test_anthropic_adapter(capture):
    calls, replies = capture
    replies.append(FakeResp(200, {"content": [{"type": "text", "text": '```json\n{"ok": 2}\n```'}]}))
    cfg = llm.LLMConfig("Anthropic Claude", "AKEY", "claude-x")
    assert llm.call_json(cfg, "sys", "user") == {"ok": 2}
    url, kw = calls[0]
    assert url == "https://api.anthropic.com/v1/messages"
    assert kw["headers"]["x-api-key"] == "AKEY" and kw["headers"]["anthropic-version"]
    assert kw["json"]["model"] == "claude-x" and kw["json"]["max_tokens"] > 0


def test_openai_compatible_adapter(capture):
    calls, replies = capture
    replies.append(FakeResp(200, {"choices": [{"message": {"content": '{"ok": 3}'}}]}))
    cfg = llm.LLMConfig("OpenAI-compatible", "OKEY", "m", base_url="https://example.test/v1/")
    assert llm.call_json(cfg, "sys", "user") == {"ok": 3}
    url, kw = calls[0]
    assert url == "https://example.test/v1/chat/completions"
    assert kw["headers"]["Authorization"] == "Bearer OKEY"
    assert kw["json"]["response_format"] == {"type": "json_object"}


def test_rate_limit_is_retried_then_succeeds(capture):
    calls, replies = capture
    replies += [FakeResp(429, {"error": "rate"}, {"retry-after": "1"}), FakeResp(503, {}),
                FakeResp(200, {"choices": [{"message": {"content": '{"ok": 4}'}}]})]
    cfg = llm.LLMConfig("OpenAI-compatible", "K", "m")
    assert llm.call_json(cfg, "s", "u") == {"ok": 4} and len(calls) == 3


def test_persistent_rate_limit_gives_clear_error(capture):
    calls, replies = capture
    replies += [FakeResp(429, {"error": "quota"})] * 3
    with pytest.raises(llm.LLMError, match="HTTP 429"):
        llm.call_json(llm.LLMConfig("OpenAI-compatible", "K", "m"), "s", "u")
    assert len(calls) == 3


def test_bad_request_is_not_retried(capture):
    calls, replies = capture
    replies.append(FakeResp(400, {"error": "bad model name"}))
    with pytest.raises(llm.LLMError, match="HTTP 400"):
        llm.call_json(llm.LLMConfig("OpenAI-compatible", "K", "m"), "s", "u")
    assert len(calls) == 1


def test_network_failure_gives_clear_error(monkeypatch):
    def boom(url, **kw):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(llm.requests, "post", boom)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    with pytest.raises(llm.LLMError, match="Could not reach the model API"):
        llm.call_json(llm.LLMConfig("Google Gemini", "K", "m"), "s", "u")


# ---------------------------------------------------------------------------
# Full pipeline over a real local HTTP server (OpenAI-compatible wire format)
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        system = body["messages"][0]["content"].split("\n\nRespond with a single valid JSON")[0]
        user = body["messages"][1]["content"]
        _Handler.seen.append((self.headers.get("Authorization"), body.get("response_format")))
        out = fake_call_json(None, system, user)
        data = json.dumps({"choices": [{"message": {"content": json.dumps(out)}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_full_pipeline_over_http(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        cfg = llm.LLMConfig("OpenAI-compatible", "LOCALKEY", "fake",
                            base_url=f"http://127.0.0.1:{srv.server_port}/v1")
        run = pipeline.run_scan(load_sample_documents("synthetic"), cfg)
        assert run["alignment"]["verifier"]["status"] == "ok"
        f = {r["id"]: r["finding"] for r in clo.analyze(run["alignment"])["rows"]}
        assert f["CLO-5"] == "Potential mismatch"
        assert len(run["competencies"]) == 3
        assert all(a == "Bearer LOCALKEY" and rf == {"type": "json_object"} for a, rf in _Handler.seen)
        assert len(_Handler.seen) == 4   # CLO extraction, blind rater, industry, course evidence
    finally:
        srv.shutdown()
