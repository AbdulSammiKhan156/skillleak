"""Provider-agnostic LLM client that always returns parsed JSON.

Supports Google Gemini, Anthropic Claude and any OpenAI-compatible API
(OpenAI, Groq, OpenRouter, ...). Uses plain HTTP so there are no SDK version
conflicts. API keys are passed in at runtime and never stored on disk.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import requests

PROVIDERS = {
    "Google Gemini": {"default_model": "gemini-2.5-flash"},
    "Anthropic Claude": {"default_model": "claude-sonnet-4-5"},
    "OpenAI-compatible": {"default_model": "gpt-4o-mini",
                          "default_base_url": "https://api.openai.com/v1"},
}

TIMEOUT = 120
MAX_RETRIES = 2                      # extra attempts for rate limits / transient server errors
RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


@dataclass
class LLMConfig:
    provider: str
    api_key: str
    model: str
    base_url: str = ""

    def label(self) -> str:
        return f"{self.provider} / {self.model}"


def _raise_for(resp: requests.Response) -> None:
    if resp.status_code >= 400:
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text[:500]
        raise LLMError(f"HTTP {resp.status_code} from model API: {detail}")


def _post(url: str, **kw) -> requests.Response:
    """POST with a small retry/backoff for rate limits (429), transient 5xx, timeouts and dropped connections."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(url, **kw)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            if attempt == MAX_RETRIES:
                raise LLMError(f"Could not reach the model API ({type(exc).__name__}). Check the internet "
                               "connection, or load a saved scan.") from exc
            time.sleep(min(2 ** attempt * 2, 20))
            continue
        if resp.status_code in RETRY_STATUS and attempt < MAX_RETRIES:
            try:
                wait = float(resp.headers.get("retry-after", ""))
            except ValueError:
                wait = 2 ** attempt * 2
            time.sleep(max(0.0, min(wait, 20)))
            continue
        return resp
    raise LLMError(f"Model API unavailable: {last_exc}")


def _call_raw(cfg: LLMConfig, system: str, user: str, max_tokens: int) -> str:
    if not cfg.api_key:
        raise LLMError("No API key provided. Add one in the sidebar.")

    if cfg.provider == "Google Gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg.model}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"responseMimeType": "application/json",
                                 "maxOutputTokens": max_tokens},
        }
        resp = _post(url, json=body, timeout=TIMEOUT,
                             headers={"x-goog-api-key": cfg.api_key})
        _raise_for(resp)
        data = resp.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError):
            raise LLMError(f"Unexpected Gemini response: {str(data)[:500]}")

    if cfg.provider == "Anthropic Claude":
        resp = _post(
            "https://api.anthropic.com/v1/messages",
            json={"model": cfg.model, "max_tokens": max_tokens, "system": system,
                  "messages": [{"role": "user", "content": user}]},
            headers={"x-api-key": cfg.api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            timeout=TIMEOUT,
        )
        _raise_for(resp)
        data = resp.json()
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    # OpenAI-compatible
    base = (cfg.base_url or PROVIDERS["OpenAI-compatible"]["default_base_url"]).rstrip("/")
    resp = _post(
        f"{base}/chat/completions",
        json={"model": cfg.model,
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": user}],
              "response_format": {"type": "json_object"}},
        headers={"Authorization": f"Bearer {cfg.api_key}"},
        timeout=TIMEOUT,
    )
    _raise_for(resp)
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError):
        raise LLMError(f"Unexpected response: {str(data)[:500]}")


def parse_json(text: str) -> dict:
    """Parse JSON from a model reply, tolerating code fences and stray prose."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def call_json(cfg: LLMConfig, system: str, user: str, max_tokens: int = 8000) -> dict:
    """Call the model and return a dict. Retries once if the JSON is invalid."""
    system = system + "\n\nRespond with a single valid JSON object only. No markdown, no commentary."
    raw = _call_raw(cfg, system, user, max_tokens)
    try:
        out = parse_json(raw)
        if not isinstance(out, dict):
            raise ValueError("top-level JSON is not an object")
        return out
    except (json.JSONDecodeError, ValueError):
        retry_user = (user + "\n\nYour previous reply was not valid JSON. "
                      "Return ONLY the JSON object, complete and properly closed.")
        raw = _call_raw(cfg, system, retry_user, max_tokens)
        try:
            out = parse_json(raw)
            if not isinstance(out, dict):
                raise ValueError("top-level JSON is not an object")
            return out
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMError(f"Model did not return valid JSON: {exc}. Start of reply: {raw[:300]}")


def list_models(cfg: LLMConfig) -> list[str]:
    """Return model ids available to this key (helps pick a valid model name)."""
    if cfg.provider == "Google Gemini":
        resp = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                            headers={"x-goog-api-key": cfg.api_key}, timeout=30,
                            params={"pageSize": 200})
        _raise_for(resp)
        return sorted(m["name"].removeprefix("models/") for m in resp.json().get("models", [])
                      if "generateContent" in m.get("supportedGenerationMethods", []))
    if cfg.provider == "Anthropic Claude":
        resp = requests.get("https://api.anthropic.com/v1/models", timeout=30,
                            headers={"x-api-key": cfg.api_key, "anthropic-version": "2023-06-01"})
        _raise_for(resp)
        return [m["id"] for m in resp.json().get("data", [])]
    base = (cfg.base_url or PROVIDERS["OpenAI-compatible"]["default_base_url"]).rstrip("/")
    resp = requests.get(f"{base}/models", timeout=30,
                        headers={"Authorization": f"Bearer {cfg.api_key}"})
    _raise_for(resp)
    return sorted(m["id"] for m in resp.json().get("data", []))
