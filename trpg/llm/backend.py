"""OpenAI-compatible chat transport for llama-server and DeepSeek."""
import json
import requests


def _strip_surrogates(s: str) -> str:
    """Remove unpaired Unicode surrogates that can't be encoded to UTF-8."""
    return s.encode("utf-8", errors="replace").decode("utf-8")


def stream_chat(base_url: str, model: str, messages: list, options: dict,
                think: bool = False, on_chunk=None, backend: str = "openai",
                timeout: int = 180, api_key: str = None) -> str:
    """Stream a chat request. Calls on_chunk(text, thinking=bool) per token.
    Returns the full assistant content string."""

    if backend != "openai":
        raise ValueError(f"Unsupported LLM backend: {backend!r}")
    return _stream_openai(base_url, model, messages, options, on_chunk, timeout, api_key, think)


def complete_chat(base_url: str, model: str, messages: list, options: dict,
                  backend: str = "openai", timeout: int = 60, api_key: str = None) -> str:
    """Non-streaming chat. Returns full content string."""

    if backend != "openai":
        raise ValueError(f"Unsupported LLM backend: {backend!r}")
    payload = _openai_payload(model, messages, options, stream=False)
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload,
                         headers=_auth_headers(api_key), timeout=timeout)
    resp.raise_for_status()
    choices = resp.json().get("choices", [])
    return choices[0]["message"]["content"].strip() if choices else ""


# ── Private helpers ────────────────────────────────────────────────────────────

def _auth_headers(api_key: str = None):
    return {"Authorization": f"Bearer {api_key}"} if api_key else None


def _stream_openai(base_url, model, messages, options, on_chunk, timeout, api_key=None, think=False):
    payload = _openai_payload(model, messages, options, stream=True, think=think)
    resp = requests.post(f"{base_url}/v1/chat/completions",
                         json=payload, headers=_auth_headers(api_key),
                         stream=True, timeout=timeout)
    resp.raise_for_status()

    full = ""
    for line in resp.iter_lines():
        if not line:
            continue
        text = line.decode("utf-8") if isinstance(line, bytes) else line
        if not text.startswith("data:"):
            continue
        payload_str = text[5:].strip()
        if payload_str == "[DONE]":
            break
        try:
            data = json.loads(payload_str)
        except json.JSONDecodeError:
            continue
        delta = data.get("choices", [{}])[0].get("delta", {})
        chunk = delta.get("content", "")
        if chunk:
            chunk = _strip_surrogates(chunk)
            full += chunk
            if on_chunk:
                on_chunk(chunk, thinking=False)
    return full


def _openai_payload(model, messages, options, stream, think=False):
    payload = {"model": model, "messages": messages, "stream": stream}
    # llama-server runs with --reasoning auto, which otherwise routes the model's
    # <think> section into a separate reasoning_content field we never read —
    # burning the token budget so small-budget agents (NPC=300, tag=200) hit the
    # max_tokens cap mid-reasoning and return an EMPTY content string. Honour the
    # Honour the think flag by telling the jinja chat template to skip thinking
    # entirely when think=False.
    payload["chat_template_kwargs"] = {"enable_thinking": bool(think)}
    if "temperature" in options:
        payload["temperature"] = options["temperature"]
    if "num_predict" in options:
        payload["max_tokens"] = options["num_predict"]
    if "repeat_penalty" in options and options["repeat_penalty"] != 1.0:
        payload["repeat_penalty"] = options["repeat_penalty"]
    return payload
