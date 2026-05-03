"""
Thin abstraction over two LLM backends:
  "ollama"  – Ollama /api/chat  (http://localhost:11434)
  "openai"  – llama-server /v1/chat/completions (http://localhost:11435)
"""
import json
import requests


def stream_chat(base_url: str, model: str, messages: list, options: dict,
                think: bool = False, on_chunk=None, backend: str = "ollama",
                timeout: int = 180) -> str:
    """Stream a chat request. Calls on_chunk(text, thinking=bool) per token.
    Returns the full assistant content string."""

    if backend == "ollama":
        return _stream_ollama(base_url, model, messages, options, think, on_chunk, timeout)
    else:
        return _stream_openai(base_url, model, messages, options, on_chunk, timeout)


def complete_chat(base_url: str, model: str, messages: list, options: dict,
                  backend: str = "ollama", timeout: int = 60) -> str:
    """Non-streaming chat. Returns full content string."""

    if backend == "ollama":
        payload = {"model": model, "messages": messages,
                   "stream": False, "think": False, "options": options}
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()
    else:
        payload = _openai_payload(model, messages, options, stream=False)
        resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout)
        resp.raise_for_status()
        choices = resp.json().get("choices", [])
        return choices[0]["message"]["content"].strip() if choices else ""


# ── Private helpers ────────────────────────────────────────────────────────────

def _stream_ollama(base_url, model, messages, options, think, on_chunk, timeout):
    payload = {"model": model, "messages": messages, "stream": True, "think": think, "options": options}
    resp = requests.post(f"{base_url}/api/chat", json=payload, stream=True, timeout=timeout)
    resp.raise_for_status()

    full = ""
    in_thinking = False
    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        msg = data.get("message", {})
        if msg.get("thinking"):
            if not in_thinking:
                in_thinking = True
                if on_chunk:
                    on_chunk("\n[思考中]\n", thinking=True)
            if on_chunk:
                on_chunk(msg["thinking"], thinking=True)
        elif msg.get("content"):
            if in_thinking:
                in_thinking = False
                if on_chunk:
                    on_chunk("\n[回答]\n", thinking=True)
            chunk = msg["content"]
            full += chunk
            if on_chunk:
                on_chunk(chunk, thinking=False)
        if data.get("done"):
            break
    return full


def _stream_openai(base_url, model, messages, options, on_chunk, timeout):
    payload = _openai_payload(model, messages, options, stream=True)
    resp = requests.post(f"{base_url}/v1/chat/completions",
                         json=payload, stream=True, timeout=timeout)
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
            full += chunk
            if on_chunk:
                on_chunk(chunk, thinking=False)
    return full


def _openai_payload(model, messages, options, stream):
    payload = {"model": model, "messages": messages, "stream": stream}
    if "temperature" in options:
        payload["temperature"] = options["temperature"]
    if "num_predict" in options:
        payload["max_tokens"] = options["num_predict"]
    if "repeat_penalty" in options and options["repeat_penalty"] != 1.0:
        payload["repeat_penalty"] = options["repeat_penalty"]
    return payload
