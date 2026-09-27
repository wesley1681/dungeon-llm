"""Configure the TRPG backend and manage a local llama-server process.

The game bootstrap imports :func:`start_server` and :func:`stop_server` so the
server follows the GUI lifecycle. This file can still be run on its own when a
standalone server is useful. DeepSeek selections only write the config.
"""

from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# Change the model here (the only place to pick a model).
# Local GGUF: qwen3.6-mtp-q4 / qwen3-30b / gemma4-google-q4 /
#             gemma4-huihui-q4 / gemma4-hauhau-q4
# DeepSeek:   deepseek:<model name>, for example deepseek:deepseek-v4-flash
#             (the API key still lives in _deepseek_config.json)
SELECTED_MODEL = "gemma4-hauhau-q4"

SCRIPT_DIR = Path(__file__).resolve().parent

# These defaults can be overridden with environment variables.
LLAMA_SERVER_EXE = Path(
    os.environ.get(
        "LLAMA_SERVER_EXE",
        SCRIPT_DIR / "../llama.cpp-tq3/build/bin/Release/llama-server.exe",
    )
)
MODEL_DIR = Path(
    os.environ.get(
        "TRPG_LLM_MODEL_DIR",
        SCRIPT_DIR / "../llama.cpp-tq3/models/gemma4-26b",
    )
)

MODEL_PATHS = {
    "qwen3.6-mtp-q4": MODEL_DIR / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    "qwen3-30b": MODEL_DIR / "Qwen3-30B-A3B-Q4_K_M.gguf",
    "gemma4-google-q4": MODEL_DIR / "gemma-4-26B-A4B-it-ollama-text.gguf",
    "gemma4-huihui-q4": MODEL_DIR / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
    "gemma4-hauhau-q4": MODEL_DIR
    / "Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf",
}

# Per-model ncmoe values (MoE expert layers on CPU). Tune per benchmark.
NCMOE_VALUES = {
    "qwen3.6-mtp-q4": 16,
    "qwen3-30b": 32,
    "gemma4-google-q4": 8,
    "gemma4-huihui-q4": 8,
    "gemma4-hauhau-q4": 8,
}

SERVER_CONFIG_PATH = SCRIPT_DIR / "_server_config.json"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8090
SERVER_STARTUP_TIMEOUT = 600.0


class ServerStartError(RuntimeError):
    """Raised when the configured local llama-server cannot be started."""


def write_server_config(*, backend: str, model: str, base_url: str) -> None:
    config = {"backend": backend, "model": model, "base_url": base_url}
    SERVER_CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def wait_for_server(
    process: subprocess.Popen[bytes],
    *,
    timeout: float = SERVER_STARTUP_TIMEOUT,
) -> None:
    """Wait until llama-server reports that its model is ready."""
    print("Waiting for llama-server to load the model...", flush=True)
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise ServerStartError(
                f"llama-server exited before becoming ready (code {return_code})."
            )

        connection = http.client.HTTPConnection(SERVER_HOST, SERVER_PORT, timeout=2)
        try:
            connection.request("GET", "/health")
            response = connection.getresponse()
            response.read()
            if 200 <= response.status < 300:
                print("llama-server is ready.", flush=True)
                return
        except (OSError, http.client.HTTPException):
            pass
        finally:
            connection.close()

        time.sleep(0.5)

    raise ServerStartError(
        f"llama-server did not become ready within {timeout:g} seconds."
    )


def start_server() -> subprocess.Popen[bytes] | None:
    """Write the backend config and start a local server when one is needed.

    Returns the local server process, or ``None`` when using DeepSeek.
    """
    if SELECTED_MODEL.startswith("deepseek:"):
        model = SELECTED_MODEL.removeprefix("deepseek:")
        write_server_config(
            backend="deepseek",
            model=model,
            base_url="https://api.deepseek.com",
        )
        print(
            f"trpg backend: DeepSeek  model={model} "
            "(api_key comes from _deepseek_config.json)"
        )
        return None

    model_path = MODEL_PATHS.get(SELECTED_MODEL, Path(SELECTED_MODEL))
    ncmoe = NCMOE_VALUES.get(SELECTED_MODEL, 32)

    if not LLAMA_SERVER_EXE.is_file():
        raise ServerStartError(
            f"llama-server was not found: {LLAMA_SERVER_EXE}\n"
            "Set LLAMA_SERVER_EXE or place llama-server.exe under "
            "llama.cpp\\build\\bin\\Release."
        )

    if not model_path.is_file():
        raise ServerStartError(
            f"GGUF model was not found: {model_path}\n"
            "Download it under models\\llm or set TRPG_LLM_MODEL_DIR."
        )

    # base_url is a bare host:port; backend.py appends /v1/chat/completions.
    write_server_config(
        backend="llamacpp",
        model=SELECTED_MODEL,
        base_url=f"http://{SERVER_HOST}:{SERVER_PORT}",
    )
    print(
        f"trpg backend: llama.cpp  model={SELECTED_MODEL}  ncmoe={ncmoe}",
        flush=True,
    )

    command = [
        str(LLAMA_SERVER_EXE),
        "-m",
        str(model_path),
        "-ngl",
        "99",
        "-fa",
        "on",
        "-ctk",
        "q8_0",
        "-ctv",
        "q8_0",
        "-ncmoe",
        str(ncmoe),
        "-c",
        "32768",
        "-b",
        "512",
        "-ub",
        "512",
        "--parallel",
        "1",
        "--cache-ram",
        "0",
        "--jinja",
        "--reasoning",
        "auto",
        "--host",
        SERVER_HOST,
        "--port",
        str(SERVER_PORT),
    ]

    try:
        process = subprocess.Popen(command)
    except OSError as exc:
        raise ServerStartError(f"Failed to start llama-server: {exc}") from exc

    try:
        wait_for_server(process)
    except BaseException:
        stop_server(process)
        raise
    return process


def stop_server(process: subprocess.Popen[bytes] | None) -> None:
    """Stop a local server started by :func:`start_server`."""
    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main() -> int:
    try:
        process = start_server()
    except ServerStartError as exc:
        print(exc, file=sys.stderr)
        return 1

    if process is None:
        return 0

    try:
        return process.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        stop_server(process)


if __name__ == "__main__":
    raise SystemExit(main())
