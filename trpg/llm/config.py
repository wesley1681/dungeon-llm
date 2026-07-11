"""Resolve which LLM backend/model trpg uses for this run.

trpg runs one backend for the whole session (GM/NPC/player agents all share
it) — there is no per-message switching, and no BACKEND/MODEL constant lives
in cli.py. The single place to pick a model is start_llama_server.ps1
(see its $selectedModel) — it always writes _server_config.json, and this
module just reads that file and turns it into the concrete
{backend, base_url, model, api_key} agents need. Run the ps1 script before
starting the game, even when picking "ollama:..." (it writes the config file
instantly without launching anything).
"""
import json
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_SERVER_CONFIG_PATH = _ROOT / "_server_config.json"
_DEEPSEEK_CONFIG_PATH = _ROOT / "_deepseek_config.json"

DEEPSEEK_DEFAULT = {
    "api_key": "",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-v4-flash",
}

# _server_config.json's "backend" value -> which wire protocol backend.py uses.
_DISPATCH = {"ollama": "ollama", "llamacpp": "openai", "deepseek": "openai"}


class BackendConfigError(RuntimeError):
    """Raised when _server_config.json (or the deepseek key it points at) is missing/incomplete."""


def resolve() -> dict:
    """Return {backend, base_url, model, api_key} from _server_config.json.

    backend is normalized to "ollama" | "openai" (what backend.py dispatches on).
    """
    if not _SERVER_CONFIG_PATH.exists():
        raise BackendConfigError(
            f"找不到 {_SERVER_CONFIG_PATH.name}，請先執行 start_llama_server.ps1 選擇模型"
            "（選 Ollama 或 DeepSeek 也要跑一次，它只會寫設定檔、不會啟動伺服器）。"
        )
    cfg = json.loads(_SERVER_CONFIG_PATH.read_text(encoding="utf-8"))
    source = cfg.get("backend")
    if source not in _DISPATCH:
        raise BackendConfigError(
            f"{_SERVER_CONFIG_PATH.name} 的 backend 欄位無效：{source!r}"
            "（合法值：ollama / llamacpp / deepseek，重跑 start_llama_server.ps1 修正）"
        )

    api_key = None
    if source == "deepseek":
        if not _DEEPSEEK_CONFIG_PATH.exists():
            _DEEPSEEK_CONFIG_PATH.write_text(
                json.dumps(DEEPSEEK_DEFAULT, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            raise BackendConfigError(
                f"已建立空白 {_DEEPSEEK_CONFIG_PATH.name}，請填入 api_key 後再啟動。"
            )
        ds_cfg = json.loads(_DEEPSEEK_CONFIG_PATH.read_text(encoding="utf-8"))
        api_key = ds_cfg.get("api_key") or None
        if not api_key:
            raise BackendConfigError(f"{_DEEPSEEK_CONFIG_PATH.name} 缺少 api_key。")

    return {"backend": _DISPATCH[source], "base_url": cfg["base_url"],
             "model": cfg["model"], "api_key": api_key}
