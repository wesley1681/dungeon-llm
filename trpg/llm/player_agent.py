import json
import pathlib
import requests
from ..engine.character import Character

OLLAMA_URL = "http://localhost:11434"
_DEBUG_DIR = pathlib.Path(__file__).parent.parent / "debug"
_DEBUG_DIR.mkdir(exist_ok=True)

_SYSTEM_TEMPLATE = """你正在扮演一個D&D角色。用第一人稱繁體中文描述你的行動或對話。

## 你的角色
姓名：{name}
種族：{race}  職業：{class_}  等級：{level}
HP：{hp}/{max_hp}  AC：{ac}
屬性：力量{STR} 敏捷{DEX} 體質{CON} 智力{INT} 感知{WIS} 魅力{CHA}
道具：{inventory}
狀態：{status}

## 角色個性
{personality}

## 行為規則
- 只描述你自己的角色做什麼，不描述其他角色
- 回應限制在2-3句話，簡短直接
- 根據你的HP和情境做出合理決定（HP低時考慮保守行動）
- 可以和隊友互動、對話
"""


class PlayerAgent:
    def __init__(self, model: str, character: Character, personality: str,
                 think: bool = False, show_thinking: bool = False,
                 options: dict = None):
        self.model = model
        self.character = character
        self.personality = personality
        self.think = think
        self.show_thinking = show_thinking
        self.options = options or {}
        self.history: list[dict] = []

    def _system_prompt(self) -> str:
        c = self.character
        return _SYSTEM_TEMPLATE.format(
            name=c.name, race=c.race, class_=c.class_, level=c.level,
            hp=c.hp, max_hp=c.max_hp, ac=c.ac,
            STR=c.stats.STR, DEX=c.stats.DEX, CON=c.stats.CON,
            INT=c.stats.INT, WIS=c.stats.WIS, CHA=c.stats.CHA,
            inventory="、".join(c.inventory) if c.inventory else "無",
            status="、".join(c.status_effects) if c.status_effects else "無",
            personality=self.personality,
        )

    def generate(self, gm_narration: str, on_chunk=None) -> str:
        """Generate AI player action. on_chunk(chunk, thinking) called per token if provided."""
        self.history.append({"role": "user", "content": gm_narration})
        messages = [{"role": "system", "content": self._system_prompt()}] + self.history
        (_DEBUG_DIR / f"{self.character.name}_context.json").write_text(
            json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": self.model, "messages": messages, "stream": True, "think": self.think, "options": self.options},
            stream=True,
            timeout=120,
        )
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
                        on_chunk(f"\n[{self.character.name} 思考中]\n", thinking=True)
                    elif self.show_thinking:
                        print(f"\n[{self.character.name} 思考中]\n", flush=True)
                chunk = msg["thinking"]
                if on_chunk:
                    on_chunk(chunk, thinking=True)
                elif self.show_thinking:
                    print(chunk, end="", flush=True)
            elif msg.get("content"):
                if in_thinking:
                    in_thinking = False
                    if on_chunk:
                        on_chunk(f"\n[{self.character.name} 回答]\n", thinking=True)
                    elif self.show_thinking:
                        print(f"\n\n[{self.character.name} 回答]\n", flush=True)
                chunk = msg["content"]
                full += chunk
                if on_chunk:
                    on_chunk(chunk, thinking=False)
                else:
                    print(chunk, end="", flush=True)
            if data.get("done"):
                break
        if not on_chunk:
            print()

        self.history.append({"role": "assistant", "content": full})
        return full
