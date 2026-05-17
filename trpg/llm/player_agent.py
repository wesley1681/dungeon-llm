import json
import pathlib
from ..engine.character import Character
from ..engine.world_state import WorldState
from .backend import stream_chat
from .log_render import render_messages

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

{tactics_section}## 行為規則
- 只描述你自己的角色做什麼，不描述其他角色
- 回應限制在1-2句話，簡短直接
- 可以和隊友互動、對話
"""


class PlayerAgent:
    def __init__(self, model: str, char_id: str, character: Character,
                 personality: str, world_state: WorldState,
                 tactics: str = "",
                 combat_tactics: str = "",
                 think: bool = False, show_thinking: bool = False,
                 options: dict = None,
                 base_url: str = OLLAMA_URL, backend: str = "ollama"):
        self.model = model
        self.char_id = char_id
        self.character = character
        self.personality = personality
        # `tactics` is the always-on guidance (exploration / conversation).
        # `combat_tactics` is added only when the combat controller calls
        # generate() — never appears in non-combat prompts. The agent itself
        # has no notion of mode; LLMPlayerController reads combat_tactics
        # off self and injects it into the nudge.
        self.tactics = tactics
        self.combat_tactics = combat_tactics
        self.world_state = world_state
        self.think = think
        self.show_thinking = show_thinking
        self.options = options or {}
        self.base_url = base_url
        self.backend = backend

    def _system_prompt(self) -> str:
        c = self.character
        tactics_section = (self.tactics.rstrip() + "\n\n") if self.tactics else ""
        return _SYSTEM_TEMPLATE.format(
            name=c.name, race=c.race, class_=c.class_, level=c.level,
            hp=c.hp, max_hp=c.max_hp, ac=c.ac,
            STR=c.stats.STR, DEX=c.stats.DEX, CON=c.stats.CON,
            INT=c.stats.INT, WIS=c.stats.WIS, CHA=c.stats.CHA,
            inventory="、".join(c.inventory) if c.inventory else "無",
            status="、".join(fx.name for fx in c.status_effects) if c.status_effects else "無",
            personality=self.personality,
            tactics_section=tactics_section,
        )

    def generate(self, nudge: str = "", on_chunk=None) -> str:
        """Generate action by reading from world_state.narrative_log (option A).

        nudge: optional one-shot user message appended after the log history,
               used for context-specific instructions like "輪到你行動，敵人..."
               or "如果保持沉默，輸出 [SILENT]". Not stored in log.

        Caller (game.py) is responsible for pushing the result back to the log
        when it should become part of the shared history.
        """
        history_msgs = render_messages(self.world_state, self.char_id)
        if not history_msgs and not nudge:
            history_msgs = [{"role": "user", "content": "（場景剛開始，請描述你的角色行動或感想）"}]

        # Character settings + any per-turn nudge go in a single user message at
        # the bottom — the LLM attends most strongly to the end of the context,
        # so identity/rules/current state arrive right where they matter most.
        bottom = self._system_prompt().rstrip()
        if nudge:
            bottom = bottom + "\n\n" + nudge
        messages = history_msgs + [{"role": "user", "content": bottom}]

        (_DEBUG_DIR / f"{self.character.name}_context.json").write_text(
            json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8", errors="replace"
        )

        def _on_chunk(chunk, thinking=False):
            if thinking:
                if self.show_thinking:
                    print(chunk, end="", flush=True)
                if on_chunk:
                    on_chunk(chunk, thinking=True)
            else:
                if on_chunk:
                    on_chunk(chunk, thinking=False)
                else:
                    print(chunk, end="", flush=True)

        full = stream_chat(
            self.base_url, self.model, messages, self.options,
            think=self.think, on_chunk=_on_chunk, backend=self.backend, timeout=120,
        )
        if not on_chunk:
            print()
        return full
