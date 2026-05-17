import json
import pathlib
from ..engine.world_state import WorldState
from .backend import stream_chat
from .log_render import render_messages

OLLAMA_URL = "http://localhost:11434"
_DEBUG_DIR = pathlib.Path(__file__).parent.parent / "debug"
_DEBUG_DIR.mkdir(exist_ok=True)

_COMBAT_SYSTEM = (
    "你是D&D遊戲主持人（GM），正在描述戰鬥中的單一事件。\n"
    "規則：\n"
    "1. 只用繁體中文輸出，禁止摻入任何其他語言的詞彙。\n"
    "2. 直接輸出一句話敘事（不超過50字），不可加「分析：」「機制：」欄位，不可寫 ---。\n"
    "3. 不要嵌入任何方括號標籤（如 [ATTACK]、[STATUS]、[END TURN] 等），結算已完成。\n"
    "4. 只描述「使用者訊息中明確發生的事件」，不要新增傷害、死亡、追加行動。\n"
    "5. 不要重複歷史內容，只描述當下這個事件。"
)

# GM persona + rules stay in system message.
# Per-turn state (room, chars, tag results) is injected at the END of the message list
# so it stays close to the model's attention window — recent context dominates generation.
_SYSTEM_BASE = """你是一位專業的D&D 5e地下城主（GM）。請用繁體中文進行沉浸式敘事。

## 你的職責
1. 根據「機制結算」中已發生的事件，撰寫流暢的繁體中文敘事（350字以內）
2. 自然地描述骰點成功/失敗的後果、移動後的場景氛圍、物品拾取的情境
3. 不替玩家做決定，只描述玩家面臨的情境與 NPC 的反應
4. NPC 的對話、表情和行為由你自由決定，豐富場景互動
5. 若無「機制結算」，則根據玩家行動推進劇情

## 強制規則（違反就是錯誤輸出）
- 「明顯可拾取物品」清單中**每一項都必須在敘事中被自然提及**（任務物品尤其重要，不可遺漏）
- 嚴禁編造任何未在「可見敵人」「可拾取物品」「NPC」「出口」清單中的角色、生物、暗影、聲音、視線、威脅或預兆
- 若「可見敵人：無」，房間就是真的安全，不要暗示有看不見的敵人
- 死亡角色（HP=0）不再出現在任何描述中
- 不替玩家做決定、不替玩家行動

直接輸出純繁體中文敘事，禁止輸出任何 [標籤] 或格式欄位。
"""

_GM_REMINDER = (
    "請以 GM 身份用繁體中文寫一段 350 字內的敘事，嚴守上方「強制規則」。\n"
    "直接輸出敘事，不要寫 [標籤]、「分析：」「機制：」等欄位。"
)


class GMAgent:
    def __init__(self, model: str, world_state: WorldState,
                 think: bool = False, show_thinking: bool = False,
                 options: dict = None,
                 base_url: str = OLLAMA_URL, backend: str = "ollama"):
        self.model = model
        self.world_state = world_state
        self.think = think
        self.show_thinking = show_thinking
        self.options = options or {}
        self.base_url = base_url
        self.backend = backend

    def _state_block(self) -> str:
        """Per-turn snapshot of room + visible chars. Injected at end of message list."""
        ws = self.world_state

        if ws.dungeon_map:
            room = ws.dungeon_map.current_room
            exits_str = "、".join(
                f"{d}（{ws.dungeon_map.rooms[rid].name}）"
                for d, rid in room.exits.items()
            ) or "無"
            room_enemies = room.alive_enemies(ws.characters)
            enemies_str = "、".join(
                f"{c.name} HP {c.hp}/{c.max_hp}" for c in room_enemies.values()
            ) or "無"
            loot_str = room.loot_state()
            location_section = (
                f"## 當前位置\n"
                f"房間：{room.name}\n"
                f"描述：{room.description}\n"
                f"出口：{exits_str}\n"
                f"可見敵人：{enemies_str}\n"
                f"明顯可拾取物品：\n{loot_str}"
            )
        else:
            location_section = f"## 當前場景\n{ws.scene}"

        current_room_hostile_ids: set[str] = set()
        if ws.dungeon_map:
            current_room_hostile_ids = {
                nid for nid in ws.dungeon_map.current_room.npc_ids
                if nid in ws.characters and ws.characters[nid].attitude == 0
            }

        status_lines = []
        for cid, char in ws.characters.items():
            if not char.is_alive():
                continue
            if ws.dungeon_map and char.is_npc and cid not in current_room_hostile_ids:
                continue
            effects = "、".join(fx.name for fx in char.status_effects) if char.status_effects else "無"
            status_lines.append(
                f"  {char.name}（{cid}）：HP {char.hp}/{char.max_hp}，AC {char.ac}，狀態 {effects}"
            )

        return location_section + "\n\n## 角色狀態\n" + "\n".join(status_lines)

    def generate(self, tag_results: list[str] | None = None,
                 on_chunk=None) -> str:
        """Generate GM narration from the unified narrative_log.

        tag_results highlights "this turn's" mechanical outcomes in the final
        user message (even though they're already in the log) so the model
        knows which events it needs to narrate now vs. recap.
        """
        ws = self.world_state

        history_msgs = render_messages(ws, "gm")
        if not history_msgs:
            history_msgs = [{"role": "user", "content": "（開場，請描述初始場景，引導玩家進入冒險）"}]

        final_parts = [f"## 當前狀態（即時）\n{self._state_block()}"]
        if tag_results:
            final_parts.append(
                "## 本回合機制結算（已執行，請在敘事中自然反映後果）\n"
                + "\n".join(f"- {r}" for r in tag_results)
            )
        final_parts.append(_GM_REMINDER)

        messages = (
            [{"role": "system", "content": _SYSTEM_BASE}]
            + history_msgs
            + [{"role": "user", "content": "\n\n".join(final_parts)}]
        )

        (_DEBUG_DIR / "gm_context.json").write_text(
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
            think=self.think, on_chunk=_on_chunk, backend=self.backend, timeout=180,
        )
        if not on_chunk:
            print()
        return full

    _COMBAT_USER_TEMPLATE = (
        "用一句（30字以內）繁體中文敘述此戰鬥結果，"
        "直接輸出敘事，不加格式欄位：\n{result_text}"
    )

    def combat_narrate(self, result_text: str, on_chunk=None) -> str:
        """Stateless short narration for a single combat event (≤50 chars).

        Caller passes the raw mechanical result_text (e.g. format_result output);
        this method composes the prompt template internally.
        """
        messages = [
            {"role": "system", "content": _COMBAT_SYSTEM},
            {"role": "user",   "content": self._COMBAT_USER_TEMPLATE.format(result_text=result_text)},
        ]
        (_DEBUG_DIR / "gm_combat_context.json").write_text(
            json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8", errors="replace"
        )
        opts = {**self.options, "num_predict": 200}
        return stream_chat(
            self.base_url, self.model, messages, opts,
            think=False, on_chunk=on_chunk, backend=self.backend, timeout=60,
        )
