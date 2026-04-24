import json
import pathlib
import requests
from ..engine.world_state import WorldState

OLLAMA_URL = "http://localhost:11434"
_DEBUG_DIR = pathlib.Path(__file__).parent.parent / "debug"
_DEBUG_DIR.mkdir(exist_ok=True)

_COMBAT_SYSTEM = "你是D&D遊戲主持人（GM）。用繁體中文描述戰鬥事件，直接輸出敘事，不加分析欄位或---分隔線。"

_SYSTEM_TEMPLATE = """你是一位專業的D&D 5e地下城主（GM）。請用繁體中文進行沉浸式敘事。

{location_section}

## 角色狀態
{char_status}

## 角色ID對照（標籤中使用ID，不用中文名）
{char_ids}

## 規則標籤（當需要觸發遊戲機制時，將標籤自然嵌入敘事中）
[INITIATIVE: start]                    ← 開始戰鬥，擲先攻（僅在進入有敵人的房間或敵人突然出現時使用）
[TRAVEL: <方向>]                       ← 玩家移動到相鄰房間，方向如 north/south/east/west
[PICKUP: <角色ID> <物品名稱>]           ← 角色從當前房間拾取物品
[ATTACK: <攻擊者ID> -> <目標ID>]        ← 攻擊判定
[DAMAGE: <骰子式> to <目標ID>]          ← 造成傷害
[HEAL: <角色ID> <骰子式>]              ← 治療
[ROLL: <角色ID> <屬性> DC<數字>]        ← 技能或豁免檢定
[STATUS: <角色ID> +/-<效果>]           ← 增加或移除狀態效果

## 技能→屬性對照（ROLL 標籤只能用縮寫）
STR：運動  DEX：特技、巧手、潛行  CON：體質豁免
INT：奧秘、歷史、調查、自然  WIS：洞察、醫療、感知、求生  CHA：欺騙、恐嚇、說服

## 移動規則
- 玩家說「往北走」「進入東邊的門」等 → 輸出 [TRAVEL: north]（或 south/east/west）
- 只能往有出口的方向移動，無效方向告知玩家
- 進入新房間後描述房間內容，若有物品可提示拾取

## 注意事項
- 每次回應控制在350字以內，敘事豐富但節奏緊湊
- 標籤要自然融入敘事，不要孤立列出
- NPC由你扮演，玩家角色只描述他們面臨的情境，不替玩家做決定
- 死亡的角色（HP=0）從戰鬥中移除

## 輸出格式（每次回應必須嚴格遵守此格式，不得省略任何欄位）

分析：（三行以內，描述玩家行動與當前情境）
機制：（一行，列出需要的標籤，或寫「無」）
---
（繁體中文敘事，350字以內，自然嵌入規則標籤）
"""


class GMAgent:
    def __init__(self, model: str, world_state: WorldState,
                 think: bool = False, show_thinking: bool = False,
                 options: dict = None):
        self.model = model
        self.world_state = world_state
        self.think = think
        self.show_thinking = show_thinking
        self.options = options or {}
        self.history: list[dict] = []

    def _system_prompt(self) -> str:
        ws = self.world_state

        # ── Location section ──────────────────────────────────────────────────
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
            loot_str = "、".join(room.loot_names()) or "無"
            location_section = (
                f"## 當前位置\n"
                f"房間：{room.name}\n"
                f"描述：{room.description}\n"
                f"出口：{exits_str}\n"
                f"可見敵人：{enemies_str}\n"
                f"可拾取物品：{loot_str}"
            )
        else:
            location_section = f"## 當前場景\n{ws.scene}"

        # ── Character status ──────────────────────────────────────────────────
        status_lines = []
        for cid, char in ws.characters.items():
            if not char.is_alive():
                continue
            effects = "、".join(char.status_effects) if char.status_effects else "無"
            status_lines.append(
                f"  {char.name}（{cid}）：HP {char.hp}/{char.max_hp}，AC {char.ac}，狀態 {effects}"
            )
        id_lines = [f"  {cid} = {char.name}" for cid, char in ws.characters.items()]

        return _SYSTEM_TEMPLATE.format(
            location_section=location_section,
            char_status="\n".join(status_lines),
            char_ids="\n".join(id_lines),
        )

    def generate(self, player_actions: list[str], on_chunk=None, stream: bool = True) -> str:
        """Generate GM narration. on_chunk(chunk, thinking) called per token if provided."""
        if player_actions:
            content = "玩家行動：\n" + "\n".join(f"• {a}" for a in player_actions)
        else:
            content = "（開場，請描述初始場景，引導玩家進入冒險）"

        self.history.append({"role": "user", "content": content})
        messages = [{"role": "system", "content": self._system_prompt()}] + self.history
        (_DEBUG_DIR / "gm_context.json").write_text(
            json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": self.model, "messages": messages, "stream": stream, "think": self.think, "options": self.options},
            stream=stream,
            timeout=180,
        )
        resp.raise_for_status()

        full = ""
        in_thinking = False
        if stream:
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
                        elif self.show_thinking:
                            print("\n[GM 思考中]\n", flush=True)
                    chunk = msg["thinking"]
                    if on_chunk:
                        on_chunk(chunk, thinking=True)
                    elif self.show_thinking:
                        print(chunk, end="", flush=True)
                elif msg.get("content"):
                    if in_thinking:
                        in_thinking = False
                        if on_chunk:
                            on_chunk("\n[回答]\n", thinking=True)
                        elif self.show_thinking:
                            print("\n\n[GM 回答]\n", flush=True)
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
        else:
            full = resp.json().get("message", {}).get("content", "")

        self.history.append({"role": "assistant", "content": full})
        return full

    def combat_narrate(self, prompt: str, on_chunk=None) -> str:
        """Lightweight GM call for in-combat narration. No format enforced."""
        self.history.append({"role": "user", "content": prompt})
        messages = [{"role": "system", "content": _COMBAT_SYSTEM}] + self.history
        (_DEBUG_DIR / "gm_context.json").write_text(
            json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": self.model, "messages": messages,
                  "stream": True, "think": False, "options": self.options},
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()

        full = ""
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            chunk = data.get("message", {}).get("content", "")
            if chunk:
                full += chunk
                if on_chunk:
                    on_chunk(chunk, thinking=False)
            if data.get("done"):
                break

        self.history.append({"role": "assistant", "content": full})
        return full

    def notify_results(self, tag_results: list[str]) -> None:
        """Inject mechanical resolution results back into GM history so it knows what happened."""
        summary = "【規則結算結果】\n" + "\n".join(f"- {r}" for r in tag_results)
        self.history.append({"role": "user", "content": summary})
