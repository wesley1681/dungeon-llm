import json
import pathlib
from ..engine.world_state import WorldState
from .backend import stream_chat, complete_chat

OLLAMA_URL = "http://localhost:11434"
_DEBUG_DIR = pathlib.Path(__file__).parent.parent / "debug"
_DEBUG_DIR.mkdir(exist_ok=True)

_COMBAT_SYSTEM = "你是D&D遊戲主持人（GM）。用繁體中文描述戰鬥事件，直接輸出敘事，不加分析欄位或---分隔線。"

_TURN_REMINDER = """---
【輪到你了，請以地下城主（GM）身份回應】

核心規則（每次都適用）：
1. 標籤只在 --- 後的敘事段才會被系統執行，機制行不執行
2. 玩家移動 → 敘事段必須包含 [TRAVEL: north/south/east/west]
3. 需要擲骰 → 直接在敘事段嵌入 [ROLL: aria DEX DC14]，不要叫玩家自己擲
4. 進入有敵人的新房間才使用 [INITIATIVE: start]
5. 只描述已知出口，不創造新房間

輸出格式（必須完整）：
分析：（你的判斷與計畫）
機制：（將使用的標籤名稱，僅備忘）
---
（繁體中文敘事，350字以內，在此嵌入執行標籤）"""

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
- 玩家說「往北走」「進入東邊的門」等 → 必須在 --- 後的敘事中嵌入 [TRAVEL: north]（或 south/east/west）
- 格式必須完整：[TRAVEL: north]，不能只寫 [TRAVEL]
- 【嚴格限制】只能往「當前位置」欄位列出的出口方向移動，禁止描述或創造任何未列出的房間、走廊、岔路
- 若玩家想往沒有出口的方向走，告知「這個方向沒有出口」
- 進入新房間後，依照系統提供的房間描述敘述，不要自行添加房間內容

## 注意事項
- 每次回應控制在500字以內，敘事豐富但節奏緊湊
- 標籤要自然融入敘事，不要孤立列出
- NPC由你扮演，玩家角色只描述他們面臨的情境，不替玩家做決定
- 死亡的角色（HP=0）從戰鬥中移除

## 輸出格式（每次回應必須嚴格遵守此格式，不得省略任何欄位）

分析：（至少十行，先確認當前劇情、玩家意圖、在場人物、應該如何推動劇情）
機制：（一行，僅列標籤名稱供人閱讀，此行不會被執行）
---
（繁體中文敘事，350字以內）
注意：[TRAVEL:] [INITIATIVE:] 等所有標籤必須寫在此敘事段才會生效。機制行只是備忘，寫在那裡不會有任何效果。
"""


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
        # When a dungeon map is active, only show NPCs that are in the current room;
        # enemies in other rooms are unknown to the party and must not appear here.
        current_room_enemy_ids: set[str] = set()
        if ws.dungeon_map:
            current_room_enemy_ids = set(ws.dungeon_map.current_room.enemy_ids)

        status_lines = []
        for cid, char in ws.characters.items():
            if not char.is_alive():
                continue
            if ws.dungeon_map and char.is_npc and cid not in current_room_enemy_ids:
                continue  # enemy in a different room — don't reveal to GM
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

    def generate(self, player_actions: list[str], on_chunk=None) -> str:
        """Generate GM narration. on_chunk(chunk, thinking) called per token if provided."""
        if player_actions:
            content = "玩家行動：\n" + "\n".join(f"• {a}" for a in player_actions)
        else:
            content = "（開場，請描述初始場景，引導玩家進入冒險）"

        self.history.append({"role": "user", "content": content})
        # _TURN_REMINDER is appended every call but never stored in history,
        # so the model sees it fresh each time without it accumulating.
        messages = (
            [{"role": "system", "content": self._system_prompt()}]
            + self.history
            + [{"role": "user", "content": _TURN_REMINDER}]
        )
        (_DEBUG_DIR / "gm_context.json").write_text(
            json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8"
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

        self.history.append({"role": "assistant", "content": full})
        return full

    def combat_narrate(self, prompt: str, on_chunk=None) -> str:
        """Lightweight GM call for in-combat narration. No format enforced."""
        self.history.append({"role": "user", "content": prompt})
        messages = [{"role": "system", "content": _COMBAT_SYSTEM}] + self.history
        (_DEBUG_DIR / "gm_context.json").write_text(
            json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        full = stream_chat(
            self.base_url, self.model, messages, self.options,
            think=False, on_chunk=on_chunk, backend=self.backend, timeout=120,
        )
        self.history.append({"role": "assistant", "content": full})
        return full

    def notify_results(self, tag_results: list[str]) -> None:
        """Inject mechanical resolution results back into GM history so it knows what happened."""
        summary = "【規則結算結果】\n" + "\n".join(f"- {r}" for r in tag_results)
        self.history.append({"role": "user", "content": summary})
