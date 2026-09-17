import json
import pathlib
from ..engine.world_state import WorldState
from ..engine.quests import objective_progress_str
from .backend import stream_chat

_DEBUG_DIR = pathlib.Path(__file__).parent.parent / "debug"

_SYSTEM = """你是D&D 5e規則裁判。根據玩家行動，決定並輸出需要執行的遊戲標籤。

## 當前狀態
{state_section}

## 角色（標籤中必須使用ID，不用中文名）
{char_info}

## 標籤語法、範例與觸發條件

[TRAVEL: <方向>]
  ⚠ 只填方向，絕對不要加角色ID
  正確：[TRAVEL: north]　[TRAVEL: east]
  錯誤：[TRAVEL: kaine east]　[TRAVEL: thor north]
  觸發：玩家明確表示*現在*移動（「我決定往北走」「進入東邊的門」「立刻前進」「衝過去」）
  不觸發：「那等等往南走」「北邊有動靜」等計劃、臆測類說詞、或是玩家意見矛盾(一個要去，一個想停留)
  限制：只能使用「當前出口」列出的方向，沒有該方向則不輸出

[PICKUP: <角色ID> <物品名>]
  觸發：玩家明確說「撿起」「拾取」，且物品在「可拾取物品」清單中
  例：[PICKUP: kaine 治療藥水]　[PICKUP: thor 火把]
  已開啟容器（如「鐵箱（已開啟）內：…」）裡的物品也可直接 PICKUP，
  直接用物品本身名字（不用寫容器名）。
  批次撿拾：若玩家說「撿拾全部」「拿光」「撿拾寶箱內的物品」「都拿走」，
  為對應的多個物品各輸出一行 PICKUP（每行一個）。
  例（玩家說「撿拾寶箱內的物品」，鐵箱已開啟內含治療藥水、急救包、金幣 30 枚）：
    [PICKUP: kaine 治療藥水]
    [PICKUP: kaine 急救包]
    [PICKUP: kaine 金幣 30 枚]

[GIVE: <給的人ID> <收的人ID> <物品名>]
  觸發：玩家把自己的武器或道具**交給隊友 / NPC**（「把手斧給老柯」「分一個急救包給索爾」「我把短劍丟給凱恩」）
  例：[GIVE: thor civilian 手斧]　[GIVE: kaine thor 急救包]
  ⚠ 兩個角色必須在同一房間且都還活著
  ⚠ 物品名必須是給的人現在身上有的東西

[CONSUME: <角色ID> <道具名>]
  觸發：玩家使用消耗品（喝藥水、用急救包等）
  例：[CONSUME: kaine 治療藥水]
  ⚠ 只需輸出 CONSUME；治療、點燈等效果由引擎依道具資料自動結算，不要再開別的標籤

[UNLOCK: <角色ID> <物件名> <屬性>]
  觸發：玩家嘗試解鎖任何有鎖的物件（寶箱、門、手銬等）
  屬性選擇：
    DEX ← 精巧操作（盜賊工具撬鎖、髮夾、精細手法），用物件原始 DC
    STR ← 蠻力破開（撞門、掰鎖、暴力拆除），DC 額外 +3
  系統內部自動擲骰，輸出成功/失敗結果，成功時直接展示內容物
  ⚠ UNLOCK 自己處理骰子，不要另外要求擲骰
  例：
    玩家說「我用盜賊工具撬開鐵箱」→ [UNLOCK: kaine 鐵箱 DEX]
    玩家說「我用蠻力砸開鐵箱」    → [UNLOCK: kaine 鐵箱 STR]
    玩家說「索爾踢開上鎖的門」    → [UNLOCK: thor 石門 STR]
    玩家說「幫我解開手銬」        → [UNLOCK: kaine 手銬 DEX]

[SEARCH: <角色ID> [屬性]]
  觸發：玩家明確說「搜索」「找找看」「翻翻雜物」「仔細看」這類**主動找隱藏物品**的行動
  屬性可選 INT（調查、檢視）或 WIS（感知、留意），預設 INT
  系統自動擲骰並揭露房內隱藏物品；不要另外要求擲骰
  例：[SEARCH: kaine]　[SEARCH: kaine WIS]　[SEARCH: thor INT]
  ⚠ 玩家只是「環顧四周」「我看看」這類被動觀察不要輸出 SEARCH，輸出無

[TALK: <npc_id>]
  觸發：玩家試圖與「可對話 NPC」或「房內已倒下」名單上的人交談
  npc_id 必須是上述兩個名單中的 ID（不在名單上的不輸出）
  例：[TALK: civilian]

[ATTACK_NPC: <npc_id>]
  觸發：玩家明確攻擊「可對話 NPC」或「房內已倒下」名單上的人
  ⚠ 此標籤只用於這兩個名單，攻擊敵人不用此標籤（戰鬥系統自動處理）
  ⚠ 與 TALK 互斥：玩家若攻擊，不要同時輸出 TALK，只輸出 ATTACK_NPC
  效果：NPC 轉為敵人並立刻進入戰鬥
  例：[ATTACK_NPC: civilian]

## 輸出規則
- 每行一個標籤
- 若本回合無需執行任何標籤，輸出「無」
- 禁止輸出任何敘事、說明或分析文字
"""


class TagAgent:
    def __init__(self, model: str, world_state: WorldState,
                 base_url: str, backend: str, options: dict = None, api_key: str = None):
        self.model = model
        self.world_state = world_state
        self.base_url = base_url
        self.backend = backend
        self.options = options or {}
        self.api_key = api_key

    def _system_prompt(self) -> str:
        ws = self.world_state

        if ws.dungeon_map:
            room = ws.dungeon_map.current_room
            exits_str = "、".join(
                f"{d}（{ws.dungeon_map.rooms[rid].name}）"
                for d, rid in room.exits.items()
            ) or "無"
            loot_str = room.loot_state()
            talk_parts, hostile_parts, fallen_parts = [], [], []
            for nid in room.npc_ids:
                if nid not in ws.characters:
                    continue
                c = ws.characters[nid]
                desc = f"{c.name}（{nid}）"
                if not c.is_alive():
                    fallen_parts.append(desc)
                elif c.attitude == 0:
                    hostile_parts.append(desc)
                else:
                    talk_parts.append(desc)
            state_section = (
                f"房間：{room.name}\n"
                f"出口：{exits_str}\n"
                f"可拾取物品：\n{loot_str}\n"
                f"可對話 NPC：{'、'.join(talk_parts) or '無'}\n"
                f"敵對 NPC（戰鬥系統處理，不要對其用 TALK/ATTACK_NPC）：{'、'.join(hostile_parts) or '無'}\n"
                f"房內已倒下：{'、'.join(fallen_parts) or '無'}（嘗試與其互動仍輸出對應標籤，系統會回報錯誤）"
            )
        else:
            state_section = f"場景：{ws.scene}"

        # Append quest status (active and completed only — what the player can act on)
        quest_lines = []
        for q in ws.quests.values():
            if q.status == "active":
                prog = objective_progress_str(q, ws)
                quest_lines.append(f"進行中：{q.title}（{prog}）" if prog else f"進行中：{q.title}")
            elif q.status == "completed":
                quest_lines.append(f"可回報：{q.title}（去找 {ws.characters[q.giver_id].name} 領獎）")
        if quest_lines:
            state_section += "\n任務狀態：" + "；".join(quest_lines)

        char_lines = []
        for cid, char in ws.characters.items():
            if char.is_npc or not char.is_alive():
                continue
            usable = [
                f"{c.name}×{c.quantity}" if c.quantity > 1 else c.name
                for c in char.consumables if c.quantity > 0
            ]
            inv = "、".join(usable) or "無"
            char_lines.append(f"{cid} = {char.name}　HP {char.hp}/{char.max_hp}　道具：{inv}")

        return _SYSTEM.format(
            state_section=state_section,
            char_info="\n".join(char_lines),
        )

    def generate_tags(self, player_actions: list[str], on_chunk=None) -> str:
        """Decide which tags to execute for the given player actions.

        Returns raw text (one tag per line, or '無').
        Stateless — no history, no side effects beyond debug file writes.
        """
        if not player_actions:
            return "無"

        content = "玩家行動：\n" + "\n".join(f"• {a}" for a in player_actions)
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": content},
        ]
        (_DEBUG_DIR / "tag_agent_context.json").write_text(
            json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8", errors="replace"
        )

        opts = {**self.options, "num_predict": 200, "temperature": 0.2}

        def _on_chunk(chunk, thinking=False):
            if not thinking:
                if on_chunk:
                    on_chunk(chunk, thinking=False)
                else:
                    print(chunk, end="", flush=True)

        result = stream_chat(
            self.base_url, self.model, messages, opts,
            think=False, on_chunk=_on_chunk, backend=self.backend, timeout=30,
            api_key=self.api_key,
        )
        if not on_chunk:
            print()

        (_DEBUG_DIR / "tag_agent_output.txt").write_text(result, encoding="utf-8", errors="replace")
        return result
