import json
import requests
from ..engine.world_state import WorldState

OLLAMA_URL = "http://localhost:11434"

_SYSTEM_TEMPLATE = """你是一位專業的D&D 5e地下城主（GM）。請用繁體中文進行沉浸式敘事。

## 當前場景
{scene}

## 角色狀態
{char_status}

## 角色ID對照（標籤中使用ID，不用中文名）
{char_ids}

## 規則標籤（當需要觸發遊戲機制時，將標籤自然嵌入敘事中）
[INITIATIVE: start]              ← 開始戰鬥，擲先攻
[ATTACK: <攻擊者ID> -> <目標ID>]  ← 攻擊判定
[DAMAGE: <骰子式> to <目標ID>]    ← 造成傷害
[HEAL: <角色ID> <骰子式>]         ← 治療
[ROLL: <角色ID> <屬性> DC<數字>]  ← 技能或豁免檢定（屬性用 STR/DEX/CON/INT/WIS/CHA）
[STATUS: <角色ID> +/-<效果>]      ← 增加或移除狀態效果

範例敘事：
「地精揮刀衝向艾里亞！[ATTACK: goblin_1 -> aria] 索爾大喊著攔截[DAMAGE: 1d6+3 to goblin_1]」

## 注意事項
- 每次回應控制在150字以內，保持節奏緊湊
- 遊戲機制標籤要自然融入敘事
- NPC由你扮演，玩家角色只描述他們面臨的情境，不替玩家做決定
- 死亡的角色（HP=0）從戰鬥中移除
"""


class GMAgent:
    def __init__(self, model: str, world_state: WorldState):
        self.model = model
        self.world_state = world_state
        self.history: list[dict] = []

    def _system_prompt(self) -> str:
        ws = self.world_state
        status_lines = []
        for cid, char in ws.characters.items():
            alive = "存活" if char.is_alive() else "已倒下"
            effects = "、".join(char.status_effects) if char.status_effects else "無"
            status_lines.append(
                f"  {char.name}（{cid}）：HP {char.hp}/{char.max_hp}，AC {char.ac}，狀態 {effects}，{alive}"
            )
        id_lines = [
            f"  {cid} = {char.name}" for cid, char in ws.characters.items()
        ]
        return _SYSTEM_TEMPLATE.format(
            scene=ws.scene,
            char_status="\n".join(status_lines),
            char_ids="\n".join(id_lines),
        )

    def generate(self, player_actions: list[str], stream: bool = True) -> str:
        """Generate GM narration. player_actions is empty on the opening turn."""
        if player_actions:
            content = "玩家行動：\n" + "\n".join(f"• {a}" for a in player_actions)
        else:
            content = "（開場，請描述初始場景，引導玩家進入冒險）"

        self.history.append({"role": "user", "content": content})
        messages = [{"role": "system", "content": self._system_prompt()}] + self.history

        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": self.model, "messages": messages, "stream": stream},
            stream=stream,
        )
        resp.raise_for_status()

        full = ""
        if stream:
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                chunk = data.get("message", {}).get("content", "")
                full += chunk
                print(chunk, end="", flush=True)
                if data.get("done"):
                    break
            print()
        else:
            full = resp.json().get("message", {}).get("content", "")

        self.history.append({"role": "assistant", "content": full})
        return full
