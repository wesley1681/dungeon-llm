import sys
import json
import requests

sys.stdout.reconfigure(encoding="utf-8")

# ── 調整這裡 ──────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
MODEL      = "gemma4-abliterix:latest"

THINK        = False  # 是否啟用思考模式
SHOW_THINKING = False  # 是否顯示思考內容

OPTIONS = {
    "temperature": 0.8,
    "num_predict": 1500,   # 思考 + 回答的 token 上限，-1 = 無限制
    # "num_ctx": 4096,     # context 視窗大小
}

USER_MESSAGE = "（開場，請描述初始場景，引導玩家進入冒險）"

# 可以在這裡加入 few-shot 示例或對話歷史來測試不同情境
HISTORY: list[dict] = [
    # {"role": "user",      "content": "..."},
    # {"role": "assistant", "content": "..."},
]

SYSTEM_PROMPT = """你是一位專業的D&D 5e地下城主（GM）。請用繁體中文進行沉浸式敘事。

## 當前場景
你們是一對受僱的冒險者，任務是從地精手中奪回小鎮鎮長的護身符。線索指向鎮外廢棄的石造地下城。

火把的光芒在潮濕的石壁上搖曳，眼前是一條向下延伸的台階走廊，空氣中瀰漫著霉味和隱約的腐臭氣息。某處傳來地精粗嘎的笑聲。

## 角色狀態
  艾里亞（aria）：HP 22/22，AC 14，狀態 無，存活
  索爾（thor）：HP 31/31，AC 16，狀態 無，存活
  地精甲（goblin_1）：HP 7/7，AC 13，狀態 無，存活
  地精乙（goblin_2）：HP 7/7，AC 13，狀態 無，存活
  地精丙（goblin_3）：HP 5/5，AC 12，狀態 無，存活

## 角色ID對照（標籤中使用ID，不用中文名）
  aria = 艾里亞
  thor = 索爾
  goblin_1 = 地精甲
  goblin_2 = 地精乙
  goblin_3 = 地精丙

## 規則標籤（當需要觸發遊戲機制時，將標籤自然嵌入敘事中）
[INITIATIVE: start]              ← 開始戰鬥，擲先攻
[ATTACK: <攻擊者ID> -> <目標ID>]  ← 攻擊判定
[DAMAGE: <骰子式> to <目標ID>]    ← 造成傷害
[HEAL: <角色ID> <骰子式>]         ← 治療
[ROLL: <角色ID> <屬性> DC<數字>]  ← 技能或豁免檢定（屬性只能用下方對照表的縮寫）
[STATUS: <角色ID> +/-<效果>]      ← 增加或移除狀態效果

## 技能→屬性對照（ROLL 標籤只能用縮寫，不能用技能名稱）
STR：運動
DEX：特技、巧手、潛行
CON：體質豁免
INT：奧秘、歷史、調查、自然、宗教
WIS：馴獸、洞察、醫療、感知、求生
CHA：欺騙、恐嚇、表演、說服

範例：
感知檢定 → [ROLL: aria WIS DC12]
潛行檢定 → [ROLL: thor DEX DC14]
說服檢定 → [ROLL: aria CHA DC15]

範例敘事：
「地精揮刀衝向艾里亞！[ATTACK: goblin_1 -> aria] 索爾大喊著攔截[DAMAGE: 1d6+3 to goblin_1]」

## 注意事項
- 每次回應控制在350字以內，敘事豐富但節奏緊湊
- 遊戲機制標籤要自然融入敘事
- NPC由你扮演，玩家角色只描述他們面臨的情境，不替玩家做決定
- 死亡的角色（HP=0）從戰鬥中移除

## 輸出格式（每次回應必須嚴格遵守此格式，不得省略任何欄位）

分析：{
三行，描述玩家行動與當前情境
}
機制：（一行，列出需要的標籤，或寫「無」）
---
（繁體中文敘事，150字以內，自然嵌入規則標籤）
"""
# ─────────────────────────────────────────────────────────────────────────────

messages = (
    [{"role": "system", "content": SYSTEM_PROMPT}]
    + HISTORY
    + [{"role": "user", "content": USER_MESSAGE}]
)

print("=" * 60)
print("SYSTEM PROMPT")
print("=" * 60)
print(SYSTEM_PROMPT)
print("=" * 60)
print(f"USER: {USER_MESSAGE}")
print("=" * 60)
print(f"think={THINK}  show_thinking={SHOW_THINKING}  options={OPTIONS}")
print("=" * 60)
print("RESPONSE")
print("=" * 60)

resp = requests.post(
    f"{OLLAMA_URL}/api/chat",
    json={
        "model":   MODEL,
        "messages": messages,
        "stream":   True,
        "think":    THINK,
        "options":  OPTIONS,
    },
    stream=True,
    timeout=180,
)
resp.raise_for_status()

in_thinking = False
for line in resp.iter_lines():
    if not line:
        continue
    data = json.loads(line)
    msg = data.get("message", {})

    if msg.get("thinking"):
        if not in_thinking:
            in_thinking = True
            if SHOW_THINKING:
                print("\n[思考中]\n", flush=True)
        if SHOW_THINKING:
            print(msg["thinking"], end="", flush=True)

    elif msg.get("content"):
        if in_thinking:
            in_thinking = False
            if SHOW_THINKING:
                print("\n\n[回答]\n", flush=True)
        print(msg["content"], end="", flush=True)

    if data.get("done"):
        break

print()
