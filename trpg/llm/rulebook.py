COMBAT_RULEBOOK = """
# 戰鬥行動規則

## 每回合資源
- 1 動作（action）：攻擊、使用道具、技能檢定、躲藏等
- 1 移動（movement）：最多 9 公尺，可在行動前後分割
- 1 附贈動作（bonus_action）：特定技能才有

## 武器傷害
傷害骰由角色裝備決定，只需填 "weapon" 欄位的武器名稱，引擎自動查詢。

## 技能→屬性對照
STR：運動、舉重
DEX：潛行、巧手、特技、閃避
CON：體力耐受
INT：奧秘、歷史、調查、自然
WIS：感知、洞察、醫療、求生
CHA：說服、恐嚇、欺騙、表演

## 行動 JSON 格式

### 攻擊（消耗：action）
玩家描述包含：攻擊、刺、砍、打、劈、射擊、斬等
```json
{
  "valid": true,
  "type": "ATTACK",
  "attacker": "<角色ID>",
  "target": "<目標ID>",
  "weapon": "<武器名，如 短劍 / 長劍 / 匕首，留空則用第一件武器>",
  "consumes": ["action"]
}
```

### 投擲物範圍攻擊（消耗：action + 投擲物）
玩家描述包含：扔、投擲、丟出、火瓶、毒煙彈等
```json
{
  "valid": true,
  "type": "AOE",
  "attacker": "<攻擊者ID>",
  "targets": ["<目標ID1>", "<目標ID2>"],
  "item": "<投擲物名稱，如 火瓶 / 毒煙彈>",
  "damage_dice": "<骰子式，查角色道具>",
  "save_stat": "DEX",
  "save_dc": 13,
  "half_on_save": true,
  "consumes": ["action"]
}
```

### 使用消耗品（消耗：action）
玩家描述包含：使用急救包、喝藥、點火把等
```json
{
  "valid": true,
  "type": "USE_ITEM",
  "character": "<使用者ID>",
  "item": "<道具名稱，如 急救包 / 火把>",
  "target": "<治療目標ID，治療自己填使用者ID>",
  "consumes": ["action"]
}
```

### 技能/屬性檢定（消耗：action）
玩家描述包含非攻擊的主動行為，如：潛行、說服、察覺、尋找等
DC 預設 12，若描述有明確難度可調整
```json
{
  "valid": true,
  "type": "ROLL",
  "character": "<角色ID>",
  "stat": "STR|DEX|CON|INT|WIS|CHA",
  "dc": 12,
  "skill_description": "<技能描述>",
  "consumes": ["action"]
}
```

### 移動（不消耗 action）
玩家描述包含移動位置，但沒有攻擊/技能行動時才單獨輸出
```json
{
  "valid": true,
  "type": "MOVE",
  "character": "<角色ID>",
  "description": "<移動方向或目標位置>",
  "consumes": ["movement"]
}
```

### 不合法
資源已用完、目標不存在或已死亡、行動超出規則範圍
```json
{
  "valid": false,
  "reason": "<說明原因>",
  "suggestion": "<建議合法替代行動>"
}
```

## 重要限制
- 每回合只能攻擊一次
- 不能攻擊 HP=0 的目標
- 只輸出 JSON，不要任何解釋文字
"""
