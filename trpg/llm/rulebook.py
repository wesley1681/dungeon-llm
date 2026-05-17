COMBAT_RULEBOOK = """# 戰鬥行動規則

把使用者的自然語言行動翻譯成 JSON 交給引擎執行。**只輸出 JSON，不要解釋**。

距離、射程、傷害、資源扣除全由引擎自動處理，不要在 JSON 裡計算或顯式寫出。
若引擎判定無法執行（超出射程、目標已倒下、資源用完等）會回報簡短原因，
你不必預先檢查，照玩家描述如實翻譯即可。

## 攻擊
玩家描述：攻擊、刺、砍、打、劈、射擊、斬等
```json
{
  "valid": true,
  "type": "ATTACK",
  "attacker": "<角色ID>",
  "target": "<目標ID>",
  "weapon": "<武器名；徒手填 無武器；留空用第一件>"
}
```

## 投擲（AOE）
玩家描述：扔、投擲、丟出、火瓶、毒煙彈等
```json
{
  "valid": true,
  "type": "AOE",
  "attacker": "<角色ID>",
  "targets": ["<目標ID1>", "<目標ID2>"],
  "item": "<投擲物名稱>",
  "save_stat": "DEX",
  "save_dc": 13,
  "half_on_save": true
}
```

## 使用消耗品
玩家描述：使用急救包、喝藥、點火把等
```json
{
  "valid": true,
  "type": "USE_ITEM",
  "character": "<使用者ID>",
  "item": "<道具名稱>",
  "target": "<治療目標ID，治療自己填使用者ID>"
}
```

## 技能/屬性檢定
玩家描述非攻擊的主動檢定，如：用感知找弱點、潛行繞後、用恐嚇喊話等
DC 預設 12，描述若有明確難度可調整。
技能對應屬性：STR=力量(運動)、DEX=敏捷(潛行/特技/閃避)、CON=體質(耐受)、
INT=智力(調查/奧秘)、WIS=感知(察覺/醫療/洞察)、CHA=魅力(說服/恐嚇/欺騙)
```json
{
  "valid": true,
  "type": "ROLL",
  "character": "<角色ID>",
  "stat": "STR|DEX|CON|INT|WIS|CHA",
  "dc": 12,
  "skill_description": "<技能描述>"
}
```

## 移動
玩家描述：衝向、靠近、後退、拉開距離、追上去、退到後排等

用語意方向：`direction` 取 "advance"（朝對立陣營靠近）或 "retreat"（遠離對立陣營）。
`distance` 預設 9（單回合上限），可更小（小步靠近）。
```json
{
  "valid": true,
  "type": "MOVE",
  "character": "<角色ID>",
  "direction": "advance",
  "distance": 9,
  "description": "<簡短描述>"
}
```

## 閃避
玩家描述：我閃避、專注防禦、舉盾防禦等
效果：下次自己回合開始前，攻擊本角色者擲劣勢
```json
{
  "valid": true,
  "type": "DODGE",
  "character": "<角色ID>"
}
```

## 躲藏
玩家描述：躲到XX後面、藏起來、潛伏等
擲 DEX DC12 豁免
```json
{
  "valid": true,
  "type": "HIDE",
  "character": "<角色ID>"
}
```

## 法術
玩家描述：施展、念咒、施法、扔火球等
範圍法術圓心可用兩種方式指定（擇一）：
  - `target`：角色 ID（圓心 = 該角色位置）或 `"self"`（圓心 = 施法者）。對應「我對地精甲扔火球」這類描述。
  - `target_position`：明確公尺座標 float。對應「我把火球扔到 6m 處」「我把火球扔到敵人和我之間」這類描述——
    玩家想避免誤傷隊友／自己時會用這個。可以從情況區的位置資訊推算座標。
若兩者都給，`target_position` 優先；都沒給則無法施法。
`slot_level` 可省略，引擎會自動挑選夠用且仍有的最低環法術位；玩家若明確指定升環施法可填入。
**AOE 法術警告**：範圍法術以圓心為中心、半徑內**所有**角色都會受影響，**包含施法者本身與隊友**——
火球術不會自動跳過自己人。施法者要自行考量站位與圓心位置，否則會誤傷盟友或自爆。
```json
{
  "valid": true,
  "type": "SPELL",
  "caster": "<施法者ID>",
  "spell_name": "<法術名稱>",
  "target": "<目標ID 或 \"self\"；不用時填 null>",
  "target_position": null,
  "slot_level": null
}
```

## 無法解析
若無法判斷玩家想做什麼：
```json
{
  "valid": false,
  "reason": "<簡短原因>",
  "suggestion": "<建議改寫>"
}
```
"""
