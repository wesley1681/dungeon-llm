# Jev 請求速查手冊（Python）

本文件供本專案的 coding agent 快速查閱。若涉及 SDK 版本、限制或進階模式，再讀根目錄 `SKILL.md` 與 [TypeSafe 官方文件](https://docs.typesafe.ai/llms.txt)。本專案使用 `typesafe-sdk>=0.7.2,<1`。

## 核心模型

```python
response = client.system_one(
    model="jev-latest",
    state=state,
    questions=questions,
)
```

- `state`：模型判斷所需的資料；使用可轉成 JSON 的字串、dict 或 list。
- `questions`：問題 ID 到 typed question 的映射。
- 問題 ID 不會傳給模型；完整意思必須寫在 `instructions`／`criteria`。
- Jev 負責語意判斷；Python 負責規則、門檻、合法性、計算與實際執行。

## 三種問題

| 需求 | 類型 | 主要答案 |
| --- | --- | --- |
| 是否成立 | `Noul` | `.noul`：答案為「是」的機率 |
| 固定選項擇一 | `Choice` | `.choice`、`.probabilities`、`.confidence` |
| 有序程度 | `Score` | `.score`、`.probabilities`、`.confidence`、`.legend` |

### Noul

只用於是非題。`0.5` 代表 yes/no 接近各半，**不是中等程度**；Noul 沒有獨立的 confidence。

```python
Noul(
    instructions="`message.text` 是否明確要求退款？",
    criteria={
        "true": "要求退回款項或撤銷收費。",
        "false": "只是抱怨、詢問，或沒有要求退款。",
    },
)
```

### Choice

用於互相比較的固定選項。可能沒有符合項目時加入 `none`／`other`；模型不能選擇未提供的候選。

```python
Choice(
    instructions="`message.text` 的主要意圖是哪一項？",
    criteria={
        "refund": "要求退款。",
        "exchange": "要求換貨。",
        "information": "只詢問資訊。",
        "none": "以上皆非。",
    },
)
```

### Score

用於可描述成階梯的程度。`criteria` 由低到高排列，索引從 0 開始；每級必須是可獨立理解的具體描述。回傳 score 是機率加權位置，可能是小數。

```python
Score(
    instructions="`bug.description` 的問題有多嚴重？",
    criteria=[
        "只有外觀問題，不影響功能。",            # level 0
        "功能受損，但有可接受的替代方法。",      # level 1
        "核心功能無法使用，而且沒有替代方法。",  # level 2
    ],
)
```

## 完整範例：社交檢定

```python
import math
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

state = {
    "player_action": {
        "actor_id": "kaine",
        "text": "告訴我密道在哪，我可以帶你安全離開。",
    },
    "npc": {
        "name": "老柯",
        "personality": "謹慎，但很想逃離地牢。",
        "attitude": "戒備",
    },
    "conversation_history": [],
}

questions = {
    "check_required": Noul(
        instructions=(
            "`player_action.text` 是否直接對 `npc` 提出具有明確目的、"
            "且結果存在不確定性的社交嘗試？"
        ),
        criteria={
            "true": "試圖以說服、欺瞞或威嚇讓 NPC 行動或改變決定。",
            "false": "只是寒暄、普通提問、陳述，或已直接動手。",
        },
    ),
    "check_type": Choice(
        # 同批問題彼此看不到答案，因此明寫假設前提。
        instructions=(
            "假設 `player_action.text` 構成社交檢定，主要使用哪種手段？"
        ),
        criteria={
            "persuade": "真誠請求、協商、交換利益或講道理。",
            "deceive": "捏造事實、虛假身分、隱瞞或誤導。",
            "intimidate": "明確威脅或可信的暴力後果。",
        },
    ),
    "effectiveness": Score(
        instructions=(
            "假設 `player_action.text` 構成社交檢定，"
            "不考慮 CHA、技能、骰點和運氣時，它對目前 `npc` 有多容易奏效？"
        ),
        criteria=[
            "幾乎不可能奏效；要求不可接受、說法荒謬或威脅不可信。",
            "很難奏效；NPC 必須承擔重大損失或風險。",
            "有明顯阻力，但仍可能成功。",
            "要求合理、可信，而且 NPC 的代價有限。",
            "幾乎沒有拒絕理由，接受是自然且低風險的選擇。",
        ],
    ),
}

with TypeSafeClient() as client:  # 預設讀取 TYPESAFE_API_KEY
    response = client.system_one(
        model="jev-latest",
        state=state,
        questions=questions,
    )

required = response.nouls["check_required"].noul
if required < 0.80:
    result = None
else:
    check_type = response.choices["check_type"].choice
    score = response.scores["effectiveness"].score
    dc_by_level = (20, 16, 14, 12, 8)
    level = max(0, min(len(dc_by_level) - 1, math.floor(score + 0.5)))
    result = {
        "actor_id": state["player_action"]["actor_id"],
        "check_type": check_type,
        "dc": dc_by_level[level],
    }
```

## 批次與依賴

- 能根據同一份 state 獨立回答的問題，放進同一次 `system_one`。
- 同批問題平行執行，彼此看不到答案。
- 分支問題可先一起問，但 `instructions` 必須寫明假設，Python 只使用適用分支。
- 只有前一答案會改變下一次的 state 或候選集合時，才拆成兩次請求。
- 多筆輸入可用 `action_0_*`、`action_1_*` 等穩定 ID，並在問題中引用 `` `actions[0].text` `` 等完整路徑。

## 門檻、錯誤與測試

- Noul 門檻由誤判成本決定；不要把單一門檻套用所有功能。
- Choice/Score 的 confidence 只表示分布集中程度，不代表一定正確。
- 保留 state、questions、typed answers、機率與最後採取動作的原因；不要記錄 API key。
- API 失敗時使用安全 fallback，不要讓不完整答案修改世界狀態。
- 測試時注入 fake client，不要真的連線。

至少測試：

1. 問題類型與一次呼叫中的問題數量。
2. Noul 門檻兩側。
3. Choice 的 `none` 與未知候選。
4. Score 的邊界、小數及 Python 映射。
5. API 例外不修改狀態。

專案範例：`trpg/llm/tag_agent.py`、`trpg/llm/dialogue_flow.py`、`tests/test_jev_tag_agent.py`、`tests/test_dialogue_flow.py`。

## 修改前檢查表

- [ ] 是非用 Noul、固定集合用 Choice、有序程度用 Score。
- [ ] state 只包含必要且最新的上下文。
- [ ] instructions 引用明確 state 路徑，criteria 定義包含與排除情況。
- [ ] Choice 候選完整，必要時有 `none`／`other`。
- [ ] Score 等級由低到高且每級具體。
- [ ] 可獨立回答的問題已合併；分支問題明寫假設。
- [ ] 門檻、映射、合法性與執行仍由 Python 控制。
- [ ] 有 fake client 測試成功、邊界與失敗路徑。

