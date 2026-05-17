# TRPG LLM Engine

本地 LLM 驅動的 D&D 5e 文字 TRPG，使用 Ollama 管理模型，Gradio 提供網頁介面。

## 需求

- Python 3.11+（建議使用專案內 `gemma-env` 虛擬環境）
- [Ollama](https://ollama.com) 服務在背景執行
- 推薦模型：`gemma4:26b`

## 快速開始

```bash
# 1. 安裝 Ollama 並下載模型
ollama pull gemma4:26b

# 2. 雙擊 start.bat（推薦，會自動帶上必要環境變數重啟 Ollama 並啟動遊戲）
start.bat            # 預設網頁版
start.bat terminal   # 終端機版（你親自玩）
start.bat claude     # Claude 測試模式（由 Claude Code 代玩）
```

不想用 bat 也可以直接呼叫：

```bash
python -m trpg                  # 網頁版（預設）
python -m trpg --mode web       # 同上
python -m trpg --mode terminal  # 終端機版
python -m trpg --mode claude    # Claude 測試模式
```

網頁版啟動後開啟 `http://localhost:7860`。

## 啟動模式

| 模式 | 用途 |
|------|------|
| `web` | Gradio 網頁介面，預設 |
| `terminal` | 純文字終端機 |
| `claude` | 由 Claude Code 透過檔案 IPC 操作凱恩的回合進行自動測試（見下方） |

### Claude 測試模式

`claude` 模式下，凱恩（玩家角色）的輸入改由 Claude Code 透過 `claude_response.txt` 提供：

1. 遊戲輪到凱恩時，stdout 印出標記 `<<<CLAUDE_TURN>>>` 並輪詢 `claude_response.txt`
2. Claude Code 透過 `Monitor` 工具讀取遊戲輸出，將行動寫入 `claude_response.txt`
3. 遊戲讀檔後刪除該檔，繼續流程

用途：跑遊戲流程驗證、找格式錯誤、觀察 GM/Arbiter 對各種行動的反應。

## 設定

主要設定在 `trpg/cli.py`：

| 變數 | 說明 |
|------|------|
| `MODEL` | Ollama 模型名稱 |
| `BACKEND` | `"ollama"` 或 `"llamacpp"` |
| `GM_THINK` | 是否啟用 GM 思考模式（gemma4 支援） |
| `GM_OPTIONS` | 溫度、token 上限等參數 |
| `TAG_OPTIONS` | TagAgent 的模型參數（低溫、小輸出） |
| `THOR_OPTIONS` | AI 玩家（索爾）的模型參數 |
| `DEBUG_ARBITER` | 顯示戰鬥判定器的 JSON 輸出 |
| `CLAUDE_TEST` | Claude 測試模式開關（一般由 `--mode claude` 自動設定） |

## 架構

```
玩家輸入
    ↓
cli.py / web.py          ← 入口（純 I/O，無遊戲邏輯）
    ↓ submit_player_input()
game.py (GameSession)    ← 遊戲核心，跑在獨立 thread
    ↓
tag_agent.py             ← 規則裁判：根據玩家行動決定執行哪些標籤
    ↓ execute_all_tags()
tag_parser.py            ← 標籤執行引擎：修改 world_state
    ↓ tag_results
gm_agent.py              ← 說書人：根據機制結算撰寫純敘事
    ↓ events
cli.py / web.py          ← 渲染輸出給玩家
```

### 職責分工

| 元件 | 職責 |
|------|------|
| `game.py` | 遊戲主邏輯：探索循環、戰鬥循環、呼叫所有 AI 代理 |
| `cli.py` | 終端機 I/O：事件 → print，input() → submit |
| `web.py` | Gradio I/O：事件 → yield UI 更新，on_submit → submit |
| `tag_agent.py` | 讀玩家行動 → 決定標籤（無歷史、低溫度） |
| `tag_parser.py` | 執行標籤 → 修改世界狀態 |
| `gm_agent.py` | 收到機制結算 → 寫敘事（不再寫任何標籤） |
| `player_agent.py` | 索爾（AI 玩家）回應 |
| `arbiter.py` | 解析戰鬥行動文字成結構化指令 |

## 規則標籤

玩家輸入後，`tag_agent.py` 根據行動意圖決定要執行哪些標籤，再由 `tag_parser.py` 實際執行並修改遊戲狀態。GM 只負責敘事，不再寫標籤。

### 探索標籤（由 TagAgent 決定）

| 標籤 | 用途 |
|------|------|
| `[TRAVEL: <方向>]` | 玩家移動到相鄰房間，自動觸發戰鬥（若房間有敵人） |
| `[ROLL: <角色ID> <屬性> DC<數字>]` | 技能或豁免檢定，例 `[ROLL: aria WIS DC12]` |
| `[PICKUP: <角色ID> <物品名>]` | 從當前房間拾取物品 |
| `[UNLOCK: <角色ID> <物件名> <屬性>]` | 解鎖有鎖的物件（DEX=撬鎖，STR=蠻力，DC+3） |
| `[CONSUME: <角色ID> <道具名>]` | 消耗 1 個消耗性道具 |
| `[HEAL: <角色ID> <骰子式>]` | 恢復 HP，通常和 CONSUME 一起使用 |
| `[STATUS: <角色ID> +/-<效果>]` | 增加或移除狀態效果 |

### 戰鬥標籤（由 Arbiter + 引擎處理）

戰鬥中的攻擊、傷害、先攻由 `arbiter.py` 解析玩家自然語言後，透過 `engine/combat.py` 執行，不走 TagAgent 流程。

## 模型使用建議

**推薦：`gemma4:26b`**

這是唯一經過完整測試、確認可以正常遊玩的模型。輸出品質明顯優於其他模型，能夠正確遵循指令格式。

> 其他模型（如 qwen3、deepseek-r1 等）普遍存在不遵循指令的問題，無法保證正常遊玩體驗，不建議使用。

**關於思考模式（`GM_THINK`）**

建議一般情況下保持 `GM_THINK = False`。思考模式會讓每一輪 GM 回應的等待時間大幅增加，而 gemma4 在不開思考的情況下輸出品質已足夠。僅在遇到複雜劇情判斷出錯時才考慮開啟。

## 已測試模型速度（RTX 5060 Ti 16GB）

| 模型 | 速度 | 備註 |
|------|------|------|
| `gemma4:26b` | 30 tok/s | 推薦，唯一完整測試 |
| `qwen3:14b` | 44 tok/s | 速度較快，但指令遵循差 |

## 已知問題

**Gemma4 + Ollama 的 Flash Attention bug**

在長 context 下 Ollama 處理 Gemma4 可能掛住。解法是讓 Ollama 帶 `GGML_FLASH_ATTENTION=0` 啟動。`start.bat` 已自動處理，手動啟動則需：

```powershell
$env:GGML_FLASH_ATTENTION=0
ollama serve
```

**Context 累積**

GM 與索爾的對話歷史會自動滑動窗口（GM 保留 12 條訊息、索爾 8 條）以避免 prompt 過長。

**GM 幻覺**

Gemma4 偶爾會描述不存在的房間或復活已死的敵人。系統的 TagParser 與 Arbiter 會正確攔截非法行動（如往不存在的方向 TRAVEL），但敘事仍可能誤導玩家。已透過清理房間描述、強化系統提示緩解，無法完全根治。

## 專案結構

```
trpg/
├── __main__.py      # 統一入口（--mode 分派）
├── game.py          # 遊戲核心邏輯（GameSession + 事件系統）
├── cli.py           # 終端機入口（事件消費者）
├── web.py           # Gradio 網頁入口（事件消費者）
├── engine/
│   ├── combat.py        # 戰鬥計算、骰子、攻擊判定
│   ├── character.py     # 角色資料結構
│   ├── items.py         # 武器、消耗品、Chest 定義
│   ├── dungeon_map.py   # 地圖與房間系統
│   └── world_state.py   # 全域遊戲狀態（單一真相）
├── llm/
│   ├── tag_agent.py     # 規則裁判：根據玩家行動決定標籤
│   ├── tag_parser.py    # 標籤執行引擎
│   ├── gm_agent.py      # GM：純敘事，不寫標籤
│   ├── player_agent.py  # AI 玩家 Agent（索爾）
│   ├── arbiter.py       # 戰鬥行動判定器
│   └── backend.py       # LLM API 後端抽象層（Ollama / llama.cpp）
├── scenarios/
│   └── dungeon.py       # 地下城場景與角色設定
└── debug/               # 每次 LLM 呼叫的完整 context（用於除錯）
    ├── gm_context.json
    ├── tag_agent_context.json
    ├── tag_agent_output.txt
    └── ...
```

---

## 實驗性：llama-server 後端（Qwen3.6-27B TQ3_4S）

> 此為測試中功能，一般使用請保持 `BACKEND = "ollama"`。

使用自訂量化格式（TQ3_4S）的 27B 模型，需要另外編譯 llama.cpp-tq3。

### 編譯需求

- CUDA Toolkit 13.2+
- Visual Studio 2022+
- CMake（`pip install cmake`）

### 啟動 llama-server

```bat
.\start_llamacpp.bat
```

等待出現 `server is listening on http://127.0.0.1:11435` 後，修改 `trpg/cli.py`：

```python
BACKEND = "llamacpp"
```

再執行 `python -m trpg`（網頁版）。

### 實測速度（RTX 5060 Ti 16GB）

| 指標 | 數值 |
|------|------|
| prompt 處理 | ~708 tok/s |
| 實際生成 | ~23 tok/s |

生成速度比 14B 模型慢，但模型能力更強，格式指令遵循較穩定。
