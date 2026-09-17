# TRPG LLM Engine

本機／API LLM 驅動的 D&D 5e 文字 TRPG。遊戲包含探索、NPC 多輪對話、任務、格狀戰鬥，以及由訓練模型或規則策略控制的非玩家角色。

## 最快啟動：桌面版

`start_llama_server.ps1` 已設定使用本機模型 `gemma4-hauhau-q4`。請開啟兩個 PowerShell 視窗，兩個視窗都先切換到本專案根目錄。

### 1. 在第一個 PowerShell 視窗啟動模型伺服器

```powershell
powershell -ExecutionPolicy Bypass -File .\start_llama_server.ps1
```

等待畫面顯示伺服器正在監聽 `http://127.0.0.1:8090`。遊戲運行期間不要關閉這個視窗。

### 2. 在第二個 PowerShell 視窗啟動桌面版

```powershell
.\gemma-env\Scripts\python.exe -m trpg.desktop
```

以上就是使用預設本機模型啟動桌面版的完整流程。README 中的 Python 命令一律明確使用專案的 `gemma-env`，不需要另外啟用虛擬環境。

## 其他操作介面

先依使用的後端啟動模型服務，再執行下列其中一個命令：

| 介面 | 啟動命令 | 說明 |
|------|----------|------|
| 桌面版 | `.\gemma-env\Scripts\python.exe -m trpg.desktop` | Tkinter 視窗、戰場圖與技能按鈕 |
| 網頁版 | `.\gemma-env\Scripts\python.exe -m trpg` | Gradio 三欄介面，開啟 `http://127.0.0.1:7860` |
| 終端機版 | `.\gemma-env\Scripts\python.exe -m trpg --mode terminal` | 純文字操作 |
| Claude 測試模式 | `.\gemma-env\Scripts\python.exe -m trpg --mode claude` | 透過 `claude_response.txt` 提供玩家輸入 |

桌面版是獨立入口，不支援 `--mode desktop` 或 `start.bat desktop`。

## 執行需求

- Windows 與 PowerShell
- 專案內已建立的 `gemma-env`，或具備相同套件的 Python 3.11+
- 執行期套件：`requests`、`gradio`、`Pillow`、`torch`
- 桌面版需要 Python 的 Tk 支援
- 下列其中一種 LLM 後端：
  - Ollama
  - OpenAI 相容的本機 `llama-server`
  - DeepSeek API

## 選擇 LLM 後端

所有介面都從專案根目錄的 `_server_config.json` 讀取 LLM 後端、位址與模型。這個檔案由 `start_llama_server.ps1` 產生，且已被 `.gitignore` 排除。

模型的唯一選擇點是 `start_llama_server.ps1` 開頭的：

```powershell
$selectedModel = "gemma4-hauhau-q4"
```

支援三種寫法：

| 寫法 | 範例 | 行為 |
|------|------|------|
| 本機 GGUF 別名或路徑 | `gemma4-hauhau-q4` | 寫入 llama.cpp 設定，並在 `127.0.0.1:8090` 啟動 `llama-server` |
| `ollama:<模型>` | `ollama:gemma4:26b` | 寫入 Ollama `localhost:11434` 設定後結束腳本 |
| `deepseek:<模型>` | `deepseek:deepseek-v4-flash` | 寫入 DeepSeek API 設定後結束腳本 |

每次切換模型或後端後都要執行：

```powershell
powershell -ExecutionPolicy Bypass -File .\start_llama_server.ps1
```

### 本機 GGUF

本機 GGUF 模式會由 `start_llama_server.ps1` 啟動 `llama-server`，因此執行腳本的 PowerShell 視窗必須保持開啟。腳本包含下列模型別名：

- `qwen3.6-mtp-q4`
- `qwen3-30b`
- `gemma4-google-q4`
- `gemma4-huihui-q4`
- `gemma4-hauhau-q4`

`llama-server.exe` 與 GGUF 使用本機絕對路徑。若專案移到另一台電腦，必須同步修改腳本內的 `$exe` 與 `$modelPaths`。

## 使用 Ollama

先安裝 Ollama 並下載模型，例如：

```powershell
ollama pull gemma4:26b
```

將 `start_llama_server.ps1` 的模型改成：

```powershell
$selectedModel = "ollama:gemma4:26b"
```

執行一次設定腳本；Ollama 模式只會產生 `_server_config.json`，不會啟動服務：

```powershell
powershell -ExecutionPolicy Bypass -File .\start_llama_server.ps1
```

接著可使用 `start.bat`。它會以 `GGML_FLASH_ATTENTION=0` 重啟 Ollama、等待服務就緒，然後啟動指定介面：

```powershell
.\start.bat            # 網頁版
.\start.bat terminal   # 終端機版
.\start.bat claude     # Claude 測試模式
```

若要使用桌面版，請先確認 Ollama 已在背景運行，再直接執行：

```powershell
.\gemma-env\Scripts\python.exe -m trpg.desktop
```

`start.bat` 內的 Ollama 與 Python 使用本機路徑；換電腦時需修改 `OLLAMA` 與 `PYTHON`。

## 使用 DeepSeek API

將 `start_llama_server.ps1` 改成所需模型，例如：

```powershell
$selectedModel = "deepseek:deepseek-v4-flash"
```

執行設定腳本：

```powershell
powershell -ExecutionPolicy Bypass -File .\start_llama_server.ps1
```

在專案根目錄的 `_deepseek_config.json` 填入 API key：

```json
{
  "api_key": "你的 API key",
  "base_url": "https://api.deepseek.com",
  "model": "deepseek-v4-flash"
}
```

若檔案不存在，第一次啟動遊戲時會自動建立空白範本並停止，填入 key 後再次啟動即可。模型與 API 位址取自 `_server_config.json`；`_deepseek_config.json` 只提供 API key。兩個檔案都不會被 Git 追蹤。

DeepSeek 不需要 `start.bat`，直接啟動所需介面即可：

```powershell
.\gemma-env\Scripts\python.exe -m trpg.desktop
# 或
.\gemma-env\Scripts\python.exe -m trpg
```

## 強化學習戰鬥模型

本專案將語言互動與戰鬥決策分成兩套系統：探索、NPC 對話與 GM 敘事由 Local／API LLM 負責；進入戰鬥後，凱恩以外的隊友與敵人預設由獨立訓練的強化學習模型控制，不需要在每個戰鬥回合呼叫 LLM。凱恩仍由玩家操作，所有角色的行動最後都交給同一套 D&D 規則引擎驗證與結算。

目前的戰鬥模型是以 PyTorch 自行實作的 PPO Actor-Critic。訓練對局混合規則式專家與近期模型快照的自我對弈，並使用線上更新的 `StrengthModel` 估計不同職業、等級、怪物及隊伍人數的相對強度，安排以勢均力敵為主、另含部分刻意優勢或劣勢的對局。

模型的主要輸入包括：

- 自己、隊友與敵人的生命、位置、狀態、屬性、抗性及剩餘行動資源。
- 各角色的技能組，以及技能的傷害類型、射程、目標類型和資源消耗等特徵。
- 30×30 戰場中的地形、角色分布、距離、視線、武器可及範圍與近戰威脅。

網路使用 Transformer 編碼技能與各角色的技能組，使用 MLP 編碼實體狀態、行動資源及 Critic value；空間部分直接使用預先計算的戰場特徵，而不是 CNN。現行 v05 架構是一個由所有非玩家戰鬥角色共用的模型，且不直接依賴職業 ID，而是從角色持有的技能和當前狀態推斷其戰鬥能力，因此同一顆模型可以操作不同職業與怪物。

模型採分層方式產生結構化行動：

```text
是否結束回合 → 選擇技能 → 選擇角色目標或戰場位置
```

推論時會先遮蔽資源不足、超出距離、沒有視線或目標無效的選項，再以 greedy argmax 選擇行動。如果訓練模型檔案不存在，遊戲會自動退回規則式 `HeuristicCombatPolicy`，不會因此無法啟動。

### 訓練目前的 v05 模型

目前唯一正式訓練入口是 `scripts/exp_matchup_train.py`。它訓練 v05 `codeword-noarch-enemyskill`，先以規則式專家提供穩定對手，再依模型相對於專家的實力自動提高近期 checkpoint 自我對弈的比例；不是在「只模仿專家」與「只跟自己打」之間二選一。

先執行不寫入 checkpoint 的管線檢查：

```powershell
.\gemma-env\Scripts\python.exe scripts\exp_matchup_train.py --smoke
```

從目前遊戲使用的模型開一個新的 warm-start 訓練 run：

```powershell
.\gemma-env\Scripts\python.exe scripts\exp_matchup_train.py `
  --init_from models\general_selfplay\mt_u0300.pt `
  --out_dir models\runs\general_v05_run1 `
  --updates 80 --steps 2048 `
  --workers 4 --threads 4 --device auto `
  --sizes "1,2,3" --max_size_gap 1 --tail_gap 2 `
  --p_monster 0.30 --p_boss 0.10 `
  --eval_every 5 --k_refresh 5 --pool_size 5
```

請使用新的 `--out_dir`，不要直接覆寫目前遊戲使用的 checkpoint。完整的訓練策略、reward、參數、checkpoint、驗收與部署方式見 [`docs/COMBAT_MODEL_TRAINING.md`](docs/COMBAT_MODEL_TRAINING.md)。其餘訓練腳本是歷史／研究用途，不是目前正式入口。

## 遊戲操作

探索與對話時可直接輸入自然語言，例如「往北走」、「搜索房間」、「撿起治療藥水」或「和老柯談談」。TagAgent 會把需要修改遊戲狀態的意圖轉成規則標籤，引擎結算後再由 GM 產生敘事。

戰鬥時：

- 網頁版與桌面版會顯示戰場、可用技能和目標控制，也可直接輸入文字。
- 終端機版會列出位置、距離、武器與剩餘行動資源。
- 輸入 `結束` 或 `end` 可結束自己的戰鬥回合。

通用命令：

| 命令 | 用途 |
|------|------|
| `短休` / `short rest` | 在探索階段短休 |
| `長休` / `long rest` | 在探索階段長休 |
| `/旁白 on` | 開啟 GM 探索敘事與戰鬥風味文字 |
| `/旁白 off` | 關閉旁白；機制仍照常結算 |
| `/旁白` | 切換旁白狀態 |
| `離開` | 結束正在進行的 NPC 對話 |
| `quit` | 結束遊戲；終端機／探索輸入使用 |
| `status` | 顯示完整隊伍狀態；僅終端機介面特別處理 |

桌面版每次啟動會覆寫根目錄的 `desktop_game_log.txt`，其中記錄戰鬥名單、位置、策略和每次行動，適合用來診斷戰鬥問題。

## Claude 測試模式

`claude` 模式把凱恩的玩家輸入改成檔案 IPC：

1. 輪到凱恩時，stdout 印出 `<<<CLAUDE_TURN>>>`。
2. 遊戲等待專案根目錄的 `claude_response.txt`。
3. 外部工具將單次行動寫入該檔案。
4. 遊戲讀取並刪除檔案後繼續。

這個模式主要用於自動跑流程、重現錯誤和觀察各代理的輸出，不是一般玩家介面。

## 主要設定位置

| 設定 | 位置 |
|------|------|
| LLM 來源與模型 | `start_llama_server.ps1` 的 `$selectedModel` |
| 實際執行期後端設定 | `_server_config.json`，由腳本產生，請勿手動維護 |
| DeepSeek API key | `_deepseek_config.json` |
| GM、TagAgent、索爾的溫度與 token 上限 | `trpg/cli.py` 的 `GM_OPTIONS`、`TAG_OPTIONS`、`THOR_OPTIONS` |
| 思考模式 | `trpg/cli.py` 的 `GM_THINK`、`THOR_THINK` |
| 顯示戰鬥結構化動作 | `trpg/cli.py` 的 `DEBUG_COMBAT_ACTION` |
| 非玩家角色的戰鬥模型 | `trpg/rl/combat_model.py` 的 `DEFAULT_COMBAT_MODEL` |
| 場景、角色與 NPC agent | `trpg/scenarios/dungeon.py` |

若 `models/general_selfplay/mt_u0300.pt` 不存在，遊戲不會因此無法啟動，而會讓非玩家角色退回規則式 `HeuristicCombatPolicy`。

## 架構

```text
desktop.py / web.py / cli.py
            │ 玩家輸入與事件顯示
            ▼
        GameSession                 遊戲主迴圈，執行於背景 thread
        ├─ controllers.py           玩家、LLM 隊友與 NPC 的探索／對話回合
        ├─ tag_agent.py             將探索意圖分類成規則標籤
        ├─ tag_parser.py            驗證並執行標籤，修改 WorldState
        ├─ dialogue_flow.py         對話檢定、事件、任務與招募判斷
        ├─ gm_agent.py              探索敘事與戰鬥風味文字
        └─ CombatPolicy
           ├─ HumanInputPolicy      凱恩的戰鬥輸入
           └─ PPO NeuralPolicy / Heuristic
                                    隊友與敵人的戰鬥決策
                     │
                     ▼
             engine/combat.py      純機制驗證與結算
```

重要元件：

| 元件 | 職責 |
|------|------|
| `trpg/llm/config.py` | 讀取 `_server_config.json`，統一解析 Ollama、llama.cpp 或 DeepSeek |
| `trpg/bootstrap.py` | 為網頁版與桌面版建立相同的 `GameSession` |
| `trpg/game.py` | 探索、對話、戰鬥、任務與事件佇列的核心流程 |
| `trpg/engine/world_state.py` | 世界狀態的單一真相來源 |
| `trpg/web_combat.py` | 網頁版與桌面版共用的戰場繪製及戰鬥命令合成 |
| `trpg/llm/backend.py` | Ollama `/api/chat` 與 OpenAI 相容 `/v1/chat/completions` 的通訊層 |

探索階段由 TagAgent 直接產生的主要機制包括：移動、拾取、給予物品、使用消耗品、解鎖、搜索、與 NPC 對話，以及攻擊非敵對 NPC。戰鬥不走探索標籤，而是由 `CombatPolicy` 產生結構化行動後交給戰鬥引擎執行。

## 常見錯誤

### 網頁已開啟，但終端顯示無法連線 `127.0.0.1:8090`

這表示 `_server_config.json` 指向 llama.cpp，但 `llama-server` 沒有運行。執行 `start_llama_server.ps1`，等待伺服器開始監聽 8090，並保持該視窗開啟。

若要使用 Ollama，請把 `$selectedModel` 設為 `ollama:<模型名稱>`，執行設定腳本，再執行 `start.bat`。

### 顯示「無法連線 Ollama」

確認 Ollama 正在 `localhost:11434` 運行，並確認 `_server_config.json` 的來源是 `ollama`。使用 `start.bat` 會自動重啟並等待 Ollama。

### 顯示找不到 Ollama 模型

設定檔中的模型名稱必須與 `ollama list` 完全一致。缺少時執行：

```powershell
ollama pull <模型名稱>
```

### 顯示找不到 `_server_config.json`

先執行：

```powershell
powershell -ExecutionPolicy Bypass -File .\start_llama_server.ps1
```

### Gemma4 + Ollama 長 context 卡住

`start.bat` 已用 `GGML_FLASH_ATTENTION=0` 啟動 Ollama。若手動啟動服務，請先設定：

```powershell
$env:GGML_FLASH_ATTENTION = "0"
ollama serve
```

### GM 敘事與機制不一致

GM 文字是 LLM 敘事，`WorldState`、TagParser 與戰鬥引擎才是實際狀態。LLM 偶爾可能描述不存在的出口、物品或已死亡角色，但非法的機制行動仍會被引擎拒絕。

## 專案結構

```text
trpg/
├── __main__.py          # web / terminal / claude 入口分派
├── bootstrap.py         # web 與 desktop 共用的遊戲組裝
├── desktop.py           # Tkinter 桌面介面
├── web.py               # Gradio 網頁介面
├── web_combat.py        # 共用戰場繪圖與 GUI 戰鬥命令
├── cli.py               # 終端機與 Claude 測試介面、代理參數
├── game.py              # GameSession 與遊戲主流程
├── engine/              # 角色、技能、物品、地圖、狀態與戰鬥規則
├── llm/                 # 後端、GM、TagAgent、NPC、對話流程與控制器
├── rl/                  # 戰鬥模型、策略、觀測與訓練程式
├── sandbox/             # 獨立戰鬥沙盒
└── scenarios/           # 地下城、角色原型與怪物資料
```
