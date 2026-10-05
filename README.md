# TRPG LLM Engine

以本機或 API LLM 驅動的 D&D 5e 文字 TRPG，包含探索、NPC 對話、任務與格狀戰鬥。

## 安裝

```powershell
py -3.13 -m venv gemma-env
.\gemma-env\Scripts\python.exe -m pip install --upgrade pip
.\gemma-env\Scripts\python.exe -m pip install -r requirements.txt
```

執行測試時另安裝：

```powershell
.\gemma-env\Scripts\python.exe -m pip install -r requirements-dev.txt
```

需要 NVIDIA CUDA 訓練時，請依 [PyTorch 官方安裝頁](https://pytorch.org/get-started/locally/) 選擇適合本機 CUDA 的安裝命令。模型權重不包含在 repository 中。

## 啟動

### Jev 行動意圖判定

探索階段的玩家行動意圖由 TypeSafe Jev 判定。啟動遊戲前設定 API key：

開發或修改 Jev 判斷前，先閱讀 [Jev 請求速查手冊](docs/JEV_REQUEST_GUIDE.md)。

```powershell
$env:TYPESAFE_API_KEY = "你的 TypeSafe API key"
```

Jev 的 `state` 只包含本回合的 `player_actions`；它目前只回傳各種行動意圖的機率與門檻判定，尚未連接角色、物品、方向、合法性或遊戲機制處理。GM、NPC 與其他生成任務仍使用下方設定的本機 GGUF 或 DeepSeek。

### 本機 GGUF

在 `start_llama_server.py` 設定模型：

```python
SELECTED_MODEL = "gemma4-hauhau-q4"
```

可使用腳本內建的模型別名或完整 GGUF 路徑，也可透過 `LLAMA_SERVER_EXE` 與 `TRPG_LLM_MODEL_DIR` 環境變數指定執行檔及模型目錄。

啟動遊戲：

```powershell
.\gemma-env\Scripts\python.exe -m trpg
```

程式會自動啟動 `llama-server`，並在 GUI 關閉時停止伺服器。

### DeepSeek API

將 `start_llama_server.py` 的模型設為：

```python
SELECTED_MODEL = "deepseek:deepseek-v4-flash"
```

在 `_deepseek_config.json` 填入 API key：

```json
{
  "api_key": "你的 API key",
  "base_url": "https://api.deepseek.com",
  "model": "deepseek-v4-flash"
}
```

接著啟動遊戲：

```powershell
.\gemma-env\Scripts\python.exe -m trpg
```

DeepSeek 模式不會啟動本機伺服器。`_server_config.json` 與 `_deepseek_config.json` 均不會被 Git 追蹤。

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

請使用新的 `--out_dir`，不要直接覆寫目前遊戲使用的 checkpoint。完整的訓練策略、reward、參數、checkpoint、驗收與部署方式見 [`trpg/rl/COMBAT_MODEL_TRAINING.md`](trpg/rl/COMBAT_MODEL_TRAINING.md)。其餘訓練腳本是歷史／研究用途，不是目前正式入口。

## 基本操作

探索與對話時直接輸入自然語言；戰鬥時可透過介面選擇技能與目標。輸入 `結束` 或 `end` 可結束自己的戰鬥回合。

| 命令 | 用途 |
|------|------|
| `短休` / `short rest` | 短休 |
| `長休` / `long rest` | 長休 |
| `/旁白` | 切換 GM 旁白 |
| `離開` | 結束 NPC 對話 |
| `quit` | 結束遊戲 |

## 專案結構

```text
trpg/
├── __main__.py          # 將 python -m trpg 轉交 bootstrap
├── bootstrap.py         # 組裝並啟動 GUI 遊戲
├── gui.py               # Tkinter 圖形介面定義
├── gui_combat.py        # GUI 戰場繪圖與戰鬥命令
├── game.py              # GameSession 與遊戲主流程
├── engine/              # 角色、技能、物品、地圖、狀態與戰鬥規則
├── llm/                 # 後端、GM、TagAgent、NPC、對話流程與控制器
├── rl/                  # 戰鬥模型、策略、觀測與訓練程式
├── sandbox/             # 獨立戰鬥沙盒
└── scenarios/           # 地下城、角色原型與怪物資料
```

#Debug
1. tag_agent_context.json : call jev時，jev看到的上下文 + 問題
