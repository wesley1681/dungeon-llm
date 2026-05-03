# TRPG LLM Engine

本地 LLM 驅動的 D&D 5e 文字 TRPG，使用 Ollama 管理模型，Gradio 提供網頁介面。

## 需求

- Python 3.11+（conda 環境）
- [Ollama](https://ollama.com) 服務在背景執行
- 推薦模型：`gemma4:26b`

## 快速開始

```bash
# 1. 安裝 Ollama 並下載模型
ollama pull gemma4:26b

# 2. 啟動遊戲（網頁介面）
python -m trpg.web
# 開啟瀏覽器：http://localhost:7860

# 或使用終端機版本
python -m trpg.main
```

## 設定

主要設定在 `trpg/main.py`：

| 變數 | 說明 |
|------|------|
| `MODEL` | Ollama 模型名稱 |
| `GM_THINK` | 是否啟用 GM 思考模式（gemma4 支援） |
| `GM_OPTIONS` | 溫度、token 上限等參數 |
| `THOR_OPTIONS` | AI 玩家（索爾）的模型參數 |
| `DEBUG_ARBITER` | 顯示戰鬥判定器的 JSON 輸出 |

## 模型使用建議

**推薦：`gemma4:26b`**

這是唯一經過完整測試、確認可以正常遊玩的模型。輸出品質明顯優於其他模型，能夠正確遵循 GM 格式指令、執行規則標籤。

已知限制：上下文累積較長後（遊戲進入後期），偶爾會出現格式混亂或行為異常，屬正常現象。

> 其他模型（如 qwen3、deepseek-r1 等）普遍存在不遵循指令的問題，無法保證正常遊玩體驗，不建議使用。

**關於思考模式（`GM_THINK`）**

建議一般情況下保持 `GM_THINK = False`。思考模式會讓每一輪 GM 回應的等待時間大幅增加，而 gemma4 在不開思考的情況下輸出品質已足夠。僅在遇到複雜劇情判斷出錯時才考慮開啟。

```bash
ollama pull gemma4:26b
```

## 已測試模型速度（RTX 5060 Ti 16GB）

| 模型 | 速度 | 備註 |
|------|------|------|
| `gemma4:26b` | 30 tok/s | 推薦，唯一完整測試 |
| `qwen3:14b` | 44 tok/s | 速度較快，但指令遵循差 |

## 專案結構

```
trpg/
├── main.py          # 終端機版主程式
├── web.py           # Gradio 網頁介面
├── engine/
│   ├── combat.py    # 戰鬥計算、骰子、攻擊判定
│   ├── character.py # 角色資料結構
│   ├── items.py     # 武器、消耗品定義
│   ├── dungeon_map.py # 地圖與房間系統
│   └── world_state.py # 全域遊戲狀態
├── llm/
│   ├── gm_agent.py      # GM LLM Agent
│   ├── player_agent.py  # AI 玩家 Agent（索爾）
│   ├── arbiter.py       # 戰鬥行動判定器
│   ├── tag_parser.py    # 規則標籤解析執行
│   ├── rulebook.py      # 判定器的規則提示
│   └── backend.py       # LLM API 後端抽象層
└── scenarios/
    └── dungeon.py   # 地下城場景與角色設定
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

等待出現 `server is listening on http://127.0.0.1:11435` 後，修改 `trpg/main.py`：

```python
BACKEND = "llamacpp"
```

再執行 `python -m trpg.web`。

### 實測速度（RTX 5060 Ti 16GB）

| 指標 | 數值 |
|------|------|
| prompt 處理 | ~708 tok/s |
| 實際生成 | ~23 tok/s |

生成速度比 14B 模型慢，但模型能力更強，格式指令遵循較穩定。
