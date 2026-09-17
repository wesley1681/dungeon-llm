# v05 戰鬥模型訓練指南

本文件只描述目前遊戲實際使用的 v05 `codeword-noarch-enemyskill` 架構。正式入口是 `scripts/exp_matchup_train.py`；其他訓練檔是歷史或研究用途，目前不使用。

## 結論：專家與自我對弈如何分工

目前採用的是漸進式混合訓練：

1. 以既有 v05 checkpoint warm-start；沒有 checkpoint 時，先面對規則式專家。
2. 使用 PPO，讓勝負與戰鬥 reward 改善策略，而不是繼續模仿專家的每個動作。
3. `StrengthModel` 估計學習方與各種隊伍組成的強度，盡量安排接近五五波的對局。
4. 模型相對於專家變強後，逐步提高近期 checkpoint 自我對弈的比例。
5. 自我對弈使用 rolling snapshot pool，不只使用最新版，降低策略循環與只會剋制單一對手的風險。

因此答案不是「先學專家」或「直接 self-play」二選一。規則式專家負責冷啟動與保留對手多樣性；PPO 與 self-play 負責突破專家上限。

## 現行訓練內容

`scripts/exp_matchup_train.py` 固定建立 v05 網路，並在以下分布收集 rollout：

- 12 個標準職業，等級 L1–L8。
- 可在目前強度範圍內配對的怪物。
- 1v1、2v2、3v3，以及人數差受限制的不對稱隊伍。
- 職業與怪物可混編。
- 目前正式入口固定使用 `open` 戰場。
- 約 70% 的目標勝率落在 `0.50 ± 0.08`；其餘對局加入受控的優勢與劣勢。
- agent 席由正在更新的模型控制；對手席由規則式專家或 snapshot pool 中的模型控制。

每個角色可有自己的等級。這一點對混編隊伍很重要，不能用單一隊伍等級取代。

## 訓練前準備

在專案根目錄執行命令。本文使用專案現有的 Windows 虛擬環境：

```powershell
.\gemma-env\Scripts\python.exe --version
.\gemma-env\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

rollout 是大量 batch=1 的戰鬥模擬，主要使用 CPU 多進程；PPO minibatch update 才適合放在 GPU。`--workers` 應依 CPU 與記憶體調整，第一次建議從 4 開始，不必直接使用預設的 16。

## 第一步：執行 smoke test

```powershell
.\gemma-env\Scripts\python.exe scripts\exp_matchup_train.py --smoke
```

它會檢查：

- v05 網路能建立。
- 職業、怪物與 per-entity level 能建立環境。
- PPO rollout 能完成並產生有限的 advantage。
- `StrengthModel` 能從對局結果更新。
- snapshot self-play 路徑能執行。

smoke test 只檢查管線是否接通。它使用未訓練模型時可能顯示低勝率或共塌警告，這不代表既有 checkpoint 的品質。

## 第二步：開始新的 warm-start run

推薦從目前遊戲使用的 v05 checkpoint 開始，但輸出到新目錄：

```powershell
.\gemma-env\Scripts\python.exe scripts\exp_matchup_train.py `
  --init_from models\general_selfplay\mt_u0300.pt `
  --out_dir models\runs\general_v05_run1 `
  --updates 80 `
  --steps 2048 `
  --lr 3e-4 `
  --epochs 4 `
  --batch 64 `
  --ent_coef 0.01 `
  --value_warmup 3 `
  --workers 4 `
  --threads 4 `
  --device auto `
  --sizes "1,2,3" `
  --max_size_gap 1 `
  --tail_gap 2 `
  --p_monster 0.30 `
  --p_boss 0.10 `
  --sp_thresh 0.0 `
  --sp_scale 1.5 `
  --sp_cap 0.60 `
  --k_refresh 5 `
  --pool_size 5 `
  --eval_every 5 `
  --health_games 48 `
  --seed 0
```

這裡明確指定 `--p_boss 0.10`，讓少量 party-content 強怪進入訓練；程式預設值是 0。正式長跑可將 `--updates` 提高到 200–400，先用 5–10 updates 的短跑確認速度、記憶體與輸出正常。

每次都應使用新的 `--out_dir`。不要直接把實驗輸出寫進 `models/general_selfplay`，否則可能覆寫遊戲目前使用的模型。

### 這不是完整 resume

`--init_from` 只載入模型權重。它不會恢復：

- Adam optimizer 狀態。
- 先前的 update 計數。
- 先前的 snapshot pool。
- 對應的 `StrengthModel`。

因此它是新的 warm-start run，不是 bit-exact 斷點續訓。輸出編號會從這次 run 的 `u0001` 重新開始，`StrengthModel` 也會從 bootstrap prior 重新校正。

## 專家與 self-play 的切換

`StrengthModel.seat_bias` 表示「目前 agent 席相對於規則式對手的整體強弱」。self-play 比例為：

```text
p_selfplay = clamp((seat_bias - sp_thresh) / sp_scale, 0, sp_cap)
```

使用預設參數時：

- `seat_bias <= 0`：只對規則式專家。
- `seat_bias` 上升：逐步混入 snapshot 對手。
- self-play 最高佔 60%。
- 每 5 updates 把目前模型加入池中，最多保留 5 份近期快照。

這讓模型弱時有穩定課程，變強後有持續進步的對手，同時仍保留規則式策略帶來的分布外多樣性。

## StrengthModel 與配對

隊伍強度使用下式估計：

```text
T(team) = sum(r[identity, level]) + g[team_size]
P(A wins) = sigmoid(T(A) - T(B) + seat_bias)
```

- `r` 分開記錄每個職業等級與怪物，所以 L4、L5 的能力斷點不會被當成平滑等級差。
- `g` 學習隊伍人數造成的 action-economy 優勢。
- `seat_bias` 吸收學習模型與固定腳本之間的整體能力差。
- 每局結束後用實際勝負更新估計。

`--max_size_gap 1` 限制接近五五波的對局最多差一人；`--tail_gap 2` 只在刻意不平衡的尾端對局放行 1v3 或 3v1。這是因為單純相加的隊伍強度在極端人數差下較不可靠。

## Reward

環境回傳的是 agent 方的 team reward：

| 項目 | 數值／預設 | 說明 |
|------|-------------|------|
| 勝利 | `+5` | 所有對手死亡，且我方至少一人生還 |
| 失敗 | `-5` | 我方全滅；同歸於盡也算失敗 |
| 超時 | `-10` | 超過依隊伍人數計算的 step 上限 |
| HP shaping | `3 × (Φ(s') - Φ(s))` | `Φ` 為我方 HP 比例總和減敵方 HP 比例總和 |
| 每次行動成本 | `0` | 預設關閉 |
| 無效移動成本 | 正式入口預設 `0` | `exp_matchup_train.py` 會在未指定時關閉 |
| 控制效果 credit | `0` | 預設關閉 |

可在啟動 Python 前調整以下環境變數：

```powershell
$env:TRPG_ACTION_STEP_COST = "0.0"
$env:TRPG_WASTED_MOVE_COST = "0.4"
$env:TRPG_MITIGATION_COEF = "0.0"
```

建議一次只改一種 reward 或分布設定，使用不同輸出目錄與相同 seed 做比較。不要為單一職業或技能加入專用獎勵，否則共用模型容易學到難以泛化的捷徑。

`TRPG_LAYOUTS` 不會改變這條訓練管線，因為目前 `env_from_specs` 明確指定 `layout="open"`。若要加入牆、困難地形或岩漿，必須先修改正式入口的場景抽樣並另外做回歸驗收。

實作注意：目前 shaping 使用 `Φ(s') - Φ(s)`，而 GAE 的 `gamma` 是 0.99。它是實用的 dense shaping，但不是嚴格的 `gamma × Φ(s') - Φ(s)` 形式，因此文件不宣稱它具有完整的 policy-invariance 保證。

## 輸出檔案

每逢 `--eval_every`：

```text
models/runs/general_v05_run1/
├── mt_u0005.pt
├── mt_u0005.pt.arch.json
├── strength_u0005.json
├── mt_u0010.pt
├── mt_u0010.pt.arch.json
└── strength_u0010.json
```

- `.pt`：模型權重。
- `.pt.arch.json`：架構代號、參數形狀與 observation schema 指紋；部署時必須和 `.pt` 放在一起。
- `strength_uXXXX.json`：當時的配對強度貨幣，供分析使用；遊戲推論不需要它。

不要只保留最後一次更新。PPO 品質可能震盪，應從通過驗收的 checkpoint 中選擇，而不是自動認定編號最大者最好。

## 如何讀訓練輸出

每個 update 會顯示：

- `R/ep`：該批每局 reward；只能用來看同一設定下的趨勢。
- `pol`、`val`、`ent`：policy loss、value loss、entropy。
- `seat`：模型相對於腳本對手的估計偏置。
- `p_sp`：本批 self-play 比例。
- `scriptWR`、`spWR`：接近平衡目標的實際勝率。

每次健康檢查還會顯示：

- `平衡 greedyWR`：配對器指定五五波時的實際 greedy 勝率。
- 各隊形勝率：避免平均值掩蓋某個 1v1、2v3 或怪物桶崩壞。
- 自我對弈打完率、逾時率與雙方末血：偵測雙方一起龜的 co-collapse。

單次小樣本的 0% 或 100% 不足以判定模型退化；先增加 `--health_games`，再跑正式驗收。

## 驗收候選 checkpoint

目前 v05 的正式入口會在每次 `--eval_every` 自動執行 `probe_health`，並在儲存 checkpoint 後立即印出結果。正式 run 建議保留完整終端輸出，依同一次訓練中的健康報告選 checkpoint。

至少檢查：

- 接近五五波的配對，greedy 勝率約在 45%–55%。
- 各主要隊形沒有明顯掉到 0%，理想上落在 30%–70%。
- 自我對弈打完率至少 90%，沒有高逾時、高末血的共塌。
- 高勝率不是靠非法動作、空轉、拖到超時或同歸於盡取得。

目前正式入口沒有獨立的 `--eval_only`，也沒有完整的候選模型對基準模型回歸守門。`probe_health` 適合訓練中篩選，但部署前仍應以固定 seed 增加 `--health_games`，並在遊戲或戰鬥沙盒中檢查代表性職業、怪物與隊形。不要拿其他架構的 evaluator 直接載入 v05 checkpoint。

## 部署到遊戲

驗收通過後：

1. 保留候選 `.pt` 旁邊的 `.pt.arch.json`。
2. 修改 `trpg/rl/combat_model.py` 的 `DEFAULT_COMBAT_MODEL` 指向候選檔。
3. 啟動遊戲，確認日誌沒有 checkpoint 架構或 observation schema 不相容錯誤。
4. 保留原本的 `models/general_selfplay/mt_u0300.pt` 作為可回退基準。

遊戲只需要模型 checkpoint 與 sidecar；`strength_uXXXX.json` 不參與推論。

## 常見問題

### 訓練很慢

先確認 `--workers` 大於 1。rollout 主要吃 CPU；GPU 只加速批次 PPO update。Windows 使用 spawn 多進程，每個 worker 都會有額外記憶體成本。

### CUDA 沒有被大量使用

這通常正常。戰鬥模擬與 batch=1 rollout 在 CPU worker 執行，只有 `ppo_update` 使用 `--device cuda/auto`。

### 一開始 `p_sp` 是 0

這是預期行為。新的 run 會重新建立 `StrengthModel`，`seat_bias` 從 0 開始；模型先和規則式專家校正，之後才提高 self-play 比例。

### checkpoint 無法載入

確認 `.pt.arch.json` 與 `.pt` 同目錄、同檔名前綴。v05 checkpoint 必須由 `codeword-noarch-enemyskill` 建立，observation schema 指紋也必須與目前程式一致。

### 平均勝率上升，但某些職業變差

不要只看聚合平均。比較各隊形、是否含怪物、平衡局勝率與 self-play 共塌指標；有回歸時先縮小取樣或 reward 改動範圍，再進行下一輪訓練。
