# 通用化網格基準 — 失敗地圖 (2026-06-13c)

目標(用戶重述):一個網,丟到**任意身份 × 任意技能組合**上,做出合理動作打贏。
本報告量測現有最接近通用的兩個 checkpoint 在整個 {identity × kit} 空間的表現,
並畫出失敗地圖。

## 儀器

`scripts/eval_generalize.py` — 複用 `eval_monster_actor.run_episode` 的成對量測
(model 臂 vs script-oracle 臂,全域 RNG 種骰、同 env 朝向/動作格、僅同進程內 Δ)。
script 臂 = 該 kit 的「合理動作」參照:標準職業→手寫專家;synth/class-chimera→
HeuristicCombatPolicy(通用 EV/greedy);monster/mon-chimera→GenericMonsterPolicy。
**Δpp = model% − script% = 「網有沒有像一個稱職的腦袋那樣把這套 kit 打出來」。**

對手 = 12 標準職業專家,公平等級(職業同級;怪物用 EQUIV_LEVEL_1V1)。games=4/配對。

身份桶:std12(分布內)、synth(已見元件的**新組合**,8 個固定種子)、
chimera_cls(打破 panel↔kit 相關)、monster(18 隻 1v1 怪)、chimera_mon(held-out 怪 kit)。

## 結果(n=2016/臂)

| bucket | pop_u0005 Δ | ma_u0032 Δ | pop model% | 含義 |
|---|---|---|---|---|
| std12 | **+3.1** | +1.2 | 43.2% | 平均**贏過職業專家** |
| synth | **+22.1** | +15.1 | 34.4% | 新 kit 組合,遠勝通用腦袋 |
| chimera_cls | **+13.9** | +4.9 | 43.1% | 同上 |
| monster | −0.8 | −3.6 | 24.3% | 與 script 平手 |
| chimera_mon | +2.1 | **−20.8** | 43.8% | pop 持平、ma 崩 |
| **OVERALL** | **+5.8** | **+1.5** | 33.4% | |

**結論 1:任意 kit 組合的泛化大致已成立。** population 訓練的網在從未見過的 kit
組合(synth +22、chimera +14)上,打得**比稱職的通用腦袋還好** → 證明是按技能
特徵行動,不是背 kit。`pop_mon/pop_u0005` = 目前最強通用底座(ma_u0032 在職業 kit
退化,且在 held-out 怪 chimera plague_wight −20.8 崩)。

## 失敗地圖(model 輸給 oracle 的格,皆由 usage-mix 量出)

1. **走位/風箏語義(最大、最一致)**:兩個游俠型最差 —— assassin −18.7、
   arcane_trickster −10.4。mix=`短弓0.34 cunning_hide0.33 move0.01` → 躲+射但
   **move0.01 幾乎不重新走位**。重現先前已知邊界(kite=負遷移)。同族:wolf −10.4。
2. **資源/冷卻排序**:war −20.8(狂噴 guiding_bolt 而非 spiritual_weapon+SG 線);
   champion −16.7(選 second_wind 小補而非 action_surge 爆發)。合法但經濟錯誤。
3. **不熟身體上的防禦惰性**:怪物身體普遍 `dodge0.05–0.26`(commoner0.25/kobold0.16/
   wolf0.15)而 script 不 dodge → 不確定怎麼用就退回 dodge≈no-op,被動局部最優復發。

## 量測注意(誠實標註)

絕對 WR 全偏低(~25–43%)因為**對手是同級調過的職業專家**,而 synth/chimera 的
參照臂是弱的通用 heuristic(synth 地板 12%)。所以正 Δ = 相對打得好,但**不等於
絕對「穩定打贏」**。要乾淨量「拿任意 kit 能不能贏」,對手應對稱(另一個任意 kit),
不是調過的專家。= 重訓前該先補的量測。

## 下一步(每條都綁住一個發現,非臆測)

- 先跑**對稱對手**版(synth-vs-synth / vs-heuristic)分離「kit 本身弱」與「網打得差」。
- 走位語義:查 rogue `move0.01` 是模型問題還是 WASTED_MOVE_COST 把「該風箏的 kit」也
  懲罰了(具體假設,需 per-turn trace 證,不要直接動架構)。
- 通用底座定為 `pop_mon/pop_u0005`。

產物:eval_results/generalize_pop_u0005.txt、generalize_ma_u0032.txt、scripts/eval_generalize.py
