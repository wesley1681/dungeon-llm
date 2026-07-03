# 模型現況紀錄（2026-06-11 更新：倒地/救起上線 + 非對稱對局；2026-06-10 多人模式 + 團隊合作）

> **2026-06-11 重大引擎變更（§0）**：實裝 5e 倒地/瀕死/救起。下方 §2-§4 的所有
> 評測數字都是**舊規則**下量的；新規則下的重新基準見 §0 與 §5（非對稱評測全在新規則下）。
> 新規則下官方重驗：team goal **BEAT 45.5% vs 39.8%（+5.7%>±2.9%）**（team_goal_after_asym.txt）。

四個目標皆達成，**同一張路由表同時服務 1v1 與 3v3**：

- **1v1**：12/12 職業 tie/beat 公平專家。合作版路由 120g 重驗 11/12 OK + champion −3%
  剛壓線（goal_1v1_after_coop_120g.txt）；champion checkpoint 本次未動、無 ALLY 技能
  （mask 改動不可能影響它），240g 穩定種子高精度重判 **−0.7%（±2.6%）= TIE**
  （eval_results/champion_240g.txt）→ champion 是平手邊界職業，120g 的 SHORT 是
  hash() 種子抽樣噪聲。**GOAL 維持 MET**。
- **3v3 多人**：合作版路由 **47.7%** vs 專家隊 **40.4%**（+7.3% > ±2.9%，2304 場/邊）
  → **BEAT 保持**（eval_results/team_final_v4_coop.txt；合作交換花掉 1.6pp 餘裕，
  前版 +8.9% 見 team_final_v3.txt）。
- **團隊合作（本次新達成）**：牧師奶隊友（治療 0.11→0.48/場、給隊友 0%→41%、
  超過專家 33%、殘血 triage）；divination 友傷 −11%（PPO 在 Φ 全額計價下的最優解仍
  保留多數火球——與「友傷迴避對勝率中性」的對照實驗互洽）。詳見 §4。
- **非對稱對局（2026-06-11 新達成）**：15 個非對稱桶（等級差 ±3 × 人數差）全部
  跟住/超越專家、擺爛指標 idle 全程 0%；battle_master 經 --asym ΔL 訓練後對稱席位
  +8.4pp、劣勢桶優勢全面擴大。詳見 §5。

---

## 0. 倒地/救起機制（2026-06-11 實裝，5e RAW 為準）

§4 末的治療收益實驗結論「撈倒地是唯一給『奶隊友>奶自己』差異化價值的機制」獲用戶
拍板後實裝。規則映射（trpg/engine/combat.py + character.py）：

- **進入瀕死**：PC（agent 隊，is_npc=False）降到 0 HP → 瀕死（昏迷），不再立即判死；
  溢出傷害 ≥ 最大 HP → 即死（5e massive damage）。對手隊是 NPC（怪物），0 HP 即死——
  PC vs 怪物的標準 D&D 不對稱。
- **死亡豁免**（共用 `roll_death_save()`，game.py 敘事循環與 env_v2 RL 循環同一套）：
  每輪輪到自己先攻位擲 d20；≥10 成功（3 次→穩定化，停擲）、<10 失敗、nat 1 雙失敗、
  nat 20 回 1 HP **且當回合可行動**；3 失敗死亡。
- **攻擊倒地者合法**（確保不被救活的戰術存在）：攻擊有優勢、1.5m 內命中自動暴擊、
  瀕死中受傷 = 1 次豁免失敗（暴擊 2 次、傷害 ≥ maxHP 即死、打斷穩定化）；
  AoE 照樣波及倒地者；昏迷者 STR/DEX 豁免自動失敗。
- **救起**：HEAL / LAY_ON_HANDS 對瀕死者合法（原 `is_alive` 閘移除，只擋真死者），
  任意治療量重置豁免、以治療量 HP 回到戰鬥；結果帶 `revived` 旗標。
- **RL 各層**：obs `partition_entities` 含瀕死（列存在 + hp=0 + alive位=0，**維度不變**、
  舊 checkpoint 不作廢）；entity mask 的 padding 判定改「全零列」（瀕死槽可選）；
  env_v2 在瀕死 agent 的先攻位 tick 豁免；專家牧師 `_find_heal_target` 含瀕死
  （HP%=0 自然最優先）+ 超距先走位；聖騎士新增 `_attempt_pickup`（lay on hands 撈人）。
- **同捆陣營相對性 bug 修正**（audit siblings）：`_find_heal_target` 原用絕對
  `is_party_ally` → 對手側牧師掃敵隊傷員、貼臉會奶敵人、自己隊友永不被奶；
  偷襲「盟友鄰接」原以玩家隊為盟友（敵方盜賊吃目標隊友的鄰接加成）；
  聖武士守護靈光同病。全部改為 actor/受保護者相對陣營。

**新規則下的平衡量測**（eval_results/diag_revive.txt，tests/test_dying_revive.py 24 測試）：

| 量測 | 結果 |
|---|---|
| 3v3 專家隊（life 補師，1152 場） | 倒地 1.80/場、**救起 0.57/場（32% 倒地被撈回）**、前排真死 16%（舊規則倒地即死 ~67-70%）、WR 43.8%±2.9（舊帶寬內） |
| 1v1 專家 vs life/war 牧師（120g/對位） | 對手牧師修 bug 後**真的會自奶**：champion vs life 95%→72%、vs war 70%→57%（同 process A/B 歸因）；巫師 vs 牧師 5-10% 新舊皆然（本來就是極差對位，非本次造成） |
| 1v1 routed 模型 sanity（8g/對位） | 無崩潰，overall 44%，各職業在歷史帶寬內 |
| 3v3 模型 vs 專家（新規則重跑，1152 場/邊） | **BEAT 保持：46.3% vs 40.5%（+5.8% > ±4.1%）**（eval_results/team_goal_revive.txt）。divination 組合的差距拉大（專家巫師隊 2-8/24 vs 模型 12-19/24）——對手牧師會自奶後，專家放風箏巫師的消耗戰更打不動 |

**尚未做（後續）**：對手專家不會補刀倒地者（引擎允許、腳本未用）——要對抗救起需加
finisher 變體；模型重訓吃進救起信號（撈人 = 行動經濟級離散收益，PBRS 天然計價：
救起立即 +治療量Φ + 隊友後續行動流）；eval_goal/MODEL_STATUS 各基準在新規則下重量測。

## 1. 多人模式定義（先前討論定案）

- 隊伍 = 角色模板「1 前排 + 1 打擊手 + 1 支援」抽樣（48 種組合全是合理隊伍）。
  角色分桶定義在 `trpg/scenarios/archetypes.py::ARCHETYPE_ROLES`（archetype 的屬性，新增職業必須宣告）。
- 同標準：模型隊與專家隊都透過 env.step 打**同一批**專家對手隊、同種子；勝 = 對面全滅且我方有人活。
- 專家 = 現狀腳本專家（v1 基準，未補強）：天生隊伍感知、牧師會奶隊友；
  但無集火（全打最近敵人）、法師 AoE 無友傷意識（會炸自己前排）、bless 只給自己。

## 2. 路由表（scripts/eval_routed.py DEFAULT_ROUTING）

**2026-06-12 obs v3 重訓潮收官**：9/12 席升級原生 v3（--asym slot-PPO 40u，配方=bm 試點），
3 個 BC 席位（berserker/devotion/life）重收示範挑戰失敗、維持現役遷移檔。
官方 120g 重驗 **12/12 GOAL MET**（goal_1v1_after_v3_wave_120g.txt，vengeance +22、evocation +32）。
換檔判準 = gate_slot.py 三閘門：pick(8g/cell)+1v1 guard(60g)、席位 1728 場/臂大樣本（同帶即換，bm 先例）、
probe_v3_features 等級敏感度。

| 職業 | checkpoint | 席位大樣本（新 vs 現役，1728/臂） | 1v1 guard / V(s) spread |
|---|---|---|---|
| battle_master | models/ppo_team_bm_v3/ppo_u0025.pt | 48.6=47.7（試點） | PASS / 1.04 |
| champion | models/ppo_team_champion_v3/ppo_u0010.pt | 48.3=48.7 par-swap | PASS / 2.29 |
| totem_bear | models/ppo_team_totem_bear_v3/ppo_u0015.pt | 66.0=65.3 par-swap | PASS / 2.97 |
| berserker | models/bc_team_berserker/bc_e01.pt（**維持**） | 新 BC 全輸 pick；56.6=56.4 | 遷移檔 / 0.00（盲） |
| evocation | models/ppo_team_evocation_v3/ppo_u0040.pt | **75.5 > 70.6（+4.9pp 真提升）**；generalist 分家 | PASS / 3.67 |
| divination | models/ppo_team_divination_v3/ppo_u0025.pt | 59.6=60.9 par-swap | PASS / 2.70 |
| life | models/bc_team_life4/bc_e08.pt（**維持**） | 新支援配方 49.8=49.4 未勝前代 | 遷移檔 / 0.18 |
| war | models/ppo_team_war_v3/ppo_u0030.pt | 59.0=57.6；generalist 分家 | PASS / 4.36 |
| assassin | models/ppo_team_assassin_v3/ppo_u0035.pt | **40.0 > 35.6（+4.5pp 真提升）**，追平專家席位 40.2 | PASS / 5.17 |
| arcane_trickster | models/ppo_team_arcane_trickster_v3/ppo_u0040.pt | **41.4 > 34.1（+7.3pp 本波最大）**，超專家席位 33.4 | PASS / 9.31 |
| devotion | models/bc_team_devotion/bc_e03.pt（**維持**） | 新 BC 全輸 pick；40.4=39.9 | 遷移檔 / 0.62(異常待查) |
| vengeance | models/ppo_team_vengeance_v3/ppo_u0015.pt | 44.9=44.3 par-swap | PASS / 1.41 |

教訓：specialist/generalist 席位全面受益於席位 PPO + asym + v3（盜賊雙席是最大贏家，
V spread 5~9 = 等級條件化最強）；in-context BC 檔三席全數拒絕重訓——「對示範分佈
過擬合的精品」不保證複現，與 obs v3 attention 教訓同根（BC 檔對 regime 最敏感）。

**這是唯一一張路由表，單挑和多人都用它**——沒有「1v1 專用」的另一套。被取代的舊 1v1
checkpoint（bm: ppo_spec_bm_v5/ppo_u0460、champion: ppo_spec_champion_wm2/ppo_u0040、
totem/berserker: ppo_endhead_gen/ppo_best、devotion: ppo_spec_devotion_v2/ppo_u0280、
vengeance: bc_endhead.pt）仍在 models/ 下，但已無理由使用：每個替換者在 1v1 都 ≈ 或 >
舊檔（guard ±2σ），且全表 120g 1v1 重驗 GOAL MET；vengeance 新檔 1v1 還比舊檔 +9pp。

## 3. 怎麼達標的（資料驅動全紀錄）

1. **Baseline（舊 1v1 路由直接上 3v3）**：42.4% vs 43.0% = 平手。
2. **逐席位消融**（team_ablate.py，MMM/EMM/MEM/MME/EEE）：
   換前排→專家 +14pp（最差組合）/+5pp（最好組合）；換打擊手/支援 ±0；
   模型法師被換掉 −25pp。→ 弱點唯一集中在前排。
3. **行為追蹤**（trace_team.py）：勝負判別變數是**交戰距離**——
   贏的前排 2.1m、輸的 3.9-4.0m（專家 2.3-2.9m）。lateral 移動次數和
   「攻擊最近敵人比例」在贏家身上同樣偏高 → 不是判別變數（修正了早期解讀）。
4. **解法 A：隊伍 PPO 微調**（trpg/rl/train_team.py + scripts/train_team_ppo.py）——
   學習者只佔一個席位，隊友=凍結路由網路（=部署配置），對手=專家隊；
   獎勵沿用一般化的隊伍 PBRS + wasted-move（無 per-class 項）；
   隊友/對手回合的 ΔΦ 累加回學習者最近 transition（保 PBRS 望遠鏡性質）。
   → battle_master/champion/totem_bear/vengeance 超越專家席位。
5. **解法 B：部署情境 BC**（scripts/bc_team_front.py）——berserker/devotion 的 PPO
   高原（devotion 最佳=第 2 個 policy update，其後越練越差）；改收
   「專家坐該席位 + 凍結模型隊友」的示範 1200 場做 BC 微調（兩段式範式的隊伍版，
   不是把規則寫進模型）→ 兩者 +7~8pp，devotion 追平專家席位。
6. **快照挑選**（scripts/pick_team_snap.py）：192 場隊伍評測排名 + 1v1 guard
   （新舊 checkpoint 各 720 場，±2σ 內才允許全域路由）。

## 4. 合作行為現況（2026-06-10 更新：量測修正 + 對照實驗）

**重要更正**：先前 trace_coop 報的「AoE 友傷 64.7/場、兩邊一樣」是**誤歸因**——
HP 差分法把「下一個角色回合開始時的敵方靈體守衛光環/地形 tick」（發生在上一個 actor 的
env.step 內）算成了友傷。改用引擎 cast 結果（target_results）正確歸因後
（trace_coop/diag_coop_value 皆已修正）：

- **evocation 友傷 = 0**：L5 有 sculpt_spells（雕塑法術，archetypes.py:197），引擎層直接
  把我方排除在 AoE 外。模型/專家都不會誤傷。
- **divination 友傷是真的**：模型隊 33.5/場、專家隊（盲放）31.8/場（fireball_div 半徑
  4.5m 蓋到交戰中的自家前排）。
- tick 傷害（光環/地形）~60-110/場，與「合作」無關，兩邊驅動本來就一樣。

**對照實驗（diag_coop_value.py，全專家隊、同種子、每臂 ≥1152 場）——
「合作行為是否與獎勵（勝率）最大化一致？」**

| 實驗 | 對照 | 勝率差 | 判定 |
|---|---|---|---|
| A：治療目標 | triage 奶隊友 vs 只奶自己 | +0.1%（2σ ±4.0%） | **中性** |
| A2：治療量 | bonus 動作高頻奶（1.62次/場）vs 原版（0.94） | +0.5%（2σ ±3.3%） | **中性**（前排死亡率 71%→70% 也沒動） |
| B：友傷迴避 | AoE 換目標/跳過 vs 盲放（divination 組合） | −0.9%（2σ ±2.0%，3456場/臂） | **中性**：省下的友傷 32.5 被少打的敵傷（52→21）抵消 |

結論：**兩個合作行為對勝率都是「中性、不衝突」**（結果檔 eval_results/
diag_coop_value.txt、diag_healvolume.txt、diag_ff_sharp.txt）：
- 奶隊友：這套數值下治療量級（5.5-7.5HP/次）相對戰鬥傷害太小，純獎勵壓力不會
  自發長出奶隊友；但做了也不虧 → in-context BC 克隆專家 triage（行為出現＋guard
  確保不回歸）是正當路徑。
- 友傷迴避：粗糙的「跳過/換目標」中性（敵傷損失抵消友傷節省）；但 PBRS 對友傷
  有稠密負梯度且同時計價敵傷 → 席位 PPO 自己搜尋「站位/落點兩全」解，若存在
  即是獎勵最大化行為。

**合作行為達成（2026-06-10 晚，trace_coop 8 4 = 384 場/邊，結果檔
eval_results/trace_coop_final.txt）**：

| 指標 | 改造前 | **現在** | 專家 |
|---|---|---|---|
| 治療/場 | 0.11 | **0.48** | 1.03 |
| 給隊友比例 | 0% | **41%** | 33% |
| 施放治療時目標 HP | 62%（近滿血自奶） | **39%**（殘血 triage） | 21% |
| AoE 友傷/場（正確歸因） | — | 8.0 | 8.0 |
| 隊伍 WR（同組合集） | — | 47% | 40% |

模型牧師「奶隊友傾向」（41%）已**超過專家**（33%）；life 組合上為 0.61/場、53% 給隊友。
官方驗收同步通過：團隊 BEAT 保持（47.7% vs 40.4%，+7.3% > ±2.9%，
eval_results/team_final_v4_coop.txt）。

**怎麼做到的（含失敗路徑，全有數據）**：
1. divination：席位 PPO（一般獎勵）→ ppo_u0025，隊伍 67>60、1v1 45=47 PASS、友傷 −11%。
   PPO 在 Φ 全額計價友傷下仍保留大部分火球（27.4/場）——這就是該局面的獎勵最大化解。
2. life：四代配方迭代——
   - v1 純隊伍 BC：合作最強（0.73/53%）但 1v1 43.5→21.9，**低於專家 32，破 1v1 目標**，棄。
   - v2 加 50% 「1v1 自我重放」（舊檔自己的 1v1 行為當標籤）：1v1 41.7 守住，但合作被稀釋到 0.30/31%。
   - v3 重放降到 30%：仍 0.34/48%。原因：治療是稀有技能類（~5% act 對），BC 欠學習。
   - v4 = 30% 重放 + **identity 逆頻率技能權重**（train_bc 既有機制）：合作 0.61/53%、
     1v1 38.9（> 專家 32，官方目標 MET；嚴格 ±2σ guard 讓 6pp，是合作的代價）、
     life 席位隊伍 −5.9pp（整體仍 BEAT，花掉 1.6pp 餘裕）。
   - 嘗試用席位 PPO 從 v4 恢復席位 WR：WR 回來了，但治療被沖回 0.21-0.28/30%——
     **獎勵中性的行為在 PPO 下不保值**（與對照實驗中性判定互洽），故 life 直接用 v4 e08。
3. war：維持原檔。war 專家本就是 battle cleric（急救閾值 25%），示範裡治療太稀，
   換檔無增益；「奶隊友」的載體是 life（領域定位）。

**「治療何時才有收益」條件實驗（diag_heal_when.py，1152 場/臂，
eval_results/diag_heal_when.txt）**——回答「為什麼自然訓練學不會奶隊友」：

| 環境 | triage − 自私 | 解讀 |
|---|---|---|
| 現狀 | −1.6%（±4.1）中性 | 治療覆蓋率僅 ~16%（補 7.3 vs 前排吃 45.4/場，52 maxHP、5.9 回合）|
| 對手集火最殘者 | −2.8% 中性偏負 | 往被圍毆目標灌奶 = 沉沒成本 |
| 治療 ×3 | +0.7% 中性；**但兩臂各 +6~7pp** | 量級夠大時「治療本身」值錢，**目標選擇仍無差**（自己也是隊員）|
| 集火 + 治療 ×3 | **−6.8% 顯著為負** | 灌了 20HP/場前排死亡率仍 69%≈67%——無腦 triage 在集火下有害 |

結論：(1) 治療量級接近單回合傷害時，治療「量」開始有勝率價值（PPO 可自然學會多奶）；
(2) 但「奶隊友優於奶自己」需要隊友生存有**差異化價值**——引擎裡被閘掉的「撈倒地」
正是唯一這種情境（瀕死=行動流全失、且自己永遠不會是可撈對象）：character.py 有完整
死亡豁免機制、apply_heal 支援撈人，但 combat.py HEAL 的 `is_alive` 閘 +
專家/obs/mask 三層都把瀕死者當死人。要讓自然訓練長出奶隊友，開撈倒地是首選引擎改動。
(3) 集火環境下正確行為是「會放棄沒救的」條件性治療，不是永遠奶最殘——這種策略正適合 RL 學。

**基礎設施修正（讓「奶隊友」可被表達，2026-06-10）**：
- `apply_resource_mask`：ALLY 目標技能不再以「某個盟友的距離」遮罩（self 永遠是合法
  觸碰目標；舊邏輯還用了未排序的 allies[0]）→ 3v3 裡 cure_wounds 不再幾乎永遠被遮。
- `apply_entity_mask` 接受 ws/agent_id：ALLY 技能的 row 遮掉敵人與超距盟友；整 row
  全遮時 argmax→slot 0→decode fallback 自奶（與 1v1 行為完全一致，1v1 重驗 life 41%≈42%）。
- PPO `_sample_action` 改存「被抽中技能 row」的 entity mask（per-row mask 之後，存
  row 0 會讓取樣/更新分佈不一致）。
- bc_team_front 收集時丟棄引擎 ERROR 的專家標籤（如超距觸碰治療）。

## 5. 非對稱對局（2026-06-11：基線量測 + ΔL 訓練管線 + 試點）

**問題**：訓練全在平衡局（等級永遠同步、3v3 微調），實戰會遇到輾壓/被輾壓。
驗收雙條件：訓練後對稱不退步＋非對稱不擺爛。

**基線量測（eval_asym.py，eval_results/asym_baseline.txt + asym_sharp.txt）**：
現有模型在全部 15 個非對稱桶**沒有擺爛、沒有缺口**——
- A（1v1 鏡像，ΔL −3..+3，720場/桶）：ΔWR 全部 ±2.4% 內跟住專家；
- B（3v3 鏡像組合等級差，288→864場/桶）：劣勢桶 ΔWR +3.7%/+1.4%，ΔL=−1 還 +9.0%；
  「優勢在劣勢桶蒸發」的初步訊號經加大樣本判為 288 場噪聲；
- C（人數差）：2v3 **+4.3%（±2.9 顯著）**、1v3 絕望局 dealt 17.6%≈專家 18.0%；
- 擺爛指標：模型 idle（首子動作就棄回合且動作未用）全程 **0.0%**（專家反而 0.4-6.6%）、
  dealt（移除敵方 HP 池比例）每桶 ≥ 專家。
機制解釋：generalist 世代訓練本就含 45% 人數不對稱局（_TEAM_CONFIGS）+ PBRS 提供
「劣勢也要換血」的稠密梯度 + BC 先驗來自永不棄戰的專家 → 沒有「放棄」吸引盆。

**管線（已裝，預設關閉）**：
- env：`reset(opp_level=...)` 跨陣營等級差（env_v2.py；順手修了重抽 episode 丟
  agent_archs 換隨機隊的舊 bug）；
- 訓練：`train_team.py::LEVEL_DELTA_TABLE`（ΔL 0:36% / ±1:18% / ±2:9% / ±3:4.5%，
  agent 等級照舊 3-8 隨機，opp = clamp(agent+ΔL, 2..8)）→ `train_team_ppo.py --asym`；
- 評測：`eval_asym.py [g1v1] [g3v3] [桶過濾 如 "B+1,B+2,C2v3"] [arch=ckpt 路由覆寫...]`。

**試點（battle_master --asym 40 updates）三門檻全過並上路由**：
對稱席位大樣本 56.8% vs 48.4%（+8.4pp，1728場/臂——混合 ΔL 反而**提升**對稱表現，
域隨機化的正則化效果）；1v1 guard 54.7=55.6 PASS；劣勢桶同種子全升
（ΔWR：B+0 +0.0→+4.1、B+1 +3.7→+6.5、B+2 +1.4→+3.4、C2v3 +4.3→+6.0，
asym_sharp_u0035.txt）。官方重驗：team goal **BEAT 45.5% vs 39.8%（+5.7%>±2.9%）**
（team_goal_after_asym.txt，revive 規則）。
其餘席位的 --asym 重訓可併入 revive 重訓潮一起做（同一訓練配方加 --asym 即可）。

**注意**：跨 run 的專家側數字會有 1-2pp 漂移（引擎骰子用全域 RNG、不吃 env seed），
同 run 內的模型 vs 專家 ΔWR 才是公平比較。

## 5.5 obs v3 世代遷移（2026-06-11：等級/威脅可見 + 槽位擴充，零回歸上線）

修掉五項觀測債（之前 §6 所列）：等級/威脅不可見、只有血量比例、archetype one-hot
是唯一威脅先驗、6 槽上限（第 4 個敵人隱形）、critic 跨難度錯置。

**Schema 變更（trpg/rl/obs.py）**
- 每實體列尾附加 6 維（舊索引全部不動）：`level/20`、`max_hp/100`、`ac/20`、
  `is_dying`、死亡豁免成功/3、失敗/3。怪物（one-hot 全零）也因此有 level/HP/AC 威脅先驗。
- 槽位 6→10：`[self, ally×3, enemy×6]`；所有槽位映射走 `N_ALLY_SLOTS`/`ENEMY_SLOT_START`
  常數（obs/action 雙向/entity mask/frontend 五處，無字面量）。
- ENTITY_DIM = 7 + 12 + N_RL_STATUS + 1 + 6；critic 輸入隨常數自動展開。

**舊 checkpoint 遷移（model.py `adapt_state_dict_for_obs_v3`，接在
`adapt_state_dict_for_perarch` 鏈頭，所有載入點自動覆蓋）**
- entity_mlp 新特徵欄零填充；critic 首層按槽位圖重散射 + 零填充。
- **attention 教訓（重要）**：skill→entity attention 的 softmax 原本不遮 padding——
  零列過 entity_mlp 是 bias 向量，pad 質量隨槽數縮放。直接放大到 10 槽打破 berserker
  （120g −8pp，diag_obsv3_shift.py：11% 狀態 move↔reckless 翻轉）；改 presence 遮罩
  打破 devotion（−13pp，24.6% 狀態退化成 dodge/hide）。兩個 BC 檔對 pad 質量敏感方向相反。
- 解法：presence 遮罩為**原生 v3 永久設計**（策略對 padding 槽數不變，未來擴槽不再重演）；
  遷移檔由手術寫入 `attn_legacy` buffer，forward 把 softmax 限制在 6 個 legacy 槽位列
  → **位元級重現遷移前 attention**（diag 驗證 devotion 0/1298、berserker 0/1347 分歧）。
  重訓腳本（train_team_ppo / bc_team_front）warm-start 時自動清旗標、改原生 v3 制。

**驗證（全在 revive 規則下）**
- 120g 官方 1v1：12/12 tie/beat GOAL MET（goal_1v1_obsv3_migration_v3_120g.txt；
  devotion +6、berserker −1、assassin −1）。
- 槽數不變性 + 遷移位元級等價有單元測試釘住（test_model.py）。
- 遷移後等級盲基線（probe_v3_features.py）：bm 檔對 opp L2~L8 V(s)/動作完全平坦
  （spread=0.0000）——確證舊模型數學上看不到等級；重訓潮後此探針應變斜。

**狀態**：遷移完成、路由表零回歸。新特徵在遷移檔上是零權重（行為保持的代價），
要真正用上 level/HP/AC/瀕死特徵 = 重訓潮（--asym + revive 信號 + 原生 v3）。

**重訓潮試點（battle_master，2026-06-11 已換檔）**：train_team_ppo --asym 40u，warm 自遷移檔
（自動清 attn_legacy）→ ppo_team_bm_v3/ppo_u0025。三閘門：席位 48.6 vs 現役 47.7
（1728/arm 持平）、1v1 guard 54=56 PASS、asym 劣勢桶錨定 ΔWR 同帶（B+1 +2.8/+1.0、
B+2 +2.1/+6.2、2v3 +3.1/+8.3，皆在合成 2σ 內；idle 全 0）。實質增益 =
**critic 等級校準出現**：probe_v3_features 顯示 V(s) 對 opp L2→L8 單調下降
（−1.72→−2.77，spread 0.0000→1.0442）——跨難度 credit 錯置的根修生效。
策略側首動作尚未按等級分化（40u 內 critic 先學會，policy 條件化需更長訓練/更強壓力）。
其餘 11 席位重訓順序：PPO 席位（champion/totem_bear/vengeance/divination）→ specialist
（assassin/arcane_trickster）→ generalist（evocation/war，需重訓 ppo_endhead_gen 或改席位 PPO）→
BC 席位最後（berserker/devotion/life，bc_team_front 會收原生 v3 新示範 + 自動清旗標）。

## 5.6 蒸餾實驗 + 縫合怪零樣本探針（2026-06-12）

**問題**：路由表 = 按身分硬路由的 MoE（12 檔、訓練成本隨身分數線性長、怪物無法路由）。
**實驗**：把 12 個路由老師蒸餾進一個 shared-head 網（model.py `n_head_groups=1`，
蒸餾是固定目標監督學習，當初逼出 per-arch heads 的梯度衝突不存在）。
兩個 student 同資料對照（scripts/distill_routed.py）：
A=身分輸入保留；B=`--blind`（self one-hot + end_features arch 尾歸零＝純能力條件化，
縫合怪/怪物的 all-zero one-hot 對 B 是分佈內）。

**結果（eval_student.py，120g/職業、同 run 專家基線）**：
- v1（13.8k pairs、8 epochs）：A mean +1.9%、8/12 過帶；**B mean +3.8%、11/12 過帶**
  （只剩 champion −9）。B 的身分盲反而全面更好：A 的 SHORT（assassin −14/AT −9/devotion −21）
  在 B 全數回帶。**身分通道讓網把容量切成 12 條窄路；盲掉才被迫共享結構。**
- 資料加倍（31.3k）+16 epochs 對照：B e08→e16 從 +3.5 掉到 +0.7（**16 epochs 過擬合**，
  identity-weight 放大稀有招）；資料 2.3× 在固定 8 epochs 下無增益（+3.5≈+3.8）→
  **殘餘 SHORT 不是樣本飢餓**，指向 greedy 老師軌跡的 covariate shift（精密序列職業
  最受傷；DAgger 是對應解，未做）。冠軍檔 = models/distill_b/distill_e08.pt。

**縫合怪零樣本（scripts/chimera_defs.py 三隻、eval_chimera.py 行為觀測，
360 場/網）**——縫合怪 one-hot 全零＝怪物條件，無任何老師示範過這些技能組合：
- **chimera_omni**（狂暴+治療+火球+瞬移+定身+action_surge）：B **56% WR**。
  自己發明了「rage(bonus)+fireball(action)」回合循環——零件層遷移成立，且 bonus/action
  仲裁是組合層湧現。失敗模式：bonus 永遠給 rage → 治療飢餓（0.02 次/場）；
  對 totem_bear 會在 3.7m 站樁丟 25 輪火球不收頭（缺集火/逼近收尾）。
- **chimera_gish**（重甲聖騎面板+法師輸出）：A 31% vs **B 52%**。A 退化（滿血狂點
  聖療 1.77/場、90% HP 施放；wasted move 5.15/場）；B 連貫（火球開場→近戰→29% HP 才急救）。
- **chimera_trickhealer**（盜賊底盤+牧師支援）：A 13% vs B 21%。兩者都**塌縮到最近
  鄰居老師**（B=assassin 劇本：hide+短弓循環，牧師整套 0 使用）——蒸餾網對遠離所有
  老師的組合會插值到最近原型，這是 population training 要隨機合成身分的實證理由。

**probe_v3_features caveat（devotion 0.617「異常」查證結果）**：probe 的開場態在對手
先手回合之後取樣；專家對手的開場動作有等級門檻（devotion L3+ 開聖武器→legacy 可見
status 位 col42、L2 沒這招），遷移檔 critic 讀的是這個合法差異，不是 v3 洩漏。
「spread=0 證明等級盲」只在跨等級開場狀態全同時成立（berserker/bm 平坦 0.0000 自洽）。

## 5.7 縫合怪退化行為收斂（2026-06-12 /goal：population + 動作空間合法性修復）

**目標**：讓 B 網在三隻縫合怪上不再出現 5.6 列出的退化行為。先量根因再動刀，
結論是「退化」其實是三種不同性質的東西，各自有不同的正確解法：

**根因一：動作空間缺口（omni 25 輪站樁的真相）**。逐步重放發現 fireball 連丟 28 輪
但法術槽完全沒消耗——`decode_action` 對「瞄到牆後/無視線的格點」回傳 None，env 把
None 當 END 靜默吞掉整回合：無 ERROR、無懲罰、無學習訊號（grid 頭從來沒有
apply_entity_mask 的對應物）。**修復＝合法性遮罩三件套**（與 decode_action 同一套
判定、單一事實來源）：
- `action.point_validity_mask`（trpg/rl/action.py）：POINT/LINE/CONE 技能的 900 格
  合法性（clamp→blocked→LoS），`pick_action` 與 `_sample_action` 都套用；
- `apply_entity_mask` 加敵方 LoS 列遮罩、`apply_resource_mask` 加「全敵無 LoS 時
  遮整個敵標技能」；
- PPO 端 `grid_masks` 存入 rollout batch、`ppo_update` 重套（與 entity_masks 同一
  機率空間契約）。回歸測試 tests/rl/test_action.py（mask↔decode 互為鏡像）。
**純修復效果（e08 權重不動，chimera_e08_maskfix.txt vs chimera_b_e08.txt 同 seeds）**：
站樁迴圈消失（火球落地、被牆卡時 dodge/misty_step 真動作）、gish 52→58%、
官方協議 routed 重驗 **12/12 GOAL MET 維持**（goal_1v1_after_maskfix_official10.txt）。
另跑了史上首次 ±3 帶寬高精度版（120/對手、n=1440/職，goal_1v1_after_maskfix_120g.txt）：
assassin −4 出 ±3 帶——同 seeds 遮罩開關 A/B（assassin_mask_ab.txt）證明**不是遮罩
回歸**（遮罩反而 +3.6pp、有牆局 +7.6pp），是高精度首次照出的既有 1v1 小缺口。
順帶踩出引擎 bug：負施法模值把 healing_word 骰串拼成 `1d4+-1` 炸 parser（標準職業
施法模恆正所以從未觸發）→ 修 `{:+d}` 格式 + apply_heal 治療下限 0。

**根因二：「不用套件」大多是正確估值，不是塌縮（trickhealer）**。強制施放 A/B
（同 seeds 對照）：強制 50% HP 以下 healing_word → WR 22.2%→16.1%（**−6.1pp，
自療在 1v1 是負 EV**）；強制 hold_person 480 場 → 16.0% vs 16.5%（**EV 中性**）；
sacred_flame 期望 4.5 vs 短弓+sneak ~14.5（資料層被輾壓）。population run 裡 PPO 把
治療機率先推高 4 倍（u05）再壓回（u25+）——那不是 credit assignment 失敗，是先探索
後正確估值。**trickhealer 打 assassin 劇本就是這套牌 1v1 的正解**；19% WR 是套牌弱
（支援套件＋脆皮底盤 vs 專家），不是行為退化。3v3 友療場景另計。

**根因三：真要訓練的部分 → population PPO（scripts/train_population.py）**。
每集身分取樣（50% 標準 12 職 / 50% `synth_identity.py` 隨機合法 ClassDef——
底盤+技能包+天賦依賴全從 CLASS_DEFS 資料池組裝、拒絕取樣保合法），縫合怪
**held-out 不入池**；blind 變換在 env 邊界；BC 錨（distill 資料回放 4×64/update）；
逐 update 梯度餘弦監測（簇間 mean −0.07~+0.09 全程，混合 batch 把單對 −0.9 的衝突
平均掉了，未見 v16 病理）。pop_b（60u，修復前）+ pop_b2（40u，修復後）。

**終驗（120g/隻，chimera_popb2_u05.txt，新冠軍檔 models/pop_b2/pop_u0005.pt）**：
| 退化行為 | 基線 | 終態 |
|---|---|---|
| omni 幻影站樁 | 25 輪原地火球 | **消失**（戰鬥→低血自療→被卡時 dodge/misty 真動作） |
| omni 治療飢餓 | 0.02 次/場 | 0.16/場、min 13% HP 會自救；WR 56→**60%** |
| gish 抖動移動 | 2.56/場 | **1.20/場**（−53%）；WR 52→57% |
| trickhealer 套件零用 | （重新分類） | 量測證明=正解，WR 19%≈基線（雜訊內） |
標準 12 職守門（同 session eval_student 120g）：u05 mean +2.6%（e08+遮罩 +4.7%）、
9/12 帶內（bm −11 出帶 vs e08 的 −2）——換檔代價 ~1.6σ，可接受但要知道。
**教訓**：「模型行為退化」要先拆成 動作空間誠實性 / 估值正確性 / 真行為缺陷 三層
量測，只有第三層該用訓練解。

## 5.8 obs v4 三合一手術（2026-06-12：怪物階段觀測基建，零回歸遷移）

怪物階段（MONSTER_CATALOG.md）暴露三個 obs 缺口，合併一次手術解掉（一次付遷移
成本，不分三刀）：

**Schema 變更（trpg/rl/obs.py）**
1. **尺度重定標**：`LEVEL_NORM 20→40`、`MAXHP_NORM 100→800`。比例特意選 2 的冪
   （×2/×8）→ 權重欄補償是純 fp 指數位移，乘積層**位元級精確**（test_obs_v4_
   rescale_compensation_exact 釘住；matmul 輸出僅剩 BLAS 求和順序漂移 ≤1e-6）。
   動機是資訊量不是穩定性（Wave 0 實測出界無害，probe_obs_bounds.txt）：不改則
   200 vs 700 血在特徵上擠在一起。
2. **能力描述子 31 維**（每實體列尾附加，v3 欄位保持嚴格前綴）：聚合自
   `skill.kit_features`（完整 kit、等級門檻保留、**不過濾資源狀態**——公開=外觀
   可推斷，剩餘次數/法術位=隱藏）。欄位：max 傷害/治療/射程/AoE/命中/DC、
   is_caster、teleport、行動爆發＋applies_status 聯集 16＋save_stat 聯集 6。
   解敵方身分盲區：陌生敵人 one-hot 全零，從此仍有機制輪廓可感知。
3. **命名欄位常量**：`I_ENT_LEVEL/I_ENT_MAXHP/ENT_DESC_START`——尾端負偏移
   在這次 append 已實際斷過一次（probe_obs_bounds），全部改名定址。

**遷移（model.py `adapt_state_dict_for_obs` = v2→v3→v4 鏈，所有載入點共用，
含 load_student/sandbox policy_loader）**
- entity_mlp/critic 首層：描述子欄零填充＋level/maxHP 欄 ×2/×8 補償；
  era 寬度凍結成 `_ENTITY_DIM_V2/_V3` 常量（活 ENTITY_DIM 不再被 era 算式引用）。
- **diag_obsv4_shift.py**（v3 視圖臂 vs v4 臂，同 ckpt 同種子）：berserker/
  devotion（v3 敏感戶）/champion/assassin/B 網共 **7,865 共享狀態 0 動作分歧**
  ——遷移在決策層面精確，無 v3 那種 attention 結構性翻轉（這次沒動 attention）。

**描述子資訊量（probe_descriptor_info.py）**
- 1-NN 跨等級還原 archetype 83%；誤判全是戰術孿生對（totem_bear↔berserker、
  evocation↔divination 描述子距離=0.000——同底盤同 kit 聚合，對手視角本來就
  同打法；差異在 trait 層）。怪物最近鄰合理（近戰怪→totem_bear、法系→evocation）。

**守門（官方 10g ×3 輪＋B 網，證據檔 goal_1v1_obsv4_official10*.txt）**
- 單輪有 1–2 戶壓 ±9 帶（bm −9/−11/−21、berserker −10/−4/−4），定性為**協議
  方差非回歸**，三重證據：函數等價（上述 0 分歧）；內部對照——專家臂程式路徑
  零變動卻同樣擺動（專家 bm 57/57/62 vs pre-v4 高精度 54）；eval_goal 的 ±9
  帶是單臂 σ 套雙臂差（正確 2σ(diff)≈±13）＋hash 種子每輪換樣本。席位級判定
  本來就需 1728/arm。B 網輪廓與 pre-v4 同形狀（assassin −22 vs −21、
  devotion −11 vs −16、mean +1.5 vs +2.6）。

**狀態**：遷移完成。術後網路描述子輸入權重=0（行為保持的代價），收益兌現在
下一波訓練（population 混怪物 / BC / DAgger）；屆時補 MONSTER_CATALOG §4.1
驗證 2（敵方陌生同種子 A/B）。BC 不落盤 obs（bc_collect 即收即用），無舊資料
集失效問題。

## 6. 已知侷限／注意事項

- **champion 是 1v1 平手邊界職業**（240g 高精度 −0.7%±2.6%）：120g 不同種子 draw 會
  在 OK/SHORT 之間翻面（eval_goal 用 hash() 種子、跨 run 抽不同棋局）。判它時用
  240g 穩定種子（scripts 見 eval_results/champion_240g.txt 的 inline 寫法）。
- **life 的合作-強度交換是量過的**：1v1 45.1→38.9（仍 > 專家 32 +6.9pp）、life 席位
  隊伍 −5.9pp。四種配方 + 席位 PPO 恢復嘗試的數據都在 §4——想拿回這 6pp 就會失去
  治療行為（獎勵中性行為在 PPO 下不保值），除非未來治療數值被 buff 到有勝率價值。

- berserker 自席位仍 −6pp（63 vs 69）——專家 berserker（魯莽攻擊+狂暴減傷）是最強前排基準；
  總體 BEAT 不靠它。最差組合 battle_master+assassin+war −25%（見 team_final_v3.txt worst-12）。
- 模型贏面部分來自專家基準的盲點（法師友傷、無集火）。把專家補強後重測會是更高的標準（未做，屬後續）。
- **種子穩定性**：隊伍腳本已把 `hash()` 換成 `stable_seed()`（crc32，跨 process 可重現）。
  1v1 舊腳本（eval_class/eval_goal/trace_champion）仍用 `hash()`——同 process 內公平、跨 run 會抽不同種子，
  跨 run 比較時要在同一次執行內比。
- BC 資料（bc_collect.py）仍是純 1v1；隊伍示範由 bc_team_front.py 另收，未合併進主 BC 集。

## 7. 怎麼跑

- 多人達標驗證：`python scripts/eval_team_goal.py 8 6`（48 組合×6 對手組合×8 場×2 驅動）
- 席位消融：`python scripts/team_ablate.py 4 6 [comp ...]`（comp 格式 front+striker+support）
- 前排行為診斷：`python scripts/trace_team.py 4 6 [comp ...]`
- 合作行為驗收：`python scripts/trace_coop.py 4 4 [comp ...]`（治療對象/集火/友傷/站位）
- 隊伍 PPO 微調：`python scripts/train_team_ppo.py --arch <front> --updates 60`
- 部署情境 BC：`python scripts/bc_team_front.py --arch <front> --episodes 1200 --epochs 6`
- 快照挑選+守門：`python scripts/pick_team_snap.py <arch> 8 60`
- 1v1（沿用）：`python scripts/eval_goal.py 120` / `python scripts/eval_class.py <ckpt|expert> <arch> <games>`
- 倒地/救起平衡探針：`python scripts/diag_revive.py 16 120`（3v3 救起頻率 + 1v1 vs 牧師基準）
- 倒地機制單元測試：`python -m pytest tests/test_dying_revive.py -q`（24 測試）
- 非對稱缺口地圖：`python scripts/eval_asym.py 60 24`（A 1v1 等級差 / B 3v3 等級差 / C 人數差；
  可加桶過濾 `"B+1,B+2,C2v3"` 與路由覆寫 `battle_master=models/....pt`）
- 非對稱席位訓練：`python scripts/train_team_ppo.py --arch <a> --asym --updates 40 --out_dir models/ppo_team_<a>_asym`
  （快照挑選：`pick_team_snap.py <a> 8 60 models/ppo_team_<a>_asym`）
- obs v3 等級敏感度探針：`python scripts/probe_v3_features.py <ckpt> [arch] [agent_level]`
  （V(s)/首動作對 opp L2~L8 掃描；遷移檔應平坦、原生 v3 重訓檔應變斜）
- obs v3 遷移分歧診斷：`python scripts/diag_obsv3_shift.py <arch> [games/opp]`
  （遷移檔 vs 位元級舊行為的逐狀態動作對照；換 attention/槽位設計時必跑）
