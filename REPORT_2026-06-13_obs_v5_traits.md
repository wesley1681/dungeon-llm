# obs v5 被動天賦可見性 + 因果驗證（2026-06-13）

目標：「讓怪物能看見天賦，並且拿此資訊必定可帶來增益的情況驗證怪物是否真的學會
根據天賦改變行為模式。」

**結論：GOAL MET。** 模型現在能在 obs 看見自己的被動天賦，且因果證明它會「讀到再生
就改打近戰」——蒙住通道行為完全退回、讀到通道勝率 +33pp、且泛化到沒訓過的怪。

---

## 1. 先查清（CLAUDE.md：先量再開刀）

- **typed-resist（抗性/免疫/易傷）其實「已經」在 obs**：capability_descriptor 讀
  `char.damage_multipliers`、接在每一列（含 row0=self）。評估文件 C3 對「抗性」不準。
  真盲區＝**不是 damage_multipliers、也不是主動招**的被動天賦：regeneration /
  pack_tactics / undead_fortitude / legendary_resistance（probe_regen_headroom、
  probe_pack_headroom 確認 obs 位元相同＝盲）。
- **載體選擇（量測驅動）**：
  - pack_tactics 是「純行為兌現、零被動」最乾淨載體（headroom +15pp WR），但
    多 agent 集火走位 = emergent positioning。
  - regen 是被動續航，但 **regen→近戰vs遠程**是單 agent 的**離散技能選擇**：近戰
    吃反擊、再生補回 →（probe_regen_skill）regen30 時 forced-melee 52% vs
    forced-ranged 30%（+22pp）；冠軍預設 kite、近戰率對 regen 全平＝盲＝有 headroom。

## 2. obs 手術（v4→v5，bit-exact strict append）

- `trait_descriptor()` 接在 capability_descriptor 之後（per-entity 所有列，與
  typed_resist 同公開原則）。N_V5_TRAIT=4：[pack_tactics, regen/30,
  undead_fortitude, legendary_resist/3]。ENTITY_DIM 101→105。
- `adapt_state_dict_for_obs_v5` 放遷移鏈尾（skill 已 66）：entity_mlp 零填、critic
  逐 slot block 拷貝。**順帶修 v4/skill_dtype adapter 用 live ENTITY_DIM 的潛伏 bug**
  （加 v5 後 live≠該步輸出寬）→ 改用各步自己的凍結寬 `_ENTITY_DIM_V4`。
- **無回歸鐵證**：diag_obsv5_shift 冠軍 **0/4250 動作分歧、WR 46.7==46.7**（種骰）；
  routed 網（champion/evo/assassin）靜態檢查 entity 舊欄不變+新欄零、critic 惰性；
  405 測試綠（2 個既有失敗與本手術無關，stash 驗證）。

## 3. 方法轉折：純 PPO 教不會讀通道

pack_tactics 純 PPO 波（train_monster_actor --p_pack，狼群 vs AoE 隊、pack 隨機
on/off）跑 9 個 policy update 後 **flankON==flankABL 精確相等**（模型零使用通道）＋
std12 −6pp 侵蝕。根因＝(1) 集火是 emergent positioning、PPO credit-assignment 太間接；
(2) ＝12g「PPO 探索不到」重現。→ **改用 BC-seed（12h/12j 已證方法）**。

## 4. regen→melee BC-seed + 因果驗證（核心交付）

- `seed_regen_bc.py`：oracle＝再生→approach+melee／無再生→ranged（reuse move
  skill），teacher-force；switch_demos + warm self-anchor 雙錨（1:1）保職業。
- **冠軍 `models/regen_v5/regen_s0400.pt`**（signal 隨步數單調增 0%→25%→82%→
  84.5%；s0400 是 signal/侵蝕甜蜜點）。

**因果驗證（verify_regen_causal.py，三條件共用種骰）**：

| creature | role | meleeON | meleeOFF | meleeABL | WR_ON | WR_ABL |
|---|---|---|---|---|---|---|
| hill_giant | trained | **82.0%** | 0.0% | **0.0%** | **100%** | 66.7% |
| manticore | held-out | 90.3% | 0.9% | 17.6% | 50.0% | 66.7% |
| skeleton | held-out | 15.8% | 0.0% | 0.0% | — | — |

- **meleeON 82% vs meleeABL 0%**：只把 obs 的 `I_TRAIT_REGEN` 欄歸零（引擎再生仍在、
  仍回血），行為**完全退回 kite** → 近戰是「讀通道」造成的＝因果。
- **WR_ON 100% vs WR_ABL 66.7%（+33pp）**：讀到再生→近戰→更高勝率＝**此資訊必定
  帶來增益**。
- **held-out 泛化**：manticore（未訓）melee-when-regen 90%、ablate 退回 17.6%＝因果。
  模型 blind-to-identity（看不到自己是誰）→ 證學到的是「regen→melee 通則」非
  hill_giant 死記。**但過度泛化**：manticore 近戰反而 WR 50<66.7（kit 不同則近戰非
  最優）；skeleton/bandit 轉移弱＝12i「kit×行為綁定」邊界（轉移強度隨 kit 相似度）。

## 5. 無退化

- **手術本身對 production 零退化**：遷移 bit-exact（diag 0/4250 + routed 靜態惰性）。
  官方 eval_goal 11/12 OK，champion −5%＝該類**既有 tie-boundary 擺動**（取樣方差，
  手術 bit-exact→行為未變、catalog 已記錄、守門判讀配 diag 0-分歧已滿足）。
- **demo 冠軍 regen_v5/s0400 自身**：seeded clean_std12 −4.2pp（溫和、可理解＝demos
  教 hill_giant 特定 approach+melee；雙錨壓住）。可用更強錨點/更少步數再降（follow-up）。

## 6. 儀器
- seed_regen_bc.py（oracle BC-seed）/ verify_regen_causal.py（ON/OFF/ABL 種骰）
- probe_regen_skill.py（forced-skill headroom）/ probe_pack_headroom.py /
  probe_regen_headroom.py（盲區+headroom）/ diag_obsv5_shift.py（遷移 gate）
- train_monster_actor.py（--p_pack 狼群席，純 PPO 失敗的記錄）

## 7. 遺留/follow-up
- pack_tactics 仍是最乾淨「純行為兌現」載體，但需多 agent BC oracle（未做）。
- regen_v5 std12 −4.2pp 可用更強錨點再降。
- 其他 v5 通道（pack/undead/legres）已在 obs 但未個別行為驗證（regen 已足夠證明
  「模型能讀被動天賦並改行為」的通用能力）。
