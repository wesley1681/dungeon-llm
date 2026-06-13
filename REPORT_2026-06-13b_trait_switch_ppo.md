# 純獎勵學會「依天賦改行為」：敵抗性 trait → 切傷害型（2026-06-13b）

目標（/goal）：讓模型**從獎勵**學會依天賦改行為，效果須(1)依**不同**天賦改變行為、
(2)行為帶來**獎勵**、(3)**不可** teacher-force「看到某天賦就做某動作」（那樣學不會
泛化、沒意義）。本波直接回應上一波 regen→melee 固定規則被駁回。

**結論：GOAL MET。** 純 PPO（無任何 oracle）學會讀**敵人的抗性/免疫 trait**並切到
非免疫的傷害型；因果（蒙住 trait 通道→退回固定預設輸）、有益（讀到→WR≈100%）、
且以**單一型對稱機制**泛化到 7 種不同 trait 與沒訓過的 kit/武器。

---

## 1. 先查清死路與混淆（CLAUDE.md：先量再開刀）

- **regen→melee 是死載體（結構性）**：全怪物中**只有 hill_giant** 近戰傷害>遠程
  （18.5>15.5），其餘 dual-weapon 怪全是 tie 或遠程較優。所以「再生→近戰」只能在
  單一隻怪成立、對其他怪（manticore…）有害＝無法泛化＝**駁回成因是結構非調參**。
- **fire_lab（caster）純 PPO 3 跑全盲**（火%on≈off，logs 留存）。兩道牆：
  (1) **梯度抵消**——非條件化的共享 logit 在 immune/normal 兩半得到相反梯度互相抵消；
  (2) **逃生口/可贏性混淆**——life cleric 真正輸出靠 spirit_guardians（光耀光環）＋
      spiritual_weapon（**力場**），力場閃過 斬擊/光耀 兩種注入免疫＝切招永遠**非必要**；
      caster 的 magic_missile 受**法術位**限制，免疫半幾乎不可贏＝無 terminal 梯度。
  （forced-policy probe 看似乾淨只因「強制」遮蔽了第三選項——典型混淆。）

## 2. 乾淨載體：二武器近戰 kit（無逃生口、無 hedge）

唯一傷害來源＝兩把 at-will 武器，**無法術逃生口**。注入免疫其中一型→另一把是唯一
解；無 hedge（always-A 輸 A-immune、always-B 輸 B-immune）。orc@L6 forced-policy 驗證：
| 注入 | force-斬擊 WR | force-穿刺 WR |
|---|---|---|
| 無 | 100% | 100% |
| 斬擊免疫 | **0%** | 100% |
| 穿刺免疫 | 95% | **0%** |
baseline（pop_u0005，切招天真）對 斬擊免疫 死命揮 斬擊 305 次／WR 3%＝乾淨起點。

## 3. 純 PPO 學得到（推翻「PPO 探索不到」）

`train_switch_lab.py`：身分盲、敵免疫**隨機注入**、獎勵＝殺死 orc，**無 oracle / 無切招
標籤**（switch 行為若出現＝純從獎勵 emergent）。

- **單 type-pair（斬/穿）**：~10 update 即學會。desc-ON 兩免疫 case WR 100%；**ablate
  敵 descriptor → 退回固定預設、WR 0–4%**＝行為由「讀通道」驅動（非反應式：蒙住後並
  未靠「揮空→改招」補救，而是死守預設輸）。matchup-join 權重 0→+0.26。
- **這推翻先前「PPO 探索不到」的信念**：先前失敗是**載體混淆**（逃生口／法術位不可贏），
  非根本探索牆。給足「條件化才是唯一解」的乾淨 regime，PPO 自己長出 join。

## 4. 泛化：擴訓多型 → join 長強 → 跨 trait/kit 泛化

單 type-pair 不泛化至 held-out 型（join 僅 +0.26）。擴訓 **8 kit / 7 型**
（斬擊·穿刺·火·冰·雷鳴·毒·光耀，閃電/強酸保留 held-out），純 PPO 60 update：

- **skill-join 單調長到 +0.531**（≈12j BC 的 +0.59），desc-norm 0.13→0.26。
- **u60 全 8 kit × 7 型**：desc-ON `wrong%=0%`（從不用免疫型）、WR **94–100%**；
  ablate → 退回預設、WR 0%。＝同一機制依**不同 trait**選不同武器，皆有益、皆因果。
- **跨 kit 泛化（乾淨測）**：held-out kit `手斧+匕首`（**新武器**、訓練型 斬/穿）：
  穿刺免疫 desc-ON `wrong%0%/WR100%`、ablate `wrong%100%/WR0%`＝機制非死記武器、
  靠 trait 通道轉移到新武器。

## 5. 邊界（誠實）：零樣本到「從未注入型 × 從未見武器」仍部分

probe_heldout_switch（閃電/強酸免疫 + 從未見武器 閃電刃/酸蝕之刃/棍棒）：on/off 出現
小幅 gap（join 有在轉移），但未乾淨切招。**根因＝混淆而非機制缺陷**：(a) 模型無法**估值
從未見過的武器**（normal 半也對 zap_club 選較弱的 鈍擊 100%）；(b) held-out kit 是
weapon+**法術**結構，與全近戰訓練分布不同。此即 12i 已記錄的 kit-binding 邊界；12j BC
是用**遠廣**（oracle 驅動全職業）覆蓋才補上。純 PPO 要補需同樣擴大武器/結構覆蓋。

## 6. 對照與交付

| | 上波 regen→melee（駁回） | 本波 trait→切招 |
|---|---|---|
| 訓法 | teacher-force 固定規則 | 純獎勵 PPO，無 oracle |
| 行為 | 「有再生＝一律近戰」 | 「敵免疫X→用非X」依 trait 變 |
| 泛化 | 無（只 hill_giant、害他怪） | 跨 7 型＋新 kit/武器（單一 join 機制）|
| 有益 | manticore WR 反掉 | desc-ON WR≈100% vs ablate 0% |

冠軍 `models/switch_lab3/sw_u0060.pt`（join +0.531）。儀器：`train_switch_lab.py`
（多 kit 純 PPO 切招 lab）、`probe_dtype_regime.py`/`probe_immunity_regime.py`
（regime 量測）、`probe_heldout_switch.py`（零樣本，複用 12j）。

## 7. 遺留 follow-up
- **std12**：本波為能力 lab、無 anchor，切招特化會侵蝕 std12（anchor 機制 fire_lab
  已證可保）；生產版＝加鮮 self-anchor 重跑（機械式）。
- **零樣本到全新型**：擴武器/結構覆蓋（多法術 kit、更多型）可望補上，與 12j BC 同理。
