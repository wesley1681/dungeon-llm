# MONSTER_CATALOG — CR 0–30 怪物目錄與機制拆解

> 建立：2026-06-12。怪物階段的錨定文件：接下來的引擎/RL 改動全部圍繞這份文件排程。
> 流程：挑機制原語 → 引擎實作＋單測 → 怪物用 ClassDef 組裝 → 進 population 訓練池 → 評測驗收。

## 0. 選取原則

- **不求全收錄，求機制空間覆蓋**：每個 CR 帶挑代表性怪物，標準是「它帶來引擎還沒有的戰鬥機制」或「它是該數值帶的典型底盤」。
- **每個新原語至少要有 2 隻怪使用**才值得實作（design vigilance：不為單一怪物寫一次性程式碼）。
- **怪物 = 合成 ClassDef**：沿用 `scripts/synth_identity.py` 的路線——底盤＋技能包＋trait，不開新的怪物類別系統。身分 one-hot 全零（B 網 blind 學生已驗證可吃此條件）。

## 1. 引擎現況基線（2026-06-12 盤點）

怪物動作要對照的「已有資產」：

| 資產 | 現況 |
|---|---|
| ClassDef 資料驅動註冊表 | 12 職業；`TRAIT_APPLIERS` 11 種：extra_attack, crit_range, sculpt_spells, sneak_attack, lay_on_hands, aura_of_protection, portent, uses_flat, uses_from_modifier, uses_table, starting_status |
| ABILITY_REGISTRY | ~60 個技能，全引擎可執行 |
| TargetType | SELF / SINGLE_ENEMY / SINGLE_ALLY / POINT / LINE / CONE / MULTI_ENEMY / MULTI_ALLY（LINE/CONE 解碼走 POINT 語義） |
| 狀態池（status.py） | prone, paralyzed, stunned, poisoned, restrained, frightened, charmed, blinded, asleep ＋增益類 ~28 種 |
| 傷害管線 | `on_incoming_damage` / `on_outgoing_damage` Modifier 掛鉤；`damage_resistance` 0..1 連續值（rage/bear_totem/uncanny_dodge 在用）；dtype 已傳入掛鉤但現有 applier 不分型 |
| 持續效果 tick | spiritual_weapon、spirit_guardians（回合 tick 基建存在） |
| 行動經濟 | action ＋ bonus action；反應類用狀態近似（shield、uncanny_dodge）；專注＋受傷 CON 豁免 |
| 瀕死規則 | 5e dying / death saves / 重擊雙失敗 / 溢傷即死 / 救起 |
| 強制位移 | pushing_attack |
| 資源 | 法術位表、uses（flat / from_modifier / table） |
| 地形 | blocked / difficult / dangerous ＋ LoS；obs 有 terrain 通道；合法性遮罩全套（grid/entity/skill） |
| 對局 | 1v1、3v3、1v3（`_TEAM_CONFIGS`）、asym ΔL（LEVEL_DELTA_TABLE） |

## 2. CR 0–30 怪物總表

「動作」列只寫戰鬥機制相關項；〔〕內是機制原語標籤，對應 §3。

### Tier 1 — 雜兵（CR 0–1）

| CR | 怪物 | 戰鬥動作與特性 | 原語標籤 |
|---|---|---|---|
| 0 | 平民 Commoner | 棍擊 | 純底盤 |
| 1/8 | 狗頭人 Kobold | 匕首／投石；**群體戰術**（鄰接盟友時優勢）；日光敏感 | 〔pack_tactics〕 |
| 1/8 | 強盜 Bandit | 彎刀／輕弩 | 純底盤 |
| 1/4 | 哥布林 Goblin | 彎刀／短弓；**靈巧逃脫**（bonus 撤離或躲藏） | 已有（cunning_action 系列） |
| 1/4 | 骷髏 Skeleton | 短劍／短弓；鈍擊**易傷**；毒**免疫** | 〔typed_resist_table〕 |
| 1/4 | 殭屍 Zombie | 重擊；**不死堅韌**（歸零時 CON 豁免留 1 HP） | 〔undead_fortitude〕 |
| 1/4 | 狼 Wolf | 撕咬＋**擊倒**（STR 豁免否則倒地）；群體戰術 | 〔on_hit_rider〕〔pack_tactics〕 |
| 1/2 | 獸人 Orc | 巨斧；**兇蠻**（bonus 朝敵衝刺） | 已有（dash pattern） |
| 1/2 | 黑影 Shadow | 接觸＋**力量吸取**（STR 永久遞減） | 〔stat_drain〕 |
| 1 | 食屍鬼 Ghoul | 爪擊＋**麻痺**（CON 豁免，回合末重豁）；撕咬 | 〔on_hit_rider〕（paralyzed 已有） |
| 1 | 恐狼 Dire Wolf | 撕咬＋擊倒；群體戰術（狼系列的高數值版） | 同狼 |

### Tier 2 — 精英（CR 2–4）

| CR | 怪物 | 戰鬥動作與特性 | 原語標籤 |
|---|---|---|---|
| 2 | 食人魔 Ogre | 巨棒（大數值單發） | 純底盤 |
| 3 | 梟熊 Owlbear | **多重攻擊**（喙＋爪） | 已有（extra_attack） |
| 2 | 石像鬼 Gargoyle | 多重攻擊；**非魔法武器抗性** | 〔typed_resist_table〕 |
| 3 | 狼人 Werewolf | 多重攻擊；**非銀武器免疫**（簡化為高額抗性） | 〔typed_resist_table〕 |
| 3 | 屍妖 Wight | 長劍×2；**生命吸取**（傷害＋maxHP 同額遞減） | 〔max_hp_drain〕 |
| 3 | 石化蜥蜴 Basilisk | 撕咬；**石化凝視**（兩段式：束縛→石化） | 〔petrify〕（可先簡化 restrained→paralyzed） |
| 3 | 蠍尾獅 Manticore | 多重攻擊；**尾刺齊射**（遠程 3 發） | 已有（scorching_ray pattern） |
| 4 | 女妖 Banshee | **慟哭**（AoE WIS 豁免，失敗直接歸零）；穿牆移動（暫緩） | 〔save_or_drop〕（AoE＋瀕死規則組合） |
| 4 | 雙頭巨人 Ettin | 多重攻擊（戰斧＋晨星）；不可背刺 | 已有 |

### Tier 3 — 戰役級（CR 5–10）

| CR | 怪物 | 戰鬥動作與特性 | 原語標籤 |
|---|---|---|---|
| 5 | 巨魔 Troll | 多重攻擊×3；**再生 10/回合**（火/酸傷該回合抑制） | 〔regeneration〕 |
| 5 | 丘陵巨人 Hill Giant | 巨棒×2；**擲石**（遠程大傷） | 已有 |
| 5 | 火元素 Fire Elemental | 接觸燃燒（**命中附加持續燃燒**）；**火免疫**；身體即火焰光環 | 〔on_hit_rider(DoT)〕〔typed_resist_table〕〔damage_aura〕（spirit_guardians pattern） |
| 6 | 雙足飛龍 Wyvern | 多重攻擊；**螫刺＋劇毒**（CON 豁免 7d6 毒，半傷） | 〔on_hit_rider〕 |
| 6 | 法師 Mage（NPC） | 火球／魔法飛彈／護盾／緩速 | 已有（純組裝） |
| 8 | 刺客 Assassin（NPC） | **死亡突襲**＋**毒刃**（命中附加大額毒傷） | assassinate 已有＋〔on_hit_rider〕 |
| 9 | 火巨人 Fire Giant | 巨劍×2；擲石；火免疫 | 〔typed_resist_table〕 |
| 10 | 幼紅龍 Young Red Dragon | 多重攻擊×3；**火焰噴吐**（CONE，**充能 5–6**） | 〔breath_recharge〕（CONE 已有） |
| 10 | 石魔像 Stone Golem | 多重攻擊；**緩速**（AoE 減速減行動）；魔法免疫＋多數狀態**免疫** | 〔condition_immunity〕 |

### Tier 4 — 傳奇入門（CR 11–16）

| CR | 怪物 | 戰鬥動作與特性 | 原語標籤 |
|---|---|---|---|
| 11 | 貝希爾 Behir | 多重攻擊；**閃電吐息**（LINE，充能）；緊抱→**吞噬**（體內持續酸傷） | 〔swallow_grapple〕〔breath_recharge〕（LINE 語義落地） |
| 12 | 大法師 Archmage（NPC） | 高環法術全套（連鎖閃電、寒冰錐…） | 已有底盤＋補高環法術 |
| 13 | 吸血鬼 Vampire | 多重攻擊；**魅惑**；**吸血**（maxHP 遞減＋等額自療）；**再生**；**傳奇行動×3** | 〔max_hp_drain〕〔regeneration〕〔legendary_actions〕（charmed 已有） |
| 13 | 眼魔 Beholder | **眼柱射線**（每回合隨機 3 發，10 種效果表）；反魔法錐（暫緩） | 〔multi_ray_table〕 |
| 14 | 成年白龍 Adult White Dragon | 多重攻擊；冰吐息；**恐懼威壓**；**傳奇行動**；**傳奇抗性** | 〔frightful_presence〕〔legendary_actions〕〔legendary_resistance〕 |
| 16 | 鐵魔像 Iron Golem | 多重攻擊；**毒氣噴吐**（充能）；**吸收火焰**（受火傷改回血） | 〔absorb_element〕〔breath_recharge〕 |

### Tier 5 — 史詩（CR 17–24）

| CR | 怪物 | 戰鬥動作與特性 | 原語標籤 |
|---|---|---|---|
| 17 | 成年紅龍 Adult Red Dragon | 完整傳奇套件：多重攻擊＋恐懼威壓＋火吐息＋傳奇行動/抗性 | Tier 4 原語的數值放大驗證 |
| 17 | 死亡騎士 Death Knight | 多重攻擊×3＋不潔懲擊；**毀滅之球**（AoE）；命令光環 | smite 已有＋〔ally_aura〕 |
| 19 | 炎魔 Balor | 火鞭（拉拽）＋火劍；**死亡爆炸**（死亡時 AoE 大傷） | 〔death_throes〕〔forced_move〕 |
| 20 | 深淵領主 Pit Fiend | 多重攻擊×4（咬附**持續毒**）；**恐懼光環**（被動，範圍內持續豁免） | 〔passive_aura〕〔on_hit_rider〕 |
| 21 | 巫妖 Lich | 高環法術＋**麻痺之觸**；傳奇行動（含插入施法）；傳奇抗性 | 施法者×傳奇套件組合驗證 |
| 21 | 熾天使 Solar | 飛行（抽象）；**滅殺長弓**（命中附帶死亡豁免）；治療之觸；傳奇行動 | 〔save_or_drop〕變體 |
| 23 | 海妖 Kraken | 觸手×3（**擒抱→吞噬**）；**閃電風暴**（多點落雷）；傳奇行動 | 〔swallow_grapple〕〔multi_point_aoe〕 |
| 24 | 遠古紅龍 Ancient Red Dragon | 紅龍套件最終數值帶（HP 546 / AC 22 / DC 24） | 數值邊界驗證用 |

### Tier 6 — 神話（CR 25–30）

| CR | 怪物 | 戰鬥動作與特性 | 原語標籤 |
|---|---|---|---|
| 26 | 木乃伊君王型 Boss（如 Marut/Solar 變體） | 高 DC 控場＋傳奇套件（佔位：此帶機制與 Tier 5 同構，主要是數值） | 數值邊界 |
| 30 | 塔拉斯克 Tarrasque | 多重攻擊×5；**吞噬**；**反射甲殼**（線/射線反彈，暫緩）；恐懼威壓；傳奇行動/抗性；HP 676 | 全原語壓力測試＋obs 出界極端案例 |

## 3. 機制原語拆解（核心表）

把 §2 所有〔標籤〕去重後按工程量分級。**這張表就是怪物階段的工作清單。**

### A 級 — 零引擎改動，純 ClassDef 組裝

| 原語 | 用例怪物 | 落點 |
|---|---|---|
| 多重攻擊 | Owlbear, Ogre+, 各巨人/龍 | `extra_attack` trait |
| bonus 衝刺/撤離/躲藏 | Goblin, Orc | cunning_action 系列 |
| 豁免型 AoE（CONE/POINT） | 龍吐息形狀, 毀滅之球 | burning_hands / fireball pattern |
| 控場條件全套 | Ghoul(麻痺), 龍(恐懼), Vampire(魅惑)… | 狀態池 9 種負面全有 |
| 持續傷害光環 | 火元素, 深淵領主 | spirit_guardians pattern |
| 自動命中多發 | Manticore 尾刺 | magic_missile / scorching_ray pattern |
| 額外傷害骰 | Assassin, Wight | sneak_attack trait |
| NPC 施法者 | Mage, Archmage, Lich(底盤) | caster chassis |
| 怪物無身分 one-hot | 全部 | blind 學生已驗證 |

### B 級 — 既有掛鉤上的小擴充（各 ≤1 天）

| 原語 | 用例怪物（≥2 檢查） | 引擎落點 |
|---|---|---|
| typed_resist_table（按傷害類型的抗性/免疫/易傷） | Skeleton, Gargoyle, 狼人, 火系×3 | `on_incoming_damage` 已收 dtype；新 trait applier 帶 {dtype: 倍率} 表 |
| on_hit_rider（命中附帶豁免/狀態/額外傷/DoT） | Wolf, Ghoul, Wyvern, Pit Fiend, 火元素 | trip_attack/menacing_attack 已是此 pattern → 泛化成資料欄位 |
| pack_tactics（鄰接盟友時優勢） | Kobold, Wolf, Dire Wolf | 優勢系統已有；新 trait applier 加鄰接判定 |
| undead_fortitude（歸零時豁免留 1） | Zombie（＋未來不死系） | `apply_damage` 歸零點 hook |
| frightful_presence（AoE 恐懼，豁免後當日免疫）✅W3 | 全部成年龍, Tarrasque | 受害者回合首 aura（tick_aura_damage＝spirit guardians 同 chokepoint，零接線）；成功＝本場免疫（victim 側 `frightful_immune_to`）；DC=8+熟練+CHA 由面板實算 |
| breath_recharge（d6 充能 5–6） | 幼龍～遠古龍, Behir, Iron Golem | uses 基建＋回合開始充能擲骰 |
| death_throes（死亡時爆炸）✅W3 | Balor（＋炸藥桶類） | `apply_damage` NPC 死亡點引爆（spec 先 pop＝鏈式爆炸防重入）；事件走 `ws.pending_events` 供顯示層汲取 |
| stat_drain / max_hp_drain | Shadow, Wight, Vampire | Modifier 寫入；**注意 obs 的 maxHP 特徵會動態變化** |
| condition_immunity（狀態免疫表） | 魔像×2, 骷髏, 不死系 | apply_status 入口處查表 |

### C 級 — 新子系統（各 2–5 天）

| 原語 | 用例怪物 | 設計要點 |
|---|---|---|
| regeneration（回合開始回血，特定傷型抑制）✅W2 | Troll, Vampire | tick 基建可借 spiritual_weapon；需「本回合受過火/酸」旗標 |
| swallow_grapple（擒抱→吞噬→體內 DoT）✅W2 | Behir, Kraken, Tarrasque | restrained 已有；簡化版：擒抱=束縛＋同格拖行，吞=束縛＋DoT＋目盲 |
| multi_ray_table（隨機效果表多射線）✅W2 | Beholder | 技能執行時擲表；效果複用狀態池 |
| legendary_resistance（X/日自動通過豁免）✅W3 | 成年龍×3, Lich, Tarrasque（海妖按 MM 無） | `make_saving_throw` 失敗點改判；**只燒在「失敗會留狀態」的豁免**（`fail_applies_status` 旗標＝保護行動經濟的 MM 戰術語義，絕不為半傷燒）；含自動失敗與 save_each 掙脫 |
| absorb_element（特定傷型轉治療）✅W1 順手 | Iron Golem | typed_resist_table 的倍率允許負值即可，順手做 |
| LINE 目標語義真落地 ✅W2 | Behir, 閃電束 | TargetType.LINE 目前 fall back POINT；補射線取目標 |
| petrify 兩段式（束縛→石化）✅W2 | Basilisk, Beholder 石化射線 | `escalates_to` 資料欄；petrified=STATUS_SLOTS 既有保留位（obs 可見、維度不動） |

### D 級 — 行動經濟/規則層（大，≥1 週）

| 原語 | 用例怪物 | 影響面 |
|---|---|---|
| legendary_actions（他人回合末插入行動）✅W3 | 龍×3, Lich, Kraken, Tarrasque, Beholder（7 用戶；Vampire 留未來純組裝） | `run_legendary_actions` orchestrator＋全 driver 回合尾接線（env_v2×5/sandbox/bc_collect/game.py）；回充走 tick self_turn_start 零接線；**obs 不加欄**——剩餘次數＝隱藏資源（§4.1 公開/隱藏原則），§4 #5 就地解決 |
| summon/spawn（戰鬥中加實體） | 召喚系, 分裂怪 | env 動態 roster；obs 10 槽上限——**怪物階段唯一未落地的 D 級原語**（無 W3 用戶；留給召喚波） |

### E 級 — 暫緩（明確不做的理由）

| 原語 | 理由 |
|---|---|
| flight 真 3D | 戰場是 2D；若需要可抽象成「無視地形＋遠程規避」，等有 ≥2 個非龍用例再議 |
| 反魔法錐、變形、轉陣營魅惑 | 規則層深、單一用例多；魅惑目前用「不能攻擊魅惑者」近似已夠 |
| 反射甲殼（Tarrasque） | 單一用例；Tarrasque 沒它也夠強 |
| 穿牆/虛體移動 | 會破壞剛建好的 LoS/blocked 合法性遮罩假設，需連動改 |

## 4. RL / obs 邊界（開工前必查清單）

> **2026-06-12d 狀態：obs v4 三合一手術已完成（遷移層）。** LEVEL_NORM 20→40、
> MAXHP_NORM 100→800（皆 2 的冪比例→權重欄補償在乘積層位元級精確）；描述子
> 31 維（`obs.capability_descriptor`，聚合自 `skill.kit_features`）附加於列尾；
> 命名欄位常量（`I_ENT_LEVEL`/`I_ENT_MAXHP`/`ENT_DESC_START`）取代尾端負偏移。
> 舊 ckpt 經 `adapt_state_dict_for_obs`（v2→v3→v4 鏈，所有 loader 共用）載入：
> **diag_obsv4_shift 6 目標（含 v3 敏感戶 berserker/devotion、battle_master、
> B 網）10,459 共享狀態 0 動作分歧**——遷移在函數層面與 v3 等價。
>
> **守門判讀（官方 10g ×3 輪＋B 網）**：單輪 10g 有 1–2 戶在 ±9 帶邊緣進出
> （bm 三輪 −9/−11/−21、berserker −10/−4/−4），但 (a) 函數等價已證、(b) 內部
> 對照——**專家臂程式路徑零變動卻同樣擺動**（專家 bm 三輪 57/57/62 vs pre-v4
> 高精度 54），(c) eval_goal 印的 ±9 帶是單臂 σ 套在雙臂差上（正確 2σ(diff)
> ≈±13）、且 hash 種子每輪換樣本——結論：**v4 零回歸，出帶是協議方差**；席位
> 級判定本來就要 1728/arm（既有準則）。B 網守門輪廓與 pre-v4 記錄同形狀
> （assassin −22 vs −21、devotion −11 vs −16、mean +1.5 vs +2.6）。
> 證據檔：diag_obsv4_shift.txt、goal_1v1_obsv4_official10{,_run2,_run3}.txt、
> student_popb2_u05_obsv4_official10.txt、probe_descriptor_info.txt。
>
> 術後網路的描述子輸入權重為零——描述子的收益要等下一波訓練（population/BC）
> 才會兌現；§4.1 驗證 2（敵方陌生 A/B 的「有描述子」臂）同樣等訓練後執行。
> rl_smoke BC→PPO 管線 v4 下全通；BC 不落盤 obs，無舊資料集失效。

1. **maxHP 出界（2026-06-12 已實測，緊急度下修）**：`MAXHP_NORM=100` 且無 clip（obs.py:262）。Wave 0 實測（`scripts/probe_obs_bounds.py`，B 網冠軍，存 `eval_results/probe_obs_bounds.txt`）：(a) 特徵審計——Wave 0 僅 hill_giant 出界（1.05），archmage 0.99 壓線；(b) 靈敏度掃描——把敵列 maxHP 特徵合成拉到 6.76（Tarrasque 等級），技能分布 TV 距離 mean ≤0.025 / max 0.066，**無失控外推**；(c) 同種子行為 A/B（n=120/臂）——裁剪 vs 不裁剪 ΔWR = 0.0%。結論：**出界不是穩定性險情，v4 重定標的動機是資訊量**（讓模型能區分 200 vs 700 血）＋描述子，與 #7 合併做，不需為 Wave 0 緊急開刀。選項仍是 log 尺度/提高 NORM/分段；動 obs = 走 v3 遷移流程（手術＋`diag_obsv3_shift` 全套）。
2. **CR→等級映射**：`LEVEL_NORM=20`，CR 30 → 1.5 出界。需定義 CR→「等效等級」表（對位用，含 asym LEVEL_DELTA_TABLE 擴充）；5e 慣例 CR≈4 人隊伍等級，1v1 對位需自己標定（用 HP×DPR 錨點），列為開工後第一個測量項。
3. **AC / DC 數值帶**：怪物 AC 至 25、DC 至 24+，超出職業帶；查 entities AC 特徵的正規化。
4. **maxHP 動態變化**（B 級 drain 原語）：obs 的 per-entity maxHP 一旦會中途下降，模型沒見過——進訓練池即可，但 probe 要加 case。
5. **傳奇行動 obs 欄位**：排 D 級時一併設計，不提前佔位。
6. **自身身分條件**：one-hot 全零＝怪物，B 網（blind）天生支援 ✅；population 訓練池直接混入怪物 ClassDef 即可。
7. **敵方身分盲區（已確認，與 #1 合併成 obs v4 一次手術）**：敵方實體列的 archetype multi-hot 只認 12 標準職業（`ARCHETYPE_OBS_LIST` 凍結），陌生敵人全零。模型對陌生敵人只看得到底盤數字（HP/maxHP/AC/等級/位置）＋已掛狀態＋專注旗標；看不到技能組、資源、抗性、武器型態，且前饋網路無跨回合記憶，無法「打著打著認出來」。歷來所有陌生身分實驗都在自己側（B 網吃陌生的自己、對手永遠是標準 12）——**敵方側陌生從未被測過**。對怪物致命：骷髏毒免疫、巨魔怕火，模型沒有任何感知通道。解法見 §4.1。

### 4.1 敵方能力描述子（capability descriptor，obs v4 核心新增）

**原則**：不擴 one-hot（每加一隻怪就改一次 obs，不可擴展），改為把實體的技能組聚合成固定長度的機制摘要，全部從 ClassDef／`SkillFeatures` 資料驅動生成——任何 synth/怪物自動獲得，零硬編碼。DnD 語境：外觀可推斷的資訊（拿什麼武器、像不像施法者、體型威脅）＝公開；資源消耗狀態（剩幾個法術位、充能、傳奇抗性餘次）＝隱藏。

**聚合來源**：實體的完整技能 kit（擁有的技能定義），**不過濾當前資源可用性**——`remaining_uses`／法術位剩量不進描述子，避免洩漏隱藏資訊。

**欄位設計**（per-entity，約 30 維）：

| 欄位 | 維度 | 聚合方式（對 kit 內所有技能） | 回答的問題 |
|---|---|---|---|
| max_hit_damage | 1 | max(expected_damage) / DMG_NORM | 最大單發威脅多大 |
| max_heal | 1 | max(expected_healing) / HEAL_NORM | 會不會治療、量級 |
| max_attack_range | 1 | max(range_m where expected_damage>0) / RANGE_NORM | 近戰還是遠程 |
| max_aoe_radius | 1 | max(aoe_radius_m) / AOE_NORM | 有無 AoE、多大 |
| best_attack_bonus | 1 | max(attack_vs_ac) / ATK_NORM | 命中威脅 |
| max_save_dc | 1 | max(save_dc) / DC_NORM | 豁免威脅 |
| is_caster | 1 | any(cost_slot_level > 0) | 像不像施法者 |
| has_teleport | 1 | any(is_teleport) | 能不能無視牆位移 |
| grants_actions | 1 | any(grants_actions > 0) | 有無行動爆發（surge 類） |
| applies_status 聯集 | 16 | OR over kit 的 applies_status | 能上哪些控場（麻痺/恐懼/…） |
| save_stat 聯集 | 6 | OR over kit 的 save_stat one-hot | 會打你哪些豁免 |
| typed_resist 摘要 | 13 | per-type (multiplier−1)：0=中性、−0.5 抗性、−1 免疫、+1 易傷、<−1 吸收；來源 `char.damage_multipliers` | 它怕什麼、不怕什麼 |

> 2026-06-12e：typed_resist 欄位已落地（描述子 31→44 維，趁訓練前零權重窗口追加；
> diag_obsv4_shift 重驗 0 分歧）。傷害類型凍結清單＝`engine/damage.py DAMAGE_TYPES`（13 型）。

**放置**：所有實體列（self/ally/enemy）對稱追加——ally 描述子順帶服務 3v3 協作（隊友會不會治療不再靠 one-hot 先驗），self 列與 skills_obs 冗餘但保持結構一致、不特判。維度成本 ~30 × 10 槽 = ~300 維。

**與 archetype multi-hot 的關係**：初版**保留共存**（描述子只加資訊，不動既有通道，避免標準 12 回退）。長期描述子應能取代 one-hot——遷移穩定後做消融：拿掉 one-hot 測標準 12 是否回退，用數據決定。

**遷移**：與 #1（MAXHP_NORM）、#2（LEVEL_NORM/CR 映射）併成 **obs v4 一次手術**，走 v3 流程——舊 ckpt 權重手術＋新欄位零初始化＋`diag_obsv3_shift` 全套門檻（v3 教訓：任何 attention 輸入改動都可能單向破壞個別職業，必跑雙向 diag）。

**驗證設計**（開工前先定，不事後補）：
1. **資訊量探針** ✅（2026-06-12d）：1-NN 跨等級還原 archetype 83%，誤判僅戰術孿生對 → 描述子攜帶身分等價資訊。
2. **敵方陌生 A/B** ⚠️ 結果反預期（2026-06-12f，§6.6）：population 訓練後 desc-on vs 敵方描述子歸零，mean Δ≈0；**同態同狀態動作翻轉率僅 0.1%** → 描述子對 **1v1 greedy actor 行為惰性**（描述子在 obs 中確有填入、權重也學起來 norm 0.65，但 argmax 幾乎不依賴它；最可能服務 critic）。
3. **行為級驗收** ⚠️ 同根因未展現（2026-06-12f）：1v1 單武器職業無「依敵變化的可選動作」，描述子無從施力。**兩者的行為收益都需要會造成可選動作的情境**（隊伍目標選擇／多傷害型 agent／狀態vs免疫／全盲對手 regime）——屬下一波針對性訓練，見 §6.6 下一步。

## 5. 落地波次

每波驗收標準：新機制單測全綠＋官方 12/12 不退（goal 協議）＋population probe 無退化。

- **Wave 0 — 純組裝（0 引擎改動）**✅：Commoner / Bandit / Goblin / Orc / Ogre / Owlbear / Ettin / Hill Giant / Mage / Archmage(部分) / Manticore。目標：把「怪物=ClassDef」管線跑通、定 CR→等級映射、確認 obs 邊界實測行為。
- **Wave 1 — B 級原語**✅：Kobold / Wolf / Skeleton / Zombie / Ghoul / Gargoyle / Wyvern / 火元素 / 幼紅龍（無傳奇）。一次解鎖 Tier 1–3 大部分怪物。
- **Wave 2 — C 級**✅（§6.7）：Troll（再生）／ Behir（吞噬＋LINE 吐息）／ Basilisk（石化兩段式）／ Beholder（射線表）。傳奇抗性**延後至 Wave 3**（名單零用戶，frightful_presence 同例）。
- **Wave 3 — D 級**✅（§6.8）：legendary_actions → 成年白龍/成年紅龍/遠古紅龍、Lich、Kraken、Tarrasque（＋Beholder 傳奇化補裝）；併入 legendary_resistance＋frightful_presence＋death_throes（Balor）；1vN Boss 對局基建＋隊伍等效標定（`calibrate_boss.py`）。
- **扮演怪物波**✅（§6.9，2026-06-13）：讓模型**自己扮演怪物**（坐 agent 席）並通用化——純 PPO、零腳本示範（腳本只當 eval 量尺）。怪席 −10/−27pp→−2.2/−2.4pp，archmage/basilisk 超腳本；held-out 同類怪零樣本泛化；冠軍 `mon_actor1/ma_u0032`。
- **obs v4 遷移（maxHP/level 尺度＋敵方能力描述子，§4.1）**：最晚 Wave 0 結束前完成——Wave 0 已含 Ogre(59 HP)、Hill Giant(105 HP)，後者已出界；描述子的敵方陌生 A/B（§4.1 驗證 2）同時補上歷史盲區。

## 6. Wave 0 落地結果（2026-06-12）

**狀態：完成。** 「怪物=ClassDef」管線跑通、CR→等效等級已標定、obs 邊界已實測（見 §4 #1）。

### 交付物

| 件 | 位置 |
|---|---|
| 11 隻 Wave 0 MonsterDef＋9 種天然武器＋註冊協議 | `trpg/scenarios/monsters.py`（`register_monsters()` 先 import obs 凍結 N_ARCHETYPES，再改註冊表——synth_identity 同協議） |
| 通用怪物策略 | `trpg/engine/combat_policy.py::GenericMonsterPolicy`——零技能名/職業名，純 SkillFeatures EV 貪心；Heuristic 基線未動 |
| 測試 | `tests/test_monsters.py` 29 項：註冊不位移 obs、statblock 對 MM（命中/DC/HP/AC 全中）、整場戰鬥 smoke、敵列 one-hot 全零、板凳凍結回歸 |
| CR 標定 | `scripts/calibrate_cr.py`；結果 `eval_results/calibrate_cr_10g.txt`（12 專家板凳 × L1–8 × n=120/級） |
| obs 邊界探針 | `scripts/probe_obs_bounds.py`；結果 `eval_results/probe_obs_bounds.txt` |

### CR→等效對位等級（1v1，n=120/級，50% 交叉內插）

| 怪物 | CR | 等效 L | 怪物 | CR | 等效 L |
|---|---|---|---|---|---|
| commoner | 0 | <1 | owlbear | 3 | 6.0 |
| bandit | 1/8 | <1 | ettin | 4 | **>8** |
| goblin | 1/4 | <1 | hill_giant | 5 | **>8** |
| orc | 1/2 | <1 | mage_npc | 6 | 5.0 |
| ogre | 2 | 4.4 | archmage | 12 | **>8** |
| manticore | 3 | 4.9 | | | |

### 發現

1. **L4→L5 斷層**：所有中型怪的板凳 WR 在 L5 跳升 30–45pp（ogre 34%→76%、owlbear 1%→45%、mage 12%→52%）——對應職業 L5 解鎖 extra_attack/fireball 的功率尖峰。等效等級因此聚在 4.4–6.0 帶。
2. **brute 的 1v1 等效遠超 CR**：ettin/hill_giant/archmage 1v1 全帶 >8（hill_giant 對 L8 專家仍 11%）。數學上自洽：CR 按 4 人隊伍標定，105 HP＋雙擊 3d8+5 對單人就是 2.5 輪秒殺。**結論：CR≥4 的 brute 是 1vN 隊伍內容**（env `_TEAM_CONFIGS` 已支援 1v3），不要塞進 1v1 訓練配對。
3. **EV 貪心自我修正了一個保真度偏差**：引擎遠程武器一律用 DEX（5e 投擲武器吃 STR），丘陵巨人 DEX 8 → 擲岩 -EV → 策略正確棄用、全程近戰。蠍尾獅（DEX 16）正常遠程風箏、法師開局連射火球。投擲武器 STR 規則＝未來小工單，目前無行為損害。
4. **板凳沙拋陷阱（已根治）**：`ARCHETYPE_LIST`/`ARCHETYPE_OBS_LIST`/`STANDARD_IDS` 原本快照「活」註冊表，import 順序在怪物註冊之後就被污染——第一版標定的「12 人板凳」實際是 23 人（怪物加入了自己的標定）。修法：`archetypes.py` 模組體內凍結 `STANDARD_ARCHETYPES` 常量，obs/env_v2/synth_identity/train_team 全改吃它；回歸測試鎖死。**教訓：「標準 12」語義必須來自凍結常量，不准快照活註冊表。**
5. obs 邊界：見 §4 #1——僅 hill_giant 出界（1.05），三層實測證明無穩定性風險，v4 動機改為資訊量。

### Wave 0 已知近似（記錄於 monsters.py docstring）

多重攻擊混合（喙+爪→單武器×N）；Orc Aggressive≈bonus dash；Bandit 輕弩→短弓；法師/大法師法術組受限於現有 L1–8 registry（大法師=「部分」版）；尾刺/擲岩無彈藥消耗。

## 6.5 Wave 1 落地結果（2026-06-12e）

**狀態：完成。** 6 個 B 級原語＋12 隻新怪物（合計 23 隻）；frightful_presence／death_throes 按「≥2 用戶」規則延後（Wave 1 名單無用戶）。

### 原語落地點（全資料驅動，零怪物名硬編碼）

| 原語 | 落點 | 驗證 |
|---|---|---|
| typed_resist_table | `engine/damage.py` 13 型凍結清單＋`Character.damage_multipliers`，apply_damage 按型倍率（0.5/0/2/負值=吸收，typo 註冊即炸） | 單測：骷髏鈍擊×2/毒免疫；吸收回血 |
| weapon on_hit rider | `Weapon.on_hit` 資料欄＋復用既有 trip/menacing rider 機制；新增傷害（save_half）/stat drain/maxHP drain 執行器 | 狼擊倒/食屍鬼麻痺（save_each 有界）/飛龍 7d6 毒半傷/黑影 STR 吸取/屍妖血上限吸取 |
| pack_tactics | `_ally_adjacent_to` helper（與 sneak attack 共用）＋resolve_attack mode 種子 | 雙狼鄰接=優勢、隊友走開=消失 |
| undead_fortitude | apply_damage 歸零點 CON 豁免留 1 HP；光耀/暴擊穿透 | 單測雙向 |
| condition_immunity | `Character.add_status` 入口查表（狀態名不存在即炸） | 殭屍毒免疫/可被擊倒 |
| breath_recharge | `tick_status_effects` self_turn_start（全 driver 唯一共同 chokepoint，零接線） | 充能 5-6 擲骰雙向＋實戰吐息/爪擊輪換 |

火焰吐息走 SPELL 管線，靠兩個新資料欄位而非特例分支：`scales_as_cantrip=False`（level=0 享免法術位/不可反制，但不吃戲法等級縮放）＋`save_dc_ability="CON"`（天然能力 DC=8+熟練+CON，非施法者豁免「不是施法者」門）。

### 【重大旁收】限次能力扣次下沉（基線斷點事件）

寫龍吐息時發現：**對手側、BC 示範收集、v1 env 從未扣過 ability_uses**——實測對手 battle_master 每場 action_surge 4–7 次（上限 1 次/短休）。也就是說 2026-06-12e 之前的所有對手與 BC 示範都是「無限資源版」。修法＝扣次下沉到 `execute_action`（唯一執行漏斗，成功才扣），移除 env_v2/sandbox/human-cmd 三處散裝副本。**影響**：所有歷史絕對 WR 不可與 e 後直接比（對手變守規後雙方絕對 WR 都上移）；within-run 的 model-vs-expert diff 指標不受影響。修復後官方面板（goal_1v1_wave1_usesfix_official10.txt）：11/12 OK，唯 champion −10 壓 ±9 帶——它是文件既載的 1v1 平手邊界職業，且在正確雙臂 2σ（±13）內。

### CR→等效對位等級（含 Wave 1，扣次修復後重標定，n=120/級）

| 怪物 | CR | 等效 L | 怪物 | CR | 等效 L |
|---|---|---|---|---|---|
| commoner/bandit/kobold | 0–⅛ | <1 | gargoyle | 2 | 4.7 |
| goblin/wolf/skeleton/zombie | ¼ | <1 | wight | 3 | 4.6 |
| orc/ghoul | ½–1 | <1 | manticore | 3 | 4.9 |
| shadow | ½ | **2.9** | owlbear | 3 | 6.6 |
| dire_wolf | 1 | 4.1 | mage_npc | 6 | 6.0 |
| ogre | 2 | 4.2 | ettin/hill_giant/fire_elemental/wyvern/young_red_dragon/archmage | 4–12 | **>8** |

**發現：抗性表是 1v1 的力量倍增器。** shadow（CR½→2.9）、fire_elemental（CR5→>8）、wyvern（CR6→>8，毒 rider）全部大幅超出 CR 對位——標準職業的輸出以物理為主，物理抗性 0.5 等於把全班 DPS 砍半，而 4 人隊伍裡會有施法者把它正規化。幼紅龍 1v1 全等級 0%（178HP/AC18/16d6 吐息，正確地不可單挑）。再次確認：**CR≥4 的 brute/元素/龍是 1vN 隊伍內容**。

### Wave 1 已知近似

屍妖以「生命吸取」為唯一武器組（MM 可選長劍；EV 貪心策略會棄用弱武器，故砍掉選項保行為保真）；血上限吸取不掛 CON 豁免（5e 有）；火元素灼燃簡化為命中附帶 1d10（MM 為回合末 tick）；幼紅龍三連擊用單一「龍之爪牙」2d8 近似咬+雙爪。

## 6.6 Population 訓練波落地結果（2026-06-12f）

**目標**：把 23 隻怪物混入 B 網（盲蒸餾單網）的 population PPO 對手/身分池，讓 obs v4 手術後歸零的描述子輸入權重開始學習；跑 §4.1 驗證 2（敵方陌生 A/B）與驗證 3（行為級抗性）。

### 配方（`scripts/train_population.py` 擴充，全資料驅動）
- **公平對位**：怪物 1v1 等效等級為**實測量**（`monsters.py:EQUIV_LEVEL_1V1`，源 `calibrate_cr_wave1_10g.txt`，含 fail-loud 覆蓋斷言——新怪未標定就無法靜默進池）。怪物用 `natural_level`、對面職業/synth 用 `round(equiv_level)`，靠 env_v2 既有非對稱 `level`/`opp_level` 達成公平。
- **池**：對手 35% 為 17 隻 1v1-可行怪（equiv≤L8），15% 身分為怪；**6 隻 equiv>L8（ettin/hill_giant/fire_elemental/wyvern/young_red_dragon/archmage）排除**——1vN 內容，1v1 無論打法都輸＝無有用梯度。
- **描述子是敵方身分的唯一通道**：怪物 one-hot 全零、B 網自盲 one-hot → 敵方怪物身分只能經描述子讀到（正是驗證 2 的設計）。
- 順帶修復：v4 手術後 distill BC 錨資料集仍是 v3 寬度（57d）→ 新增 `obs.migrate_entities_v3_to_v4`（尺度重定標＋零填描述子），錨點重播才能過遷移後的網。
- **未動獎勵幾何**（BFS 路徑距離工單刻意分離，避免混淆「描述子是否學起來」與「獎勵幾何是否改變行為」的歸因——CLAUDE.md 方法論）。

### 結果（30 更新，eval/5；`eval_results/pop_mon_train.log`）
| 更新 | std12 | vs-怪 | desc-norm | grad-cos |
|------|-------|-------|-----------|----------|
| base | 41.7% | 67.2% | **0.0000** | — |
| 5    | 43.1% | 66.2% | 0.1338 | +0.000 |
| 10–25| 32–38%| 60–68%| 0.28–0.47 | −0.03~−0.06 |
| 30   | 34.7% | 62.7% | **0.6498** | −0.050 |

1. **描述子通道確實可學**：norm 單調 0→0.65。冠軍 = `pop_mon/pop_u0005`（std12 43.1% 不退基線、vs-怪≈base、grad-cos 仍 ~0；u5 後是已知的 population PPO std12 侵蝕＋梯度衝突轉負）。
2. **但描述子對 1v1 greedy actor 行為惰性**（與訓練前預期不符，已照 CLAUDE.md 查清）：
   - 驗證 2（A/B，desc-on vs 敵方描述子歸零，同種子）：mean Δ = **−0.8%（u5）/ +0.5%（u30）**，皆在噪音內（n=612/臂）。
   - **決定性量測**（`probe_descriptor_ab.py --divergence`，同態同狀態比較 greedy 動作）：消去敵方描述子，動作翻轉率 **0.1%**（8549 決策）；唯一非零是 shadow（物抗怪）0.9%。
   - 非空探針確認：敵方描述子在 obs 中確實填入（shadow norm 2.25、14/44 非零）→ 惰性非「沒東西可消」的假象。
   - 機制（最可能，未能完全隔離）：entity 編碼器 actor/critic 共用＋3 輪 value-warmup → 描述子權重成長主要服務 **critic**（敵方威脅→價值基線），不改 actor 的 argmax。1v1 中動作多由 HP/射程/技能可用性遮罩決定。
3. **根因**：1v1 物理對拼＋單武器職業幾乎無「依敵方能力而變」的可選動作 → 描述子無從施力。驗證 3（對免疫停用該傷害型）同根因，1v1 配方下不可展現（職業無替代傷害型/狀態可切換）。

### 描述子資訊增益定律（2026-06-12f 反事實實驗，`probe_resist_choice.py`）
追問「哪些情境 descriptor 絕對有資訊增益」→ 用引擎實傷（含命中/豁免/倍率）量「讀 descriptor 選最優」相對「最佳固定策略」的後悔值（regret）：

| agent | 兩選項 | 對手 | 最佳固定策略 regret＝descriptor 真實價值 |
|---|---|---|---|
| life 牧師 | 長劍(斬) vs 神聖光輝(光耀) | 全 23 | **0.3 HP**（光耀幾乎全場最優、無怪抗光耀 → 永不需切換） |
| evocation | 火球(火) vs 魔法飛彈(力場) | 全 23 | **21.3 HP**（火免疫怪 fire_elemental/young_red_dragon 把火球歸零 → 強迫切力場，每回合省 ~10.5 HP） |

**定律**：descriptor 行為增益 ∝「敵方抗性把 agent 的*原本最優*選項壓到比次優更差」的頻率。需同時 (a) agent ≥2 種可用傷害型、(b) 存在剋制 agent 主力傷害型的敵人。shadow 對牧師看似戲劇（斬×0.5 光耀×2 = 4× 擺幅）但**不構成決策**——它只是強化本就最優的光耀；真正製造決策的是「免疫主力型」（火 vs 火元素）。**關鍵閉環**：能讓 descriptor 行動化的敵人（火免疫龍/元素）恰是本波因 1vN 而排除的怪 → agent 訓練時從沒遇過唯一可施力的情境 → 這就是描述子行為惰性（0.1% 翻轉）的根因，不是「1v1 一律無選擇」這麼粗。

### 【已撤回 → 改判】模型學不會切招：是探索瓶頸，不是結構缺口（2026-06-12g）
> 2026-06-12f 曾下「結構性根因＝`as_vector` 缺 damage_type，故再多訓練也學不會」的結論。**該結論已撤回**——它建立在一個被快取吃掉信號的壞實驗上。逼問流程與改判如下：

1. 現役 pop_mon 扮 evocation vs 火免疫：火球 ~100%、descriptor on/off 無差（`probe_fire_switch.py`）——但這些真怪 1v1 不可贏（`probe_winnability.py`：magic_missile L20 WR 仍 0%）→ 無終端梯度，只是「沒被訓過」，非「學不到」。
2. 改用**注入火免疫 orc @L6**（可贏：always-mm WR 100%、always-fireball 0%）的 fire-lab 訓 40 更新，仍 on≈off。一度據此下「結構根因」結論。
3. **致命 bug 出土**（`probe_desc_cache.py`）：`capability_descriptor` 的快取鍵只看 (level, n_abilities, n_weapons)，**漏了 `damage_multipliers`**。`reset()` 在注入前已 build 一次 obs 把 orc 的「非免疫」描述子快取掉 → 注入後重建讀回舊值。**注入的免疫從沒進到 obs，fire-lab 兩半 obs 完全相同**，on≈off 是必然，測不到任何東西。→ 已修快取鍵（納入 `damage_multipliers` 簽章；正常對局抗性恆定故零開銷；38 測試通過）。
4. **信號修正後重跑** fire-lab（B=−1.00 確認免疫已進 obs）：40 更新**仍** on≈off（u40 IMMUNE 火% on=77/off=76），但 desc-norm 0.13→0.55（描述子權重確在長）。
5. **判別實驗**（`probe_fire_bcfit.py`）分離「表徵 vs 探索」：用 oracle 標籤（免疫→mm、正常→fireball）監督式擬合 skill head（無探索問題）→ acc 0.98；擬合後 **IMMUNE(完整描述子) 火% 39% vs normal(完整描述子) 100%**。由 #3 已證注入免疫 orc 與普通 orc 的 obs **唯一差異就是 typed_resist[火] 那一格**，故這 61pp 行為差**只能**經由該格傳遞。

**改判結論**：網路**能**用現有特徵（AoE/豁免/目標型 當代理）表徵出「依抗火描述子切招」，監督式一擬就會。**skill 側缺 damage_type 不是硬性障礙；damage_type 手術不需要。** 真正卡住的是 **PPO 探索/credit-assignment**——信號在、可贏在、終端梯度在，但從 BC 的 ~85% 火球先驗出發，40 更新探索不到免疫半的切招；監督式（繞過探索）立刻找到。

### 切招 BC-seed 波（2026-06-12h，GOAL MET）——「依敵情判斷打法」正式裝進模型
上節的修法已落地並驗收通過。**冠軍：`models/seed_switch4/seed_s1600.pt`**（warm 自 pop_u0005）。

**最終配方**（`scripts/seed_switch_bc.py`，四版迭代逼出，每一項都對應一個被數據否決的前版）：
1. **通用 oracle 標籤**：抗性調整 EV argmax（`expected_damage × 武器attacks_per_action × 敵damage_multiplier[該招傷害型]`），候選集取自 `_sample_action` 的資源遮罩（引擎可行性同路）、動作經 `build_action→encode_action`（與 BC 收集同編碼）。傷害型從 weapon/spell 資料讀（`action_damage_type`），零招式名單。preflight 七案例全對（含「抗性0.5 仍留火球 14>10.5」「真火元素下 ice_storm 9.0 輸 mm 10.5」的比價細節）。
2. **毒標籤修復（ev_eps=1.0）**：最佳可行傷害 EV<1 時不標、不 teacher-force——v1/v2 在「mm 位耗盡 vs 火免疫」的死局裡讓 max() 亂標火球，immune[火] 桶 42.8% 是 POINT 毒標籤（`probe_demo_fit.py` 實測），v3 起 12k 樣本攔掉 1,004 個。
3. **flip-targeted 注入（flip_p=0.5）**：一半注入瞄準 oracle 當下主力型（mult=0）保證翻轉劇本，每職業切招密度拉到 34%（v1 均勻注入下 life 只有 ~2%）。
4. **自適應加權（switch_boost=3, adaptive）**：加權「網路當前答錯」的樣本——有自平衡性（邊際過衝→confirm 變錯→自動拉回）。v3 用真值旗標固定加權＝34%×4 倍質量壓倒邊際，整網翻成無條件用備胎招，在兩個邊際間震盪不收斂。
5. **greedy 自蒸餾 anchor（1:1）**：滾 warm 網 greedy 行為（全 12 職業×標準+怪物對手、無注入、**真描述子狀態**）當行為保存錨。v2 教訓：舊 distill anchor 的 obs 是 v4 遷移零填——**描述子全零空間的錨鎖不住真描述子空間的行為**（std12 −5.6pp 就掉在那裡）；v3 教訓：採樣式標籤帶熵噪（argmax 對採樣天花板 ~0.3-0.5），錨變成分布匹配漿糊。
6. 3e-4×1600 步（800 步不夠，條件式 s1200 才浮現後三連穩）。

**終局驗收**（`probe_seed_accept.py`，60 局/臂＋std×3=432 局，單進程 warm vs 候選）：
| 臂 | warm | seed_s1600 |
|---|---|---|
| evo vs 注入火免疫 orc（on） | WR 37%／火 83% | **WR 100%／力場 100%（火 0%）** |
| 同（off 遮描述子） | WR 37%／火 81% | WR 60%／火 69% ←分化 |
| life vs 注入光耀免疫 orc（on/off） | 0%↔2%，光耀 100% 兩臂 | **on WR 85%／斬擊 100%** ↔ off WR 2%／光耀 100% |
| 控制（evo/life vs 普通 orc） | 火 100%／光耀 100% | **無損**（火 100%、光耀 100%、WR 98-100%） |
| 真 fire_elemental（行為軸） | 火 98% | 火 68%（off 78%）——**部分轉移** |
| std12 | 44.0% | 42.4%（**−1.6pp，n=432 噪音帶內**） |
| vs-mon | 65.7% | 62.5%（−3.2pp，~1.3σ 邊界） |

**守門量測教訓（重要）**：同一凍結 warm 網的 std12 在四個進程量出 36.8/42.4/43.1/45.8——引擎骰子吃全域 RNG，跑前工作負載改變骰流；games=1（144 局）二項噪音 ±4pp 再疊流分歧。**單進程 games=1 守門曾誤報 −6.2pp，n=432 收斂到 −1.6pp**。終局判定一律單進程高局數（本波協議：std×3＋60 局/臂）。

**遺留**：①真怪 fire_elemental 只有部分轉移（10pp on/off 分化、未全切）——免疫怪 1v1 不可贏故 demos 中其狀態稀少，補法＝把不可贏免疫怪的 demos 加密（標籤不需勝利，BC 可用）；②未來任何 PPO 波 anchor 必含切招 demos（PPO 侵蝕 BC 行為是既證模式）；③條件免疫軸（敵 condition_immunity 不在描述子）仍另案。

### 舉一反三 held-out probe（2026-06-12i）——seed 學到的是「kit 綁定的關聯」，不是通用規則
用戶追問「能不能不靠手動教、舉一反三」。`probe_heldout_switch.py`：兩個執行期合成 kit（全零 one-hot＝B 網原生域）vs 注入主力型免疫 orc，oracle preflight 先驗證方向（fireball 28>劍 7.5；閃電刃 20>mm 10.5，免疫後翻轉 ✓）：
| 臂 | 設計 | seeded s1600 結果 |
|---|---|---|
| A 長劍＋fireball vs 火免疫 | **held-out kit × 教過的火類型**（無訓練職業有此組合） | **不切**：on 火 94% ≈ off 95%，WR 2%（normal 控制臂火球 100% 正常） |
| B 閃電刃＋mm vs 閃電免疫 | **held-out 類型**（閃電從未當免疫教過；「武器免疫→換 mm」結構在穿刺軸教過） | **不切**：on 閃電 88% ≈ off 89%，WR 0%（結構預測命中） |

**結論（claim 層級，與 12g 不矛盾）**：12g 證明「damage_type 不是*學會被教配對*的必要條件」（關聯記憶夠用）；12i 證明「damage_type（或等價的技能側類型通道）是*零樣本泛化到沒教過配對*的必要條件」——學到的規則既綁 kit 上下文（A）也綁抗性表的特定列（B），obs 裡沒有任何通道能把「閃電刃」連到「閃電列」。**手術回到桌面、但理由換了**：as_vector 補 13 維傷害型 one-hot（53→66、skill_proj 零填遷移）讓切招規則可表達為「技能類型 one-hot ⋅ 敵 typed_resist」的跨類型共享對齊 → 少數類型 seed 後應零樣本轉移其餘類型；驗收＝重跑本 probe，A/B 臂翻綠。PPO 自主發現已排除（fire_lab2/3）。等用戶定奪。

### skill-dtype 手術＋matchup join（2026-06-12j）——讓「舉一反三」結構上可能
用戶以 /goal 拍板：「讓模型學會看資訊，必須用與訓練不同的戰鬥情境檢測，否則直接視為訓練失敗」＝ 12i probe 即驗收。動手前先釘死一個數學事實：**one-hot 列彼此正交，「位元 i × 抗性列 i」的交互權重只有類型 i 在訓練中出現過才會有梯度——純加 one-hot 最多救 A 臂（教過類型×新 kit），結構上不可能救 B 臂（沒教過類型）**。所以手術是兩件事一起做：

**①引擎側（53→66）**：`SkillFeatures.damage_types`＝傷害佔比加權**軟 one-hot**（對齊 `DAMAGE_TYPES` 13 維、append-only）。資料來源全引擎：武器列＝`weapon.damage_type`（＋on_hit 異型 rider 按 EV 佔比分權，wyvern 尾刺＝穿刺/毒分擔）；法術列＝資料填寫 18 筆；戰技 6 個用 `@weapon` 哨兵（materialize 時解析成持有者武器型，鏡像引擎 `get_weapon` 解析）；divine_smite 列＝光耀（其 expected_damage=9 本來就只描述 2d8 rider）；spirit_guardians/hunters_mark＝狀態機傷害，型別釘在 catalog（光耀/穿刺＝引擎 literal 鏡像）。守門＝`tests/test_skill_dtype.py`：掃全部可達 (carrier×skill)，宣告型別必 ⊆ 引擎實際傷害包（`action_damage_types` 提取器）、佔比和=1、@weapon 必已解析——資料寫錯會大聲炸。

**②模型側（matchup join，型別對稱的接地）**：forward 內從**原始 obs** 算 join_se＝技能列 dtype 軟 one-hot ⋅ 敵列 typed_resist 切片（mult−1 編碼）≈「對該敵用該招的期望傷害乘數−1」；skill_head 收「對在場敵列（is_enemy **位元**判定，非 slot 索引——佈局無關）取 max」、entity_head 收逐 (skill,entity) 原始值，各 +1 輸入維。**單一共享權重跨 13 型通用**＝零樣本的結構載體；desc-off 時 resist 切片全零 → join=0，因果測試語意不變。mul+sum 而非 bmm（K 歸約順序與 slot 數無關，bit 級 padding 不變性）。

**③遷移（bit-exact）**：`adapt_state_dict_for_skill_dtype` 入鏈（v3→v4→dtype，所有 loader 自動覆蓋）：skill_proj [64,53]→[64,66] 尾零填、critic 第一層尾 +13 零列、雙 head 各 +1 零列（含 pre-perarch 單數形 key）；舊 adapter 的 tail 算式改用凍結 `_SKILL_DIM_V1=53`（否則活常量會讓 v3/v4 偵測失配）。驗證：318 測試綠（2 紅＝stash 對照證實為手術前既有債：test_combat 攻擊骰算術/test_tag_parser）＋`diag_skill_dtype_shift.py`（`_dtype_era_v1` 診斷旗重建舊算術臂，同程序同種子逐狀態影子比對）：student **0/3645**、routed berserker **0/1210** 決策翻轉。

**④管線升級**：oracle `damaging_options` 改讀 materialized features（軟 one-hot 加權乘數）——順帶修掉 v4 的 smite 誤判（smite 對斬擊免疫敵 EV 曾被算成 0 而 poison-skip，現在正確＝光耀 9 → 聖騎士成為第 4 個訓練翻轉情境，multi-dtype pool 5→7 職業）；`--inject_types` 白名單＝顯式 held-out 邊界（本波訓練 {火,光耀,穿刺,斬擊}，held-out={閃電,強酸,冰,...}；注入直方圖落 log 供審計）；訓練打印 join 權重（`skill_heads[0].weight[0,-1]`）追蹤共享規則是否真的被學進去。probe 加 C 臂（酸蝕之刃＝第二個全 held-out 型）＋`--ablate_dtype`（歸零技能側 dtype 尾＝join+one-hot 同滅，驗證行為由新通道驅動）。

**驗收（2026-06-12j，`models/seed_dtype1/seed_s1600.pt`，`eval_results/heldout_dtype1b.log`）——舉一反三成立，機制釘死**：

訓練軌跡：demo 10k（switch 35%、注入嚴格限白名單{火,光耀,穿刺,斬擊}×{0,.5,2}，直方圖落 log）；join-w 0→**+0.591** 單調成長；條件式 **s800 即浮現**（v4 要 s1200+）；s1600 分布內 battery 全綠（evo on WR100%/力場100%、off 火71%；normal 控制火100% 雙臂；life on WR95%/斬擊100%、off 光耀100%；fire_elemental on/off 分化 61/75 優於 v4 的 68/78）。

held-out 三層判決（40 局/臂；B/C 臂=mm fallback 設計失效——seeded 網對含 mm 合成 kit 一律只丟 mm、無預設用刀行為可切，改 D/E 雙武器臂為有效儀器；WARM 在 D/E 預設 100% 用帶型武器且免疫下完全不切＝理想對照）：
| 臂 | desc-on | desc-off | dtype 消融(on) | 判讀 |
|---|---|---|---|---|
| A 長劍+火球 vs 火免疫（新kit×教過型） | **斬擊75%、WR75%** | 火94%、WR8% | 斬擊51%、WR22%（殘餘） | 翻綠；消融殘餘=12g 代理特徵路徑只服務教過型 ✓ |
| D 閃電刃+棍棒 vs 閃電免疫（**從未注入的列**） | **鈍擊32%、WR48%** | 鈍擊12%、WR0% | **鈍擊4%、WR2%（消失）** | 零樣本翻綠；通道因果 ✓ |
| E 酸刃+棍棒 vs 強酸免疫（**從未注入的列**） | **鈍擊36%、WR55%** | 鈍擊12%、WR0% | **鈍擊2%、WR2%（消失）** | 零樣本翻綠；通道因果 ✓ |

三重因果：看得到情報→切（WR 0→48/55）；遮描述子→不切；只斷技能側 dtype 通道（描述子可見）→不切——**held-out 類型的切換 100% 由本手術通道承載**，正交性數學預測完全命中（教過型有代理路徑、沒教過型只有 join）。誠實註記：D/E 份額為部分切換（32-36%，非分布內的 100%）——join 單權重的 logit 位移（±0.59）對兩條相似 ATTACK 列的嵌入偏好只能部分扳動，但勝率已從永敗到過半；normal 控制臂無損（D'/E' WR 98-100%）。

守門（`probe_seed_accept` 單進程，std×3＋mon×2＋battery 60 局/臂，`eval_results/seed_dtype1_accept.log`）：**std12 WARM 43.1% → CANDIDATE 44.0%（+0.9pp）、vs-mon 66.7% → 67.6%（+0.9pp）——零退化**；worst3 同形（AT/assassin/champion）；訓練中 games=1 守門曾報 36.8%＝12h 已記錄的噪音陷阱再次驗證。分布內 60 局/臂：evo on WR100%/力場100%・off 43%、normal 火100% 雙臂、life on 77%/斬擊100%・off 2%。**新冠軍：`models/seed_dtype1/seed_s1600.pt`**（取代 seed_switch4；未來 PPO 波 anchor 必含本波 switch demos）。遺留：D/E 部分切換的份額提升（kit 多樣化 demo 或 join 通道加寬）屬可選優化；B/C mm-fallback 臂留檔作「合成施法 kit mm 偏好」行為註記。

## 6.7 Wave 2 落地結果（2026-06-12k）

**狀態：完成。** 4 個 C 級原語＋LINE 語義＋4 隻新怪（合計 27 隻）；legendary_resistance **延後至 Wave 3**（Wave 2 名單零用戶——用戶＝成年龍/Lich/Tarrasque 全在傳奇行動波，frightful_presence 同例）。**obs 維度全程凍結**（檢查點零遷移）。

### 原語落地點（全資料驅動，零怪物名硬編碼）

| 原語 | 落點 | 驗證 |
|---|---|---|
| regeneration | trait `regeneration {amount, blocked_by}`＋`apply_damage` 記錄 `recent_damage_types`（後抗性：免疫歸零=沒受傷）＋tick self_turn_start 回血並清窗（recharge 同 chokepoint、零額外 RNG=不擾動標準對局骰流） | 單測：回血/火・強酸抑制/他型不抑制/免疫歸零不抑制/滿血不過量/0HP 不復活/全員清窗 |
| swallow_grapple | `Ability.requires_target_status / blocked_by_target_status`（policy 閘＋ATTACK handler 雙重驗證）＋action 側 rider 欄位（`rider_save_each/rider_rounds/rider_metadata`，與 weapon rider 對稱）＋`Swallowed` 複合狀態（束縛+目盲語義+metadata DoT）＋`tick_aura_damage` 通用 status-tick（`tick_damage_dice/type`、`ends_if_source_dead`=吞噬者死亡吐出） | 單測：未束縛 ERROR/單咬非 multiattack/複合狀態+metadata/已吞噬阻擋/體內 6d6 強酸 tick 記 attacker/STR 豁免掙脫/源死釋放 |
| petrify 兩段式 | `Spell.escalates_to` 資料欄＋共用 `apply_named_status` helper（SPELL 與射線表同路）；`Petrified` 類掛 MODIFIER_CLASSES（**STATUS_SLOTS 既有保留位→obs 可見且維度不動**）＝麻痺語義+全傷害減半；失能判定收斂單一 `Character.is_incapacitated()`（execute_action/env×2/sandbox/bc_collect 五處改吃同一份） | 單測：束縛→石化升級/豁免成功抵抗/stage1 免疫不升級/石化=失能+減半+近戰不自動暴擊（5e：auto-crit 僅 paralyzed/unconscious） |
| multi_ray_table | 新 action 型 `EYE_RAYS`：擲表取 n 道**互異**射線（5e RAW 重擲重複）、每道隨機瞄準範圍內可見敵、DC=8+熟練+dc_stat；表=ability 資料（`EYE_RAY_TABLE` 10 道，import 時 fail-loud 驗證型別/狀態名）；效果複用狀態池＋傷害（save_half/save 無傷） | 單測：3 道互異全瞄敵/DC16/無目標 ERROR/傷害豁免半傷・無傷/狀態射線/石化射線兩段式（與凝視共用語義）/緩速=引擎側新狀態 |
| LINE 語義 | `Spell.aoe_shape="line" + line_length_m/line_width_m`；受影響=「距 caster→端點線段 ≤ 寬/2」（`vec2.point_segment_distance`，policy 友軍檢查同一幾何）＋逐目標 LoS（光束被全遮擋止）；sculpt 條件同步含 line | 單測：共線命中/側偏 1.5m 不中/超線長不中/瞄自身 ERROR/超射程 ERROR/免疫目標 0 傷但次數照扣 |
| 新狀態 obs 凍結紀律 | `RL_STATUS_NAMES = STATUS_SLOTS ∪ MODIFIER_CLASSES` 直接決定 ENTITY_DIM——**任何新狀態名入 MODIFIER_CLASSES 都會位移所有檢查點**。`petrified` 用既有保留位（可見）；`swallowed/slowed` 入 `ENGINE_ONLY_STATUS_CLASSES`（引擎全功能、obs 暫不可見，下次 obs 手術整批收編）；`ALL_STATUS_CLASSES`=施加類單一查找入口（rider/SPELL/射線/condition_immunity 驗證共用） | 單測 pin：N_RL_STATUS==31、swallowed/slowed ∉ RL_STATUS_NAMES |

### GenericMonsterPolicy v1（控場施放——含一次被數據打回的設計）

v0 文件明載「不打控場」。v1 第一版「控場無條件優先」讓 basilisk 正常運作，但 **mage_npc 等效等級 6.0→4.8 反向移動**。控制變因 A/B（同引擎、policy 開/關控場、n=120/級/臂）：控場開場讓專家 WR **+18pp@L5／+7pp@L6**——hold_person 落地率 54% 不是問題，問題是 save_each 逃脫＋火球節奏損失，對高 DPR 腳本專家不划算。修正＝**控場優先只給「控場主導」kit**（最佳傷害 EV ≤ 武器 EV；零名單、零閾值魔數）：basilisk（咬 10=最佳傷害）→凝視循環 ✓；mage/archmage（火球 28≫匕首）→回到 v0 純轟。重標 mage_npc **6.1**（pre-wave 6.0，骰噪聲帶內＝v0 行為還原證明）。其餘 v1 件：requires/blocked_by_target_status 閘（吞噬）＋LINE 友軍線段幾何。

### CR→等效對位等級（Wave 2 增量，n=120/級，`calibrate_cr_wave2_10g.txt`）

| 怪物 | CR | 等效 L | 判讀 |
|---|---|---|---|
| basilisk | 3 | **4.4** | 石化把 CR3 底盤打到 gargoyle 量級；**入 1v1 池**（首隻帶兩段式控場的池內怪） |
| troll | 5 | **>8**（L1-6 全 0%） | 再生 10/回＝抗性表力量倍增器定律再現（對單人≈血量翻倍；火/酸抑制唯 evocation 可解 L7-8 2%）→ 1vN 內容 |
| behir | 11 | >8 | 1vN |
| beholder | 13 | >8 | 1vN |
| mage_npc | 6 | 6.1（重標） | 控場閘門後 v0 等價 |
| archmage | 12 | >8（重標，+閃電束） | 不變 |

### 守門（單進程 `probe_seed_accept`＋官方 eval_goal）

- B 網（warm pop_u0005 + 冠軍 seed_dtype1，`wave2_bnet_accept.log`）：分布內 battery 與 12j 記錄同形——candidate evo+火免疫 on **WR100%/力場100%**・off 48%、life on **87%/斬擊100%**、控制臂無損、desc-norm 2.081 同值；std12 WARM 44.1 / CANDIDATE 42.0（12j：43.1/44.0；n=288 雙向 ±2pp 噪音帶內）；vs-mon WARM 66.9 / CANDIDATE **67.6＝12j 同值**（且 vs-mon 池已含 basilisk）——**Wave 2 引擎對 B 網零退化、切招行為完整保留**。
- 官方 12/12（10g×3 輪，`goal_1v1_wave2_official10{,_run2,_run3}.txt`）：run1 11/12＋champion −12；**run2、run3 整面 GOAL MET 12/12**；champion 三輪 −12/+0/−6 中位數 −6＝既載平手邊界戶的協議方差（obs-v4 守門同例）。
- 全測試套件 358 綠＋2 紅（`git stash` 對照＝手術前既有債：test_combat 攻擊骰/test_tag_parser，與 12j 記錄同名同模式）。

### Wave 2 已知近似（記錄於 monsters.py docstring）

巨魔三連擊=爪×3、無 0HP 再生復活（引擎怪 0=死，火/酸決策保留在逐回合抑制）；貝希爾緊勒命中即擒（無對抗檢定，STR DC16 save_each=MM 逃脫 DC）、吞噬逃脫用豁免替代 30 傷吐出、體內酸 tick 在受害者回合首（MM=貝希爾回合首，每輪節奏等價）；石化凝視=主動動作（MM 為被動回合首觸發）；眼魔不飛、反魔法錐緩議（§3 E）、緩速射線≈半速−2AC、念力射線≈束縛 1 回合、魅惑/沉睡不建模「受傷即解」。

### 旁收

- **materialize 戲法縮放閘**：`cost_slot_level=0` 且有傷害的技能一律吃等級倍率——fire_breath（非戲法天然能力）在 nat10 龍身上 EV 虛報 ×2（policy/描述子層；引擎骰一直是平的）。修＝`Ability.scales_as_cantrip=False`（四個天然能力標旗，預設 True=既有技能 obs 值位元不變）。
- **sandbox MULTI_ATTACK 顯示修復**：driver 讀 `hits` 鍵但引擎回傳 `attacks`——所有多重攻擊怪在 sandbox 一直顯示「0/0 命中」；＋EYE_RAYS 顯示分支（逐射線豁免/傷害/狀態）。
- **game.py 戰役迴圈補 `tick_aura_damage`**：五個 driver 唯獨它漏掛——靈體守護光環在 CLI 戰役從不結算（吞噬 DoT 順帶修通）。
- 大法師補閃電束（LINE 第二活體用戶；池外、守門不可見）。

### 遺留

1. **legendary_resistance → Wave 3**（與 legendary_actions/frightful_presence/death_throes 同波，用戶到齊）。
2. 武器 rider EV 不入 `from_weapon.expected_damage`（53 維凍結語義）：描述子對 rider 武器（wyvern 螫刺/緊勒）低報威脅——既有偏置非本波引入，下次 obs 手術一併處理；本波繞法＝behir 武器欄只放緊勒。
3. swallowed/slowed obs 不可見（引擎側 registry）；模型對「被吞」的感知=束縛+目盲位（吞噬施加的可見組件）——足夠 1v1，3v3 救援語義等 obs 手術。
4. 1v1 池新增 basilisk（4.4）：下一個 population 波的對手池自動擴為 18 隻；petrify 兩段式=模型首見的「不可逆控場」——值得在該波驗證 B 網是否學會優先打斷（嚴格說是跑不掉就打死）。

## 6.8 Wave 3 落地結果（2026-06-13）——怪物階段收尾波

**狀態：完成。** D 級 legendary_actions＋三個延後原語（legendary_resistance／frightful_presence／death_throes）＋7 隻傳奇帶怪物（合計 34 隻）＋Beholder 傳奇化補裝＋1vN Boss 對局基建與標定。**obs 維度全程凍結且本波零新狀態名**（frightened/prone/paralyzed/restrained/swallowed 全是既有詞彙——連 ENGINE_ONLY registry 都沒動，檢查點零遷移）。

### 原語落地點（全資料驅動，零怪物名硬編碼）

| 原語 | 落點 | 驗證 |
|---|---|---|
| legendary_actions | trait `legendary_actions {per_round, options}`（選項＝`{"ability": id}` 或 `{"weapon": name}`＋cost；fail-loud：未授予/吃法術位/未知武器/cost<1 註冊即炸）；回充＝tick self_turn_start（recharge 同 chokepoint 零接線）；觸發＝`combat_policy.run_legendary_actions(ws, ended_id, rnd)` 接在**每個 driver 的回合尾**（env_v2×5 含失能/瀕死跳過槽、sandbox、bc_collect、game.py；skill_probe 為合成探針免接）；選擇＝EV/cost 貪心複用 `_damage_options`（decide step-2 逐字抽出共用，零骰子零行為位移）；每觸發一選項（5e RAW）、武器選項強制 n_attacks=1、不花移動（搆不到就棄權） | 單測：回充/單選項每觸發/自己回合不觸發/失能跳過/超距棄權不扣/預算不足不放/翼擊自心 nova 排除施法者/巫妖戲法 L21 ×4 縮放/validation 四炸 |
| legendary_resistance | `make_saving_throw` 失敗點（擲骰失敗＋自動失敗皆覆蓋）＋`fail_applies_status` 旗標——**只在「失敗會留狀態」的豁免燒次數**（SPELL applies_status/武器 rider/射線狀態道/save_each 掙脫四類站點傳旗；純傷害半傷豁免永不觸發＝MM 戰術語義）；per-combat 池 | 單測：狀態豁免燒次改判/傷害豁免不燒/自動失敗也救/池耗盡失效/束縛 save_each 經 LR 掙脫 |
| frightful_presence | trait 存 `{radius_m, dc, rounds}`（DC=8+熟練+CHA 面板實算：白14/紅19/古21/塔17 全中 MM）；受害者回合首 aura 檢查（tick_aura_damage＝spirit guardians 同 chokepoint）：未免疫且未恐懼者擲 WIS——成功＝對該源**本場免疫**、失敗＝frightened（save_each 重豁）；condition_immunity 早跳不耗骰 | 單測：失敗上恐懼帶 save_each/成功進免疫集且不再擲/狀態免疫零擲/半徑外零擲/已恐懼不重複 |
| death_throes | trait 存 `{damage_dice, damage_type, radius_m, save_stat, dc}`（DC 面板實算：炎魔 20=8+6+CON6）；`apply_damage` NPC 死亡點引爆——**spec 先 pop**（鏈式爆炸防重入）、雙方陣營皆波及（5e）、事件入 `ws.pending_events`（sandbox/game.py 汲取顯示，env 無視） | 單測：爆炸全傷+半徑外無傷+事件/豁免半傷/無 ws 不炸不崩/pop 後不重爆 |
| EYE_RAYS 泛化 | `distinct_rays` action 旗（預設 True＝眼魔互異重擲；False＝海妖落雷允許重複；單款表零擲表 RNG） | 單測：3 落雷同名/隨機敵目標/DC22 |
| SPELL 自心 nova | `Spell.excludes_caster`（預設 False＝火球踩腳趾行為位元不變）——龍翼拍擊/生命擾亂用 | 單測：翼擊敵中招＋施法者不在 target_results |

### 怪物（7 新＋1 補裝，合計 34）

| 怪物 | CR | 傳奇套件 | statblock 對 MM |
|---|---|---|---|
| 成年白龍 | 13 | LA3（尾1/翼2）＋LR3＋威壓 DC14＋寒冰吐息 12d8 | +11/200/AC18 ✓ |
| 成年紅龍 | 17 | LA3＋LR3＋威壓 DC19＋火吐息 18d6 DC21 | +14/256/AC19 ✓ |
| 遠古紅龍 | 24 | LA3＋LR3＋威壓 DC21＋火吐息 26d6 DC24 | +17/546/AC22 ✓ |
| 巫妖 | 21 | LA3（戲法1/麻痺觸2/生命擾亂3）＋LR3＋法術 DC20＋麻痺之觸 DC18 | 135/AC17 ✓ |
| 海妖 | 23 | LA3（觸手1/閃電風暴2）＋**無 LR（MM 如此）**＋吞噬鏈 | +17/472/AC18 ✓ |
| 炎魔 | 19 | 非傳奇（MM 如此）；death_throes 20d6 DC20＋火劍閃電 rider | +14/262/AC19 ✓ |
| 塔拉斯克 | 30 | LA3（爪1/吞噬2）＋LR3＋威壓 DC17＋巨顎擒抱→吞噬 16d6 | +19/676/AC25 ✓（obs：HP 676/800、L30/40 ＝ v4 重定標預留入界 ✓） |
| 眼魔（補裝） | 13 | LA3（單射線×3/回合＝MM 傳奇射線） | 不變 |

行為煙測（3 專家 L8 隊 vs Boss）：白龍場均 4 傳奇尾擊＋威壓觸及 2-3 人＋LR 燒在控場；巫妖戲法連發 7.3 LA/場；海妖落雷/觸手混用；眼魔 16 LA/場（逐回合補射線）；炎魔死亡爆炸敘事鏈（sandbox/game.py）可視。

### 1vN Boss 對局（`calibrate_boss.py`，3 專家隊 battle_master+life+evocation，n=30/級）

「CR≥4＝1vN 內容」自 Wave 0 起是 inf 標記，本波給出實測座標（`calibrate_boss_wave3_30g.txt`）：

| 怪物 | CR | 隊伍(3人)等效 L | 判讀 |
|---|---|---|---|
| ettin / hill_giant | 4 / 5 | **2.6 / 3.2** | 1vN 帶的入門戶 |
| fire_elemental / wyvern / troll / archmage | 5-12 | **4.3 / 4.3 / 4.5 / 4.4** | 中段聚帶（L5 功率尖峰再現） |
| young_red_dragon / behir / beholder | 10-13 | >8 | 3×L8 天花板外 |
| 成年龍×2 / 遠古紅龍 / lich / kraken / balor / tarrasque | 13-30 | **>8（全 0%）** | 與 5e 標定一致：CR≈四人同級隊；職業註冊表上限 L8 → 傳奇 Boss＝更大隊伍或更高等級表的內容；**Boss 訓練波的對手池下限＝CR≤12 段** |

### 已知近似（記錄於 monsters.py docstring）

龍多重攻擊＝爪牙 2d8×3（三龍與 MM 混合差 ≤6%）、紅龍咬的火 rider 不建模；龍翼拍擊共用條目取紅龍 2d6+8（白/古 ±2）、拍後半速飛行不建模；威壓＝受害者回合首 aura（MM 為多重攻擊附掛）、save_each 逃脫不給免疫（下回合首再擲）、無 LoS 閘；LR 不覆蓋專注豁免；傳奇選項「偵測/移動」削除（本引擎無感知/定位收益）；麻痺之觸＝精巧武器 +10（MM 法術攻擊 +12 的最近管線近似）；塔拉斯克多重攻擊＝巨顎×4 帶逐擊擒抱 rider（MM 僅咬擒）、反射甲殼緩議（§3 E）；海妖觸手 9m 擒抱即中、Fling/墨雲不建模、落雷 DC22（MM 23）、海妖/巫妖/塔拉斯克非魔法物理免疫→0.5 抗性代理（狼人先例）；炎魔火鞭（拉拽）不建模——forced_move pull 延後至 ≥2 用戶（Kraken Fling 亦未建模，單用戶不開原語）。

### 守門

- 全測試套件 **405 綠＋2 紅**（test_combat 攻擊骰/test_tag_parser＝12j 起既載既有債）；本波新增 35 項單測（154 monster 測試全綠）。
- B 網單進程 probe（warm pop_u0005＋冠軍 seed_dtype1，`wave3_bnet_accept.log`）：CANDIDATE battery 與 12j/W2 紀錄**逐臂同形**——evo+火免疫 on WR100%/力場100%・off 43%、life+光耀免疫 on 87%/斬擊100%、控制臂 98-100% 無損、desc-norm 2.081 同值；std12 41.7（W2 42.0，n=288 噪音帶）、vs-mon **67.1**（W2 67.6，−0.5pp 帶內；池不變＝W3 怪全 inf 不入 1v1 池）——**Wave 3 引擎對 B 網零退化、切招行為完整保留**。
- 官方 12/12（10g×3 輪，`goal_1v1_wave3_official10{,_run2,_run3}.txt`）：run2 **整面 GOAL MET 12/12**（bm −3/champion −3）；run1/run3 在 bm（−16/−10）與 champion（−14/−13）出帶——**已按 CLAUDE.md 證偽「真回歸」假說**：同進程、逐局種全域骰的 200 局標準對局指紋，real 接線 vs orchestrator 全 stub **位元級相同**（`fingerprint_wave3_proof.txt`，sha256 一致）＝Wave 3 對標準路徑零骰零控制流變化；其餘新分支全是欄位閘死路（LR uses=0／無威壓源／death_throes None／EYE_RAYS 不可達）。擺動＝取樣方差：eval_goal 以進程鹽 `hash()` 取樣（每輪樣本不同）＋專家臂自身同步擺動（bm 專家 70/63/69 vs W2 期 57）＝obs-v4 守門同型判例（當時 bm −9/−11/−21 亦判協議方差）。插曲教訓：第一版指紋跨進程比對忘了引擎骰吃全域 RNG（記憶既載），實驗本身無效——種骰後才是有效儀器。

### 旁收

- game.py 的 tick_aura 顯示迴圈假設每事件必有 `damage` 鍵——FRIGHTFUL_PRESENCE 事件會 KeyError，已分支處理（W2 剛補上這條迴圈，本波第一個非傷害 aura 事件就踩到）。
- sandbox 回合首 aura 事件原本完全靜默（spirit guardians 光環傷害在 sandbox 從不顯示）——順手補敘事（AURA_DAMAGE/STATUS_TICK_DAMAGE/FRIGHTFUL_PRESENCE 三分支）。

### 遺留（怪物階段收尾後的地圖）

1. **summon/spawn**＝唯一未落地的 D 級原語（零 W3 用戶；env 動態 roster＋obs 10 槽是已知影響面）。
2. **扮演怪物波已執行**（§6.9）：模型坐 agent 席自己扮怪、純 PPO 通用化＝GOAL MET；遺留三個下一步＝①吐息泛化子波（補「技能×身體」分布缺口）②boss held-out kite 負遷移③std12 −3.5 完全復原。
3. 武器 rider EV 不入 from_weapon、swallowed/slowed obs 不可見、傳奇次數 obs 隱藏（設計如此）——全部等下次 obs 手術整批收編。
4. 未組裝的目錄怪（Vampire/Iron Golem/Death Knight/Pit Fiend/Solar/Banshee/石魔像…）＝既有原語的純組裝戶（A 級工作量），按用戶需求隨時加。
5. E 級緩議清單不變（飛行 3D/反魔法錐/變形/反射甲殼/穿牆）。

## 6.9 扮演怪物波落地結果（2026-06-13）——模型自己扮怪 + 通用化

**狀態：GOAL MET。** 用戶 /goal：「讓模型學會扮演怪物，必須真正通用化，而非靠
手寫 BC 腳本然後模仿 BC。」此前所有守門只測模型坐**職業**席；怪物 100% 由
GenericMonsterPolicy 腳本驅動，模型從沒當過怪。本波讓模型坐 **agent 席自己扮怪**，
**純 PPO、零腳本示範**（腳本只當 eval 量尺，從不當老師）。錨定報告
`REPORT_2026-06-13_monster_actor.md`。

### 儀器（本波新建）
- `eval_monster_actor.py`：模型扮怪 vs 腳本扮怪**配對評測**（同 env、同動作格點、
  每局 crc32 種全域骰 → within-run Δ 純 policy）；1v1（18 怪×12 職業）＋ 1v3 boss。
- `chimera_monsters.py`：3 隻 eval-only 組裝怪（frost_troll/storm_ogre/plague_wight），
  訓練從不 import ＝零樣本儀器（class 側 chimera_defs 的怪物對應）。
- `clean_std12.py`：**種骰 std12 正確儀器**（見下「測量缺陷」）。
- `train_monster_actor.py`：怪席 population PPO；`PARTY3_EQUIV_LEVEL` measured 常量。

### 結果（冠軍 `models/mon_actor1/ma_u0032.pt`，純 PPO 72 更新中第 32）
| 軸 | 基線（seed_s1600 零訓練）| 冠軍 | 判定 |
|---|---|---|---|
| 1v1 怪席平均（n=1728）| −10.2pp | **−2.2pp** | 大幅學會 |
| boss 訓練池內（排 held-out）| −27.1pp | **+5.2pp** | 超腳本 |
| archmage 施法 boss | +20~+25 | **+25~+33** | 完全學會換法術 |
| basilisk 1v1 | +6.9 | **+22.9** | 咬+走位超腳本（非凝視——模型自摸打法）|
| held-out 真怪 ghoul/garg/mant | −31.9/−11.1/−2.8 | **−10.4/+2.1/−3.1** | **同類零樣本泛化** |
| chimera plague_wight/frost_troll | −31.7/−40 | −17.7/−25 | 部分泛化 |
| chimera storm_ogre 吐息 | 0% | **0%** | 未泛化（見下）|

### 通用化邊界（誠實，按 CLAUDE.md 查清根因）
- **held-out 同類怪泛化**＝真通用化（近戰/遠程 1v1 機制訓練中有覆蓋 → 遷移到新實例）。
- **兩個未泛化已查清，皆非架構缺陷**：
  1. **hill_giant boss −35（held-out）**：唯一同時 held-out＋帶遠程武器的 boss；
     model 擲岩×0.32/巨棒×0.01＝1v1 弱怪席學的「能遠程就 kite」過度遷移，boss 1v3
     貼臉局 kite 跑不掉。對照同池近親 ettin（純近戰）boss @L2 100% 平手。
  2. **storm_ogre 吐息 0%**：訓練 boss/1v1 池**沒有任何怪用過 `lightning_breath`**
     （吐息怪全 CR≥10=1vN inf 不入池；archmage 用 `lightning_bolt` 法術=異 skill）。
     ＝12i「kit×行為綁定、非通用規則」在 PPO 波重現＝**訓練分布未覆蓋該配對**，
     可由「降一隻吐息怪入可訓帶」補。

### 停滯解剖（mon-atk 率 18 更新平於 0.10-0.18，`diag_mon_actor.py`）
H1 advantage 不偏攻擊？**否決**（怪席攻擊槽 adv +2.58）；H3 慢？**證實**（單更新
攻擊−move logit +0.40，累計爬升）；H2 anchor 拔河？證實但可承受（−0.095/4 批）。
配方不動，預測的 S 曲線在 u28-32 應驗（troll boss 0→75%）。續波 +36 證飽和無增益。

### 守門 + std12 退化的測量缺陷查清（CLAUDE.md 關鍵）
- **官方 eval_goal 10g GOAL MET 12/12**；切招 battery 完整（evo 火免疫 on 力場100%/
  off 55%、life 光耀免疫 on 斬擊100%/off 0%）。
- **std12「−6.6pp」是髒數據**：`standard_probe`/`monster_opp_probe` **每局未種全域
  引擎骰** → warm 先跑漂移 RNG、candidate 接著跑骰流不同（同進程差三測 −6.6/−6.6/
  −1.9）。修正儀器 `clean_std12.py`（復用已種骰的 `run_episode`、warm/cand 每格同骰）
  測得**真實退化 −3.5pp**（溫和；seed-wave −1.6 與 pop-wave −5~−12 之間）。
- 救援 `rescue_std12.py`（post-PPO 三 anchor BC）只挽回 1.1pp std12 卻賠 7pp 怪席＝
  不划算；u0032 定案。**教訓：終局 std12 判定必用種骰 `clean_std12`，不可信
  `standard_probe` 的跨快照絕對值**。

### 設計紀律
- 怪席行為錨定用「u0032 自己的 greedy 自蒸餾」（非腳本 demos）＝鎖定 PPO 所學、
  不教新，合 /goal「非模仿」。
- `PARTY3_EQUIV_LEVEL`/`party3_boss_monsters()` fail-loud（覆蓋斷言＝1v1-inf 才有
  party 標定），與 `EQUIV_LEVEL_1V1` 同協議。
- held-out 雙側檢疫（rollout＋fresh anchor 皆無）；warm 網歷史對手側曝光已揭露。

## 7. 變更記錄

- 2026-06-13：**扮演怪物波 GOAL MET**（§6.9）：模型坐 agent 席自己扮怪、純 PPO 零腳本示範（腳本只當 eval 量尺）。儀器＝`eval_monster_actor.py`（模型 vs 腳本配對、每局種全域骰）＋`chimera_monsters.py`（零樣本組裝怪）＋`train_monster_actor.py`（怪席 population PPO，held-out 雙側檢疫）。冠軍 `mon_actor1/ma_u0032`：怪席 1v1 −10.2→−2.2pp、boss 訓練池 −27.1→+5.2pp、archmage +25~33／basilisk +22.9 超腳本、held-out 同類真怪零樣本泛化（ghoul +21.5改善/garg・mant parity）。**通用化邊界查清**：全新「技能×身體」配對不泛化（storm_ogre 吐息 0%＝訓練池無任何怪用過 lightning_breath、hill_giant kite 負遷移）＝分布覆蓋問題非架構缺陷。**std12 測量缺陷查清**：`standard_probe` 每局未種全域骰 →「−6.6pp」是髒數據（同進程差三測 −6.6/−6.6/−1.9）；新 `clean_std12.py`（種骰、warm/cand 同骰）測得真實退化 **−3.5pp**（溫和）；官方 eval_goal **GOAL MET 12/12**、切招 battery 完整。停滯解剖 `diag_mon_actor.py`（H1 否決/H3 慢但在動/H2 anchor 24%）。救援 `rescue_std12.py` 不划算（1.1pp 換 7pp 怪席）。新增常量 `PARTY3_EQUIV_LEVEL`/`party3_boss_monsters()`（measured，fail-loud）。
- 2026-06-13：**Wave 3（D 級傳奇套件）完成＝怪物階段收尾波**（§6.8）：legendary_actions（trait 選項表＋`run_legendary_actions` orchestrator＋全 driver 回合尾接線；EV/cost 貪心複用 `_damage_options` 抽出；obs 不加欄=隱藏資源原則）／legendary_resistance（`fail_applies_status` 旗標＝只為狀態後果燒，含自動失敗與 save_each 掙脫）／frightful_presence（受害者回合首 aura，成功=本場免疫；DC 面板實算全中 MM）／death_throes（NPC 死亡點引爆＋`ws.pending_events` 顯示通道）；EYE_RAYS `distinct_rays` 旗＋SPELL `excludes_caster` 自心 nova；7 新怪（成年白龍/成年紅龍/遠古紅龍/巫妖/海妖/炎魔/塔拉斯克，statblock 全中 MM）＋眼魔傳奇化（合計 34）；`calibrate_boss.py` 1vN 標定（ettin 2.6→archmage 4.4 入帶、CR≥10 對 3×L8 全 >8＝Boss 訓練波對手池下限 CR≤12）；**守門**：405 測試綠＋2 既有債、B 網 battery 與 12j 逐臂同形（vs-mon 67.1≈紀錄）、官方 run2 GOAL MET 12/12＋run1/3 擺動經 200 局種骰指紋證明為取樣方差（real vs stub 位元級相同）；旁收 game.py aura 顯示 KeyError（非傷害事件）＋sandbox 回合首 aura 靜默；遺留：summon/spawn（唯一未落地 D 級）、Boss 訓練波、obs 手術批次（rider EV/swallowed 可見性）。新增 35 單測＋`fingerprint_std.py`。
- 2026-06-12k：**Wave 2（C 級）完成**（§6.7）：regeneration／swallow_grapple（requires/blocked_by_target_status＋action-rider 對稱欄＋Swallowed 複合狀態＋tick_aura_damage 通用 status-tick）／petrify 兩段式（`Spell.escalates_to`＋`apply_named_status` 共用 helper＋Petrified 用 STATUS_SLOTS 保留位=obs 維度凍結）／multi_ray_table（EYE_RAYS action＋10 道資料表 fail-loud 驗證）／LINE 真語義（`aoe_shape="line"`＋`point_segment_distance`＋逐目標 LoS）；troll/basilisk/behir/beholder 4 怪（合計 27）；**GenericMonsterPolicy v1 控場＝控場主導 kit 限定**（無條件控場被 A/B 打回：mage 6.0→4.8、專家 WR +18pp@L5；閘門後 mage 重標 6.1=v0 等價）；CR 標定 basilisk **4.4 入 1v1 池**、troll >8（再生=倍增器定律）；失能判定收斂 `is_incapacitated()`；旁收：materialize 戲法縮放閘（fire_breath EV 虛報 ×2 修復）、sandbox MULTI_ATTACK 顯示鍵錯誤、game.py 戰役迴圈漏 tick_aura_damage、大法師補閃電束；**守門全綠**（官方 run2/3 GOAL MET 12/12、B 網 std12/vs-mon 12j 同值、358 測試+2 既有債）；legendary_resistance 延後 Wave 3（零用戶）。新增 35 單測；`calibrate_cr.py` 支援怪物子集。
- 2026-06-12j：**skill-dtype 手術＋matchup join 落地，舉一反三 GOAL MET**（§6.6）：SKILL_FEATURE_DIM 53→66（傷害佔比軟 one-hot，引擎資料全覆蓋＋交叉驗證測試）；模型雙 head +1 matchup join（單共享權重、型別對稱＝零樣本載體；is_enemy 位元遮罩、mul+sum bit 級不變性）；`adapt_state_dict_for_skill_dtype` 遷移鏈（零填 bit-exact，diag 0/3645＋0/1210 零決策翻轉）；oracle features 化（修 smite 誤判、pool 5→7）；`--inject_types` held-out 白名單。**驗收**：新冠軍 `seed_dtype1/seed_s1600.pt`——held-out A 臂翻綠（斬擊75%/WR75）、D/E 雙武器臂（閃電/強酸=從未注入列）WR 0→48/55% 且 desc-off 回退＋dtype 消融歸零＝通道因果；std12 43.1→44.0、vs-mon 66.7→67.6 零退化；B/C mm-fallback 臂設計失效（改 D/E 為有效儀器）。新增 `diag_skill_dtype_shift.py`/`probe_ability_dtypes.py`/`tests/test_skill_dtype.py`。
- 2026-06-12：初版。引擎基線盤點＋42 隻怪物＋原語分級 A–E＋波次規劃。
- 2026-06-12b：§4 新增敵方身分盲區確認（#7）＋ §4.1 敵方能力描述子設計（欄位表/公開隱藏原則/與 one-hot 共存策略/obs v4 合併遷移/三層驗證）。
- 2026-06-12c：**Wave 0 落地完成**（§6）：monsters.py 11 隻＋GenericMonsterPolicy＋29 測試＋CR 標定＋obs 邊界實測（§4 #1 緊急度下修）；梟熊 CR 2→3 修正；發現並根治註冊表快照陷阱（STANDARD_ARCHETYPES 凍結常量）；彎刀補「精巧」（5e finesse 資料修正）。
- 2026-06-12e：**Wave 1 完成**（§6.5）：6 個 B 級原語（typed_resist/on_hit rider+drains/pack_tactics/undead_fortitude/condition_immunity/breath_recharge）＋fire_breath（SPELL 管線資料欄位）＋12 隻新怪；描述子補 typed_resist 13 維（31→44，零權重窗口、diag 0 分歧）；frightful/death_throes 延後（無用戶）；**限次扣次下沉 execute_action（歷史基線斷點：對手/BC 過去全是無限資源版）**；修復後官方 11/12 OK（champion −10=既載平手邊界戶）；B 網 mean +4.4%、短戶輪廓與歷來同形。
- 2026-06-12f：**Population 訓練波完成**（§6.6）：23 怪混入 B 網 population PPO（17 隻 1v1-可行對手＋身分，公平等效等級實測對位；6 隻 1vN 排除）；描述子權重 0→0.65 證明可學；但驗證 2 A/B≈0、動作翻轉率 0.1% → **描述子對 1v1 greedy actor 行為惰性**（最可能服務 critic；根因＝1v1 單武器職業無依敵變化的可選動作）；冠軍 `pop_mon/pop_u0005`（std12 不退、描述子預熱）；新增 `obs.migrate_entities_v3_to_v4`（修 v4 後 BC 錨資料集寬度）＋`probe_descriptor_ab.py`（A/B＋divergence 模式）；BFS 獎勵幾何刻意未動以保歸因。行為收益待針對性下一波（隊伍目標選擇/多傷害型/狀態vs免疫/全盲對手）。
- 2026-06-12i：**舉一反三 held-out probe**（§6.6）：合成 kit（劍+火球／閃電刃+mm）vs 注入免疫——seeded 網兩臂皆不切（on≈off），證明 seed 學到的是 kit 綁定關聯而非通用規則；**damage_type 技能側通道＝零樣本泛化的必要條件**（與 12g「非學會必要」不矛盾，claim 層級不同）；手術（53→66）以泛化為由回到桌面，驗收＝本 probe 翻綠；新增 `probe_heldout_switch.py`。
- 2026-06-12h：**切招 BC-seed 波 GOAL MET**（§6.6）：通用 oracle（抗性調整 EV argmax，零招式名單）＋毒標籤修復（ev_eps）＋flip-targeted 注入＋自適應加權＋greedy 自蒸餾 anchor（真描述子空間行為錨）→ 雙職業條件式切招裝上（evo 火→力場 WR 37→100、life 光耀→斬擊 WR 0→85，desc-off 即退回＝描述子因果）、控制臂無損、std12 −1.6pp@n=432 噪音帶內；冠軍 `seed_switch4/seed_s1600.pt`；四版迭代教訓與守門噪音協議（同網跨進程 ±4.5pp→終局判定必單進程高局數）入 §6.6；遺留：真火元素部分轉移、PPO 波 anchor 必含切招 demos、條件免疫軸另案。新增 `seed_switch_bc.py`/`probe_seed_accept.py`/`probe_demo_fit.py`/`probe_kit_dtypes.py`。
- 2026-06-12g：**撤回 damage_type 結構根因，改判為探索瓶頸**（§6.6）：①出土並修復描述子快取 bug（`capability_descriptor` 快取鍵漏 `damage_multipliers`，導致注入式免疫實驗信號被吃 → fire-lab 兩半 obs 等同，前一輪「結構根因」結論失效）；②信號修正後 fire-lab 仍 on≈off（探索未及），但 `probe_fire_bcfit.py` 監督式擬合 acc 0.98、IMMUNE 39% vs normal 100% 火球（唯一差異格＝typed_resist[火]）證明**網路能用既有特徵表徵切招，damage_type 手術不需要**；③真瓶頸＝PPO 從 BC 火球先驗探索不到免疫半切招，解法＝BC-seed 切招後 PPO 維持（非架構）。新增 `probe_desc_cache.py` / `probe_fire_bcfit.py`。
- 2026-06-12d：**obs v4 三合一手術完成（遷移層）**：尺度重定標（2 的冪）＋能力描述子 31 維＋CR 等級帶入界；`adapt_state_dict_for_obs` v2→v3→v4 鏈；diag_obsv4_shift 0/7865 動作分歧；描述子資訊量探針 1-NN 83%（誤判僅戰術孿生對 bear↔berserker、evo↔div，描述子距離=0——kit 視角下語義正確）；官方 12/12＋B 網守門結果見 §4 狀態塊；待辦：下一波訓練啟用描述子權重＋敵方陌生 A/B。
