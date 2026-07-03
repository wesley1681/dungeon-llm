# 模型評估覆蓋地圖（EVAL_COVERAGE）

目的：如實記錄「我們現在用什麼手段評估一個模型好不好」、「涵蓋了哪些對戰排列組合」，
並列出**已知看不出來的情境（盲區）**。本檔用於「找出當前評估看不見的戰鬥情況」的迭代審查。

> 背景教訓（2026-07-01）：一個 dodge-collapse 缺陷（法師對弱免疫敵 0 傷）在守門指標
> （std12 WR / held-out 切招）上**完全看不出來**，是用戶在 GUI 手動測出來的。而且它是
> **回歸**（uni_v2~v6 正確 → uni_v7 的 rsw 手術弄壞），**沒有任何評估腳本抓到這個回歸**。
> 這份檔案就是要系統性地補上這類盲點。

---

## 第一部分：現有偵測手段（量什麼行為/指標）

| 手段 | 腳本 | 量什麼 | 侷限 |
|---|---|---|---|
| 勝率 WR（對腳本專家，配對種子） | eval_goal / eval_v2 / eval_routed / eval_student / eval_class / eval_generalize | 能不能贏稱職專家 | **對「0 傷但沒輸/平手」不敏感**；勝率高不代表沒退化 |
| Δpp（model% − script%） | eval_generalize | 有沒有像一個通用腦一樣駕馭這個 kit | 只在 std12/synth/chimera/monster 桶 |
| 動作使用分布 mix | eval_generalize / diag_class | 用了哪些招、比例 | 需人工判讀 |
| 退化：致命卡位/裝飾空轉/拒戰/滿血補 | diag_degen_audit（agent 席）/ diag_degen_selfplay（對手席） | 全場戰鬥的退化指紋 | 對手是**侵略性腳本**；開闊地為主 |
| 死動作（execute→ERROR） | diag_dead_action | 選到引擎會拒的動作（chill_touch 類） | 只統計 ERROR 類 |
| 逐通道因果讀取（翻轉率） | diag_info_channels | 敵每個資訊子通道歸零→動作翻不翻 | 只在 monster 對手、1v1、L≤8 |
| 對免疫敵的傷害輸出 | eval_immune_weak / diag_ranged_standoff | 主招被免疫時會不會用次級招 | 火法師 vs 少數弱怪 |
| 走位/視線重建（繞牆） | diag_wall_los / eval_wall_combat / eval_selfplay_walls | 敵在牆後會不會繞出視線 | 單柱、靜止敵為主 |
| held-out 傷害切招 | probe_heldout_switch | 沒見過的 kit/型上會不會避開免疫傷害型 | 合成 2 選項 kit、對 orc |
| 天賦因果（regen/pack） | verify_regen_causal / verify_pack_causal | 天賦通道是否因果影響行為 | 單一天賦 |
| 1vN 行為品質 | diag_1vN | 0% 是「輸給人數」還是「打得爛」（model vs script 同局） | 1vN |
| 劣勢不划水 | eval_asym | 優劣勢/人數差下不idle、追戰局 | LEVEL_DELTA 桶 |
| 侵蝕守門 | 訓練內建 eval（std12 / vs-mon） | 補新能力有沒有弄壞舊能力 | **只看聚合 WR**，抓不到角落回歸 |

---

## 第二部分：涵蓋的對戰排列組合

- **陣型 arrangement**：1v1（主力）、1vN／劣勢（eval_asym / diag_1vN / eval_team_arrangements）、
  Nv1／優勢（eval_team_arrangements / diag_degen_audit 2v1）、NvN 隊伍（eval_team / eval_team_goal）。
- **身分 identity**：12 標準職業、1v1-viable 怪物、職業縫合怪、怪物縫合怪、synth 隨機 kit（eval_generalize）。
- **對手驅動**：稱職腳本專家（主力）、self-play 凍結快照（eval_selfplay_walls）、**靜止被動**（diag_wall_los / diag_ranged_standoff）。
- **注入條件**：對「我方主傷害型」注入免疫（probe_heldout_switch / diag_ranged_standoff / eval_immune_weak / seed_switch）、怪物天然抗性/免疫。
- **地形 terrain**：開闊地（主力）、牆（diag_wall_los / eval_wall_combat / eval_selfplay_walls）。
- **等級 level**：**3–8**（所有評估都在這帶）；少量 level-delta（eval_asym）。
- **傷害切招情境**：主招免疫→改次級（合成 kit 為主）。

---

## 第三部分：已知盲區（我的分析，2026-07-01）

> 「屬實」判準：能否構造一個**會影響勝率／輸出**、而**現有手段測不出**的情境。

1. **等級 > 8**：所有評估都在 L3–8；用戶 GUI 測 L20，整段外推從沒被評估。
2. **強弱錯配**：有能力的角色（L8 法師）vs 很弱的敵（15HP orc）。免疫只在**勢均力敵**測過→maxHP-gate dodge 崩的角落沒覆蓋。
3. **傷害輸出當主指標**：多數評估用 WR，對「0 傷但存活/平手」不敏感——dodge-collapse 就藏在 WR 底下沒被抓到。
4. **跨版本回歸偵測**：沒有任何腳本比對「這版在某角落有沒有比前一版最佳退步」。v6→v7 的 dodge 侵蝕就這樣漏掉。
5. **被動對手 × 全身分**：只有 diag_ranged_standoff（法師）測被動敵；沒掃過所有身分。
6. **分布外自訂 kit**：怪招移植到職業（cold_breath 上法師＝GUI kit 型），不在任何桶。
7. **瀕死/倒地敵/復活機制**：沒有評估「補刀倒地敵 / 撿倒地隊友」。
8. **條件免疫行為槓桿**：condition_immunity 通道行為面沒評估（1v1 零槓桿但團隊控場未測）。
9. **狀態 vs 免疫**：敵免疫我要施加的狀態（浪費控制）——行為面沒評估。
10. **反應/傳奇決策品質**：在真實戰鬥中的反應（shield/counterspell）決策沒有通用評估。
11. **GUI 部署制式本身**：人類被動玩家 + 高等級 + 任意手挑 kit（真實部署條件）不是任何評估桶。

---

## 第四部分：Subagent 迭代新增的盲區（append log）

> 每輪 subagent 讀本檔、提出未涵蓋情境；我（主控）驗證是否屬實，屬實才追加於此。

<!-- ITERATION LOG START -->

### 第 1 輪（已驗證屬實）— 主題：模型作為怪物攻勢/控制的「承受方」

現有評估**全部**把模型當攻擊方去打被動/腳本怪，**從沒把模型放在「被怪物控制/攻擊」的承受席**並用能抓壞行為的指標量。以下 5 條機制已在引擎核對存在、且無任何評估涵蓋：

12. **被擒抱/吞噬（kraken 觸手 / behir 緊勒 / tarrasque 巨顎）＋ obs 隱形**：模型被 grapple→`restrained`（速度 0），或被 `swallowed`。**已驗證：`swallowed`/`slowed` 在 `ENGINE_ONLY_STATUS_CLASSES`，不在 `RL_STATUS_NAMES`＝obs 完全看不見**（只有附掛的 restrained/blinded bit 露出）。壞行為＝速度 0 還一直 MOVE/kite 空轉挨 DoT，而非攻擊掙脫。量：被吞噬時 MOVE%（速度 0＝浪費）vs 攻擊掙脫率、DoT 校正存活回合 vs 腳本 oracle。
13. **恐懼光環 frightful_presence（白/紅龍、tarrasque）**：模型回合首被迫 WIS 存，失敗上 `frightened`。壞行為＝被恐懼後輸出崩/不再接戰。量：frightened 回合的攻擊嘗試率與傷害 vs 消融光環的同種子基線。（受害者側確切懲罰＝劣勢/不可趨近，待引擎核對；但「模型受敵光環」這方向確無評估。）
14. **再生怪 troll（每回合 +10，僅火/酸抑制）＝切招的鏡像**：模型有火/酸＋物理兩型，正解＝**每回合持續補火/酸壓制再生**（非「避開被抗的型」的一次性 argmax）。切招評估全是「對被動 orc 一次選型」，無任何評估把再生怪放對手席量「持續施型率」。量：troll 每個回合週期前有無模型的火/酸傷害、troll HP 淨侵蝕斜率 vs oracle。
15. **basilisk 石化凝視兩段式（restrained→petrified，不搶就鎖死）**：**已驗證 line 29「two-stage restrained→petrified per failed save」、basilisk 在 1v1 池（equiv 4.4）**。正解＝被restrained後搶在第二次失敗前爆殺；壞行為＝被石化前還在 kite。量：石化率、restrained 後到石化窗口內對 basilisk 的傷害曲線 vs oracle。
16. **beholder 眼射線（隨機多效，`slowed` obs 隱形）＋ balor 死亡爆裂 death_throes（死點 20d6 火 9m）**：**已驗證 eye_rays（line 485）、death_throes（line 627）存在**。(a) 中 `slowed`＝obs 看不見的速度/AC 懲罰→誤判走位/搆不到；(b) 貼著 balor 補刀＝吃 ~35 AoE，贏了也可能被帶走。量：(a) slowed 下「預期 reach vs 實際搆到」落差、搆不到而 whiff 的攻擊數；(b) balor 死亡瞬間模型與其距離＋吃到的 death_throes 傷害 vs 會先撤到 >9m 的 oracle。


### 第 2 輪（已驗證屬實）— 主題：隊伍協同 / 資源時序 / 對手席 boss / 目標選擇

17. **非 evocation 法師隊伍的 AoE 誤傷自己人**：模型駕駛 `divination+battle_master+life` 隊伍，divination 有 `fireball_div`（POINT 6m）但**無 sculpt_spells（已驗證：只有 evocation 有，archetypes.py:413；引擎 AoE 會打到自己人 combat.py:3082）**。把火球中心放在貼著我方前排的敵人＝炸掉自家前排+牧師、可能斷專注。**現有 FF 儀器 diag_coop_value 的模型臂只跑 evocation comps（sculpt-免疫，EXP C）**＝非 evocation 隊伍的火球落點沒人量。指標：引擎歸屬的 aoe_ally_dmg/場、concentration_breaks_self/場 vs 會重新瞄準的 oracle。
18. **多回合的 burst/專注「時機」**：模型駕駛 `vengeance`（smite 有限位）或 `evocation`（web/ice 專注）在 6+ 回合對稱局。壞行為＝第一回合把 smite 全丟進滿血敵、或在已專注時重丟別的專注（浪費）、或讓專注無謂中斷。**所有 WR 評估只給終局 bit；diag_degen_audit 明確把 buff/專注當非退化**＝抓不到「時機錯」（非「沒用」）。指標：有限資源花費的回合索引＋花費時目標 HP%、專注 uptime、`concentration_wasted` vs 專家的花費時機分布。（＝記憶標記的 resource-timing 殘留缺口，但**無任何評估儀器**。）
19. **1vN 角色差異化目標分流**：模型單體 vs **混色敵隊 [life 治療 / evocation 脆皮爆發 / battle_master 坦]**。有活著的敵方治療者時，不先秒治療＝磨不死＝平/負。**diag_1vN 的對手是同質輪替同職（STANDARD_IDS[(k+i)%len]，line 72）**，身分無差別＝測不出「打坦沒打治療」。指標：前 K 次攻擊落在治療/施法者 vs 坦的比例、`敵治療者在第 R 回合前被殺` 率 vs 「總打治療」oracle。
20. **模型當 BOSS 對稱職隊伍的目標選擇（部署制式）**：模型坐 agent 席駕駛 boss（manticore/mage_npc/basilisk 或帶隊 boss）vs 含治療(life)+脆皮(evocation)+坦的 3-PC 隊。Boss 的**主動作目標由 entity_head 控制**（傳奇動作硬寫打最近 combat_policy.py:1335，反應也是）；若 boss 老是打最近（通常是坦）就輸掉聰明 boss 會贏的局。**eval_monster_actor 用固定隊伍(line 66)＋只報 WR/usage-mix，從不記 boss 打了哪個 PC**＝「稱職用招但打錯目標」的 boss 與「打對」的同分。指標：boss 攻勢落在治療+施法者 vs 坦的比例、`敵治療被擊倒` 率 vs greedy-nearest 基線與 focus-healer oracle。
21. **脆皮遠程對「主動逼近」近戰的整場走位結果**（開闊地，非牆）：模型駕駛 L5 evocation/arcane_trickster vs 主動逼近的近戰猛獸（berserker/dire_wolf）。壞行為＝站進近戰 reach 被爆殺，而非拉開＋遠程輸出。**注意部分重疊**：`probe_kite_decision` 已探「approach vs kite 的決策方向」，但**沒量「對主動逼近者、整場有無被逼進 reach＋存活」的結果**；`diag_ranged_standoff` 是被動敵。精修缺口＝主動近戰下的**回合末落在敵 reach 內比例（明明有移動力可撤）＋存活/standoff 距離** vs kiting oracle。


### 第 3 輪（已驗證屬實）— 主題：感知極限 / 反應決策 / 地形光環 / 資源耗盡

22. **敵人實際 reach > 1.5m 對 obs 隱形**：巨人巨棒 3.0m、龍尾 4.5m、kraken 觸手 9.0m 等 reach 武器存在，但 **obs 的近戰威脅格寫死 `ENEMY_THREAT_RADIUS_M = 1.5+1.5`（obs.py:383，註解明寫「Fixed melee reach, not the enemy's equipped-weapon range」）**＝模型把所有近戰敵當 1.5m reach。壞行為＝站在 2-4m 以為安全、吃滿 3m 巨人攻擊；當 kiter 也退不夠遠。所有 kite/standoff 儀器都用 1.5m-reach 對手。指標：回合末與 reach 敵的距離 vs 該敵**實際** range_normal、「回合末落在敵實際 reach 內（明明有移動力可退出）」比例 vs reach-aware oracle。
23. **戰鬥中資源耗盡（箭/法術位）後不 fallback**：弓手 20 箭用完後 ATTACK 回 `ERROR「沒有箭了」`（combat.py:1697）、施法者法術位乾後只剩戲法/武器。**自身法術位/彈藥/次數 obs 刻意不讀（obs.py:509，公開資訊原則）**＝模型看不到自己快乾，可能整回合空放被拒的攻擊（0 傷）而非改近戰/戲法。diag_dead_action 只聚合 ERROR、不綁耗盡情境；WR 評估局短、20 箭很少用完。指標：強制少箭/少位起手，量耗盡後浪費回合率、到首次 fallback 的回合、耗盡前後 DPR vs 立即切換 oracle。
24. **站在持續傷害光環/危險地形不移開**：模型 vs fire_elemental（火光環）或敵牧師 spirit_guardians（4.5m 光環，tick_aura_damage），或在 `layout="lava"`（3×3 危險格，tick_terrain_damage 1d4/回，env_v2.py:377）。**已驗證無任何 eval 用 lava 或把模型放光環裡**（grep 空；diag_degen_audit 只在註解提光環、且開闊地）。壞行為＝回合末停在光環/熔岩上吃重複傷害，而安全格只差一步。光環半徑/中心也未編碼（只有施法者狀態 bit）。指標：回合末停在敵光環半徑/危險格內（有移動力可離）的比例、累計自身光環+地形傷害 vs 會離開的 oracle。
25. **通用反應決策品質（真實戰鬥，非手造判別局）**：模型駕駛能 shield/uncanny_dodge/counterspell 的身分，跨多種一般對局，看反應**觸發時機對不對**（大命中才 uncanny、危險法術才 counterspell，別浪費在小命中/戲法）。反應接縫由 `NeuralReactionDecider` 控制。**唯一反應儀器是 probe_reaction_headroom（只 shield、單一 evocation-vs-bm 配對）＋ train_reaction_lab（手造判別 lab）**＝沒有跨身分、跨三種反應、量觸發合適度的通用評估；記憶已測 shield「always」100% take-rate（不讀觸發）。指標：各反應型的 take-rate 按觸發價值分層（shield/uncanny 按 attack_margin 翻窗、counterspell 按 incoming spell_level）、浪費反應數 vs 價值感知 oracle，≥6 身分×對手。

> （第 3 輪 #5「補刀瀕死敵/撿倒地隊友」＝ Part 3 已列的 gap #7，不重計；本輪確認 `diag_revive.py` 是 **expert-vs-expert**、不驅動模型，佐證 #7 確為未涵蓋。）


### 第 4 輪（已驗證屬實）— 主題：分級傷害倍率 / 利用敵既有狀態 / 攻擊性打斷專注 / 移動地形 / kit 覆蓋

26. **敵人抗性(×0.5)/易傷(×2)——非全免疫——的選型**：模型駕駛雙傷害型（life：斬擊/光耀）vs `shadow`（斬擊×0.5、光耀×2）等分級倍率敵。正解＝選未被抗/被加倍的型，排序和「全免疫×0」不同。**所有行為切招測試只注 ×0 免疫（二元「避開死型」）；probe_resist_choice 是無模型的 info_gain 資訊天花板探針（不驅動策略）**＝沒有腳本量「模型選的型是否跟著 0.5/1/2 倍率走」。指標：每次攻擊所選型的倍率、用較高倍率選項的比例 vs oracle、淨 HP 侵蝕斜率 vs 固定最高基傷策略。
27. **利用敵人「已有」的優勢狀態（prone/restrained/stunned/paralyzed）**：敵人已被（腳本隊友/地形/預設）弄成 prone/paralyzed 等——攻擊它給**優勢**、近戰打 paralyzed＝**自動暴擊**。模型該優先打那個敵、且對 paralyzed 選近戰吃暴擊。**現有狀態相關項全是鏡像**（item12-16、24＝模型被控；item9＝敵免疫我要施的狀態）；「敵有狀態、你會不會利用」的攻擊槓桿**沒測**。指標：預設一敵 prone/paralyzed、另一敵滿血，量前 K 次攻擊落在受狀態敵的比例＋對 paralyzed 是否近戰，消融狀態驗因果。
28. **攻擊性打斷敵人專注**：敵施法者正專注某控制/增益（hold_person/spirit_guardians…），**任何傷害觸發 CON 存 DC max(10,dmg//2)、失敗清 `concentrating_on` 並移除該來源所有效果（combat.py:284-285,387）**。模型該集火那個專注者打斷法術（大傷害拉高 DC），而非分散/打坦。**item18 是模型浪費自己專注的防守面；「敵在專注、你會不會集火打斷」的攻擊面沒測**。指標：模型傷害落在專注者的比例、`concentration_break` 次數/回合 vs focus-caster oracle 與 nearest 基線；消融專注者狀態 bit 驗因果。
29. **difficult 地形的移動規劃（2× 移動花費）**：`layout="difficult"`（env_v2.py:375，移動花費加倍 combat.py:637）。kiter 要 2 倍預算才能拉開同樣距離、近戰可能搆不到。地形在 obs 可見（0.5/格），所以是**技能缺口非感知缺口**。**無任何 eval 傳 difficult 給模型（grep 空）；item24 是危險地形「離開」DoT、這是「距離規劃被地形花費弄壞」**。指標：difficult vs open 配對種子下，回合末與敵距離 vs 意圖、回合末落在敵 reach 內比例、whiff 率。
30. **縫合怪/synth 的 kit 覆蓋率退化當一級 gate**：大跨職 kit（chimera_omni/trickhealer）常**結構性棄用半套 kit**（記憶：trickhealer 塌回最近老師、omni 亂長 rage+fireball）。**eval_chimera 印每招使用率、diag_chimera_logits 分類為何休眠，但只在少數手挑狀態、只 3 個固定 chimera、只對 12 專家 L5**；**無聚合「M 招裡 N 招整場沒放」的覆蓋率 gate、無 synth 隨機 kit 覆蓋、無跨對手掃來區分「這裡用不到」vs「從不可用」**。WR 高但半套 kit 死掉的回歸今天過關。指標：全 chimera＋一批 synth × 多對手，per-identity **kit-coverage=（用過≥1 的非 move 招）/（合法≥1 的非 move 招）**＋合法-vs-選擇拆分，獨立於 WR 設門檻。


### 第 5 輪（已驗證屬實）— 主題：評估「方法學」漏洞（為什麼腳本測不出模型好壞、為什麼回歸沒被抓）

> 這輪不是新戰鬥情境，是 **HOW 的漏洞**——直指用戶「腳本沒辦法評估模型表現」與本場 v6→v7 回歸無人察覺。

31. **truncation/平手被併入「敗」，「把能贏的局拖到時限」對所有守門隱形**：既不贏也不死、把**能贏**的局拖到 step cap（dodge 崩/永遠 kite/拒戰）＝和誠實戰敗同分。**已驗證**：env 分 terminated/truncated，但每個守門都 `done=term or trunc` 然後 `won=敵死 and 我活`（eval_goal.py:46-48 等）；`measure_draws.py` 是唯一拆平手的、但 std12-1v1-only、未 gate、不看「當時是否領先」。修法：shared run_episode 加 `stalled_while_winning`（truncated 且終局 我方 HP-frac − 敵 HP-frac > τ），逐身分 gate ≤ ε，並在 eval_goal/eval_generalize 報 我勝/敵勝/同歸/平手 完整拆分。
32. **無「逐角落不侵蝕」守門——per-class 資料被直接丟棄（v6→v7 回歸盲的煙槍）**：**已驗證 `standard_probe` 算出 per-class 字典但呼叫端 `std0, _ = standard_probe(...)`（train_population.py:698）直接丟棄**，只用 12 職**聚合平均**對凍結 base 比。一個手術把某職某角落從 100%→0%、聚合平均只動 ±3-5pp＝完全藏住。compare_snapshots 印並排但無 per-corner Δ/不 fail；gate_slot 有 2σ 但只 team-slot 聚合。修法：`regression_gate.py --prev_best --candidate`，同種子跑 std12+怪+退化角落，`min over corners (cand−prev) < −2σ_cell` 即 FAIL、印最崩角落（保留那個被丟的 per 字典）。
33. **守門在 1–2 局/格、無信賴區間＝在純 RNG 噪音上做決策**：**已驗證 standard_probe 預設 games=1**（per-cell std ±50pp）、eval_generalize 預設 2/pairing；只有 gate_slot/eval_goal 算 2σ 且只對聚合。真 ±5pp 角落回歸可被「噪音內」無據駁回，false-adopt/false-clear 都活著，而「不侵蝕」宣稱全靠這些數字。修法：守門內建檢定力——給目標可偵測回歸 δ 算最小 games/cell、每角落印 Wilson CI，判「不侵蝕」須聲明實際能偵測的最小回歸；CI 太寬的角落標 UNDERPOWERED 而非默默 OK。
34. **對手席（部署 GUI 的真實席位）的play品質從沒逐身分評估**：模型可能 agent 席稱職、**對手席**（`_run_opponent_turn`→NeuralCombatPolicy）退化，跨多數身分，而守門全綠（記憶已有一個對手席專屬 bug＝漏空轉防呆）。**已驗證**多數 eval 把待測網放 agent 席；diag_degen_selfplay 驅兩席但只記 noop-move+trunc**兩席合併**、無法歸屬單席勝負/DPR/拒戰。修法：`eval_opponent_seat.py`——稱職專家在 agent 席、候選網用 use_self_play_opponent 駕對手席，跨全標準身分+怪樣本，報對手席 WR/DPR/拒戰/滿血補/0傷 逐身分（＝diag_degen_audit 的對手席鏡像）。
35. **無 DPR/擊殺時間一級指標＝慢殺/近拖與乾淨勝同分；開闊地自對打互相被動未量**：**已驗證 eval_v2 印 avg_rounds/avg_self_damage 但不 gate、且 self_damage 是「我受的傷」非「我造成的傷」＝根本沒算 DPR-dealt**。慢殺（資源亂用）和乾淨勝同分；DPR 崩是回歸的**領先指標**（早於 WR/平手崩）。自對打只在牆/鏡像測 trunc%，無開闊地跨身分非鏡像互相被動掃。修法：shared run_episode 加 DPR=（敵損 HP）/（我方回合）與 time-to-kill，逐身分對腳本臂 gate（DPR_model ≥ DPR_script − 2σ）；自對打掃擴到開闊地非鏡像對，用「兩席 DPR≈0 且 trunc% 升」聯合訊號抓互相退化均衡。

---

## 小結（5 輪迭代後）

現有評估的**根本缺陷**：以 **agent 席、開闊地、勢均力敵、L≤8、聚合 WR** 為主，對「模型作為承受方/對手席、面對怪物機制、在角落回歸、拖平手、慢殺」大面積盲。最關鍵的方法學洞（31-35）解釋了**為什麼一個能力回歸能無人察覺**——尤其 #32（per-class 資料被丟）。優先補：**逐角落不侵蝕守門 + 平手/DPR 一級指標 + 對手席評估**，因為它們是「守門本身可信」的前提。

---

## 第五部分：統一驗收入口 `scripts/eval_gate.py`（2026-07-01 落地）

把本檔收斂成**單一腳本**：給一個模型、跑一次、印一張全面驗收表。取代「人肉挑散落腳本、人肉讀數字、人肉判過不過」——那正是 v6→v7 dodge 回歸無人察覺的根因。

```
python scripts/eval_gate.py <model.pt> [--base <old_best.pt>] [--games N] [--only ...] [--list]
python scripts/eval_gate.py <model.pt> --self_test    # 測試這些測試：合成壞 agent 確認 check 有牙齒
```

**已接上的可執行檢測（會判 PASS/FAIL/UNDERPOWERED；20 個，涵蓋 goal 五大支柱＋承受方＋地形＋高階＋恐懼＋再生＋隊伍協同＋NvM 部署形狀＋治療者/怪席目標選擇）**：
- `std12`：12 職對稱腳本專家 WR + **逐職 DPR**（#35）+ **平手/拖平拆分**（#31），保留 per-class（#32）
- `immune`：法師 vs 弱小免疫敵的傷害輸出＝**dodge-collapse 偵測器**（Part1 immune / #2 / #3 / #23）
- `degen`：整場致命退化審計（卡死/被夾/真拒戰/滿血補；含 chimera/monster 身分）→ **零退化**支柱
- `oppseat`：**候選網駕對手席**會不會退化空轉（#34，部署 GUI 真實席位）
- `general`：扮 **怪/職業縫合怪/怪縫合怪/synth 隨機 kit**（重用 eval_generalize 多桶 Δpp vs 駕同 kit 腳本；#6/#30）→ **扮任意身分**支柱
- `walls`：走位／繞牆視線重建（牆地形 WR 不該相對開闊地崩塌；#21）→ **走位**支柱
- `asym`：優劣勢不划水（劣勢 1v2／低階 Δ-3、優勢 2v1；首手 idle 率）→ **面對優劣勢**支柱
- `switch`：依敵抗性資訊切傷害型（self-calibrating：主型被免疫→是否改砸非免疫型；Part1 held-out 切招 / #26）→ **依資訊選招**支柱
- `passive`：對消極/被動敵（GUI 人類玩家式）不空轉（全身分 vs 靜止敵、量 dmg%/idle；#5 / #11）→ 用戶最初手抓的 **Evoker disengage-loop** 正是此類（近零傷害硬 FAIL；建議 games≥4 才穩定）
- `exploit`：利用敵既有優勢狀態（一被動敵預設 prone→數攻擊落點傾斜，純目標選擇＋on/off 因果；#27）→ 軟能力，只在「明顯避開可利用敵(超2σ)」或 vs base 回歸時 FAIL，否則攤開 Δ 當能力指標
- `victim`：**承受方**——被 restrained(速度0)＋敵貼臉→該攻擊而非空移浪費回合（#12 被擒抱/緊勒的行為面）。引擎事實：restrained 不減 movement 資源→MOVE 不被 mask、可空轉挨打；空移率>15%＝FAIL
- `dead`：死動作（Part1）——暫時 patch execute_action 數「agent 執行時回 ERROR」的動作＝合法性遮罩漏網＝浪費回合；任一身分 >2% 死動作＝FAIL
- `infochan`：逐通道因果（Part1）——敵每個資訊通道歸零→行為翻不翻（重用 diag_info_channels，描述子豐富的怪取樣）。實測 uni_v7：typed_resist flip 22%、passive_traits 7%（天賦因果）、WHOLE_desc sanity 14%、**condition_immunity 0%（不讀→#8/#9 浪費控制的根源，數據坐實）**。一個儀器覆蓋 逐通道因果＋天賦因果＋#8/#9。主靠回歸守門抓「本讀某資訊、後不讀」
- `terrain`：特殊地形（#29 difficult / #24 lava）——difficult(2×移動)功能是否保留（gate：diff/open 傷害比<40%＝崩）；lava 自傷增量**只報不判**（實測正負混雜＝局部危險格低訊號，當 gate 會 flaky）
- `highlevel`：高等級外推（#1 L>8）——L20 對稱局核心行為（傷害/勝/空轉）。實測 uni_v7 全身分正常接戰（最低傷害 49%、0% 空轉）＝obs 外推未崩。用戶在 L20 玩、訓練只到 L8，這是唯一驗證高階的 check
- `frighten`：承受方——被 frightened 後**遠程** agent 是否還接戰輸出（#13）。引擎：frightened=攻擊 disadvantage＋不能朝源移動、但不阻止攻擊。on/off 因果；攻擊率崩<30%且明顯低於常態＝#13 輸出崩。uni_v7 恐懼下攻擊率≈常態 PASS
- `regen`：再生怪壓制（#14）——對 troll(每回合再生、火/酸抑制) regen-ON/OFF 淨侵蝕差。火法師天生壓制→regen-ON 淨傷 ~0=壓不住=FAIL。uni_v7 火法師 regen-ON 淨傷 73-75%(≈OFF)=壓制有效 PASS
- `teamff`：非evo 隊伍 AoE 誤傷（#17）——放 aoe_radius>0 招時用 act[2]→世界座標，量活著隊友是否落半徑內。**軟判**（設定把敵叢集隊友旁→FF 率部分被迫）：報＋回歸守門，極端>60%或回歸才 FAIL。uni_v7 divination 50% AoE 炸到隊友(真發現、軟判)
- `nvm`：**NvM 部署形狀**（07-03 補；怪1v隊3／隊3v怪1／2v2／劣勢1v3／3v3）——舊儀器最多 2 實體、「怪 vs 冒險隊」主形狀從未被驗過的結構性盲區。雙門：首手 idle ≤10%（asym 同門檻）＋接戰輸出 ≥ 同席**逐席身分**腳本專家的一半（`_multi_expert_driver`；專家 dealt<15% 的桶＝對局不可贏、只考 idle＝miner 專家仲裁同原則）。標定教訓：07-03 前探針把「零傷害場」(速死/全 miss，專家同席同零) 誤計成空轉→誤判 1vN 退化；本檢查兩個指標都分得開匹配現實與模型退化。實測 uni_v10 五桶全綠（dealt 追平或超專家）＋always_end 牙齒證明。bug_miner 生成空間同步擴 NvM（na/no∈{1..3} 獨立抽、混編敵隊、隊友從全目錄抽）
- `healerfocus`：**敵隊含治療者／怪席（BOSS 視角）目標選擇**（07-03b 補；#19/#20 轉正）——桶＝1v2敵治療/2v2敵治療/3v3敵治療（敵含 life）＋怪1v隊3（怪席 vs 含治療者隊）。**設計上尊重舊「刻意不做」的 flaky 批評**：不對「必須先秒治療者」設硬門檻（NvM Phase-1 三個 oracle 介入實驗已證目標紀律在部署形狀**因果惰性**＝硬門檻是偽科學）；門＝nvm 同款雙門（不划水＋接戰≥同席專家半），hFirst（首殺=治療者率）/hShare（治療者活著時傷害決策集火率）**只印＋入回歸 cells**——實測 uni_v10 集火治療者 hShare 79-86% **遠超**腳本專家 39-67%（含怪席 70/31）＝從沒訓過的**湧現能力**，回歸守門保它不流失。治療者判定＝特徵式（kit 含 ally-target 治療技；second_wind 自療不算——第一版踩過「全敵皆治療者」的坑）。儀器本體＝`scripts/diag_nvm_behavior.py`（成對種子、意圖解碼、含 low/sticky/healer 三個 oracle 介入臂）

> 實測（uni_v7 @games=2）：9 PASS / 2 FAIL——只有 `immune`(dodge-collapse) 與 `degen`(滿血補) 掛，其餘全綠。即「丟模型、跑一次、得全面驗收結果」。

### 剩餘項目的定案（清單「解決」＝每項都有明確歸屬，無懸空）

**已建（本輪補完）：** #13→`frighten`、#14→`regen`、#17→`teamff`（見上）。**07-03b：#19/#20→`healerfocus` 從「刻意不做」轉正**——原 flaky 批評（無乾淨因果開關、硬判目標智能會 flaky）被新設計繞開：行為指標只印＋回歸守門、判定門全在結果層（見上）。

**做不成乾淨 check／低值（nuanced，硬建會 flaky，故只列不建，附理由）：**
| 項目 | 理由 |
|---|---|
| #7 補刀倒地敵/撿隊友 | 「補刀 vs 忽略（敵已中立化）」**無單一正解**＝像 #19 混淆；撿倒地隊友需 revive 機制+team 設定，訊號稀 |
| #16 death_throes 補刀距離 | 需先在 1v1 打贏特定 boss(balor)＋slowed obs 隱形＝極 niche、樣本稀、低值 |
| #18 burst/專注時機 | 「smite 該第幾回合丟、目標 HP% 多少才對」＝**主觀門檻**，無客觀對錯線可判＝硬定門檻會 flaky |

**刻意不做（做不成乾淨 check；硬做會 flaky／obs 天生看不見）：**
| 項目 | 理由 |
|---|---|
| #22 敵 reach>1.5m | obs 寫死 1.5m，模型**根本收不到**真 reach＝obs 缺陷非行為缺陷，改行為也沒用 |
| #28 打斷專注 | 「集火打斷」與「避開傷害光環」正解相反＝無單一正確答案可判 |
| #10 / #25 反應決策品質 | 記憶已量測：目前 always≈greedy＝**無 discrimination headroom**，造不出有意義 pass/fail |
| #15 石化第二段 | petrified＝incapacitated＝不能行動＝無行為可測；第一段 restrained 已由 `victim` 涵蓋 |

**已部分涵蓋（附註即足）：** #26(switch 做 ×0，分級 0.5/2 未做)、#30(general 有 Δpp，獨立 kit-coverage 指標未做)、#33(已印 n/σ/UNDERPOWERED，最小局數推導未做)、usage-mix(general/std12 印 mix，無自動門檻)。

**結論**：48 項全部定案＝**18 個可執行 check（覆蓋 36 項）＋3 項 nuanced 不建（附理由）＋6 項刻意不做（附理由）＋部分涵蓋若干**。沒有任何一項停在「未決」。

### 第六部分：驗證狀態（test the tests，2026-07-01）

「可信」＝該 check 被證明**好壞模型都判對**（不只跑得動）。用版本歷史當 known-bad 樣本兩面驗證：

| check | 兩面驗證 | 證據 |
|---|---|---|
| `immune` | ✅ 兩面成立 | uni_v6 PASS 100%傷 ／ uni_v7 FAIL 0%傷（dodge-collapse 回歸） |
| `switch` | ✅ 兩面成立 | uni_v5(rsw前) FAIL：注免疫後仍砸 84% ／ uni_v7(rsw後) PASS：切到 0%。正確區分 rsw 手術邊界 |
| `infochan` | ✅ 指標兩面 | typed_resist flip：v5 **5.4%** ／ v7 **23.5%**（sanity 3.8%/14.5%）＝正確反映「v7 讀抗性、v5 不讀」。但 gate 是回歸式，需 `--base` 才判 FAIL |
| `degen` | ✅ 兩面成立（區分版本） | uni_v1 **8** 個致命退化 ／ uni_v7 **6** 個（v1 多 evocation/orc 空轉、v7 修掉）＝隨版本變＝measure 真行為。且抓到跨版本**持續真退化**：battle_master 滿血硬補 second_wind（v1/v7 都有）、basilisk 怪席空轉 |
| `passive` | ⚠ 會開火、未深驗 | 抓到 uni_v6 vengeance 對被動 orc 4 局全 0 傷；真偽未深追 |
| `walls` | ✗ **無法用現有 ckpt 驗** | 連最舊 uni_v1 都 PASS＝版本歷史裡沒有「牆大崩」樣本。且 LoS 手術(v2→v3)只動 +3pp、遠低於本 check 的 35pp 崩潰門檻＝**粗檢，抓不到細微 LoS，假陰性率未知** |
| 其餘 9 個 | 未驗 | 需構造 known-bad（擾動權重）或找對應版本邊界；多數只證了「跑得動＋對正常模型 PASS」 |

**誠實總結**：**4 個 check（immune/switch/infochan/degen）以真模型兩面驗證成立**；1 個（passive）會對真問題開火但未深驗；1 個（walls）證明是粗檢且無樣本可驗。

#### 6.1 `--self_test`（合成壞 agent、確認 check 有牙齒）

不必等版本歷史剛好有壞模型——`python scripts/eval_gate.py <任一模型> --self_test` 會**包住真 net、竄改輸出＝合成明顯壞掉的 agent**，確認該抓到它的 check 真的印 FAIL。目前的壞 agent：
- `always_end`（每回合直接結束、什麼都不做）→ 應被 `std12/immune/passive/degen/asym/highlevel/regen/terrain/oppseat/general` 這 **10 個** check 抓到。

**這一跑就抓到一個真 bug**：`oppseat` 對 always_end 竟印 PASS（說對手席有 17% 攻勢）＝瞎了。追下去發現**根因**：`use_self_play_opponent` 只設 `_opponent_override` 旗標、由 **reset() 才讀它建 NeuralCombatPolicy 對手席**（env_v2.py:257-266），而 `drive_episode` 之前是**先 reset 後才呼叫**＝override 沒生效、對手席一直是腳本對手。**故 oppseat 先前所有結果（含 uni_v6/v7 PASS）都是測錯席位＝假的。** 修法：把 `use_self_play_opponent` 移到 reset 之前。修後 self_test 全 10 個 check 有牙齒；oppseat 真實結果 uni_v7 對手席攻勢 44-96%、uni_v6 最低 41%（皆 PASS，這次是真數據）。

**self_test 抓到的第二個真缺陷——`victim` 是瞎的**：原 victim 測「被 restrained 還空移浪費」，但 `available_skills`（skill.py:575-582）在 speed×0 時**直接不列 MOVE**＝該退化引擎根本不可能發生＝victim 測不可能事件＝永遠 PASS。已**重寫**成量「被 restrained＋敵貼臉時的拒戰率（能打卻選 END 放棄）」＝真退化；並加進 always_end 的 should_fail 兩面驗證（always_end 被 restrained→100% 拒戰→victim FAIL 有牙齒；真 uni_v7 0% 拒戰＝有意義 PASS）。

**全套合成壞 agent（每個 check 都證明有牙齒）**：`always_end`(什麼都不做→抓 std12/immune/passive/degen/asym/highlevel/regen/terrain/oppseat/general/victim 11 個)、`freeze_when_frightened`(被恐懼就停→frighten)、`avoid_prone`(反向利用敵狀態→exploit)、`freeze_on_walls`(看到牆就停→walls，證明 walls 能抓『開闊正常/牆上崩』的粗崩)、`aoe_on_ally`(AoE 強制打隊友→teamff)、`broken_mask`(模擬遮罩破掉→dead 抓 ERROR)。

**驗證後帳目**：**全 18 個 check 都有牙齒證明**＝16 個靠合成壞 agent（上列）＋ switch/infochan 靠真模型兩面（v5/v7）。immune/degen 亦真模型兩面成立。

> **「測試這些測試」共抓到三個真缺陷**：`oppseat`（呼叫順序錯→測腳本對手非候選網）、`victim`（測引擎不可能發生的事件→永遠假 PASS）、`entity_logits 誤當一維`（合成 agent 目標覆寫沒作用、順帶證原 pick_action 目標維度理解錯）。前兩者是「跑一次給答案、但答案沒意義」的瞎 check——正是用戶最初擔心的「檢測腳本沒辦法評估模型」。都已修。
> **合成也揭露弱檢**：`teamff` 故意亂炸(~71%) vs 正常非-sculpt(~60%) 差距薄＝S/N 低（軟判，threshold 0.5）；`walls` 對細微 LoS 仍粗（只抓粗崩，不抓 3pp 級 LoS 差異）。這些是誠實的能力邊界，不是假裝全能。

**核心＝逐角落不侵蝕守門**（#4 / #32）：給 `--base` 舊最佳，任一角落比 base 退步 > 2σ（或明顯全崩 ≥30pp 硬地板）即紅旗回歸 FAIL、印出最崩角落。已驗證：`eval_gate uni_v6` 全綠、`eval_gate uni_v7 --base uni_v6` 精準 FAIL 並標「evocation/orc dmg% −100%」＝當初存在就會擋下該回歸的儀器。

**低局數誠實**（#33）：Δpp 這類噪音大的指標在低 n 只標 `UNDERPOWERED`（超出 2σ 才硬 FAIL），並印 n/cell 與 σ；不在噪音上假判。

**顯性盲區**：覆蓋表把本檔**每一項**都列出，未接上可執行檢測的顯性標 `NOT_COVERED`＋原因——「漏一個」從隱形變成表上一行。仍未接：反應品質(#10/#25)、承受方怪物機制(#12-16)、reach>1.5(#22)、危險地形(#24)、利用敵狀態(#27)、打斷專注(#28)、瀕死/復活(#7)、等級>8(#1)…（下一批接入目標）。

<!-- ITERATION LOG END -->

## 第七部分：生成式偵錯 `scripts/bug_miner.py`（2026-07-02 落地）

**動機（用戶的結構性批評）**：eval_gate 的 18 個 check 都是「人先想到失敗模式→寫死一個 check」＝列舉式；沒被想到的失敗模式結構上永遠測不到（吐息怪的遮罩漏洞就是實例）。bug_miner 反轉：**隨機生成情境 × 逐決策機械審計**，bug 型別不預先歸類。

**三層儀器**（與 eval_gate 完全分離，eval_gate 不動、可隨時退回）：
- **A 逐決策健全不變量**（引擎事實判「這一步確定錯」；定義沿用退化審查波驗證過的 diag_degen_audit 版本）：`refuse / stall_move / blocked_move / heal_full / dead_action(引擎ERROR) / immune_hit(EV≈0攻擊而有正EV替代) / null_control(對條件免疫敵施控) / no_engage / opp_no_engage`
- **B 軟事件**：`avoid_easy_target(放著prone敵打健康敵) / aoe_ally / idle_first`
- **C 成對條件差分**（同種子雙胞胎）：layout↔open、注入免疫↔無、模型↔腳本專家同席

**分類問題的解**（不重新引入列舉）：每事件掛機械 context 標籤（layout/突變/敵我狀態位/隊形），報告對每偵測器算**標籤 lift**＝觸發條件由數據自己說；致命事件當場跑**通道掃描**（逐 obs 通道歸零→模型動作翻不翻＝讀資訊軸根因）＋ immune_hit 附 oracle 端依賴；簽名比對指紋庫（`bug_miner_fingerprints.json`）→ KNOWN(名)/NEW。沒見過的組合自動成為新 (偵測器×高lift標籤) 列。每簽名附 `--replay <scen_key>` 一鍵重現（逐決策 verbose＋通道詳情）。

**V1 盲測（合成壞 agent，miner 不被告知植入什麼）＝6/6 全挖出**：always_end→refuse×433+no_engage；freeze_when_frightened→**lift 自動定位 self:frightened**（refuse×45[self:frightened]登頂）；avoid_prone→avoid_easy_target；freeze_on_walls→致命事件集中 layout:walls；aoe_on_ally→aoe_ally；broken_mask→dead_action×118。

**V2 真版本邊界（n=80 seed=3 同劇本）＝已知 bug 全部盲重現＋排序重現版本史**：
| 偵測器 | v5 | v6 | v7 | v9 | 對應已知 |
|---|---|---|---|---|---|
| no_engage | 28 | **6** | **36** | 18 | immune 崩世代 v7 最糟、v6 最乾淨、v9 半恢復 ✓ |
| heal_full | 34 | 15 | 20 | 9 | bm 滿血補家族，逐版下降 ✓ |
| immune_hit(+oracle_dep=typed_resist) | 4 | 2 | 20 | 8 | 不切招家族 ✓ |
| opp_no_engage | 0 | 0 | 1 | 0 | 對手席停滯家族 ✓ |

**首日新發現（全部 replay 驗證、eval_gate 看不到）**：
1. **引擎 bug#1**：LINE 吐息可瞄自己格→引擎拒「需要一個方向」→浪費回合（storm_ogre；check_dead 身分池無吐息怪＝永遠測不到）。
2. **引擎 bug#2**：SINGLE_ENEMY 攻擊的**實體目標層遮罩不做距離判定**（技能層只保證「有人在射程內」）→模型選 1.6m 外目標、reach 1.5m→ERROR 迴圈（v5 一場 75 回合 0 傷平手＝絕對影響勝率）。
3. **v9 disengage 角落**：evocation vs 被動 evocation L4 開闊地，t25 起原地空移到 50 回合 0 傷平手（`stall_move`×9＋no_engage；v6 同情境也 0 傷）＝用戶手抓的 disengage-loop 家族仍活在現任最佳模型的角落；check_passive（固定 orc 敵）沒看到。
4. **v9 heal_full 殘留**：2v1+out_reach 情境 9 次（degen check 固定面板判 v9=0 ＝角落沒采到）。
5. **v9 mirror no_engage 叢集**：18 場、敗率 89%、top lift opp:mirror——待下一波修模時追根因。

**儀器自身的健全性修正（開發中抓到）**：immune 注入對單傷害型 kit＝遊戲不可贏局（champion 注斬擊免疫 100→0 不是模型錯）→加**公平性守門**（注入後無有意義替代 EV 即撤回、tag mutation:immune_skipped）＋ no_engage 要求「kit 級可贏性」（傷害型×敵抗性全 0 倍率的天生死局不算退化）——1vN game-impossible 教訓同族。

**誠實邊界（尚不能取代 eval_gate 的部分）**：std12 專家同席差分目前采樣稀（n=1/run，需拉高 twin_expert 采樣或全 12 職固定配額）；regen/troll 壓制這類「trait 因果開關」儀器未接；oppseat 只有場級 opp_no_engage（無 eval_gate 的對手席 WR 深度）；general 的 Δpp 桶聚合未做。**現行分工＝eval_gate 是驗收回歸門（守既有能力），bug_miner 是發現與根因儀器（挖沒人想到的洞）**；要完全取代需先補上述四項並在同批版本上證明逐項不劣。

用法：`python scripts/bug_miner.py <m.pt> --n 200 --seed 0 [--out rep.json]`；`--self_test`；`--replay "<scen_key>" --n/--seed 同報告那次`。

### 7.1 首日發現的修復（2026-07-02，「解決BUG吧」波）

**兩個引擎遮罩洞已修（模型無關、全版本受益）**，皆 TDD（失敗測試→修→轉綠）＋replay 翻盤證據＋守門零侵蝕：
1. **LINE 自指**：修在 `point_validity_mask`（action.py）——LINE 技且施法者站格心時把自己格標非法（精確鏡射引擎 aim<1e-6 拒絕條件；在無牆 early-return 之前）。證據：mine|1|0016 storm_ogre 從「12 回合 0 傷落敗＋6 ERROR」→「4 回合 30 傷勝、零事件」。測試 `test_point_mask_blocks_line_zero_aim_own_cell`。
2. **實體目標超距**：修在 `apply_entity_mask`（model.py）——ENEMY 列加引擎同源 per-(skill,slot) 合法性門（ATTACK 用 `attack_range_check`＝近戰 reach/遠程 range_long＋LoS＋requires/blocked_by_target_status；帶 range_m 的單體法術用引擎逐位元同條件），**每列全遮則還原該列**＝永不製造「無合法目標」新狀態。證據：mine|3|0004 zombie 從「75 回合 0 傷平手＋180 ERROR」→「37 回合勝」，且該場 avoid_easy_target 一併消失（遠端非法目標被遮後 argmax 落回貼臉 prone 敵）。測試 `test_entity_mask_blocks_out_of_reach_enemy_slot`。
   驗證：post-fix 重挖 v9 **dead_action 38→0、其餘簽名數字不動**；eval_gate 17/18 PASS 唯 immune FAIL＝與修前完全相同＝零侵蝕；全套件 470 綠。

**儀器健全性 #3**：`no_engage` 加 `chose_offense`——dmg=0 分不開「從不攻擊」與「攻了全 miss」（實測 L1 wolf vs BM mirror 兩咬全 miss＝匹配非退化）；修後 v9 no_engage 18→6、**倖存 6 場全是真的**。`opp_no_engage` 同縫隙但對手席看不到逐決策動作＝標註弱定義。

**v9 模型行為簽名根因（下一波的靶）**：
- **resist→dodge 塌縮（真決策 bug、immune-FAIL 家族本體）**：totem_bear L10 滿血貼臉 L1 shadow（16 HP、全物理 0.5×抗性），長劍/reckless 合法卻永遠 move→dodge→END 50 回合。**因果錘定：ablate 敵 typed_resist → 立刻改選 rage**。＝模型學到「有抗性→消極」而非「打折仍最佳就照打」；與 v7 切招手術 anchor=0 診斷同族。修法＝switch-lab 式隔離手術（resist 子欄+join 凍結配方）教「抗性下攻擊」，`eval_gate v10 --base v9` 守門。
- 0009 stall＝magic_missile×4 射進對面 shield 反應（反應盲＝reaction M3 已知）＋ L4 evocation 無傷害戲法、法術位耗盡後結構性零輸出（kit 事實非決策 bug）。
- heal_full ×9＝已知 second_wind 滿血補家族（低影響、勝場照贏）。
- avoid_easy_target（不優先打 prone 敵）＝軟能力缺口，待訓練波。

**旁收**：兩個預存在測試失敗（HEAD 乾淨 worktree 證實）修復——test_combat 兩測 patch 錯接縫（resolve_attack 走 roll_d20 非 roll＝live 骰 flaky）；test_tag_parser 字串比對 StatusEffect 物件（adds 永遠 False、removes 永遠 True＝瞎測試，重寫走引擎 has_status）。

## 第七部分之二：v10 抗性手術波（2026-07-02，goal=雙儀器全過）

### 根因鏈（全數據錘定，probe_resist_curve.py）
1. **行為層**：v9 對「外型抗性條目存在」全面消極（A@物理=1.0 dodge 塌縮），對「自身型 ×0 免疫」照打（B@0.0）＝13 個獨立開關、不讀值不讀符號（--decomp：僅光耀=2.0 易傷也躲）。
2. **版本鏈**：v3 語義大致正確 → v3→v5 裝入亂打免疫 → **v6→v7 rsw 手術裝入塌縮**（4攻勝→0攻敗）→ v7=v8=v9。
3. **權重考古**：skill_head 輸入=`sk(64)+h(128)+ctx(32)+typed-join(1)+blocked(64)+cimmun(1)`——`ncol-1` 位置漂移 bug 讓 rsw 的凍結 hook/監看**全訓錯欄**（cimmun 特徵恆 0＝零梯度）；typed join v7=+0.314=v3 原值＝**全程凍結**，pooled 13 欄（norm 2.1→7.5）成唯一載體＝捷徑本體。修復＝model.py `net.tjoin_col` 具名索引。

### Step 0：載體歸零 `_v10_reset.pt`（make_v10_reset.py）
只歸零 pooled 13 欄；typed join +0.314 保留（DAgger 底座健全語義）。驗證：無抗性局 5 場逐決策 logits **位元級同 v9**（恆等式+實測）；vs shadow 恢復攻擊獲勝。reset 模型 miner n=16 smoke：**致命事件 0**。

### 訓練：train_resist_grid.py（純 PPO、無 oracle、凍結手術）
網格=幅度(0/0.25/0.5/2)×符號(易傷)×外型欄×多欄疊加×單/雙型kit×真職業(berserker/champion/devotion)×天然怪(shadow/gargoyle/skeleton/zombie)×1v2目標選擇；公平性=單型永不×0、注入後無 EV 逐步抬回 0.5；hold-out=wight/ghoul（零樣本守門）。**run1 診斷**：uniform lr 1e-4 下 join 20u 只長 +0.07、wrong0 卡 63%、pooled 漂移 → **run2 修正**＝join 欄獨立 lr 5e-3＋×0 格加密：u5 即 join +0.642、switch WR 94%、wrong0 8%、pooled 僅 0.086。

### 同波儀器/引擎修復
- **heal_full 家族引擎修**：apply_resource_mask 滿血遮純治療技（特徵謂詞 expected_healing>0/無傷害/無狀態，SELF 看自己、ALLY 看全隊，0.95 門檻對齊 degen 偵測器；dying=hp0 自動算受傷→復活可用）＋失敗測試轉綠。註：degen 波記憶中「已修」在本分支不存在＝舊記憶需驗證的實例。
- **miner 健全性 #4**：stall_move 加 offense_now 門（available_skills 已資源過濾→法術耗盡+無武器=不可贏局；武器不受距離影響→近戰追擊空轉照抓）＝0009 evocation 假陽性關閉。
- **miner 突變池 +resist 0.5**（#26 後半進回歸門）＋指紋 v9-resist-dodge-collapse＋pair-diff 維度通用化。
- **v9 no_engage×6 叢集收斂**：replay 證全是天然 shadow 的 resist 家族（lift 的 mirror/dlvl:3 是混淆因子）——四個致命家族只剩 resist 一個真模型 bug。

### v10 波終局（2026-07-02）：GOAL MET —— 雙儀器正式雙過
**交付物 `models/unified/uni_v10.pt`**（=resist_grid7b/rg_u0005，鏈：uni_v9 → pooled 歸零 reset → 網格 PPO run7(+L1/L3)+7b 續訓 15u）＋**決策層守門套件**（隨代碼、對所有模型生效）。

**正式驗收（同一檢查點）**：
- `eval_gate uni_v10 --base uni_v9`：**18/18 全 PASS、逐角落零回歸、總判定驗收通過**（scripts/_gate_uni_v10_final.txt）
- `bug_miner uni_v10 --n 200 --seed 0`：**致命不變量 0 事件**（6098 決策；軟事件僅 avoid_easy_target 0%敗率影響＋2 件 expert_parity 情境事件）（scripts/_miner_uni_v10_final.txt）
- 探針：resist 曲線單調、--decomp 全欄健康（含光耀+2 易傷照打）、heldout wight/ghoul 零樣本攻擊獲勝、六案例 switch（合成×2/mage_npc/devotion L20/vengeance/life）WR 8/8 wrong0=0

**守門套件（引擎真值、特徵謂詞、零技能名，全 TDD）**：
1. AoE 支配格受限重選（pick_action）：乾淨落點（零隊友、敵覆蓋≥）存在時按模型 logits 重選；肉搏團不干預——teamff 60%→0%
2. 重複 dodge 零壓力遮罩（resolve_attack 計數 `_incoming_attempts`）：連續 dodge 且期間零來襲＝實證 null——解凍 berserker 對被動敵全部凍結格
3. 免疫空砸遮罩（apply_resource_mask pre-pass）：純傷害技對全體在世敵份額加權倍率≤0.05 且存在 >0.05 替代＝嚴格支配——關閉聖騎士/施法者 attack-into-×0 尾巴（單共享 join 純量的結構張力：join>2 蝕 through，run4/5/6 三現＝訓練không收尾）
4. env 失能守門（step 強制 END，對稱 _run_opponent_turn）——麻痺 dead_action 噪音歸零

**儀器升級**：miner +resist(0.5)/vuln(2.0) 突變＋專家基準仲裁（#5：專家同零輸出→expert_parity 降級；專家有輸出→維持致命）；eval_gate infochan 回歸規則對齊語義（跨 5% 讀取門檻才算退化）、teamff 反事實乾淨落點量測、self-test guard_holds 語義；stall_move offense_now 門（#4）。SELF-TEST 全過、477 測試綠。

**誠實邊界（透明記錄）**：seed-3 複測餘 1 既存口袋 `defensive-inertia-under-fire`（berserker 劣勢對主動施法者龜縮；v9 位元級同值＝零回歸；敵無抗性條目＝本波載體零梯度路徑）→ 指紋歸檔，屬防禦惰性/走位波（generalize_grid 缺口#3）。expert_parity 三情境（雙法師風箏怪／shield 吃 mm／永久定身）＝引擎可證 game-impossible。avoid_easy_target 軟事件（prone 目標優先偏好）0% 敗率影響、待目標選擇波。

### v10 守門補遺波（2026-07-02 深夜，用戶 seed666 抽查觸發）
**用戶 n=500/seed666 逼出 no_engage×7（100%敗）→ 解剖成三件事：**
1. **儀器健全性 #6**：單輪速死局（衝滿移動仍差 0.5m、行動點無合法攻擊、下輪前被轟死）「從無可攻擊時刻」＝匹配/速度事實非拒戰（miner turns 計子決策非回合、>2 門檻失義）→ no_engage 加 ever_attackable 前提；seed666 致命 7→2。
2. **接戰承諾失敗家族（真 bug、v9 位元級同值既存）**：變體 A＝berserker 進 reach 每輪 dodge（受壓＝repeat-dodge 守門正確放行）；變體 B＝門口 2.0m 站樁（行動花自療、移動不花）。觸發域=mirror 神經施法者（腳本施法者誘發不了）。**DAgger/BC 修復線 5 次實驗全數失敗後放棄**——END 標籤毒（refuse×1355）→被動格→END過濾→被動錨→覆蓋錨（miner 分布），全網 BC 每輪都在未錨定角落漏新洞；佐證記憶：berserker＝obs_v3 波已判「拒絕重訓」職業。**治本＝純龜縮守門**（apply_resource_mask）：單挑＋整場零出手嘗試（execute_action 計 _outgoing_attempts，含救投法術）＋敵在 reach+剩餘移動內＋行動在手 → 遮 dodge/hide；出手一次永久解鎖（坦克-有輸出合法）。索引病例 0119（40傷勝）/0236（18傷接戰）/掃描格 A0-A2 全修；A3＝無決策錯誤（正確接近+移動耗盡後 brace，輸在等級差爆發）。
3. **決定論護欄**：批量/隔離同種子軌跡分岔懸案（0236、0101）錘定＝多重程序並行時 MKL 動態執行緒改變浮點歸約順序→近平手 argmax 翻面。miner 加 MKL_DYNAMIC=FALSE + torch.set_num_threads(1)；**驗收跑必須單獨執行**。
（附帶：expert 仲裁 #5 已在本波前落地——no_engage/stall 類致命事件由同身分腳本專家同席同種子仲裁，專家同零輸出＝game-impossible 降級 expert_parity。）

**守門補遺波終驗（同夜）**：0477 對手席 50 回合停擺真因＝**距離遮罩邊界錯殺**（`>= reach-0.01` 遮掉引擎合法的 d==reach；attack_range_check 是 `d > reach` 才拒）＋完美風暴（定身敵不動＝距離恆 1.5、move 空轉歸零、dodge/hide 被龜縮守門遮、只剩 END）→ 修＝引擎同源 `> reach + 1e-6`（Vec2 同源浮點＝位元安全），TDD 479 綠。**最終串行終驗（單獨執行＝決定論）：miner seed666 n=500 致命 0（14274 決策）＋ miner seed0 n=200 致命 0（6124 決策）＋ eval_gate 18/18 --base v9 零回歸驗收通過。**
