"""生成式偵錯 (bug miner)：不預設 bug 型別，隨機生成情境、逐決策抓「機械可判的錯誤」。

與 eval_gate 的關係（eval_gate 不動、可隨時退回）：
  eval_gate = 固定 18 個 check、每個 check 綁一個人想出來的失敗模式（列舉式）。
  bug_miner = 三層通用儀器，不列舉失敗模式：
    A) 逐決策健全不變量（sound invariants）：只用引擎事實判「這一步確定是錯的」，
       定義沿用退化審查波已數據驗證過的版本（diag_degen_audit），零 buff 假陽性：
         refuse／stall_move／blocked_move／heal_full（DD 四項）
         dead_action（引擎回 ERROR＝合法性遮罩漏網）
         immune_hit（選了 EV≈0 的攻擊、同回合有 EV 正常的攻擊合法＝浪費招在免疫敵）
         null_control（對 condition_immunities 免疫該狀態的敵施控場＝null 效果）
         no_engage／opp_no_engage（整場 0 傷害；後者＝對手席管線退化）
    B) 軟事件（EV 支配/目標選擇/誤傷）：不是「確定錯」但值得排名上報：
         avoid_easy_target（放著有 prone/restrained 的敵不打、改打同 EV 的健康敵）
         aoe_ally（AoE 半徑蓋到隊友）、idle_first（回合首手棄 action）
    C) 成對條件差分（同種子雙胞胎）：layout≠open ↔ open、注入免疫 ↔ 不注入、
       模型 ↔ 腳本專家（同席同種子）——不變量抓不到的「還在動但輸掉」型崩塌由此浮現。

分類問題的解（報告不預先歸類 bug 型別）：
  每個事件掛「機械 context 標籤」（layout/突變/敵我當下狀態位/隊形/席位），報告對每個
  偵測器算標籤 lift（P(標籤|事件)/P(標籤|全決策)）——高 lift 標籤＝這個 bug 的觸發條件，
  由數據自己說；沒見過的組合自動變成新的 (偵測器×高lift標籤) 列，不會被塞進舊分類。
  致命事件當場跑通道掃描（diag_info_channels 的逐通道歸零）：模型對哪條 obs 通道
  有反應/沒反應＝「讀資訊」軸的根因；immune_hit 另附 oracle 端依賴（EV 剝掉抗性
  後最佳招變不變＝正解本來就依賴 typed_resist 的證據）。
  簽名比對指紋庫（bug_miner_fingerprints.json）：對上＝KNOWN(名字)，對不上＝NEW。

用法:
  python scripts/bug_miner.py models/unified/uni_v9.pt --n 200 --seed 0 --out rep.json
  python scripts/bug_miner.py <m> --self_test --n 60     # V1：合成壞 agent 盲測（有牙齒證明）
  python scripts/bug_miner.py <m> --replay "mine|0|0042" --seed 0 --n 200   # 逐決策重現一場
"""
from __future__ import annotations
import sys, os, argparse, random, json, math
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
# 決定論護欄（2026-07-02 指紋獵殺定案）：多重程序並行時 MKL 動態執行緒會改變
# 浮點歸約順序→近平手 argmax 翻面→同種子軌跡分岔（0236 批量/隔離不一致懸案、
# 0101 手動重現對不上同因）。固定執行緒＋關動態＝同種子位元級可重現。
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")
torch.set_num_threads(1)

from trpg.scenarios.monsters import (MONSTER_DEFS, EQUIV_LEVEL_1V1,
                                     register_monsters, onev1_viable_monsters)
register_monsters()
import trpg.rl.env_v2 as _envmod
from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.action import encode_action, _entity_id_at_slot
from trpg.rl.obs import (RL_STATUS_NAMES, N_ARCHETYPES, ENEMY_SLOT_START,
                         N_GRID, GRID_CELL_SIZE_M)
from trpg.engine.skill import available_skills, TargetType, STATUS_SLOTS
from trpg.engine.combat import attack_range_check
from trpg.engine.damage import DAMAGE_TYPES
from trpg.engine.status import Prone, Restrained, Frightened
from trpg.engine.combat_policy import make_archetype_policy
from trpg.engine.vec2 import Vec2

from distill_routed import load_student
from eval_routed import stable_seed
from train_population import blind_np_single
from seed_switch_bc import damaging_options
import diag_info_channels as IC
# GUI 身分組裝＝與 play_gui 共用同一份（gui_identity），保證「GUI 湊得出的組合
# miner 就湊得出」是結構性成立、不是靠運氣（2026-07-02 用戶硬性要求）。
from gui_identity import (build_identity, identity_catalog, DMG_TYPES,
                          RESIST_MODES, BOOL_TRAITS)
from trpg.engine.abilities import ABILITY_REGISTRY

CLASSES = list(ARCHETYPE_LIST)
_STATUS_START = 7 + N_ARCHETYPES          # 實體列狀態多熱區塊起點（同 eval_gate）
_ADVANTAGE_STATUS = {"prone", "restrained", "paralyzed", "stunned", "blinded"}
FATAL = {"refuse", "stall_move", "blocked_move", "heal_full", "dead_action",
         "immune_hit", "null_control", "no_engage", "opp_no_engage"}
FPRINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "bug_miner_fingerprints.json")

# 身分池：12 職＋1v1 可用怪＋縫合怪（能扮任意身分支柱＝生成器該覆蓋全部）
_EQUIV = dict(EQUIV_LEVEL_1V1)
try:
    from chimera_defs import register_chimeras, CHIMERA_IDS
    register_chimeras(); CLS_CHIM = list(CHIMERA_IDS)
except Exception:
    CLS_CHIM = []
try:
    from chimera_monsters import register_chimera_monsters, CHIMERA_EQUIV_1V1
    register_chimera_monsters(); _EQUIV.update(CHIMERA_EQUIV_1V1)
    MON_CHIM = list(CHIMERA_EQUIV_1V1)
except Exception:
    MON_CHIM = []
# 身分池＝GUI 下拉選單同源（全部職業＋全部怪物，含 1vN 內容）＋縫合怪（超集）。
# 用同一份 identity_catalog()＝「GUI 湊得出的身分，miner 一定抽得到」不會漂移。
# 過強怪物在 1v1 不可贏的假陽性由 winnable/ever_attackable/專家仲裁三道守門兜底。
GUI_IDENTS = [i for (_, i, _, _) in identity_catalog()]
IDENTS = GUI_IDENTS + CLS_CHIM + MON_CHIM
GRAFT_SKILLS = sorted(ABILITY_REGISTRY.keys())    # 加掛用：整個能力登錄表（含 swallow 家族）
OPP_POOL = ["battle_master", "champion", "evocation", "vengeance",
            "orc", "ogre", "shadow", "mage_npc", "ghoul"]
LAYOUTS = ["open", "walls", "pillar", "difficult", "lava"]


# ── 情境 ─────────────────────────────────────────────────────────────────────

@dataclass
class Scenario:
    key: str
    agents: list
    opps: list
    lvl: int
    opp_lvl: int
    layout: str
    opp_kind: str            # script / passive / mirror
    mutation: str            # none / immune / enemy_prone / self_frightened / self_restrained
    pair_id: str = ""        # 成對差分：同 pair_id 的 main/twin 同種子
    pair_role: str = "main"  # main / twin_open / twin_noimmune / twin_expert
    tags: list = field(default_factory=list)


def _ident_level(rng, ident, dlvl):
    if ident in _EQUIV:
        eq = _EQUIV[ident]
        base = 8 if not math.isfinite(eq) else max(1, int(round(eq)))
    else:
        base = rng.randint(4, 8)
    return max(1, base + dlvl)


def _pick(rng, table):
    r = rng.random(); acc = 0.0
    for item, w in table:
        acc += w
        if r < acc:
            return item
    return table[-1][0]


def _sample_graft(rng, p=0.35):
    """GUI 身分加掛的隨機取樣（模型/agent 席）：隨機技能（含 swallow 家族等
    target-status 前置能力）＋抗性/免疫/易傷/布林天賦，全走 gui_identity.build_identity
    ＝與 GUI 完全同一組合空間。不硬寫任何能力名——均勻抽 registry 讓每種組合都會
    在隨機中出現（[[feedback_design_vigilance]]）。回 None 或 {'kwargs','tags'}。"""
    if rng.random() >= p:
        return None
    kwargs, tags = {}, []
    ks = []                                    # 1~2 個加掛技能
    while len(ks) < 2 and (not ks or rng.random() < 0.30):
        sid = rng.choice(GRAFT_SKILLS)
        if sid not in ks:
            ks.append(sid)
    if ks:
        kwargs["extra_skill_ids"] = ks
        tags += [f"graft_skill:{s}" for s in ks]
    if rng.random() < 0.40:                    # 抗性天賦（型×模式）
        mode = rng.choice([m for m, v in RESIST_MODES.items() if v is not None])
        kwargs["dmg_type"] = rng.choice(DMG_TYPES)
        kwargs["resist_mult"] = RESIST_MODES[mode]
        tags.append(f"graft_resist:{kwargs['dmg_type']}:{mode}")
    bt = [k for k in BOOL_TRAITS if rng.random() < 0.15]   # 布林天賦
    if bt:
        kwargs["bool_traits"] = bt
        tags += [f"graft_trait:{k}" for k in bt]
    return {"kwargs": kwargs, "tags": tags} if kwargs else None


def sample_scenarios(n, seed):
    """生成 n 個主情境＋自動掛雙胞胎（成對差分）。決定論：同 (n,seed) 同劇本。"""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        key = f"mine|{seed}|{i:04d}"
        base_ident = rng.choice(IDENTS)
        mutation = _pick(rng, [("none", 0.50), ("immune", 0.13),
                               ("resist", 0.05), ("vuln", 0.03),
                               ("enemy_prone", 0.09), ("self_frightened", 0.10),
                               ("self_restrained", 0.05), ("none", 0.05)])
        opp_kind = _pick(rng, [("script", 0.60), ("passive", 0.20), ("mirror", 0.20)])
        # 形狀＝雙側人數獨立抽樣（NvM 全覆蓋，含 1v3/3v1/2v2/3v3…）。07-03 部署
        # 評估實測：退化起始點在 3 敵——同一隻怪 1v1/1v2 正常接戰、1v3 站著發呆；
        # 舊三形狀表（1v1/1v2/2v1）上限 2 實體＝儀器全綠但「怪 vs 冒險隊」這個
        # DND 部署主形狀從未被生成過的結構性盲區。
        na = _pick(rng, [(1, 0.72), (2, 0.16), (3, 0.12)])
        no = _pick(rng, [(1, 0.72), (2, 0.16), (3, 0.12)])
        layout = _pick(rng, [("open", 0.40), ("walls", 0.22), ("pillar", 0.08),
                             ("difficult", 0.15), ("lava", 0.15)])
        dlvl = _pick(rng, [(0, 0.60), (-3, 0.20), (+3, 0.10), ("L20", 0.10)])
        opp = rng.choice(OPP_POOL)
        # enemy_prone 突變＝目標選擇儀器：雙同型被動敵、其一 prone（乾淨可比）
        if mutation == "enemy_prone":
            na, no, opp_kind = 1, 2, "passive"
        if dlvl == "L20":
            if base_ident not in CLASSES:
                dlvl = 0
            else:
                opp = rng.choice(["champion", "battle_master", "evocation"])
        # GUI 加掛（模型/agent 席）：等級用 base_ident 定（grafted 臨時 id 不在 _EQUIV），
        # 再經 build_identity 產出最終身分＝與 GUI resolve_identity 同一條路。
        graft = _sample_graft(rng)
        ident = base_ident
        graft_tags = []
        if graft is not None:
            ident = build_identity(base_ident, **graft["kwargs"])
            graft_tags = graft["tags"]
        # 隊友/敵隊逐槽位獨立抽＝混編隊伍（不硬寫任何隊友名，池＝完整身分目錄，
        # [[feedback_design_vigilance]]）。enemy_prone 是唯一的同型複製例外：
        # 目標選擇儀器需要「雙同型敵其一 prone」的乾淨可比前提。
        agents = [ident] + [rng.choice(IDENTS) for _ in range(na - 1)]
        if mutation == "enemy_prone":
            opps = [opp] * no
        else:
            xpool = (["champion", "battle_master", "evocation"]
                     if dlvl == "L20" else OPP_POOL)
            opps = [opp] + [rng.choice(xpool) for _ in range(no - 1)]
        shape = f"{na}v{no}"
        if dlvl == "L20":
            lvl = opp_lvl = 20
        else:
            lvl = _ident_level(rng, base_ident, dlvl)
            opp_lvl = (MONSTER_DEFS[opp].natural_level if opp in MONSTER_DEFS
                       else max(1, lvl - dlvl))   # dlvl<0＝我方劣勢
        tags = [f"layout:{layout}", f"opp:{opp_kind}", f"shape:{shape}",
                f"mutation:{mutation}", f"dlvl:{dlvl}"] + graft_tags
        sc = Scenario(key, agents, opps, lvl, opp_lvl, layout, opp_kind,
                      mutation, pair_id=key, tags=tags)
        out.append(sc)
        # 雙胞胎（同種子、單一維度翻轉）：不變量抓不到的崩塌走成對差分
        # 雙胞胎 key 必須跟主情境不同（事件/場次統計才不會混同一場），
        # 但引擎種子取自 pair_id＝同種子成對可比
        if layout != "open":
            tw = Scenario(key + "~open", agents, opps, lvl, opp_lvl, "open",
                          opp_kind, mutation, pair_id=key,
                          pair_role="twin_open",
                          tags=[t for t in tags if not t.startswith("layout")]
                               + ["layout:open"])
            out.append(tw)
        if mutation in ("immune", "resist", "vuln"):
            tw = Scenario(key + "~noimm", agents, opps, lvl, opp_lvl, layout,
                          opp_kind, "none", pair_id=key,
                          pair_role="twin_noimmune",
                          tags=[t for t in tags if not t.startswith("mutation")]
                               + ["mutation:none"])
            out.append(tw)
        # 腳本專家同席差分（std12 的生成式對應）：同情境、專家駕同一席
        if (ident in CLASSES and opp_kind == "script" and mutation == "none"
                and shape == "1v1" and layout == "open" and dlvl == 0
                and rng.random() < 0.5):
            tw = Scenario(key + "~exp", agents, opps, lvl, opp_lvl, layout,
                          "script", "none", pair_id=key,
                          pair_role="twin_expert", tags=list(tags))
            out.append(tw)
    return out


# ── 事件 ─────────────────────────────────────────────────────────────────────

@dataclass
class Event:
    detector: str
    fatal: bool
    scen_key: str
    ident: str
    turn: int
    tags: list
    detail: str = ""
    chan: dict | None = None     # 通道掃描：{channel: 模型動作翻不翻}
    oracle_dep: str = ""         # 正解端依賴（immune_hit 專用）
    lost: bool = False           # 該場最終輸/停滯（事後回填）
    actor: str = ""              # 實際席位 id（NvM 下 ident 可能重複）


def _model_driver(net):
    """回 (act, blind_ob, masked_skill_logits)。apply_resource_mask 走模組全域＝
    self_test 可 patch 成 pass-through 模擬「遮罩破掉」。"""
    def drv(env, obs, actor):
        ob = blind_np_single(obs)
        ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob.items()}
        with torch.no_grad():
            el, s, e, g = net(ot)
        s = apply_resource_mask(s, env.resources, env.ws, actor)
        e = apply_entity_mask(e, ot, env.ws, actor)
        act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
        return act, ob, s
    return drv


def _expert_driver(archs):
    """archs＝單一身分或與 agent 席逐位對齊的身分 list。NvM 形狀下每席各配
    自己身分的腳本（單策略駕全部席位會讓隊友席拿錯 kit 的專家＝仲裁失真）。"""
    if isinstance(archs, str):
        archs = [archs]
    pols = [make_archetype_policy(a) for a in archs]
    def drv(env, obs, actor):
        aids = list(env.agent_ids)
        i = aids.index(actor) if actor in aids else 0
        pol = pols[min(i, len(pols) - 1)]
        a = env.ws.characters[actor]
        dec = pol.decide(actor, a, env.ws, env.resources, env.ws.combat.round_number)
        if dec.action is None or getattr(dec, "fled", False):
            return [0, 0, 0], None, None
        return list(encode_action(dec.action, env.ws, actor)), None, None
    return drv


class _StationaryPolicy:
    def decide(self, opp_id, opp, ws, resources, round_number):
        from types import SimpleNamespace
        return SimpleNamespace(action=None, fled=False, ended=True)


def _status_tags(ob):
    """從 obs 讀敵我當下狀態位＝機械 context（模型看得到什麼就標什麼）。"""
    ent = ob["entities"]
    tags = []
    n_st = len(RL_STATUS_NAMES)
    for j, name in enumerate(RL_STATUS_NAMES):
        if ent[0, _STATUS_START + j] > 0.5:
            tags.append(f"self:{name}")
        if (ent[ENEMY_SLOT_START:, _STATUS_START + j] > 0.5).any():
            tags.append(f"enemy:{name}")
    return tags


def _channel_scan(net, ob, env, actor, a_on):
    """事件當下逐通道歸零→模型動作翻不翻（讀資訊軸的根因）。"""
    out = {}
    for name, st, ln in IC.CHANNELS:
        off = IC.zero_slice(ob, st, ln)
        if np.array_equal(ob["entities"], off["entities"]):
            continue                                  # 通道本來就 0＝無信號
        out[name] = (IC.greedy(net, off, env, actor) != tuple(a_on))
    ter = ob.get("terrain")
    if ter is not None and float(ter.max()) > 0:
        off = dict(ob); off["terrain"] = np.zeros_like(ter)
        out["terrain"] = (IC.greedy(net, off, env, actor) != tuple(a_on))
    ent = ob["entities"]
    if float(ent[0, _STATUS_START:_STATUS_START + len(RL_STATUS_NAMES)].max()) > 0:
        off = dict(ob); e2 = ent.copy()
        e2[0, _STATUS_START:_STATUS_START + len(RL_STATUS_NAMES)] = 0.0
        off["entities"] = e2
        out["self_status"] = (IC.greedy(net, off, env, actor) != tuple(a_on))
    return out


def _apply_mutation(env, scen):
    """回傳突變細節標籤。immune＝針對 agent 主 EV 傷害型注入（同 seed_switch 的 flip 注入）。"""
    ws = env.ws
    if scen.mutation == "immune":
        aid = env.agent_ids[0]; oid = env.opp_ids[0]
        opts = damaging_options(ws, aid, oid)
        if not opts:
            return "mutation:immune:n/a"
        dt = max(opts, key=lambda t: t[4])[3]
        best_before = max(t[4] for t in opts)
        for o in env.opp_ids:
            ws.characters[o].damage_multipliers[dt] = 0.0
        # 公平性守門：注入後若無有意義的替代傷害型（純單型 kit）＝遊戲不可贏，
        # 0 輸出會是「局不可贏」而非模型退化（1vN game-impossible 教訓同族）→撤回注入
        after = damaging_options(ws, aid, oid)
        best_after = max((t[4] for t in after), default=0.0)
        if best_after < max(2.0, 0.2 * best_before):
            for o in env.opp_ids:
                ws.characters[o].damage_multipliers.pop(dt, None)
            for j, t in enumerate(scen.tags):
                if t == "mutation:immune":
                    scen.tags[j] = "mutation:none"
            scen.tags.append("mutation:immune_skipped")
            return ""
        return f"mutation:immune:{dt}"
    if scen.mutation == "resist":
        # 分級抗性 0.5（#26 後半）：EV 減半仍可贏＝「照打才對」的情境——
        # v9 塌縮家族（resist→dodge）的直接觸發型態；×0 免疫走 immune 突變
        aid = env.agent_ids[0]; oid = env.opp_ids[0]
        opts = damaging_options(ws, aid, oid)
        if not opts:
            return "mutation:resist:n/a"
        dt = max(opts, key=lambda t: t[4])[3]
        for o in env.opp_ids:
            ws.characters[o].damage_multipliers[dt] = 0.5
        return f"mutation:resist:{dt}"
    if scen.mutation == "vuln":
        # 易傷 2.0（#26 後半的正號側）：任意欄+2 只該讓攻擊更好——run3 訓練中期
        # 曾長出「單獨外型正欄→消極」捷徑；消極由 no_engage/refuse/stall 不變量抓
        dt = DAMAGE_TYPES[stable_seed(scen.key) % len(DAMAGE_TYPES)]
        for o in env.opp_ids:
            ws.characters[o].damage_multipliers[dt] = 2.0
        return f"mutation:vuln:{dt}"
    if scen.mutation == "enemy_prone":
        ws.characters[env.opp_ids[0]].add_status(Prone())
    elif scen.mutation == "self_frightened":
        ws.characters[env.agent_ids[0]].add_status(Frightened())
    elif scen.mutation == "self_restrained":
        ws.characters[env.agent_ids[0]].add_status(Restrained())
    return ""


# 引擎 ERROR 捕捉（dead_action）：包住 execute_action，事件記到當前 ctx
_EXEC_ORIG = None
_CTX = {"sink": None, "scen": None, "turn": 0, "tags": (), "seat_of": {}}


def _exec_wrap(action, ws):
    res = _EXEC_ORIG(action, ws)
    sink = _CTX["sink"]
    if sink is not None and isinstance(res, dict) and res.get("type") == "ERROR":
        aid = (action.get("attacker") or action.get("caster")
               or action.get("character"))
        seat = _CTX["seat_of"].get(aid)
        if seat:                                   # 只記模型駕駛的席位
            sink.append(Event("dead_action", True, _CTX["scen"].key,
                              _CTX["scen"].agents[0], _CTX["turn"],
                              list(_CTX["tags"]) + [f"seat:{seat}"],
                              detail=f"[{seat}] {action.get('skill_id','?')}:"
                                     f"{str(res.get('message',''))[:40]}"))
    return res


def audited_episode(driver, scen, net=None, events=None, scan_budget=None,
                    dec_tags=None, verbose=False):
    """跑一場＋逐決策審計。回 outcome dict。events=None 時只記 outcome（雙胞胎）。"""
    seed = stable_seed(scen.pair_id or scen.key)   # 雙胞胎與主情境同種子＝成對可比
    random.seed(seed)
    env = CombatEnvV2(seed=seed ^ 0x5A5A5A, n_agents=len(scen.agents),
                      n_opps=len(scen.opps))
    if scen.opp_kind == "mirror" and net is not None:
        env.use_self_play_opponent(net, blind=True)     # 必須在 reset 前
    obs, _ = env.reset(agent_archs=list(scen.agents), opp_archs=list(scen.opps),
                       level=scen.lvl, opp_level=scen.opp_lvl, layout=scen.layout)
    if scen.opp_kind == "passive":
        for o in env.opp_ids:
            env._opp_policies[o] = _StationaryPolicy()
    mut_tag = _apply_mutation(env, scen)
    scen_tags = list(scen.tags) + ([mut_tag] if mut_tag else [])

    def _kit_can_damage(att_ids, def_ids):
        """kit 級可贏性（無視距離/資源，只看傷害型×敵抗性）：全 0 倍率＝天生不可贏局，
        0 輸出不算該席退化（no_engage／opp_no_engage 的健全性前提——兩席同一條）。"""
        for a_ in att_ids:
            ach = env.ws.characters[a_]
            if not ach.is_alive():
                continue
            for sk in available_skills(ach, env.ws):
                f = sk.features
                if getattr(f, "expected_damage", 0) <= 0:
                    continue
                pairs = list(f.iter_damage_types())
                if not pairs:
                    continue
                for o_ in def_ids:
                    oc = env.ws.characters[o_]
                    mult = sum(sh * float((oc.damage_multipliers or {})
                                          .get(tok, 1.0)) for tok, sh in pairs)
                    if mult > 0.05:
                        return True
        return False
    winnable = _kit_can_damage(env.agent_ids, env.opp_ids)
    # 對手席同款前提：agent 免疫掉對手全部傷害型時（0021：黯蝕免疫嫁接 vs
    # shadow），對手 0 輸出是引擎必然，不是管線退化。
    opp_winnable = _kit_can_damage(env.opp_ids, env.agent_ids)
    aids = list(env.agent_ids)
    ident_of = dict(zip(aids, scen.agents))
    opp_hp0 = {o: env.ws.characters[o].hp for o in env.opp_ids}
    team_hp0 = sum(env.ws.characters[a].hp for a in aids)
    _CTX["scen"] = scen
    _CTX["tags"] = scen_tags
    _CTX["sink"] = events
    _CTX["seat_of"] = {a: "agent" for a in aids}
    if scen.opp_kind == "mirror":
        _CTX["seat_of"].update({o: "opp" for o in env.opp_ids})

    turns = 0; steps = 0; done = False
    prev_owner = None
    had_offense = False
    chose_offense = False    # 整場是否「嘗試過」任何攻擊（打了全 miss ≠ 拒戰）
    offense_by: set = set()  # 逐席位攻擊嘗試（NvM：stall_move 健全性用）
    # 儀器健全性 #6：整場是否「曾有可攻擊時刻」——atk_legal（含距離遮罩後合法）
    # 或 reach+剩餘移動可及且行動在手。單輪速死（衝滿移動仍在 reach 外、下輪
    # 開始前被轟死）從頭到尾沒有可攻擊時刻＝匹配/速度事實，不是拒戰
    # （seed666 0067：ghoul 衝 9m 落在 2m、行動點無合法攻擊、被 L6 法師速殺）。
    ever_attackable = False
    opp_turns = 0        # 本模型「回合首手仍握行動＋移動＋有活敵」的回合數（見下方 #6）
    last_turn_round = -1
    ep_events = []

    def emit(detector, tags_extra, detail="", ob=None, act=None, actor=None):
        if events is None:
            return
        ev = Event(detector, detector in FATAL, scen.key,
                   ident_of.get(actor, scen.agents[0]), turns,
                   scen_tags + tags_extra, detail=detail, actor=actor or "")
        # 致命/目標選擇事件當場通道掃描（限額控成本）
        if (net is not None and ob is not None and scan_budget is not None
                and detector in FATAL | {"avoid_easy_target"}):
            bk = (detector, ev.ident)
            if scan_budget[bk] > 0:
                scan_budget[bk] -= 1
                try:
                    ev.chan = _channel_scan(net, ob, env, actor, act)
                except Exception:
                    ev.chan = None
        events.append(ev); ep_events.append(ev)
        if verbose:
            print(f"    ⚑ turn{turns:3d} [{detector}] {detail}  "
                  f"+tags={tags_extra}")

    while not done:
        actor = env.current_agent_id
        if actor not in aids:
            obs, _, term, trunc, _ = env.step([0, 0, 0])
            done = term or trunc
            continue
        ch = env.ws.characters[actor]
        sks = available_skills(ch, env.ws)
        first = actor != prev_owner
        prev_owner = actor
        mv_avail = env.resources.get("movement", 0.0)
        act_avail = env.resources.get("action", 0) > 0
        pos0 = ch.position
        try:
            reach = float(ch.get_weapon().range_normal)
        except Exception:
            reach = 1.5
        live = [env.ws.characters[o] for o in env.opp_ids
                if env.ws.characters[o].is_alive()]
        nd = min((ch.position.distance_to(e.position) for e in live), default=99.0)
        in_reach = nd <= reach + 0.05
        # 每回合首手（round 變了）且仍握行動＋移動＋有活敵＝一次「可交戰機會」。
        # 累積機會回合＝多回合能縮短距離的量度（單回合 reach 抓不到的龜縮退化）。
        cur_round = env.ws.combat.round_number
        if cur_round != last_turn_round:
            last_turn_round = cur_round
            if act_avail and mv_avail >= 1.0 and live:
                opp_turns += 1
        _CTX["turn"] = turns

        act, ob, s_masked = driver(env, obs, actor)
        dtags = _status_tags(ob) + (["in_reach"] if in_reach else ["out_reach"]) \
            if ob is not None else []
        if dec_tags is not None and ob is not None:
            dec_tags.update(scen_tags + dtags)
            dec_tags["__total__"] += 1

        # 引擎真值 EV 選項（damaging_options＝切招波驗證過的 oracle 基座）。
        # 逐目標「當下合法性」過濾（07-03 NvM 波）：immune_hit 的聲明是
        # 「同回合有 EV 正常的攻擊**合法**」，但 damaging_options 的 skm 是
        # per-skill（最近敵）遮罩、EV 逐目標不驗距——2v2 裡唯一搆得到的敵
        # 是免疫者時，reach 外的高 EV 目標被當成「合法替代」＝假陽性
        # （mine|666|0032/0150）。同 apply_entity_mask 的引擎同源檢查。
        def _tgt_legal_now(sk_, oid_):
            tch_ = env.ws.characters[oid_]
            bf_ = env.ws.combat.battlefield if env.ws.combat else None
            if bf_ is not None and not bf_.has_line_of_sight(
                    ch.position, tch_.position):
                return False
            try:
                a_ = sk_.build_action(actor, oid_,
                                      (tch_.position.x, tch_.position.y))
            except Exception:
                return False
            if not isinstance(a_, dict):
                return False
            if a_.get("type") == "ATTACK":
                w_ = ch.get_weapon(a_.get("weapon", ""))
                ok_, _, _ = attack_range_check(ch, tch_, w_, bf_)
                if not ok_:
                    return False
                rts_ = a_.get("requires_target_status")
                if rts_ and not tch_.has_status(rts_):
                    return False
                bts_ = a_.get("blocked_by_target_status")
                if bts_ and tch_.has_status(bts_):
                    return False
            elif "range_m" in a_:
                if (ch.position.distance_to(tch_.position)
                        > float(a_["range_m"]) + 1e-6):
                    return False
            return True
        opts_all = []
        atk_legal = False
        if ob is not None and s_masked is not None:
            skm = [bool(s_masked[0, i].item() <= -1e8)
                   for i in range(s_masked.shape[-1])]
            for o in env.opp_ids:
                if env.ws.characters[o].is_alive():
                    try:
                        for t in damaging_options(env.ws, actor, o, skm):
                            if _tgt_legal_now(t[1], o):
                                opts_all.append(t + (o,))
                    except Exception:
                        pass
            for i in range(min(len(sks), len(skm))):
                if not skm[i] and i > 0 and sks[i].features.target_type in (
                        TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY):
                    atk_legal = True
        # #6：可攻擊時刻＝攻擊當下合法，或單回合走得到，或**累積 ≥3 個機會回合**
        # （手握行動＋移動＋有活敵）。第三條是關鍵：模型龜在遠處、對手又不逼近，
        # 明明多走幾回合就搆得到卻選擇不縮短距離＝拒戰真陽性（被動對手退化，GUI
        # 抽查命中的正是這型）；舊版單回合 reach 把它壓成「搆不到」假陰性。真的搆不到
        # （速度被秒＝機會回合 <3、對手風箏比你快＝機會回合多但專家也追不到）分別由
        # 門檻與專家仲裁兜底，不會誤傷。
        if atk_legal or (act_avail and nd <= reach + mv_avail) or opp_turns >= 3:
            ever_attackable = True
        best_ev = max((t[4] for t in opts_all), default=0.0)
        # 當下攻擊技存在性（資源感知、range 無關）：available_skills 已過濾
        # 沒位的法術/用盡的能力，但武器不受距離影響仍列出——所以此旗＝
        # 「走過去就有輸出」為真、「法術耗盡永久零輸出」為假（不可贏局）。
        offense_now = any(
            s_.skill_id != "move" and s_.features.target_type in (
                TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY,
                TargetType.POINT, TargetType.LINE, TargetType.CONE)
            for s_ in sks[1:])
        had_offense = had_offense or offense_now

        chosen_sk = sks[act[0]] if 0 < act[0] < len(sks) else None
        is_move = chosen_sk is not None and chosen_sk.skill_id == "move"
        if chosen_sk is not None and chosen_sk.skill_id != "move":
            tt_ = chosen_sk.features.target_type
            if (tt_ in (TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY,
                        TargetType.POINT, TargetType.LINE, TargetType.CONE)
                    or getattr(chosen_sk.features, "expected_damage", 0) > 0):
                chose_offense = True
                offense_by.add(actor)

        if events is not None and ob is not None:
            # A: refuse（DD 驗證過的定義：敵在 reach 內+攻擊合法+action 在手，卻棄回合）
            if act[0] == 0 and act_avail and in_reach and atk_legal:
                emit("refuse", dtags, f"END while atk_legal nd={nd:.1f}",
                     ob=ob, act=act, actor=actor)
            elif first and act[0] == 0 and act_avail and live:
                emit("idle_first", dtags, "first sub-action END w/ action",
                     ob=ob, act=act, actor=actor)
            # A: heal_full（DD 定義）
            if chosen_sk is not None and \
                    getattr(chosen_sk.features, "expected_healing", 0) > 0:
                allies = [ch] + [env.ws.characters[a] for a in aids
                                 if a != actor and env.ws.characters[a].is_alive()]
                if all(a.hp >= a.max_hp * 0.95 for a in allies):
                    emit("heal_full", dtags, chosen_sk.skill_id,
                         ob=ob, act=act, actor=actor)
            # A: immune_hit（EV≈0 的攻擊、有 EV 正常替代）＋ oracle 端依賴
            if chosen_sk is not None and opts_all and best_ev >= 2.0:
                mine = [t for t in opts_all if t[0] == act[0]]
                if mine:
                    tgt = _entity_id_at_slot(env.ws, actor, act[1])
                    exact = [t for t in mine if t[5] == tgt]
                    c_ev = (exact[0][4] if exact
                            else max(t[4] for t in mine))
                    if c_ev <= 0.05 * best_ev:
                        # oracle 端依賴：把抗性剝掉（EV=raw expected_damage）後
                        # 最佳招是否改變＝正解本來就依賴 typed_resist 的機械證據
                        raw_best = max(opts_all,
                                       key=lambda t: float(
                                           t[1].features.expected_damage))[0]
                        ev_best = max(opts_all, key=lambda t: t[4])[0]
                        emit("immune_hit", dtags,
                             f"{chosen_sk.skill_id} ev={c_ev:.1f} "
                             f"best={best_ev:.1f}",
                             ob=ob, act=act, actor=actor)
                        if ep_events and ep_events[-1].detector == "immune_hit":
                            ep_events[-1].oracle_dep = (
                                "typed_resist" if raw_best != ev_best else "")
            # A: null_control（施加狀態到 condition_immunities 免疫的目標）
            if (chosen_sk is not None and chosen_sk.features.target_type
                    == TargetType.SINGLE_ENEMY
                    and any(chosen_sk.features.applies_status)
                    and getattr(chosen_sk.features, "expected_damage", 0) <= 0):
                tgt = _entity_id_at_slot(env.ws, actor, act[1])
                tch = env.ws.characters.get(tgt)
                if tch is not None:
                    applied = {STATUS_SLOTS[j] for j, b in
                               enumerate(chosen_sk.features.applies_status) if b}
                    imm = set(getattr(tch, "condition_immunities", ()) or ())
                    if applied and applied <= imm:
                        emit("null_control", dtags,
                             f"{chosen_sk.skill_id}→{tgt} imm={sorted(applied)}",
                             ob=ob, act=act, actor=actor)
            # B: avoid_easy_target（有優勢狀態敵可打、卻打同 EV 的健康敵）。
            # 優勢敵用 obs 位元判定（模型看得到的資訊＝跟牠的視角一致）
            if chosen_sk is not None and chosen_sk.features.target_type in (
                    TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY) and opts_all:
                ent_ = ob["entities"]
                adv_idx = [RL_STATUS_NAMES.index(nm) for nm in _ADVANTAGE_STATUS
                           if nm in RL_STATUS_NAMES]
                adv_ids = []
                for r in range(ENEMY_SLOT_START, ent_.shape[0]):
                    if ent_[r, 5] > 0.5 and any(
                            ent_[r, _STATUS_START + j] > 0.5 for j in adv_idx):
                        cid = _entity_id_at_slot(env.ws, actor, r)
                        if cid is not None:
                            adv_ids.append(cid)
                if adv_ids:
                    tgt = _entity_id_at_slot(env.ws, actor, act[1])
                    if tgt not in adv_ids:
                        c = [t for t in opts_all if t[0] == act[0] and t[5] == tgt]
                        a_opts = [t for t in opts_all if t[5] in adv_ids]
                        if c and a_opts and max(t[4] for t in a_opts) >= c[0][4] - 1e-6:
                            emit("avoid_easy_target", dtags,
                                 f"hit {tgt} not {adv_ids[0]}",
                                 ob=ob, act=act, actor=actor)
            # B: aoe_ally（AoE 半徑蓋隊友；同 eval_gate teamff 幾何）
            if chosen_sk is not None and chosen_sk.skill_id != "move":
                rad = float(getattr(chosen_sk.features, "aoe_radius_m", 0.0) or 0.0)
                if rad > 0 and getattr(chosen_sk.features, "expected_damage", 0) > 0:
                    gx = act[2] // N_GRID; gy = act[2] % N_GRID
                    tx = (gx + 0.5) * GRID_CELL_SIZE_M
                    ty = (gy + 0.5) * GRID_CELL_SIZE_M
                    for al in aids:
                        if al != actor and env.ws.characters[al].is_alive():
                            p = env.ws.characters[al].position
                            if ((p.x - tx) ** 2 + (p.y - ty) ** 2) ** 0.5 <= rad:
                                emit("aoe_ally", dtags, chosen_sk.skill_id,
                                     ob=ob, act=act, actor=actor)
                                break

        turns += 1
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
        # A: 空轉移動（DD 定義：位移<0.1 且能動；stall=reach 外+action 在手）。
        # offense_now 門（2026-07-02 儀器健全性 #4）：法術耗盡+無武器＝永久零
        # 輸出＝不可贏局，繞圈不算退化（mine|3|0009 evocation 假陽性家族）
        if events is not None and is_move:
            disp = ch.position.distance_to(pos0)
            if disp < 0.1 and mv_avail >= 1.0:
                if act_avail and not in_reach and offense_now:
                    emit("stall_move", dtags, f"nd={nd:.1f}", actor=actor)
                elif act[2] != (max(0, min(N_GRID - 1, int(pos0.x / GRID_CELL_SIZE_M)))
                                * N_GRID
                                + max(0, min(N_GRID - 1, int(pos0.y / GRID_CELL_SIZE_M)))):
                    emit("blocked_move", dtags, f"nd={nd:.1f}", actor=actor)
        steps += 1
        if steps > 400:
            break

    _CTX["sink"] = None
    team_alive = any(env.ws.characters[a].is_alive() for a in aids)
    opp_alive = any(env.ws.characters[o].is_alive() for o in env.opp_ids)
    dmg = sum(opp_hp0[o] - max(0, env.ws.characters[o].hp) for o in env.opp_ids)
    taken = team_hp0 - sum(max(0, env.ws.characters[a].hp) for a in aids)
    win = team_alive and not opp_alive
    loss = not team_alive
    out = dict(win=win, loss=loss, draw=(not win and not loss),
               dmg=dmg, taken=taken, turns=turns)
    # 場級不變量
    if events is not None:
        # no_engage 健全性（2026-07-02 假陽性修正）：dmg=0 分不開「從不攻擊」與
        # 「攻了全 miss」（實測 L1 wolf vs BM mirror：兩次咬擊全 miss 落敗＝匹配
        # 非退化）。只有整場「從未嘗試任何攻擊」才是退化。
        if (had_offense and turns > 2 and dmg <= 0 and winnable
                and not chose_offense and ever_attackable):
            ev = Event("no_engage", True, scen.key, scen.agents[0], turns,
                       scen_tags, detail=f"turns={turns} 0攻擊嘗試")
            events.append(ev); ep_events.append(ev)
        # 已知弱點：對手席由 env 內部驅動、看不到逐決策動作，taken=0 無法區分
        # 「從不攻擊」與「攻了全 miss」——比 no_engage 弱一級，簽名判讀時須 replay。
        if (scen.opp_kind == "mirror" and turns > 4 and taken <= 0
                and opp_alive and opp_winnable):
            ev = Event("opp_no_engage", True, scen.key, scen.agents[0], turns,
                       scen_tags, detail=f"對手席(同網) {turns} 回合 0 輸出(弱定義)")
            events.append(ev); ep_events.append(ev)
    # stall_move 健全性（07-03 NvM 波實測兩個假陽性家族，DD 定義在 1v1 驗證時
    # 沒有的形態）：①召喚/光環流站樁（war 牧師靈體武器+守衛光環 148-0 獲勝，
    # 站著不動＝正確打法，null move 是免費子動作裡的廢步、行動照樣花在施法）
    # ——該席整場有攻擊嘗試＝降級；②瞬態（同席 <3 次、勝局裡開場猶豫 1-2 步）
    # ——降級。真停車者（零嘗試＋≥3 次原地）保持致命。
    if events is not None and ep_events:
        n_stall = Counter(ev.actor for ev in ep_events
                          if ev.detector == "stall_move")
        for ev in ep_events:
            if ev.detector != "stall_move" or not ev.fatal:
                continue
            if ev.actor in offense_by:
                ev.fatal = False
                ev.detector = "stall_move(with_output)"
            elif n_stall.get(ev.actor, 0) < 3:
                ev.fatal = False
                ev.detector = "stall_move(transient)"
    bad = loss or (out["draw"] and dmg < taken)
    for ev in ep_events:
        ev.lost = bad
    return out


# ── 聚合與報告 ────────────────────────────────────────────────────────────────

def _tag_lift(events, dec_tags):
    """對每個偵測器：各 context 標籤的 lift = P(tag|事件)/P(tag|全決策)。"""
    total = max(1, dec_tags.get("__total__", 1))
    base = {t: c / total for t, c in dec_tags.items() if t != "__total__"}
    per_det = defaultdict(list)
    for ev in events:
        per_det[ev.detector].append(ev)
    out = {}
    for det, evs in per_det.items():
        cnt = Counter(t for ev in evs for t in set(ev.tags))
        lifts = []
        for t, c in cnt.items():
            cov = c / len(evs)
            b = base.get(t, 0.0)
            if b <= 0 or cov < 0.3:
                continue
            lifts.append((t, cov / b, cov))
        lifts.sort(key=lambda x: -x[1])
        out[det] = lifts
    return out


def _chan_verdicts(evs):
    """聚合事件的通道掃描：每通道 模型翻轉率（n）＋ oracle 依賴計數。"""
    agg = defaultdict(lambda: [0, 0])
    for ev in evs:
        if ev.chan:
            for name, flip in ev.chan.items():
                agg[name][1] += 1
                agg[name][0] += int(flip)
    dep = Counter(ev.oracle_dep for ev in evs if ev.oracle_dep)
    return {k: (v[0], v[1]) for k, v in agg.items()}, dep


def load_fingerprints():
    if os.path.exists(FPRINT_PATH):
        with open(FPRINT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def match_fingerprint(fps, det, top_tags):
    for fp in fps:
        if fp["detector"] != det:
            continue
        t = fp.get("tag")
        if not t or any(t in tag for tag in top_tags):
            return fp["name"]
    return None


@dataclass
class Report:
    n_scen: int = 0
    n_dec: int = 0
    events: list = field(default_factory=list)
    outcomes: dict = field(default_factory=dict)      # scen_key(pair_role) → outcome
    scens: dict = field(default_factory=dict)
    dec_tags: Counter = field(default_factory=Counter)
    det_rows: list = field(default_factory=list)      # 印出的簽名列（機械）
    pair_rows: list = field(default_factory=list)
    expert_rows: list = field(default_factory=list)

    def det_count(self, name):
        return sum(1 for ev in self.events if ev.detector == name)

    def fatal_top_tag(self):
        f = [ev for ev in self.events if ev.fatal]
        if not f:
            return None
        tl = _tag_lift(f, self.dec_tags)
        best = None
        for det, lifts in tl.items():
            for t, lift, cov in lifts[:3]:
                if best is None or lift > best[1]:
                    best = (t, lift)
        return best


def mine(net, n_scen, seed, scan_cap=4, verbose=False, progress=True):
    global _EXEC_ORIG
    rep = Report(n_scen=n_scen)
    scens = sample_scenarios(n_scen, seed)
    rep.scens = {(s.key, s.pair_role): s for s in scens}
    scan_budget = defaultdict(lambda: scan_cap)
    if _EXEC_ORIG is None:
        _EXEC_ORIG = _envmod.execute_action
    _envmod.execute_action = _exec_wrap
    try:
        for k, sc in enumerate(scens):
            if progress and k % 25 == 0:
                print(f"  … {k}/{len(scens)} 情境（事件 {len(rep.events)}）",
                      flush=True)
            if sc.pair_role == "twin_expert":
                drv = _expert_driver(sc.agents)
                out = audited_episode(drv, sc, net=None, events=None)
            else:
                drv = _model_driver(net)
                out = audited_episode(drv, sc, net=net, events=rep.events,
                                      scan_budget=scan_budget,
                                      dec_tags=rep.dec_tags, verbose=verbose)
            rep.outcomes[(sc.key, sc.pair_role)] = out
    finally:
        _envmod.execute_action = _EXEC_ORIG
        _CTX["sink"] = None
    rep.n_dec = rep.dec_tags.get("__total__", 0)
    _expert_arbitration(rep, scens)
    _aggregate(rep)
    return rep


# 儀器健全性 #5（2026-07-02）：接戰類致命事件的專家基準仲裁。
# no_engage/stall_move 的「可贏性」前提（kit 傷害型×倍率）看不見反應經濟
# （mm 全被 shield 吃）、風箏距離（雙施法者 vs 慢速怪）、永久定身等實戰不可贏
# ——判準交給引擎事實：同身分腳本專家、同種子同席跑一場，專家也零輸出＝
# 這局沒有可責怪的接戰路徑（0082 雙法師風箏／0116 shield 吃 mm／0141 定身
# 全數專家同零）→ 降級為情境事件（expert_parity）。專家打得出傷害而模型
# 零嘗試＝真拒戰，維持致命（0087 專家 14 傷保留）。
_ARBITRATED = ("no_engage", "stall_move", "blocked_move")


def _expert_arbitration(rep, scens):
    keys = {ev.scen_key for ev in rep.events
            if ev.fatal and ev.detector in _ARBITRATED}
    if not keys:
        return
    by_key = {s.key: s for s in scens}
    parity = set()
    for k in sorted(keys):
        sc = by_key.get(k)
        if sc is None:
            continue
        try:
            out = audited_episode(_expert_driver(sc.agents), sc,
                                  net=None, events=None)
        except Exception:
            continue                       # 專家跑不了＝不仲裁、維持致命
        if out["dmg"] <= 0:
            parity.add(k)
    for ev in rep.events:
        if (ev.scen_key in parity and ev.detector in _ARBITRATED
                and ev.fatal):
            ev.fatal = False
            ev.detector = f"{ev.detector}(expert_parity)"
            ev.tags.append("expert_parity")


def _aggregate(rep):
    fps = load_fingerprints()
    tl = _tag_lift(rep.events, rep.dec_tags)
    per_det = defaultdict(list)
    for ev in rep.events:
        per_det[ev.detector].append(ev)
    order = sorted(per_det, key=lambda d: (d not in FATAL, -len(per_det[d])))
    for det in order:
        evs = per_det[det]
        games = {ev.scen_key for ev in evs}
        lost = sum(1 for k in games
                   if any(ev.lost for ev in evs if ev.scen_key == k))
        chans, dep = _chan_verdicts(evs)
        lifts = tl.get(det, [])[:4]
        top_tags = [t for t, _, _ in lifts]
        known = match_fingerprint(fps, det, top_tags)
        rep.det_rows.append(dict(
            detector=det, fatal=det in FATAL, n=len(evs), games=len(games),
            lost_in_affected=lost / max(1, len(games)),
            lifts=[(t, round(l, 1), round(c, 2)) for t, l, c in lifts],
            chans={k: f"{a}/{b}" for k, (a, b) in sorted(chans.items())
                   if b > 0},
            oracle_dep=dict(dep),
            known=known,
            examples=[f"{ev.scen_key}#t{ev.turn}" for ev in evs[:3]],
        ))
    # 成對差分（twin 的 key 帶後綴；配對靠 pair_id）
    pair_acc = defaultdict(lambda: [0, 0, 0])   # (dim, ident) → [main_w, twin_w, n]
    for (key, role), out in rep.outcomes.items():
        if role == "main":
            continue
        sc = rep.scens.get((key, role))
        main = rep.outcomes.get((sc.pair_id, "main")) if sc else None
        if main is None:
            continue
        msc = rep.scens[(sc.pair_id, "main")]
        if role == "twin_expert":
            rep.expert_rows.append((msc.agents[0], int(main["win"]),
                                    int(out["win"]), sc.pair_id))
            continue
        dim = ("layout:" + msc.layout if role == "twin_open"
               else f"mutation:{msc.mutation}")
        acc = pair_acc[(dim, msc.agents[0])]
        acc[0] += int(main["win"]); acc[1] += int(out["win"]); acc[2] += 1
    for (dim, ident), (mw, tw, n) in sorted(pair_acc.items()):
        rep.pair_rows.append(dict(dim=dim, ident=ident, n=n,
                                  main_wr=mw / n, twin_wr=tw / n,
                                  drop=(mw - tw) / n))   # 條件開啟(main)−關閉(twin)


def print_report(rep, model_name=""):
    print("\n" + "=" * 78)
    print(f"生成式偵錯報告  model={model_name}  情境={rep.n_scen}  "
          f"決策={rep.n_dec}  事件={len(rep.events)}")
    print("=" * 78)
    fatal_rows = [r for r in rep.det_rows if r["fatal"]]
    soft_rows = [r for r in rep.det_rows if not r["fatal"]]
    if not fatal_rows:
        print("\n-- 致命不變量：0 事件（此劇本分布下無確定性錯誤決策）--")
    else:
        print("\n-- 致命不變量事件（每一筆都是引擎事實判定的確定錯誤）--")
    for r in fatal_rows + soft_rows:
        tagstr = "  ".join(f"{t}×{l}({c:.0%})" for t, l, c in r["lifts"])
        known = f"  ⟶ KNOWN[{r['known']}]" if r["known"] else "  ⟶ NEW"
        sev = "FATAL" if r["fatal"] else "soft "
        print(f"\n[{sev}] {r['detector']:18s} n={r['n']:<4d} 場={r['games']:<3d} "
              f"敗率(受影響場)={r['lost_in_affected']:.0%}{known}")
        if tagstr:
            print(f"        觸發條件(lift): {tagstr}")
        if r["chans"]:
            inert = [k for k, v in r["chans"].items()
                     if v.startswith("0/") and int(v.split("/")[1]) >= 3]
            live = [f"{k}={v}" for k, v in r["chans"].items()
                    if not v.startswith("0/")]
            if inert:
                print(f"        通道掃描: 帶值不讀 → {', '.join(inert)}")
            if live:
                print(f"        通道掃描: 有反應 → {', '.join(live)}")
        if r["oracle_dep"]:
            print(f"        oracle 端依賴: {r['oracle_dep']}")
        print(f"        例: {'  '.join(r['examples'])}   "
              f"(--replay \"{r['examples'][0].split('#')[0]}\")")
    shown = [pr for pr in rep.pair_rows if pr["n"] >= 2 or pr["drop"] != 0]
    if shown:
        print("\n-- 成對條件差分（同種子雙胞胎；Δ=條件開啟後 WR 變化；n=1 全平的列不印）--")
        for pr in shown:
            flag = "  ⚠崩" if (pr["n"] >= 3 and pr["drop"] <= -0.5) else ""
            print(f"   {pr['dim']:18s} {pr['ident']:16s} n={pr['n']:<2d} "
                  f"WR {pr['twin_wr']:.0%}→{pr['main_wr']:.0%} "
                  f"Δ={pr['drop']*100:+.0f}pp{flag}")
    if rep.expert_rows:
        agg = defaultdict(lambda: [0, 0, 0])
        for ident, mw, ew, _ in rep.expert_rows:
            agg[ident][0] += mw; agg[ident][1] += ew; agg[ident][2] += 1
        print("\n-- 腳本專家同席差分（同情境同種子；模型 vs 專家勝場）--")
        for ident, (mw, ew, n) in sorted(agg.items()):
            flag = "  ⚠低於專家" if ew - mw >= 2 else ""
            print(f"   {ident:16s} n={n:<2d} model={mw} expert={ew}{flag}")
    print()


def save_json(rep, path, model_name):
    data = dict(model=model_name, n_scen=rep.n_scen, n_dec=rep.n_dec,
                det_rows=rep.det_rows, pair_rows=rep.pair_rows,
                expert_rows=rep.expert_rows,
                events=[asdict(ev) for ev in rep.events])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"(json → {path})")


# ── V1 自測：合成壞 agent 盲測（miner 不被告知 bug、必須自己挖出正確簽名） ─────────

def _st_always_end(rep):
    return rep.det_count("refuse") + rep.det_count("no_engage") >= 5


def _st_frightened(rep):
    evs = [ev for ev in rep.events if ev.fatal]
    with_fr = [ev for ev in evs if any("frightened" in t for t in ev.tags)]
    return len(with_fr) >= 2 and len(with_fr) / max(1, len(evs)) >= 0.6


def _st_avoid_prone(rep):
    return rep.det_count("avoid_easy_target") >= 2


def _st_walls(rep):
    evs = [ev for ev in rep.events if ev.fatal]
    on_wall = [ev for ev in evs
               if any(t in ("layout:walls", "layout:pillar") for t in ev.tags)]
    pair_crash = any(pr["dim"].startswith("layout") and pr["n"] >= 2
                     and pr["drop"] <= -0.5 for pr in rep.pair_rows)
    return (len(on_wall) >= 3 and len(on_wall) / max(1, len(evs)) >= 0.6) \
        or pair_crash


def _st_aoe_ally(rep):
    # 2026-07-03 語義升級（同 eval_gate teamff 的 guard_holds）：pick_action 的
    # AoE 支配格重選守門把「強制炸隊友格」在決策層結構性擋掉（守門單元證明＝
    # tests/rl/test_model.py 的 aoe relocate/scrum 雙測），端到端 ≈0 事件＝守門
    # 有效而非偵測器瞎；守門失守時事件湧現→本測翻紅。偵測器自身牙齒由 07-01
    # V1 盲測（守門落地前）背書；真模型在無乾淨落點的貼身團戰仍可觸發此事件。
    return rep.det_count("aoe_ally") <= 2


def _st_broken_mask(rep):
    return rep.det_count("dead_action") >= 3


SELF_TEST = {
    "always_end": ("refuse/no_engage 大量觸發", _st_always_end),
    "freeze_when_frightened": ("致命事件集中在 frightened 標籤", _st_frightened),
    "avoid_prone": ("avoid_easy_target 觸發", _st_avoid_prone),
    "freeze_on_walls": ("致命事件集中牆面/成對差分崩", _st_walls),
    "aoe_on_ally": ("AoE 支配格守門擋住強制亂炸（aoe_ally≈0＝守門有效）", _st_aoe_ally),
}


def run_self_test(base_net, n_scen, seed):
    from eval_gate import _EvilNet
    print("=" * 78)
    print("MINER SELF-TEST：合成壞 agent 盲測——miner 不知道植入了什麼，必須自己把")
    print("正確的機械簽名挖出來（生成式版的 test-the-tests）")
    print("=" * 78)
    ok_all = True
    for mode, (expect, pred) in SELF_TEST.items():
        print(f"\n== [{mode}] 期望：{expect}", flush=True)
        evil = _EvilNet(base_net, mode)
        rep = mine(evil, n_scen, seed, progress=False)
        ok = pred(rep)
        ok_all &= ok
        top = sorted(rep.det_rows, key=lambda r: (not r["fatal"], -r["n"]))[:3]
        brief = "; ".join(f"{r['detector']}×{r['n']}"
                          f"[{','.join(t for t, _, _ in r['lifts'][:2])}]"
                          for r in top) or "(無事件)"
        print(f"   {'✓ 挖到' if ok else '✗ 沒挖到!!'}  top簽名: {brief}")
    # broken_mask：patch 本模組的 apply_resource_mask＝遮罩破掉→dead_action 該出現
    print(f"\n== [broken_mask] 期望：dead_action 觸發", flush=True)
    global apply_resource_mask
    orig = apply_resource_mask
    try:
        apply_resource_mask = lambda s, *a, **k: s
        rep = mine(base_net, max(20, n_scen // 3), seed, progress=False)
        ok = _st_broken_mask(rep)
        ok_all &= ok
        print(f"   {'✓ 挖到' if ok else '✗ 沒挖到!!'}  "
              f"dead_action×{rep.det_count('dead_action')}")
    finally:
        apply_resource_mask = orig
    print("\n" + "=" * 78)
    print("SELF-TEST 通過：全部合成 bug 都被盲挖出來" if ok_all
          else "SELF-TEST 未過：有合成 bug 沒被挖出（見 ✗）——miner 還沒資格取代 eval_gate")
    print("=" * 78)
    return ok_all


def replay(net, key, n_scen, seed):
    scens = sample_scenarios(n_scen, seed)
    sc = next((s for s in scens if s.key == key and s.pair_role == "main"), None)
    if sc is None:
        print(f"!! 找不到情境 {key}（--n/--seed 必須跟產生報告那次相同）"); return
    print(f"重現 {key}: {sc.agents} vs {sc.opps} L{sc.lvl}/{sc.opp_lvl} "
          f"{sc.layout} opp={sc.opp_kind} mut={sc.mutation}")
    events = []
    global _EXEC_ORIG
    if _EXEC_ORIG is None:
        _EXEC_ORIG = _envmod.execute_action
    _envmod.execute_action = _exec_wrap
    try:
        out = audited_episode(_model_driver(net), sc, net=net, events=events,
                              scan_budget=defaultdict(lambda: 99),
                              dec_tags=Counter(), verbose=True)
    finally:
        _envmod.execute_action = _EXEC_ORIG
    print(f"結局: {out}")
    for ev in events:
        print(f"  [{ev.detector}@t{ev.turn}] {ev.detail}  "
              f"tags={[t for t in ev.tags if ':' in t][:6]}")
        if ev.chan:
            inert = [k for k, v in ev.chan.items() if not v]
            live = [k for k, v in ev.chan.items() if v]
            print(f"      通道: 翻={live} 不翻={inert}"
                  + (f"  oracle_dep={ev.oracle_dep}" if ev.oracle_dep else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=200, help="主情境數（雙胞胎自動另加）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scan_cap", type=int, default=4,
                    help="每 (偵測器×身分) 的通道掃描次數上限")
    ap.add_argument("--out", default=None)
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--replay", default=None, help="重現單一情境（逐決策 verbose）")
    args = ap.parse_args()

    net = load_student(args.ckpt); net.eval()
    if args.self_test:
        ok = run_self_test(net, args.n, args.seed)
        sys.exit(0 if ok else 1)
    if args.replay:
        replay(net, args.replay, args.n, args.seed)
        return
    rep = mine(net, args.n, args.seed, scan_cap=args.scan_cap)
    print_report(rep, os.path.basename(args.ckpt))
    if args.out:
        save_json(rep, args.out, os.path.basename(args.ckpt))
    sys.exit(1 if any(r["fatal"] for r in rep.det_rows) else 0)


if __name__ == "__main__":
    main()
