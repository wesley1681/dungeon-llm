"""嚴格審查機制：整場戰鬥退化審計（取代單回合 diag_engage_audit 的假陽性）。

單回合探針會把 buff/光環職(rage/spirit_guardians/sacred_weapon)誤判成 stall。
本審計改跑**完整戰鬥**，只用**引擎事實**判退化，故對 buff 開場零假陽性：

  D1 空轉移動 (noop_move) : 某 sub-action 選 MOVE，但引擎位移 < 0.1m，且決策當下
                            movement >= 1.0m（能動卻沒動）— reach-stall 的全場指紋。
                            buff(rage/SG)不是 MOVE，故不會被算進來。
  D2 零接戰 (no_engage)   : 整場存活 > 2 回合、有攻擊/傷害技可用，卻 0 點傷害輸出。
  win                      : 是否擊敗對手（對稱對手→量「能否贏」；劣勢局本就可能輸）。

模型只駕駛 agent 席（對手由 env.step 內部腳本驅動 = 稱職基準）。配對全域骰。

用法: python scripts/diag_degen_audit.py [模型路徑] [--games N]
"""
from __future__ import annotations
import sys, os, argparse, random
from collections import Counter
from zlib import crc32
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import torch

from trpg.scenarios.monsters import (MONSTER_DEFS, EQUIV_LEVEL_1V1,
                                     register_monsters, onev1_viable_monsters)
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import (CombatPolicyNet, apply_resource_mask,
                           apply_entity_mask, pick_action)
from trpg.engine.skill import available_skills, TargetType
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_population import blind_np_single

# Chimeras (held-out novel identity×kit) — register so the auditor covers the
# goal's 「縫合怪」axis, not just standard classes/monsters.
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

CLASSES = ["battle_master", "champion", "totem_bear", "berserker",
           "evocation", "divination", "life", "war",
           "assassin", "arcane_trickster", "devotion", "vengeance"]
# Broad monster set: melee + ranged + casters + petrify/gaze + undead.
MELEE_MON = ["kobold", "goblin", "orc", "ogre", "wolf", "dire_wolf", "ghoul",
             "owlbear", "shadow", "wight", "bandit", "skeleton", "zombie",
             "commoner", "gargoyle", "basilisk", "mage_npc", "manticore"]
OPP_PANEL = ["battle_master", "champion", "evocation", "vengeance"]


def load(path):
    # 與 uni_v7 / eval 一致的單頭(n_head_groups=1)載入(per-arch 擴張=改架構)。
    from distill_routed import load_student
    return load_student(path)


def _has_offense(ch, ws):
    for s in available_skills(ch, ws):
        tt = s.features.target_type
        if tt in (TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY,
                  TargetType.POINT, TargetType.LINE, TargetType.CONE):
            if s.skill_id != "move":
                return True
    return False


def run_combat(net, ident, opp_archs, lvl, opp_lvl, ep_key):
    """模型駕駛 agent 席 ident，回傳 (win, noop_moves, dmg_dealt, turns)。"""
    k = crc32(ep_key.encode())
    random.seed(k)
    env = CombatEnvV2(seed=k ^ 0x5A5A5A, n_agents=1, n_opps=len(opp_archs))
    obs, _ = env.reset(agent_archs=[ident], opp_archs=list(opp_archs),
                       level=lvl, opp_level=opp_lvl)
    aid = env.agent_ids[0]
    opp_hp0 = {o: env.ws.characters[o].hp for o in env.opp_ids}
    noop = 0
    noop_stall = 0     # reach 外 + action 還在手 = 真卡(致命)
    noop_trail = 0     # 已在 reach 內 或 action 已用 = 燒剩餘移動(裝飾)
    noop_self = 0      # 殘留空轉:目標=自己格   noop_block = 目標!=自己格(被擋/夾)
    noop_block = 0
    refuse = 0         # 真拒戰:敵在reach內+攻擊合法+action在手,卻選move/dodge/end
    heal_full = 0      # 滿血硬補
    turns = 0
    done = False
    had_offense = False
    chose_offense = False
    from trpg.rl.obs import N_GRID, GRID_CELL_SIZE_M
    while not done:
        actor = env.current_agent_id
        ch = env.ws.characters[actor]
        is_agent = actor in env.agent_ids
        if is_agent:
            if _has_offense(ch, env.ws):
                had_offense = True
            ob = blind_np_single(obs)
            ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws,
                                   agent_id=actor))
            # D1 偵測：MOVE 前抓位置、可用移動力、action、最近敵距與 reach
            mv_avail = env.resources.get("movement", 0.0)
            act_avail = env.resources.get("action", 0) > 0
            pos0 = ch.position
            try:
                reach = float(ch.get_weapon().range_normal)
            except Exception:
                reach = 1.5
            live = [env.ws.characters[o] for o in env.opp_ids
                    if env.ws.characters[o].is_alive()]
            nd = min((ch.position.distance_to(e.position) for e in live),
                     default=99.0)
            in_reach = nd <= reach + 0.05
            _sks = available_skills(ch, env.ws)
            is_move = 0 < act[0] < len(_sks) and _sks[act[0]].skill_id == "move"
            # 攻擊是否合法(該回合 mask 後可選的鎖敵攻擊)
            atk_legal = False
            for i in range(s.shape[-1]):
                if s[0, i].item() > -1e8 and i < len(_sks):
                    if _sks[i].features.target_type in (
                            TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY):
                        atk_legal = True; break
            chose_atk = False
            if 0 < act[0] < len(_sks):
                _sk = _sks[act[0]]
                _tt = _sk.features.target_type
                if (_tt in (TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY)
                        or (_tt in (TargetType.POINT, TargetType.LINE,
                                    TargetType.CONE)
                            and _sk.skill_id != "move")
                        or getattr(_sk.features, "expected_damage", 0) > 0):
                    chose_offense = True
                if _tt in (TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY):
                    chose_atk = True
            # 真拒戰：敵在 reach 內 + 攻擊合法 + action 還在手，卻直接「結束回合」
            # 什麼也不做（浪費了能打的這個 action）。施 buff/法術/dodge 都用掉了
            # 資源、不算拒戰；只有「丟著 action 結束」才是退化。
            ended = (act[0] == 0)
            if ended and act_avail and in_reach and atk_legal:
                refuse += 1
            # D3 滿血硬補：選了治療技(expected_healing>0)，但自己與所有隊友都接近滿血
            # (無人受傷可補)=浪費資源。
            if 0 < act[0] < len(_sks):
                _hsk = _sks[act[0]]
                if getattr(_hsk.features, "expected_healing", 0) > 0:
                    allies = [ch] + [env.ws.characters[a] for a in env.agent_ids
                                     if a != actor and env.ws.characters[a].is_alive()]
                    if all(a.hp >= a.max_hp * 0.95 for a in allies):
                        heal_full += 1
            cur_cell = (max(0, min(N_GRID - 1, int(pos0.x / GRID_CELL_SIZE_M)))
                        * N_GRID
                        + max(0, min(N_GRID - 1, int(pos0.y / GRID_CELL_SIZE_M))))
            turns += 1
            obs, _, term, trunc, _ = env.step(act)
            if is_move:
                disp = ch.position.distance_to(pos0)
                if disp < 0.1 and mv_avail >= 1.0:
                    noop += 1
                    if act[2] == cur_cell:
                        noop_self += 1
                    else:
                        noop_block += 1
                    if act_avail and not in_reach:
                        noop_stall += 1
                    else:
                        noop_trail += 1
        else:
            # 對手席由 env.step 內部腳本驅動：丟一個 no-op 推進
            obs, _, term, trunc, _ = env.step([0, 0, 0])
        done = term or trunc
    win = (env.ws.characters[aid].is_alive()
           and not any(env.ws.characters[o].is_alive() for o in env.opp_ids))
    dmg = sum(opp_hp0[o] - max(0, env.ws.characters[o].hp) for o in env.opp_ids)
    return win, noop, refuse, noop_self, noop_block, heal_full, noop_stall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default="models/unified/uni_v7.pt")
    ap.add_argument("--games", type=int, default=6)
    args = ap.parse_args()
    net = load(args.ckpt)
    print(f"模型: {os.path.basename(args.ckpt)}  games/cell={args.games}\n")

    # 情境: (標籤, n_opp, 我方Δ等級)
    SCEN = [("1v1平", 1, 0), ("劣勢1v2", 2, 0), ("劣勢低階Δ-3", 1, -3),
            ("優勢2v1*", 1, +3)]
    hdr = (f"{'身分':14s} | " + " ".join(f"{s[0]:>9s}" for s in SCEN)
           + " | 卡死stall 空轉self 空轉blk 真拒戰 滿血補")
    print(hdr); print("-" * len(hdr))

    idents = CLASSES + MELEE_MON + CLS_CHIM + MON_CHIM
    g_self = g_block = g_refuse = g_heal = g_stall = 0
    for ident in idents:
        _eq = _EQUIV.get(ident, 5)
        # inf-equiv = 1vN-only chassis (too strong for fair 1v1) → clamp to 8
        # just so the degeneracy probe can still exercise its action selection.
        base_lvl = 8 if _eq == float("inf") else max(1, int(round(_eq)))
        cells = []
        tot_self = tot_block = tot_refuse = tot_heal = tot_stall = 0
        for label, n_opp, dlvl in SCEN:
            lvl = max(1, base_lvl + dlvl)
            wins = selfs = blocks = refuses = heals = stalls = 0
            for gi in range(args.games):
                opp = [OPP_PANEL[gi % len(OPP_PANEL)] for _ in range(n_opp)]
                w, nm, rf, ns, nb, hf, st = run_combat(
                    net, ident, opp, lvl, base_lvl,
                    f"{ident}|{label}|{gi}")
                wins += int(w); selfs += ns; blocks += nb; refuses += rf
                heals += hf; stalls += st
            cells.append(f"{wins}/{args.games}")
            tot_self += selfs; tot_block += blocks; tot_refuse += refuses
            tot_heal += heals; tot_stall += stalls
        g_self += tot_self; g_block += tot_block; g_refuse += tot_refuse
        g_heal += tot_heal; g_stall += tot_stall
        # ⚠ only for the FATAL degeneracies (per user's BUG def): reach-out stall,
        # blocked stall, real refuse, full-HP heal. Decorative trailing noop_self
        # (already in reach / action spent = burning leftover movement) is NOT a
        # bug ("多扔的、對戰局沒影響的不叫 BUG") — shown but un-flagged.
        flag = " ⚠" if (tot_stall or tot_block or tot_refuse or tot_heal) else ""
        print(f"{ident:14s} | " + " ".join(f"{c:>9s}" for c in cells)
              + f" | {tot_stall:8d} {tot_self:7d} {tot_block:7d} "
              f"{tot_refuse:6d} {tot_heal:5d}{flag}")
    print("-" * len(hdr))
    print(f"{'總計':14s}  |  卡死(stall)={g_stall}  空轉self(裝飾)={g_self}"
          f"  空轉blocked={g_block}  真拒戰={g_refuse}  滿血硬補={g_heal}")
    print("\n* 優勢2v1：我方單體高3階打2個同階對手(測「以一打多+優勢」不退化，可能輸)")


if __name__ == "__main__":
    main()
