"""受控「接敵」行為審計：把通用模型放在 (身分 × 起始距離 × 角度) 的網格上，
對一個靜止假人跑一個完整回合，自動分類結果並標出退化行為。

退化判準（在「一回合內理論上接得到」的情境，即 dist <= 移動力+觸及）：
  STALL_NOOP   : 從沒攻擊，且回合內出現過原地空轉移動（grid-head 微距病）
  STALL_DODGE  : 從沒攻擊，最後停在能接觸的範圍卻選 dodge/end（防禦惰性）
  STALL_FAR    : 從沒攻擊也沒接近到觸及（接近能力不足）
  OK           : 至少攻擊到一次
遠到一回合接不到的(dist > 移動力+觸及)標 UNREACHABLE，不計入退化率。

用法: python scripts/diag_engage_audit.py [模型路徑]
"""
from __future__ import annotations
import os
import sys
import math

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
from trpg.rl.env_v2 import (CombatEnvV2, MOVE_BUDGET_M,  # noqa: E402
                            _MAX_SUB_ACTIONS_PER_TURN)
import trpg.rl.env_v2 as E  # noqa: E402
from trpg.rl.model import CombatPolicyNet  # noqa: E402
from trpg.rl.neural_policy import NeuralCombatPolicy  # noqa: E402
from trpg.engine.vec2 import Vec2  # noqa: E402
from trpg.engine.skill import available_skills, TargetType  # noqa: E402
from trpg.scenarios.monsters import register_monsters  # noqa: E402


def _is_offensive(skill_id, skills):
    """offensive = 鎖敵(SINGLE/MULTI_ENEMY) 或 非 move 的 AOE/點/線/錐傷害技。
    通用判定，不寫死任何技能名（move 是引擎移動技、dodge/hide/disengage 是 SELF）。"""
    for s in skills:
        if getattr(s, "skill_id", "") != skill_id:
            continue
        tt = getattr(getattr(s, "features", None), "target_type", None)
        if tt in (TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY):
            return True
        if tt in (TargetType.POINT, TargetType.LINE, TargetType.CONE) \
                and skill_id != "move":
            return True
        return False
    return False

register_monsters()
exec_action = E.execute_action
consume = E.consume_resources

MELEE = ["kobold", "goblin", "orc", "ogre", "troll", "hill_giant",
         "wolf", "dire_wolf", "ghoul", "owlbear", "shadow", "wight"]
CLASSES = ["battle_master", "champion", "totem_bear", "berserker",
           "evocation", "divination", "life", "war",
           "assassin", "arcane_trickster", "devotion", "vengeance"]
RANGED = ["mage_npc", "manticore"]   # 有遠程武器/法術，行為基準
IDENTS = MELEE + CLASSES + RANGED

DISTS = [1.2, 1.6, 1.8, 2.0, 2.4, 3.0, 4.0, 6.0, 9.0, 12.0]
ANGLES = [0.0, 27.0, 45.0]   # 角度→製造偏心斜對角(觸及邊緣)情境


def load(path):
    net = CombatPolicyNet(hidden=128)
    sd = torch.load(path, map_location="cpu")
    sd = CombatPolicyNet.adapt_state_dict_for_perarch(sd)
    net.load_state_dict(sd, strict=False)
    net.eval()
    return net


def reach_of(ch):
    try:
        return float(ch.get_weapon().range_normal)
    except Exception:
        return 1.5


def run_turn(net, ident, dist, ang_deg):
    """模型驅動 opp_0(ident) 對靜止 agent_0(假人) 跑一回合，回傳分類。"""
    env = CombatEnvV2(n_agents=1, n_opps=1)
    env.reset(level=5, opp_level=5, layout="open",
              agent_archs=["battle_master"], opp_archs=[ident], seed=7)
    dummy = env.ws.characters["agent_0"]
    mob = env.ws.characters["opp_0"]
    cx, cy = 15.0, 15.3   # 偏心，製造非整數距離
    dummy.position = Vec2(cx, cy)
    a = math.radians(ang_deg)
    mob.position = Vec2(cx + dist * math.cos(a), cy + dist * math.sin(a))
    reach = reach_of(mob)
    pol = NeuralCombatPolicy(net, blind=True)
    res = {"action": 1, "bonus_action": 1, "movement": MOVE_BUDGET_M}
    attacked = False
    had_noop_move = False
    actions = []
    hp0 = dummy.hp
    for _ in range(_MAX_SUB_ACTIONS_PER_TURN):
        skills = available_skills(mob, env.ws)   # 決策當下的可用招(供 offensive 判定)
        d = pol.decide("opp_0", mob, env.ws, res, 1)
        if d.action is None:
            actions.append("END")
            break
        atype = d.action.get("type")
        sid = d.action.get("skill_id") or atype
        r = exec_action(d.action, env.ws)
        if r.get("type") != "ERROR":
            consume(res, d.action, r)
        if atype == "MOVE":
            moved = r.get("distance", 0.0)
            if moved < 0.01:
                had_noop_move = True
            if moved < 0.01 or res["movement"] < 0.5:
                res["movement"] = 0.0
        # engaged = 選了任何 offensive 行動(含遠程/法術/AOE) 或 敵人實際掉血
        if _is_offensive(sid, skills) or dummy.hp < hp0:
            attacked = True
        actions.append(sid)
        if d.ended:
            break
    final_dist = mob.position.distance_to(dummy.position)
    reachable = dist <= MOVE_BUDGET_M + reach + 0.5   # 一回合理論上接得到
    if not reachable:
        cls = "UNREACHABLE"
    elif attacked:
        cls = "OK"
    elif had_noop_move:
        cls = "STALL_NOOP"
    elif final_dist <= reach + 0.3:
        cls = "STALL_DODGE"
    else:
        cls = "STALL_FAR"
    return cls, final_dist, reach, actions


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _ROOT, "models", "unified", "uni_v7.pt")
    net = load(path)
    print(f"模型: {os.path.basename(path)}  (一回合移動力 {MOVE_BUDGET_M}m)\n")
    hdr = f"{'身分':14s} {'觸及':>4s} | {'OK':>3s} {'NOOP':>4s} {'DODGE':>5s} {'FAR':>3s} | 退化率  代表失敗樣本"
    print(hdr)
    print("-" * len(hdr))
    grand = {"OK": 0, "STALL_NOOP": 0, "STALL_DODGE": 0, "STALL_FAR": 0}
    for ident in IDENTS:
        counts = {"OK": 0, "STALL_NOOP": 0, "STALL_DODGE": 0, "STALL_FAR": 0,
                  "UNREACHABLE": 0}
        sample = ""
        reach = None
        for dist in DISTS:
            for ang in ANGLES:
                cls, fd, reach, acts = run_turn(net, ident, dist, ang)
                counts[cls] += 1
                if cls.startswith("STALL") and not sample:
                    sample = f"d={dist}@{ang:.0f}°→{cls}(留{fd:.2f}m) {acts}"
        reachable_total = sum(counts[k] for k in
                              ("OK", "STALL_NOOP", "STALL_DODGE", "STALL_FAR"))
        stalls = reachable_total - counts["OK"]
        rate = (stalls / reachable_total * 100) if reachable_total else 0.0
        for k in grand:
            grand[k] += counts[k]
        print(f"{ident:14s} {reach:4.1f} | {counts['OK']:3d} "
              f"{counts['STALL_NOOP']:4d} {counts['STALL_DODGE']:5d} "
              f"{counts['STALL_FAR']:3d} | {rate:5.1f}%  {sample}")
    tot = sum(grand.values())
    stalls = tot - grand["OK"]
    print("-" * len(hdr))
    print(f"{'總計':14s}      | {grand['OK']:3d} {grand['STALL_NOOP']:4d} "
          f"{grand['STALL_DODGE']:5d} {grand['STALL_FAR']:3d} | "
          f"{stalls/tot*100:5.1f}% 退化(可接敵情境)")


if __name__ == "__main__":
    main()
