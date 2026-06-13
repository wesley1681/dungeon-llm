"""盤點：哪些招模型在訓練中完全沒「用過」（沒坐該席扮演過）。

模型「用過」一個招 = 訓練中它坐 agent 席、那個身份的 kit 裡有該招。訓練可扮
身份 = train_monster_actor 的 1v1 池 + boss 池 + 標準 12 職業（synth 用職業招池，
不引入新招）。對比遊戲全部怪+職業的招，差集 = 模型動作空間裡「沒用過」的招。

注意：傳奇行動（龍尾/傳奇招）不在 available_skills（由 orchestrator 腳本執行，
根本不在模型動作空間）——另行列出。

Usage: python scripts/probe_unseen_skills.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

from trpg.scenarios.monsters import MONSTER_DEFS, register_monsters
register_monsters()

from trpg.rl.env_v2 import CombatEnvV2
from trpg.engine.skill import available_skills

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_identity import STANDARD_IDS
from train_monster_actor import train_pools, HELD_OUT_1V1, HELD_OUT_BOSS


def skills_of(cid, level):
    env = CombatEnvV2(seed=1, n_agents=1, n_opps=1)
    env.reset(agent_archs=[cid], opp_archs=["commoner"],
              level=level, opp_level=1)
    aid = env.agent_ids[0]
    return {s.skill_id for s in available_skills(env.ws.characters[aid], env.ws)
            if s.features.expected_damage > 0 or s.features.expected_healing > 0
            or s.features.applies_status}   # drop generic move/dodge/dash...


def lvl(cid):
    return MONSTER_DEFS[cid].natural_level if cid in MONSTER_DEFS else 6


def legendary_options(cid):
    """傳奇行動招（不在 available_skills、不在模型動作空間）。"""
    md = MONSTER_DEFS.get(cid)
    if not md:
        return set()
    out = set()
    for t in md.traits:
        if t.trait_id == "legendary_actions":
            for o in t.params.get("options", []):
                out.add(o.get("ability") or f"weapon:{o.get('weapon')}")
    return out


def main():
    t1, tb = train_pools()
    trained = set(t1) | set(tb) | set(STANDARD_IDS)
    allids = set(MONSTER_DEFS) | set(STANDARD_IDS)
    excluded = allids - trained

    print(f"訓練可扮身份：{len(trained)}（1v1 怪 {len(t1)} + boss 怪 {len(tb)} "
          f"+ 職業 {len(STANDARD_IDS)}）")
    print(f"排除身份：{len(excluded)}\n")

    seen = set()
    for cid in trained:
        seen |= skills_of(cid, lvl(cid))

    # 哪些招完全沒在訓練 agent 席出現（含 reason 分類）
    reason = {}
    for cid in HELD_OUT_1V1 + HELD_OUT_BOSS:
        reason[cid] = "held-out（故意留作泛化測試）"
    import math
    from trpg.scenarios.monsters import EQUIV_LEVEL_1V1, PARTY3_EQUIV_LEVEL
    for cid in excluded:
        if cid in reason:
            continue
        e1 = EQUIV_LEVEL_1V1.get(cid, math.inf)
        ep = PARTY3_EQUIV_LEVEL.get(cid, math.inf)
        if math.isinf(e1) and math.isinf(ep):
            reason[cid] = "1vN inf（1v1 與 3 人隊都打不過＝太強、無公平訓練對手）"
        elif math.isinf(e1):
            reason[cid] = "1v1 inf（只在 boss 池，但 party-equiv>8 也排除）"
        else:
            reason[cid] = "其他"

    print("=== 模型動作空間裡「完全沒用過」的招（按身份）===")
    unseen_all = set()
    for cid in sorted(excluded, key=lambda c: reason.get(c, "")):
        extra = skills_of(cid, lvl(cid)) - seen
        if extra:
            unseen_all |= extra
            tag = "怪" if cid in MONSTER_DEFS else "職業"
            print(f"  [{tag}] {cid:<20} 沒見過招：{sorted(extra)}")
            print(f"        排除原因：{reason.get(cid, '?')}")

    print(f"\n  完全沒用過的「主動招」合計：{len(unseen_all)} 種")
    print(f"  {sorted(unseen_all)}\n")

    print("=== 傳奇行動招（不在模型動作空間 = orchestrator 腳本執行）===")
    leg_all = set()
    for cid in MONSTER_DEFS:
        lo = legendary_options(cid)
        if lo:
            leg_all |= lo
            print(f"  {cid:<20} {sorted(lo)}")
    print(f"\n  傳奇招合計：{len(leg_all)} 種（模型永遠不會選，因為不是它的動作）")
    print(f"  {sorted(leg_all)}")


if __name__ == "__main__":
    main()
