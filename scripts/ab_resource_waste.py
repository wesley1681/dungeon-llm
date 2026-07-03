"""量測「搆不到/滿血時亂燒 surge+second_wind」到底吃不吃勝率(=是否真 BUG)。

A=原模型；B=同模型+測量用 gate(僅測量,非交付)：當選到的 skill 是
  - action_surge 但無敵在 reach 內(surge 拿了也打不到) → 遮掉重選
  - 純治療技 但自己+隊友皆 >40% HP(不需補) → 遮掉重選
重選=把該 skill mask 後重跑 pick_action(讓模型挑次優,通常 end/dodge/move)。
同一批 seed 配對比 WR。WR 顯著上升=真 WR-BUG 值得訓練修；持平=非 WR-BUG。

用法: python scripts/ab_resource_waste.py [模型] [--games N]
"""
from __future__ import annotations
import sys, os, argparse, random
from zlib import crc32
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import torch

from trpg.scenarios.monsters import register_monsters, EQUIV_LEVEL_1V1
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.engine.skill import available_skills, TargetType
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_population import blind_np_single

try:
    from chimera_defs import register_chimeras
    register_chimeras()
except Exception:
    pass


def load(path):
    from distill_routed import load_student
    return load_student(path)


def _reach_of(ch):
    try:
        return float(ch.get_weapon().range_normal) if ch.weapons else 1.5
    except Exception:
        return 1.5


def _wasteful(skills, idx, ch, ws, agent_ids, opp_ids):
    """選到的 skill 是否屬於『此刻燒了沒用』的資源浪費。"""
    if not (0 < idx < len(skills)):
        return False
    sk = skills[idx]
    sid = sk.skill_id
    # 純治療技 + 無人受傷
    if getattr(sk.features, "expected_healing", 0) > 0:
        allies = [ch] + [ws.characters[a] for a in agent_ids
                         if ws.characters[a] is not ch and ws.characters[a].is_alive()]
        if all(a.hp >= a.max_hp * 0.4 for a in allies):
            return True
    # action_surge 但無敵在 reach 內(拿了第二 action 也打不到)
    if sid == "action_surge":
        reach = _reach_of(ch)
        live = [ws.characters[o] for o in opp_ids if ws.characters[o].is_alive()]
        nd = min((ch.position.distance_to(o.position) for o in live), default=99)
        if nd > reach + 0.05:
            return True
    return False


def _attack_legal(s_row, skills):
    """該回合 mask 後是否還有合法的鎖敵攻擊(=浪費資源時是否本可攻擊)。"""
    for i in range(s_row.shape[-1]):
        if i < len(skills) and s_row[i].item() > -1e8:
            if skills[i].features.target_type in (
                    TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY):
                return True
    return False


def run_combat(net, ident, opp_archs, lvl, opp_lvl, ep_key, gate, counters=None):
    k = crc32(ep_key.encode()); random.seed(k)
    env = CombatEnvV2(seed=k ^ 0x5A5A5A, n_agents=1, n_opps=len(opp_archs))
    obs, _ = env.reset(agent_archs=[ident], opp_archs=list(opp_archs),
                       level=lvl, opp_level=opp_lvl)
    aid = env.agent_ids[0]
    done = False
    while not done:
        actor = env.current_agent_id
        ch = env.ws.characters[actor]
        if actor in env.agent_ids:
            ob = blind_np_single(obs)
            ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
            sks = available_skills(ch, env.ws)
            if counters is not None and _wasteful(sks, act[0], ch, env.ws,
                                                  env.agent_ids, env.opp_ids):
                counters["waste"] += 1
                if _attack_legal(s[0], sks):
                    counters["waste_atk_legal"] += 1   # 決定性:浪費時本可攻擊
            if gate:
                guard = 0
                while _wasteful(sks, act[0], ch, env.ws, env.agent_ids, env.opp_ids) and guard < 8:
                    s[0, act[0]] = -1e9
                    act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
                    guard += 1
            obs, _, term, trunc, _ = env.step(act)
        else:
            obs, _, term, trunc, _ = env.step([0, 0, 0])
        done = term or trunc
    return (env.ws.characters[aid].is_alive()
            and not any(env.ws.characters[o].is_alive() for o in env.opp_ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default="models/unified/uni_v8.pt")
    ap.add_argument("--games", type=int, default=40)
    args = ap.parse_args()
    net = load(args.ckpt)
    # surge/heal 浪費集中在 fighter/paladin/cleric 類；用對稱同階對手量「能否贏」。
    IDENTS = ["champion", "battle_master", "war", "devotion", "vengeance",
              "life", "totem_bear", "berserker"]
    OPP = ["battle_master", "champion", "vengeance", "evocation"]
    print(f"模型={os.path.basename(args.ckpt)} games={args.games} (對稱同階)\n")
    print(f"{'身分':14s} | A原始 | B+gate | Δ | 浪費次數(其中攻擊本合法)")
    print("-" * 60)
    tA = tB = tn = 0
    cnt = {"waste": 0, "waste_atk_legal": 0}
    for ident in IDENTS:
        eq = EQUIV_LEVEL_1V1.get(ident, 5)
        lvl = 8 if eq == float("inf") else max(1, int(round(eq)))
        a = b = 0
        c = {"waste": 0, "waste_atk_legal": 0}
        for gi in range(args.games):
            opp = [OPP[gi % len(OPP)]]
            key = f"{ident}|{gi}"
            a += int(run_combat(net, ident, opp, lvl, lvl, key, gate=False, counters=c))
            b += int(run_combat(net, ident, opp, lvl, lvl, key, gate=True))
        tA += a; tB += b; tn += args.games
        cnt["waste"] += c["waste"]; cnt["waste_atk_legal"] += c["waste_atk_legal"]
        d = b - a
        flag = " ⚠" if d != 0 else ""
        print(f"{ident:14s} | {a:3d}/{args.games} | {b:3d}/{args.games} | {d:+d}{flag}"
              f" | {c['waste']:4d} ({c['waste_atk_legal']})")
    print("-" * 60)
    print(f"{'總計':14s} | {tA:3d}/{tn} | {tB:3d}/{tn} | {tB-tA:+d}"
          f"  (A={tA/tn*100:.1f}% B={tB/tn*100:.1f}%)")
    print(f"\n浪費總數={cnt['waste']}  其中『攻擊本合法』(真會吃WR)={cnt['waste_atk_legal']}")


if __name__ == "__main__":
    main()
