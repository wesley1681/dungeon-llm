"""釘死 probe vs 審計矛盾：重跑審計裡 champion 觸發 heal_full 的戰鬥，
在每次「滿血補」觸發的當下 dump 完整狀態(回合/HP%/resources/skill/敵數/敵距)，
看清審計到底在數什麼狀態。"""
from __future__ import annotations
import sys, os, random
from zlib import crc32
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import torch

from trpg.scenarios.monsters import register_monsters, EQUIV_LEVEL_1V1
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.engine.skill import available_skills
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_population import blind_np_single


def load(path):
    from distill_routed import load_student
    return load_student(path)


def run(net, ident, opp_archs, lvl, opp_lvl, ep_key):
    k = crc32(ep_key.encode()); random.seed(k)
    env = CombatEnvV2(seed=k ^ 0x5A5A5A, n_agents=1, n_opps=len(opp_archs))
    obs, _ = env.reset(agent_archs=[ident], opp_archs=list(opp_archs),
                       level=lvl, opp_level=opp_lvl)
    done = False; turn = 0
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
            if 0 < act[0] < len(sks):
                sk = sks[act[0]]
                if getattr(sk.features, "expected_healing", 0) > 0:
                    allies = [ch] + [env.ws.characters[a] for a in env.agent_ids
                                     if a != actor and env.ws.characters[a].is_alive()]
                    if all(a.hp >= a.max_hp * 0.95 for a in allies):
                        live = [env.ws.characters[o] for o in env.opp_ids
                                if env.ws.characters[o].is_alive()]
                        nd = min((ch.position.distance_to(o.position) for o in live),
                                 default=99)
                        # 量「實際補了幾點 HP」:治療前後 HP 差(被 max_hp 夾住)
                        hp_before = ch.hp
                        turn += 1
                        obs, _, term, trunc, _ = env.step(act)
                        hp_after = env.ws.characters[actor].hp
                        print(f"  [{ep_key}] turn{turn} HP={hp_before:.0f}/{ch.max_hp:.0f}"
                              f"({hp_before/ch.max_hp*100:.0f}%) skill={sk.skill_id}"
                              f" → 實際補了 {hp_after-hp_before:+.1f} HP"
                              f"  (res a={env.resources.get('action')},b={env.resources.get('bonus_action')})"
                              f" 敵距={nd:.1f}")
                        done = term or trunc
                        continue
            turn += 1
            obs, _, term, trunc, _ = env.step(act)
        else:
            obs, _, term, trunc, _ = env.step([0, 0, 0])
        done = term or trunc


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "models/unified/uni_v8.pt"
    net = load(path)
    OPP_PANEL = ["battle_master", "champion", "evocation", "vengeance"]
    SCEN = [("1v1平", 1, 0), ("劣勢1v2", 2, 0), ("劣勢低階Δ-3", 1, -3), ("優勢2v1*", 1, +3)]
    for ident in ["champion", "battle_master", "war", "devotion", "chimera_gish"]:
        try:
            from chimera_defs import register_chimeras
            register_chimeras()
        except Exception:
            pass
        eq = EQUIV_LEVEL_1V1.get(ident, 5)
        base = 8 if eq == float("inf") else max(1, int(round(eq)))
        print(f"== {ident} (base L{base}) ==")
        for label, n_opp, dlvl in SCEN:
            lvl = max(1, base + dlvl)
            for gi in range(3):
                opp = [OPP_PANEL[gi % len(OPP_PANEL)] for _ in range(n_opp)]
                run(net, ident, opp, lvl, base, f"{ident}|{label}|{gi}")


if __name__ == "__main__":
    main()
