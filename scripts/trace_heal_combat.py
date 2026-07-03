"""逐 sub-action trace 單場戰鬥，看清為什麼模型卡在 reach 外、movement 歸 0、最後 heal。"""
from __future__ import annotations
import sys, os, random
from zlib import crc32
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import torch

from trpg.scenarios.monsters import register_monsters
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.engine.skill import available_skills
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_population import blind_np_single


def load(path):
    from distill_routed import load_student
    return load_student(path)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "models/unified/uni_v8.pt"
    ident = sys.argv[2] if len(sys.argv) > 2 else "champion"
    ep_key = sys.argv[3] if len(sys.argv) > 3 else "champion|1v1平|1"
    lvl = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    net = load(path)
    k = crc32(ep_key.encode()); random.seed(k)
    env = CombatEnvV2(seed=k ^ 0x5A5A5A, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[ident], opp_archs=["battle_master"],
                       level=lvl, opp_level=lvl)
    aid = env.agent_ids[0]
    done = False; turn = 0; last_actor = None
    while not done and turn < 60:
        actor = env.current_agent_id
        ch = env.ws.characters[actor]
        is_agent = actor in env.agent_ids
        if actor != last_actor:
            print(f"--- {'AGENT' if is_agent else 'OPP'} {actor} 回合 ---")
            last_actor = actor
        if is_agent:
            foe = [env.ws.characters[o] for o in env.opp_ids
                   if env.ws.characters[o].is_alive()]
            nd = min((ch.position.distance_to(o.position) for o in foe), default=99)
            ob = blind_np_single(obs)
            ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            endp = torch.sigmoid(el[0]).item()
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
            sks = available_skills(ch, env.ws)
            sid = sks[act[0]].skill_id if 0 < act[0] < len(sks) else ("END" if act[0] == 0 else "?")
            p0 = (ch.position.x, ch.position.y)
            r = env.resources
            print(f"   pos=({p0[0]:.1f},{p0[1]:.1f}) 敵距={nd:.2f} "
                  f"res(a={r.get('action')},b={r.get('bonus_action')},mv={r.get('movement'):.1f})"
                  f" endp={endp:.3f} → {sid} grid={act[2]}")
            obs, _, term, trunc, _ = env.step(act)
            if sid == "move":
                p1 = (ch.position.x, ch.position.y)
                disp = ((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)**0.5
                print(f"        移動後 pos=({p1[0]:.1f},{p1[1]:.1f}) 位移={disp:.2f} mv剩={env.resources.get('movement'):.1f}")
            turn += 1
        else:
            obs, _, term, trunc, _ = env.step([0, 0, 0])
        done = term or trunc
    print("結束")


if __name__ == "__main__":
    main()
