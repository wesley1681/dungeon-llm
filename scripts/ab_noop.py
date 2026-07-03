"""no-op 抑制的配對 A/B 侵蝕量測：同模型同種子，抑制 ON vs OFF，比 1v1 vs 腳本專家 WR。
WR 應幾乎不變（move-to-self 無作用，移除它只換成真動作）→ 證零侵蝕。
用法: python scripts/ab_noop.py [games]
"""
import sys, os, random
from zlib import crc32
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import torch
from trpg.scenarios.monsters import register_monsters, EQUIV_LEVEL_1V1, MONSTER_DEFS
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import (CombatPolicyNet, apply_resource_mask,
                           apply_entity_mask, pick_action)
from trpg.engine.skill import available_skills
from train_population import blind_np_single

CLASSES = ["battle_master", "champion", "totem_bear", "berserker", "evocation",
           "divination", "life", "war", "assassin", "arcane_trickster",
           "devotion", "vengeance"]
MON = ["kobold", "orc", "ogre", "wolf", "ghoul", "owlbear", "shadow", "wight"]
OPP = ["battle_master", "champion", "evocation", "vengeance"]
GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 16


def load(p):
    net = CombatPolicyNet(hidden=128); sd = torch.load(p, map_location="cpu")
    net.load_state_dict(CombatPolicyNet.adapt_state_dict_for_perarch(sd), strict=False)
    net.eval(); return net


def play(net, ident, opp, lvl, ep):
    k = crc32(ep.encode()); random.seed(k)
    env = CombatEnvV2(seed=k ^ 0x5A5A5A, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[ident], opp_archs=[opp], level=lvl, opp_level=lvl)
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    done = False
    while not done:
        actor = env.current_agent_id
        if actor == aid:
            ob = blind_np_single(obs)
            ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
            obs, _, term, trunc, _ = env.step(act)
        else:
            obs, _, term, trunc, _ = env.step([0, 0, 0])
        done = term or trunc
    return int(env.ws.characters[aid].is_alive()
               and not env.ws.characters[oid].is_alive())


def run(net):
    out = {}
    for ident in CLASSES + MON:
        lvl = max(1, round(EQUIV_LEVEL_1V1.get(ident, 5))) if ident in MONSTER_DEFS else 5
        w = sum(play(net, ident, OPP[g % len(OPP)], lvl, f"{ident}|{g}")
                for g in range(GAMES))
        out[ident] = w
    return out


if __name__ == "__main__":
    net = load("models/unified/uni_v7.pt")
    os.environ["TRPG_NOOP_SUPPRESS"] = "0"; off = run(net)
    os.environ["TRPG_NOOP_SUPPRESS"] = "1"; on = run(net)
    print(f"\n1v1 vs 腳本專家  games={GAMES}  (OFF=舊 / ON=抑制)\n")
    print(f"{'身分':16s} {'OFF':>5s} {'ON':>5s} {'Δ':>4s}")
    print("-" * 34)
    toff = ton = 0
    for ident in CLASSES + MON:
        d = on[ident] - off[ident]; toff += off[ident]; ton += on[ident]
        flag = "" if abs(d) <= 1 else (" ↑" if d > 0 else " ↓")
        print(f"{ident:16s} {off[ident]:5d} {on[ident]:5d} {d:+4d}{flag}")
    print("-" * 34)
    n = len(CLASSES + MON) * GAMES
    print(f"{'總勝':16s} {toff:5d} {ton:5d} {ton-toff:+4d}  / {n}")
    print(f"WR: OFF={toff/n*100:.1f}%  ON={ton/n*100:.1f}%")
