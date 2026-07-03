"""WHAT is the agent doing during a self-play wall stalemate?

The LoS-row fix only redirects the destination of a move the agent ALREADY
chose. If freeze episodes are dominated by the agent NOT moving (ending the
turn, or attacking with no LoS = a no-op), then the cause is the skill/value
decision (frozen in train_grid_los_only), not the grid cell — and no amount of
LoS-row training can fix it. This logs, over self-play episodes split by
outcome (frozen=truncated-no-death vs decisive), per agent decision:
    blocked%   turns with NO enemy in LoS
    of BLOCKED turns, the action-TYPE the agent picked:
       move%   tried to reposition (the only thing that can regain LoS)
       end%    ended the turn / did nothing
       atk%    attacked (a no-op when blocked — can't hit through a wall)
       other%  buff/point/etc.
A high end%/atk% on blocked turns = the agent isn't even trying to move =
the freeze is a policy-choice problem above the grid head.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import argparse, random
from collections import Counter
from zlib import crc32
import torch

from trpg.scenarios.monsters import register_monsters
register_monsters()
from trpg.engine.vec2 import Vec2  # noqa
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.obs import partition_entities
from trpg.rl.action import decode_action
from trpg.engine.skill import available_skills

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single
from synth_identity import STANDARD_IDS


def _enemy_los(ws, aid):
    bf = ws.combat.battlefield if ws.combat else None
    if bf is None:
        return True
    a = ws.characters[aid]
    _, enemies = partition_entities(ws, aid)
    return any(ws.characters[e].is_alive()
               and bf.has_line_of_sight(a.position, ws.characters[e].position)
               for e in enemies)


def act_type(act, ws, aid):
    skills = available_skills(ws.characters[aid], ws)
    si = int(act[0])
    if si >= len(skills) or skills[si].skill_id == "end":
        return "end"
    ad = decode_action(act, ws, aid)
    if ad is None:
        return "end"
    t = ad.get("type", "")
    if t == "MOVE":
        return "move"
    if t in ("ATTACK", "WEAPON_ATTACK") or "attack" in str(t).lower():
        return "atk"
    return "other"


def run(net, ident, opp, layout, ep_key, max_turns=40):
    k = crc32(ep_key.encode()); random.seed(k)
    env = CombatEnvV2(seed=k ^ 0x11, n_agents=1, n_opps=1)
    env.use_self_play_opponent(net)
    obs, _ = env.reset(agent_archs=[ident], opp_archs=[opp],
                       level=5, opp_level=5, layout=layout)
    aid = env.agent_ids[0]
    blk = 0; tot = 0
    blk_types = Counter()
    done = False; turns = 0
    while not done:
        actor = env.current_agent_id
        ob = blind_np_single(obs)
        ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
        with torch.no_grad():
            el, s, e, g = net(ot)
        s = apply_resource_mask(s, env.resources, env.ws, actor)
        e = apply_entity_mask(e, ot, env.ws, actor)
        act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
        if actor == aid:
            tot += 1
            if not _enemy_los(env.ws, aid):
                blk += 1
                blk_types[act_type(act, env.ws, aid)] += 1
        obs, _, term, trunc, _ = env.step(act)
        turns += 1
        done = term or trunc
    frozen = trunc and env.ws.combat is not None
    return frozen, blk, tot, blk_types


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/pop_mon/pop_u0005.pt")
    p.add_argument("--layout", default="pillar")
    p.add_argument("--games", type=int, default=40)
    args = p.parse_args()
    net = load_student(args.ckpt); net.eval()
    fro = Counter(); dec = Counter()
    fro_blk = [0, 0]; dec_blk = [0, 0]   # [blocked, total]
    nf = nd = 0
    for g in range(args.games):
        ident = STANDARD_IDS[g % len(STANDARD_IDS)]
        opp = STANDARD_IDS[(g + 4) % len(STANDARD_IDS)]
        frozen, blk, tot, types = run(net, ident, opp, args.layout, f"sf|{g}")
        if frozen:
            nf += 1; fro += types; fro_blk[0] += blk; fro_blk[1] += tot
        else:
            nd += 1; dec += types; dec_blk[0] += blk; dec_blk[1] += tot

    def show(tag, n, blkacc, types):
        if n == 0:
            print(f"{tag}: 0 episodes"); return
        b = blkacc[0] / max(1, blkacc[1])
        tt = sum(types.values()) or 1
        print(f"{tag} ({n} eps)  blocked%={b:.0%}  | of blocked turns: "
              f"move={types['move']/tt:.0%} end={types['end']/tt:.0%} "
              f"atk={types['atk']/tt:.0%} other={types['other']/tt:.0%}")
    print(f"ckpt={args.ckpt} layout={args.layout} games={args.games}\n")
    show("FROZEN ", nf, fro_blk, fro)
    show("DECISIVE", nd, dec_blk, dec)


if __name__ == "__main__":
    main()
