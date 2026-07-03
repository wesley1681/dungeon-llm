"""WHY does the grid head plateau at the pillar corner (never completes flank)?

Run champion (argmax) against a stationary pillar until it stops moving, then
at that stuck state dump, for the MOVE skill, the top cells by grid_logit and
by proximity, annotated with: is it a legal move target (point_validity_mask)?
is it reachable by a STRAIGHT walk (engine moves straight, stops at walls)?
does it restore LoS? distance-to-enemy. This reveals whether the plateau is
(a) good cells masked out, (b) engine straight-move can't round the corner,
(c) the distance channel out-scoring the proximity channel.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import random
from zlib import crc32
import torch

from trpg.scenarios.monsters import register_monsters
register_monsters()
from trpg.engine.vec2 import Vec2, TerrainType
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.obs import N_GRID
from trpg.rl.action import (decode_action, point_validity_mask,
                            _grid_cell_to_xy, available_skills)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single, _StationaryPolicy


def setup(ident="champion"):
    k = crc32(f"plat|{ident}".encode()); random.seed(k)
    env = CombatEnvV2(seed=k ^ 0x33, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[ident], opp_archs=["champion"],
                       level=5, opp_level=5, layout="open")
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    bf = env.ws.combat.battlefield; W, H = bf.width, bf.height
    nx = len(bf.cells[0]); ny = len(bf.cells)
    bf.cells = [[int(TerrainType.NORMAL)] * nx for _ in range(ny)]
    bf.add_rect_obstacle(W*0.5-0.75, H*0.5-3.0, W*0.5+0.75, H*0.5+3.0)
    env.ws.characters[aid].position = Vec2(W*0.20, H*0.5)
    env.ws.characters[oid].position = Vec2(W*0.80, H*0.5)
    env._opp_policies[oid] = _StationaryPolicy()
    return env, aid, oid, obs


def main():
    ck = sys.argv[1] if len(sys.argv) > 1 else "models/pop_mon/pop_u0005.pt"
    ident = sys.argv[2] if len(sys.argv) > 2 else "champion"
    net = load_student(ck); net.eval()
    env, aid, oid, obs = setup(ident)
    ag = env.ws.characters[aid]; en = env.ws.characters[oid]; bf = env.ws.combat.battlefield
    prev = None; stuck_at = None
    for t in range(20):
        p = ag.position
        if prev is not None and p.distance_to(prev) < 0.3:
            stuck_at = p; break
        prev = p
        ob = blind_np_single(obs)
        ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
        with torch.no_grad():
            el, s, e, g = net(ot)
        s = apply_resource_mask(s, env.resources, env.ws, aid)
        e = apply_entity_mask(e, ot, env.ws, aid)
        act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=aid))
        obs, _, term, trunc, _ = env.step(act)
        if term or trunc:
            break
    print(f"ckpt={ck} ident={ident}")
    print(f"agent stuck at ({ag.position.x:.1f},{ag.position.y:.1f}) "
          f"enemy ({en.position.x:.1f},{en.position.y:.1f}) "
          f"LoS={bf.has_line_of_sight(ag.position, en.position)}\n")
    # analyse the MOVE skill grid logits at the stuck state
    ob = blind_np_single(obs)
    ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
    with torch.no_grad():
        el, s, e, g = net(ot)
    s2 = apply_resource_mask(s, env.resources, env.ws, aid)
    e2 = apply_entity_mask(e, ot, env.ws, aid)
    chosen = list(pick_action(el[0], s2[0], e2[0], g[0], ws=env.ws, agent_id=aid))
    sk = available_skills(ag, env.ws)
    print(f"CHOSEN action {chosen} -> skill="
          f"{sk[chosen[0]].skill_id if chosen[0] < len(sk) else 'end'}\n")
    mi = next(i for i, x in enumerate(sk) if x.skill_id == "move")
    reach = sk[mi].features.range_m
    inval = point_validity_mask(env.ws, aid, sk[mi])   # [N_GRID*N_GRID] bool
    row = g[0, mi]
    # proximity channel = los_grid at each cell (from obs)
    prox = torch.from_numpy(ob["los_grid"][0]).flatten()
    order = torch.argsort(row, descending=True)
    print(f"move reach={reach:.1f}m  legal cells={int((~inval).sum())}")
    print(f"{'cell':>5} {'logit':>7} {'prox':>5} {'legal':>5} {'straight':>8} "
          f"{'dEN':>5} {'restoreLoS':>10}")
    shown = 0
    for ci in order.tolist():
        if inval[ci]:
            continue
        x, y = _grid_cell_to_xy(ci); c = Vec2(x, y)
        straight = bf.has_line_of_sight(ag.position, c)   # clear straight walk?
        dEN = c.distance_to(en.position)
        restores = bf.has_line_of_sight(c, en.position)
        within = ag.position.distance_to(c) <= reach + 1e-6
        print(f"{ci:>5} {float(row[ci]):>7.2f} {float(prox[ci]):>5.2f} "
              f"{'Y':>5} {('Y' if straight and within else 'n'):>8} "
              f"{dEN:>5.1f} {('Y' if restores else 'n'):>10}")
        shown += 1
        if shown >= 12:
            break


if __name__ == "__main__":
    main()
