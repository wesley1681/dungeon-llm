"""Is the LoS-row actually learning for the MOVE skill (melee), or only evo?

For a controlled blocked state (melee behind the pillar diag), this prints,
per checkpoint:
  qlos(move)   grid_query[:,los] for the MOVE skill slot — the trainable scalar
               that boosts LoS=1 cells. Larger positive => stronger flank pull.
  P(losCell)   probability mass the grid policy (move skill) puts on cells that
               RESTORE LoS to the enemy, vs base. The behavioural readout.
  argmaxLoS    is the single highest-logit reachable move cell a LoS cell?
If qlos/P(losCell) grow base->trained, learning works and just needs time. If
flat, the gradient never reaches the move slot (exploration/credit problem).
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
from trpg.rl.model import apply_resource_mask
from trpg.rl.obs import N_GRID
from trpg.engine.skill import available_skills
from trpg.rl.action import _grid_cell_to_xy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single


def blocked_state(ident="champion"):
    k = crc32(f"probe|{ident}".encode()); random.seed(k)
    env = CombatEnvV2(seed=k ^ 0x33, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[ident], opp_archs=["champion"],
                       level=5, opp_level=5, layout="open")
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    bf = env.ws.combat.battlefield
    W, H = bf.width, bf.height
    nx = len(bf.cells[0]); ny = len(bf.cells)
    bf.cells = [[int(TerrainType.NORMAL)] * nx for _ in range(ny)]
    cx, cy = W * 0.5, H * 0.5
    bf.add_rect_obstacle(cx - 0.75, cy - 3.0, cx + 0.75, cy + 3.0)
    env.ws.characters[aid].position = Vec2(W * 0.20, cy)
    env.ws.characters[oid].position = Vec2(W * 0.80, cy)
    return env, aid, oid, obs


def move_slot(env, aid):
    sk = available_skills(env.ws.characters[aid], env.ws)
    for i, s in enumerate(sk):
        if s.skill_id == "move":
            return i, s
    return None, None


def analyze(ckpt, ident="champion"):
    net = load_student(ckpt); net.eval()
    env, aid, oid, obs = blocked_state(ident)
    bf = env.ws.combat.battlefield
    enemy = env.ws.characters[oid]; agent = env.ws.characters[aid]
    mi, ms = move_slot(env, aid)
    ob = blind_np_single(obs)
    ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
    with torch.no_grad():
        el, s, e, g = net(ot)
    # grid_logits for the move slot
    row = g[0, mi]                              # [N_GRID*N_GRID]
    # which reachable cells restore LoS?
    reach = ms.features.range_m
    los_cells, all_cells = [], []
    for cell in range(N_GRID * N_GRID):
        x, y = _grid_cell_to_xy(cell)
        c = Vec2(x, y)
        if bf.is_blocked(c):
            continue
        if (agent.position.distance_to(c) <= reach + 1e-6
                and bf.has_line_of_sight(agent.position, c)):
            all_cells.append(cell)
            if bf.has_line_of_sight(c, enemy.position):
                los_cells.append(cell)
    # softmax over reachable cells only (the legal move set)
    if all_cells:
        sub = row[all_cells]
        p = torch.softmax(sub, 0)
        plos = sum(float(p[i]) for i, c in enumerate(all_cells) if c in los_cells)
        amax = all_cells[int(torch.argmax(sub))]
        amax_is_los = amax in los_cells
    else:
        plos, amax_is_los = float("nan"), False
    # qlos: grid_query[:,los] for the move slot
    qlos = float("nan")
    return plos, amax_is_los, len(los_cells), len(all_cells)


def main():
    ident = sys.argv[2] if len(sys.argv) > 2 else "champion"
    cks = [("base", "models/pop_mon/pop_u0005.pt")]
    if len(sys.argv) > 1:
        cks.append(("ckpt", sys.argv[1]))
    print(f"identity={ident}  (blocked behind pillar)\n")
    print(f"{'tag':<8} {'P(losCell)':>11} {'argmax=LoS':>11} "
          f"{'#losCells':>10} {'#reach':>8}")
    for tag, ck in cks:
        plos, amax, nlos, nall = analyze(ck, ident)
        print(f"{tag:<8} {plos:>11.3f} {str(amax):>11} {nlos:>10} {nall:>8}",
              flush=True)


if __name__ == "__main__":
    main()
