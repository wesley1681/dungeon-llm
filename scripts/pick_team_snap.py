"""Pick the best team-mode snapshot for one archetype, with a 1v1 guard.

Evaluates every snapshot in models/ppo_team_<arch>/ (plus the current routed
checkpoint as the do-nothing candidate) on the team metric (eval_team_slot,
big sample), ranks them, then checks the winner's 1v1 WR vs the fair expert
so the already-met 1v1 goal is not silently regressed.

Usage: python scripts/pick_team_snap.py <arch> [team_games] [v1_games] [snap_dir]
       snap_dir overrides the default models/ppo_team_<arch> snapshot folder
       (e.g. an --asym training run written to its own directory).
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path

import torch

from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.model import (CombatPolicyNet, apply_resource_mask,
                           apply_entity_mask, pick_action)
from trpg.rl.action import encode_action
from trpg.engine.combat_policy import make_archetype_policy
from trpg.scenarios.archetypes import ARCHETYPE_ROLES

from eval_routed import DEFAULT_ROUTING, load_net, stable_seed
from train_team_ppo import eval_team_slot, ROLE_SLOT

arch = sys.argv[1]
team_games = int(sys.argv[2]) if len(sys.argv) > 2 else 8
v1_games = int(sys.argv[3]) if len(sys.argv) > 3 else 60
snap_dir_arg = sys.argv[4] if len(sys.argv) > 4 else None

slot = ROLE_SLOT[ARCHETYPE_ROLES[arch]]

cache = {}
mate_nets = {}
for a, path in DEFAULT_ROUTING.items():
    if a == arch:
        continue
    if path not in cache:
        cache[path] = load_net(path)
    mate_nets[a] = cache[path]


def one_v1_wr(net) -> float:
    """eval_class-equivalent: 1v1 vs all 12 opponents, fixed seeds."""
    wins = n = 0
    for opp in ARCHETYPE_LIST:
        base = stable_seed(f"{arch}_{opp}")
        for i in range(v1_games):
            env = CombatEnvV2(seed=base + i, n_agents=1, n_opps=1)
            obs, _ = env.reset(agent_archs=[arch], opp_archs=[opp], level=5)
            aid = env.agent_ids[0]; oid = env.opp_ids[0]
            done = False
            while not done:
                actor = env.current_agent_id
                ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
                with torch.no_grad():
                    el, s, e, g = net(ot)
                s = apply_resource_mask(s, env.resources, env.ws, actor)
                e = apply_entity_mask(e, ot, env.ws, actor)
                act = list(pick_action(el[0], s[0], e[0], g[0],
                                       ws=env.ws, agent_id=actor))
                obs, _, term, trunc, _ = env.step(act)
                done = term or trunc
            n += 1
            wins += int((not env.ws.characters[oid].is_alive())
                        and env.ws.characters[aid].is_alive())
    return wins / n


candidates: list[tuple[str, str]] = [("current-routing", DEFAULT_ROUTING[arch])]
snap_dir = Path(snap_dir_arg or f"models/ppo_team_{arch}")
for p in sorted(snap_dir.glob("ppo_u*.pt")):
    candidates.append((p.stem, str(p)))
if snap_dir_arg is None:
    for p in sorted(Path(f"models/bc_team_{arch}").glob("bc_e*.pt")):
        candidates.append((f"bc_{p.stem}", str(p)))

print(f"=== {arch} (slot={slot}) — team eval {team_games} games/cell ===")
rows = []
for name, path in candidates:
    net = load_net(path)
    wr = eval_team_slot(net, arch, slot, mate_nets, games=team_games)
    rows.append((wr, name, path))
    print(f"  {name:18s} team WR={wr:5.1%}", flush=True)

expert_ref = eval_team_slot("expert", arch, slot, mate_nets, games=team_games)
print(f"  {'EXPERT-IN-SLOT':18s} team WR={expert_ref:5.1%}")

rows.sort(reverse=True)
best_wr, best_name, best_path = rows[0]
print(f"\nbest: {best_name} ({best_path})  team WR={best_wr:.1%} "
      f"vs expert-in-slot {expert_ref:.1%}")

if best_name != "current-routing":
    print(f"\n1v1 guard ({v1_games} games x 12 opps):")
    new_wr = one_v1_wr(load_net(best_path))
    old_wr = one_v1_wr(load_net(DEFAULT_ROUTING[arch]))
    n_tot = v1_games * 12
    sig = ((new_wr*(1-new_wr) + old_wr*(1-old_wr)) / n_tot) ** 0.5
    print(f"  new {new_wr:.1%}  vs current-routing {old_wr:.1%}  "
          f"(2-sigma ±{2*sig:.1%})")
    if new_wr >= old_wr - 2 * sig:
        print("  1v1 GUARD PASS — safe to route this checkpoint everywhere")
    else:
        print("  1v1 GUARD FAIL — use it for TEAM routing only")
