"""Pick a support-slot checkpoint from a replay-mixed BC run.

Selection rule: the LATEST epoch that passes the 1v1 guard (later epochs fit
the team demos more; earlier ones sit closer to the 1v1-strong warm start).
Then sanity-check the winner's team WR (big sample) against current routing.

Raw team WR cannot rank these candidates: healing was measured WR-neutral
(diag_coop_value), so the cooperation behaviour itself is selected by
trace_coop afterwards — this script only enforces the two no-regression
gates (1v1 guard, team WR).

Usage: python scripts/pick_support_v2.py <arch> <snap_dir> [v1_games] [team_games]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path

import torch

from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.scenarios.archetypes import ARCHETYPE_ROLES

from eval_routed import DEFAULT_ROUTING, load_net, stable_seed
from train_team_ppo import eval_team_slot, ROLE_SLOT

arch = sys.argv[1]
snap_dir = Path(sys.argv[2])
v1_games = int(sys.argv[3]) if len(sys.argv) > 3 else 60
team_games = int(sys.argv[4]) if len(sys.argv) > 4 else 8

slot = ROLE_SLOT[ARCHETYPE_ROLES[arch]]


def one_v1_wr(net) -> float:
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


old_path = DEFAULT_ROUTING[arch]
print(f"=== {arch} (slot={slot}) — 1v1 guard baseline: {old_path}")
old_wr = one_v1_wr(load_net(old_path))
n_tot = v1_games * 12
print(f"  old 1v1 WR = {old_wr:.1%} ({n_tot} games)", flush=True)

chosen = None
for p in sorted(snap_dir.glob("bc_e*.pt"), reverse=True):
    net = load_net(str(p))
    wr = one_v1_wr(net)
    sig = ((wr * (1 - wr) + old_wr * (1 - old_wr)) / n_tot) ** 0.5
    ok = wr >= old_wr - 2 * sig
    print(f"  {p.name}: 1v1 {wr:.1%} vs {old_wr:.1%} (2σ ±{2*sig:.1%}) "
          f"→ {'PASS' if ok else 'FAIL'}", flush=True)
    if ok:
        chosen = (str(p), net)
        break

if chosen is None:
    print("\nNO epoch passes the 1v1 guard — keep current routing.")
    sys.exit(0)

path, net = chosen
cache = {}
mate_nets = {}
for a, pth in DEFAULT_ROUTING.items():
    if a == arch:
        continue
    if pth not in cache:
        cache[pth] = load_net(pth)
    mate_nets[a] = cache[pth]

print(f"\nteam WR check ({team_games} games/cell):")
new_team = eval_team_slot(net, arch, slot, mate_nets, games=team_games)
cur_team = eval_team_slot(load_net(old_path), arch, slot, mate_nets,
                          games=team_games)
print(f"  candidate {Path(path).name}: team WR={new_team:.1%}  "
      f"current-routing: {cur_team:.1%}")
print(f"\nCHOSEN: {path}")
