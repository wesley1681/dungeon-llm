"""Does melee EVER discover a flank under exploration (sampling)?

If the policy, sampling actions (not argmax), occasionally walks around the
pillar and regains LoS to a STATIONARY enemy, then training against
hold-position opponents would reinforce that success -> flanking is learnable
by curriculum. If it essentially never does even when sampling, exploration is
too weak and we also need a grid-entropy boost. Also contrasts a MOVING expert
opponent (the current train condition): there LoS returns because the ENEMY
walks around -> the attack reward credits WAITING, not flanking.

Reports, per identity, over N sampled episodes:
    stat:regain%   vs STATIONARY enemy, episodes where agent regained LoS
    stat:byMove%   of those, fraction where the AGENT moved to regain (true flank)
    move:enemyLoS% vs MOVING expert, episodes where LoS returned via ENEMY move
                   (= free reward for waiting, the anti-signal)
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import random
from types import SimpleNamespace
from zlib import crc32
import torch

from trpg.scenarios.monsters import register_monsters
register_monsters()
from trpg.engine.vec2 import Vec2, TerrainType
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.train_ppo import _sample_action

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single


class Stationary:
    def decide(self, *a, **k):
        return SimpleNamespace(action=None, fled=False, ended=True)


def episode(net, ident, ep_key, stationary, max_turns=20):
    k = crc32(ep_key.encode()); random.seed(k)
    env = CombatEnvV2(seed=k ^ 0x33, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[ident], opp_archs=["champion"],
                       level=5, opp_level=5, layout="open")
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    bf = env.ws.combat.battlefield
    W, H = bf.width, bf.height
    nx = len(bf.cells[0]); ny = len(bf.cells)
    bf.cells = [[int(TerrainType.NORMAL)] * nx for _ in range(ny)]
    bf.add_rect_obstacle(W*0.5-0.75, H*0.5-3.0, W*0.5+0.75, H*0.5+3.0)
    ag = env.ws.characters[aid]; en = env.ws.characters[oid]
    ag.position = Vec2(W*0.20, H*0.5); en.position = Vec2(W*0.80, H*0.5)
    if stationary:
        env._opp_policies[oid] = Stationary()
    start_y = ag.position.y
    regained = False; agent_moved_at_regain = False
    # The opponent is played INSIDE env.step (current_agent_id stays the agent
    # in 1v1), so there is no turn boundary to key on — check LoS every step.
    steps = 0; done = False
    while not done and steps < max_turns * 6:
        steps += 1
        los = bf.has_line_of_sight(ag.position, en.position)
        if los and not regained:
            regained = True
            # a true flank moved laterally off the blocked sightline (the enemy
            # is stationary, so any LoS we get we earned by repositioning)
            agent_moved_at_regain = abs(ag.position.y - start_y) > 1.0
        actor = env.current_agent_id
        ob = blind_np_single(obs)
        ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
        with torch.no_grad():
            action, *_ = _sample_action(net, ot, env.resources, env.ws, actor)
        obs, _, term, trunc, _ = env.step(action.numpy().tolist())
        done = term or trunc
    return regained, agent_moved_at_regain


def main():
    ck = sys.argv[1] if len(sys.argv) > 1 else "models/pop_mon/pop_u0005.pt"
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    net = load_student(ck); net.eval()
    idents = ["assassin", "champion", "war", "battle_master", "evocation"]
    print(f"ckpt={ck}  games={games} (SAMPLED actions)\n")
    print(f"{'identity':<14} {'stat:regain%':>13} {'stat:byMove%':>13} "
          f"{'move:enemyLoS%':>15}")
    for ident in idents:
        sr = sm = 0
        for g in range(games):
            r, m = episode(net, ident, f"s|{ident}|{g}", stationary=True)
            sr += r; sm += (r and m)
        # moving-expert: LoS returned but agent did NOT move => enemy gave it
        me = 0
        for g in range(games):
            r, m = episode(net, ident, f"m|{ident}|{g}", stationary=False)
            me += (r and not m)
        regp = sr / games
        bym = (sm / sr) if sr else float("nan")
        print(f"{ident:<14} {regp:>13.0%} {bym:>13.0%} {me/games:>15.0%}",
              flush=True)


if __name__ == "__main__":
    main()
