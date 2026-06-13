"""验证-2: does the BLIND student actually USE the enemy capability descriptor?

Monsters read all-zero in the archetype one-hot, and the student blinds its OWN
one-hot, so the v4 descriptor is the ONLY channel carrying enemy identity. This
probe runs the same trained net over the same monster matchups twice:

  arm A (desc-on):   normal v4 obs.
  arm B (enemy-desc-zeroed): the descriptor columns of the ENEMY rows are zeroed
        every step (self descriptor kept — the student needs it to know its own
        kit). Any WR gap A−B is causally the enemy descriptor's contribution.

Same seeds in both arms, fair pairing levels (class @ round(equiv), monster @
natural_level). A positive A−B means the net reads the stranger's kit and plays
better for it; ~0 means the channel learned weights but isn't behaviourally used.

Usage:
  python scripts/probe_descriptor_ab.py models/pop_mon/pop_u0005.pt --games 4
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse
import numpy as np
import torch

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import (apply_resource_mask, apply_entity_mask, pick_action)
from trpg.rl.obs import ENEMY_SLOT_START, ENT_DESC_START, N_V4_DESC
from trpg.scenarios.monsters import (register_monsters, onev1_viable_monsters,
                                     EQUIV_LEVEL_1V1, MONSTER_DEFS)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from synth_identity import STANDARD_IDS
from eval_routed import stable_seed


def blind_self_one_hot(obs):
    """Zero the student's own archetype one-hot (entity 0 + end_features)."""
    from distill_routed import _ARCH_OH_START, _END_ARCH_START
    from trpg.rl.obs import N_ARCHETYPES
    out = dict(obs)
    ent = obs["entities"].copy()
    ent[0, _ARCH_OH_START:_ARCH_OH_START + N_ARCHETYPES] = 0.0
    out["entities"] = ent
    ef = obs["end_features"].copy()
    ef[_END_ARCH_START:_END_ARCH_START + N_ARCHETYPES] = 0.0
    out["end_features"] = ef
    return out


def zero_enemy_descriptor(obs):
    """Zero the v4 capability descriptor on every ENEMY row (self kept)."""
    out = dict(obs)
    ent = out["entities"].copy()
    ent[ENEMY_SLOT_START:, ENT_DESC_START:ENT_DESC_START + N_V4_DESC] = 0.0
    out["entities"] = ent
    return out


def play(net, agent, opp, seed, level, opp_level, kill_enemy_desc):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[agent], opp_archs=[opp],
                       level=level, opp_level=opp_level)
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    done = False
    while not done:
        actor = env.current_agent_id
        ob = blind_self_one_hot(obs)
        if kill_enemy_desc:
            ob = zero_enemy_descriptor(ob)
        ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob.items()}
        with torch.no_grad():
            el, s, e, g = net(ot)
        s = apply_resource_mask(s, env.resources, env.ws, actor)
        e = apply_entity_mask(e, ot, env.ws, actor)
        act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws,
                               agent_id=actor))
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    return (not env.ws.characters[oid].is_alive()
            and env.ws.characters[aid].is_alive())


def play_divergence(net, agent, opp, seed, level, opp_level):
    """Play ONE game under desc-on; at every agent decision, also compute the
    greedy action under enemy-desc-zeroed at the SAME state. Returns
    (n_decisions, n_action_diff, n_skill_diff). No compounding RNG — both views
    score the identical state, so this is the pure 'does the enemy descriptor
    flip the chosen action' rate."""
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[agent], opp_archs=[opp],
                       level=level, opp_level=opp_level)
    n = diff = skdiff = 0
    done = False
    while not done:
        actor = env.current_agent_id
        ob_on = blind_self_one_hot(obs)
        ob_off = zero_enemy_descriptor(ob_on)
        acts = {}
        for key, ob in (("on", ob_on), ("off", ob_off)):
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            acts[key] = tuple(pick_action(el[0], s[0], e[0], g[0],
                                          ws=env.ws, agent_id=actor))
        if actor in env.agent_ids:           # only the learner's own decisions
            n += 1
            if acts["on"] != acts["off"]:
                diff += 1
            if acts["on"][0] != acts["off"][0]:
                skdiff += 1
        obs, _, term, trunc, _ = env.step(list(acts["on"]))
        done = term or trunc
    return n, diff, skdiff


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=str)
    p.add_argument("--games", type=int, default=4)
    p.add_argument("--mon_max_level", type=float, default=8.0)
    p.add_argument("--divergence", action="store_true",
                   help="report action-flip rate (desc-on vs enemy-desc-off "
                        "at identical states) instead of the WR A/B")
    args = p.parse_args()

    register_monsters()
    net = load_student(args.ckpt); net.eval()
    mon_pool = onev1_viable_monsters(args.mon_max_level)
    print(f"ckpt={args.ckpt}  games/matchup={args.games}  "
          f"monsters={len(mon_pool)}\n", flush=True)

    if args.divergence:
        print(f"{'monster':16s} {'decisions':>9s} {'act-flip':>9s} "
              f"{'skill-flip':>10s}")
        T_n = T_d = T_s = 0
        rows = []
        for mon in mon_pool:
            lvl = max(1, round(EQUIV_LEVEL_1V1[mon]))
            m_lvl = MONSTER_DEFS[mon].natural_level
            n = d = s = 0
            for a in STANDARD_IDS:
                base = stable_seed(f"abdiv_{a}_{mon}")
                for k in range(args.games):
                    gn, gd, gs = play_divergence(net, a, mon, base + k,
                                                 lvl, m_lvl)
                    n += gn; d += gd; s += gs
            rows.append((mon, n, d, s))
            T_n += n; T_d += d; T_s += s
        rows.sort(key=lambda r: (r[2] / max(1, r[1])), reverse=True)
        for mon, n, d, s in rows:
            print(f"{mon:16s} {n:9d} {d/max(1,n):9.1%} {s/max(1,n):10.1%}")
        print("-" * 48)
        print(f"{'MEAN':16s} {T_n:9d} {T_d/max(1,T_n):9.1%} "
              f"{T_s/max(1,T_n):10.1%}")
        return

    tot_a = tot_b = tot_n = 0
    rows = []
    for mon in mon_pool:
        lvl = max(1, round(EQUIV_LEVEL_1V1[mon]))
        m_lvl = MONSTER_DEFS[mon].natural_level
        wa = wb = n = 0
        for a in STANDARD_IDS:
            base = stable_seed(f"abdesc_{a}_{mon}")
            for k in range(args.games):
                seed = base + k
                wa += int(play(net, a, mon, seed, lvl, m_lvl, False))
                wb += int(play(net, a, mon, seed, lvl, m_lvl, True))
                n += 1
        rows.append((mon, wa / n, wb / n, (wa - wb) / n))
        tot_a += wa; tot_b += wb; tot_n += n
    rows.sort(key=lambda r: r[3], reverse=True)
    print(f"{'monster':16s} {'desc-on':>8s} {'desc-off':>9s} {'Δ(on-off)':>10s}")
    for mon, a, b, d in rows:
        print(f"{mon:16s} {a:8.1%} {b:9.1%} {d:+10.1%}")
    print("-" * 46)
    print(f"{'MEAN':16s} {tot_a/tot_n:8.1%} {tot_b/tot_n:9.1%} "
          f"{(tot_a-tot_b)/tot_n:+10.1%}  (n={tot_n}/arm)")


if __name__ == "__main__":
    main()
