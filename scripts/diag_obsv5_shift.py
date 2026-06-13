"""obs v5 migration behaviour gate (2026-06-13, passive-trait descriptor).

Same process, same seeds, same checkpoint, two arms:
  v5 : production path — ckpt through the FULL adapter chain (…→v5: N_V5_TRAIT
       trait columns zero-padded), fed live v5 obs.
  v4 : pre-migration computation — ckpt at its on-disk v4 shapes (chain stopped
       before v5), fed a v4 VIEW of the v5 obs (the trait tail sliced off; NO
       rescale — v5 was a pure strict append).

v5 added zero NORM changes, so the migration is exact by construction (the new
columns meet only zero weights). This script confirms it end-to-end: expected
action disagreement ≈ 0 on real games, WR identical per arm. Tests TROLL and
WOLF first on purpose — their trait columns are NON-zero (regen / pack tactics),
so a leak in the zero-pad would surface as a decision flip here.

Usage: python scripts/diag_obsv5_shift.py [games_per_opp]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

from collections import Counter
import torch

import trpg.rl.model as M
from trpg.scenarios.monsters import register_monsters
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.model import (CombatPolicyNet, apply_resource_mask,
                           apply_entity_mask, pick_action, _ENTITY_DIM_V4)
from trpg.engine.skill import available_skills

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student, blind_self_identity_t

CKPT = "models/mon_actor1/ma_u0032.pt"
AGENTS = ["troll", "wolf", "battle_master", "evocation", "life"]


class _V4Consts:
    def __enter__(self):
        self.saved = M.ENTITY_DIM
        M.ENTITY_DIM = _ENTITY_DIM_V4
    def __exit__(self, *a):
        M.ENTITY_DIM = self.saved


def build_v4_net(path: str):
    """Load the student ckpt at its on-disk v4 shapes: full chain EXCEPT the v5
    step (the champion is already v4-entity + 66-skill, so v3/v4/skill_dtype are
    all passthrough → raw sd into a v4-width net)."""
    sd = torch.load(path, map_location="cpu")
    sd = CombatPolicyNet.adapt_state_dict_for_skill_dtype(
        CombatPolicyNet.adapt_state_dict_for_obs_v4(
            CombatPolicyNet.adapt_state_dict_for_obs_v3(sd)))
    with _V4Consts():
        net = CombatPolicyNet(hidden=128, n_head_groups=1)
    net.load_state_dict(sd)
    net.eval()
    return net


def v4_view(ot: dict) -> dict:
    out = dict(ot)
    out["entities"] = ot["entities"][:, :, :_ENTITY_DIM_V4].clone()
    return out


def make_actor(net, view):
    def act(ot, env, actor):
        o = view(ot) if view else ot
        o = blind_self_identity_t({k: v.clone() for k, v in o.items()})
        with torch.no_grad():
            el, s, e, g = net(o)
        s = apply_resource_mask(s, env.resources, env.ws, actor)
        e = apply_entity_mask(e, o, env.ws, actor)
        return list(pick_action(el[0], s[0], e[0], g[0],
                                ws=env.ws, agent_id=actor))
    return act


def play(arch, opp, seed, actor_fn, shadow_fn=None, disagree=None):
    import random
    random.seed(seed)          # global engine dice — both arms face same stream
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[arch], opp_archs=[opp], level=5)
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    done = False
    while not done:
        actor = env.current_agent_id
        ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
        act = actor_fn(ot, env, actor)
        if shadow_fn is not None:
            alt = shadow_fn(ot, env, actor)
            if alt != act:
                sks = available_skills(env.ws.characters[actor], env.ws)
                nm = lambda a: (sks[a[0]].skill_id if a[0] < len(sks) else "end")
                disagree[(nm(act), nm(alt))] += 1
            disagree["__states__"] += 1
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    return (not env.ws.characters[oid].is_alive()
            and env.ws.characters[aid].is_alive())


def main():
    gpo = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    net_v5 = load_student(CKPT)
    net_v4 = build_v4_net(CKPT)
    act_v5 = make_actor(net_v5, None)
    act_v4 = make_actor(net_v4, v4_view)

    print(f"ckpt={CKPT}  {gpo} games × {len(ARCHETYPE_LIST)} opps/arm")
    print(f"agents (traited first): {AGENTS}")
    wins_v4 = wins_v5 = n = 0
    disagree = Counter()
    from eval_routed import stable_seed
    for arch in AGENTS:
        for opp in ARCHETYPE_LIST:
            base = stable_seed(f"diagv5_{arch}_{opp}")
            for i in range(gpo):
                wins_v4 += play(arch, opp, base + i, act_v4)
                wins_v5 += play(arch, opp, base + i, act_v5,
                                shadow_fn=act_v4, disagree=disagree)
                n += 1
    states = disagree.pop("__states__", 0)
    mismatches = sum(disagree.values())
    print(f"\nWR v4-view = {wins_v4/n:.1%}   WR v5 = {wins_v5/n:.1%}   (n={n}/arm)")
    print(f"action disagreement: {mismatches}/{states} states "
          f"({mismatches/max(1,states):.2%})")
    for (a, b), c in disagree.most_common(10):
        print(f"  v5={a:<24s} v4={b:<24s} x{c}")
    print("PASS (bit-exact migration)" if mismatches == 0
          else "!! non-zero disagreement — investigate the zero-pad")


if __name__ == "__main__":
    main()
