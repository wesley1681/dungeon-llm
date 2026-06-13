"""obs v4 migration behaviour gate (MONSTER_CATALOG §4, 2026-06-12).

Two arms, SAME process, SAME seeds, SAME checkpoint:
  v4 : the production path — ckpt loaded through the full adapter chain
       (v3→v4: descriptor columns zero-padded, level/max_hp weight columns
       scaled ×2/×8), fed v4 obs.
  v3 : the pre-migration computation — ckpt at its on-disk v3 shapes
       (chain stopped before the v4 adapter), fed a v3 VIEW of the v4 obs
       (descriptor columns sliced off; level/max_hp features rescaled back
       ×2/×8 — power-of-2, exact).

The migration math is exact per-product (test_obs_v4_rescale_compensation_
exact); the only residual is BLAS summation-order drift (≤1e-6 logits).
This script measures whether that drift EVER flips a decision: expected
action disagreement ≈ 0, WR identical per arm.

Usage: python scripts/diag_obsv4_shift.py <arch>|student [games_per_opp]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

from collections import Counter
import torch

import trpg.rl.model as M
from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.obs import (ENTITY_DIM, N_V4_DESC, I_ENT_LEVEL, I_ENT_MAXHP,
                         LEVEL_NORM, MAXHP_NORM, V3_LEVEL_NORM, V3_MAXHP_NORM)
from trpg.rl.model import (CombatPolicyNet, apply_resource_mask,
                           apply_entity_mask, pick_action, _ENTITY_DIM_V3)
from trpg.engine.skill import available_skills

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_routed import DEFAULT_ROUTING, DEFAULT_HIDDEN, load_net, stable_seed

LVL_S = LEVEL_NORM / V3_LEVEL_NORM    # 2.0
HP_S = MAXHP_NORM / V3_MAXHP_NORM     # 8.0


class _V3Consts:
    def __enter__(self):
        self.saved = M.ENTITY_DIM
        M.ENTITY_DIM = _ENTITY_DIM_V3
    def __exit__(self, *a):
        M.ENTITY_DIM = self.saved


def build_v3_net(path: str, hidden: int, n_groups: int | None):
    """Load the ckpt at its v3 shapes: v3 adapter + per-arch tiling only."""
    sd = torch.load(path, map_location="cpu")
    sd = CombatPolicyNet.adapt_state_dict_for_obs_v3(sd)
    if n_groups is None:    # routed ckpt — tile shared heads if pre-perarch
        from trpg.rl.obs import N_ARCHETYPES
        for ob, nb in (("skill_head", "skill_heads"),
                       ("entity_head", "entity_heads"),
                       ("grid_query_proj", "grid_query_projs")):
            for suf in ("weight", "bias"):
                k = f"{ob}.{suf}"
                if k in sd:
                    v = sd.pop(k)
                    for i in range(N_ARCHETYPES):
                        sd[f"{nb}.{i}.{suf}"] = v.clone()
    with _V3Consts():
        net = CombatPolicyNet(hidden=hidden,
                              n_head_groups=n_groups)
    net.load_state_dict(sd)
    net.eval()
    return net


def v3_view(ot: dict) -> dict:
    out = dict(ot)
    ent = ot["entities"][:, :, :_ENTITY_DIM_V3].clone()
    ent[:, :, I_ENT_LEVEL] *= LVL_S
    ent[:, :, I_ENT_MAXHP] *= HP_S
    out["entities"] = ent
    return out


def make_actor(net, view):
    def act(ot, env, actor, blind):
        o = view(ot) if view else ot
        if blind:
            from distill_routed import blind_self_identity_t
            o = blind_self_identity_t({k: v.clone() for k, v in o.items()})
        with torch.no_grad():
            el, s, e, g = net(o)
        s = apply_resource_mask(s, env.resources, env.ws, actor)
        e = apply_entity_mask(e, o, env.ws, actor)
        return list(pick_action(el[0], s[0], e[0], g[0],
                                ws=env.ws, agent_id=actor))
    return act


def play(arch, opp, seed, actor_fn, blind, shadow_fn=None, disagree=None):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[arch], opp_archs=[opp], level=5)
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    done = False
    while not done:
        actor = env.current_agent_id
        ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
        act = actor_fn(ot, env, actor, blind)
        if shadow_fn is not None:
            alt = shadow_fn(ot, env, actor, blind)
            if alt != act:
                sks = available_skills(env.ws.characters[actor], env.ws)
                def nm(a):
                    return (sks[a[0]].skill_id if a[0] < len(sks) else "end")
                disagree[(nm(act), nm(alt))] += 1
            disagree["__states__"] += 1
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    return (not env.ws.characters[oid].is_alive()
            and env.ws.characters[aid].is_alive())


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "berserker"
    gpo = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    if target == "student":
        from distill_routed import load_student
        path = "models/pop_b2/pop_u0005.pt"
        net_v4 = load_student(path)
        net_v3 = build_v3_net(path, hidden=128, n_groups=1)
        arch_pool = ["battle_master", "evocation", "life", "assassin"]
        blind = True
    else:
        path = DEFAULT_ROUTING[target]
        net_v4 = load_net(path, DEFAULT_HIDDEN[target])
        net_v3 = build_v3_net(path, DEFAULT_HIDDEN[target], n_groups=None)
        arch_pool = [target]
        blind = False

    act_v4 = make_actor(net_v4, None)
    act_v3 = make_actor(net_v3, v3_view)

    print(f"target={target} ckpt={path}  {gpo} games x {len(ARCHETYPE_LIST)} opps/arm")
    wins_v3 = wins_v4 = n = 0
    disagree = Counter()
    for arch in arch_pool:
        for opp in ARCHETYPE_LIST:
            base = stable_seed(f"diagv4_{arch}_{opp}")
            for i in range(gpo):
                wins_v3 += play(arch, opp, base + i, act_v3, blind)
                wins_v4 += play(arch, opp, base + i, act_v4, blind,
                                shadow_fn=act_v3, disagree=disagree)
                n += 1
    states = disagree.pop("__states__", 0)
    mismatches = sum(disagree.values())
    print(f"\nWR v3-view = {wins_v3/n:.1%}   WR v4 = {wins_v4/n:.1%}   (n={n}/arm)")
    print(f"action disagreement: {mismatches}/{states} states "
          f"({mismatches/max(1,states):.2%})")
    for (a, b), c in disagree.most_common(10):
        print(f"  v4={a:<28s} v3={b:<28s} x{c}")


if __name__ == "__main__":
    main()
