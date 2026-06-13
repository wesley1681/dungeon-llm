"""skill-dtype migration behaviour gate (MONSTER_CATALOG §6.6, 2026-06-12i).

Two arms, SAME process, SAME seeds, SAME checkpoint:
  new : production path — ckpt through the full adapter chain (… → skill
        dtype: skill_proj 53→66 zero-pad, heads +1 zero join column), fed
        live 66-wide obs.
  v1  : the pre-migration computation — ckpt at its on-disk pre-dtype shapes
        (chain stopped before the dtype adapter), modules rebuilt at the old
        widths, ``_dtype_era_v1`` forward flag (53-slice, no join concat).

Zero-padded columns contribute exactly +0.0 per product; the only residual
is BLAS summation-order drift across the wider rows (the v4 surgery measured
the same class at ≤1e-6 logits). This script measures whether that drift
EVER flips a decision: expected action disagreement ≈ 0, WR identical.

Usage: python scripts/diag_skill_dtype_shift.py <arch>|student [ckpt] [games_per_opp]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

from collections import Counter
import torch
import torch.nn as nn

from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.obs import N_ENTITY_SLOTS, ENTITY_DIM, END_FEATURES_DIM
from trpg.rl.model import (CombatPolicyNet, apply_resource_mask,
                           apply_entity_mask, pick_action, _SKILL_DIM_V1)
from trpg.engine.skill import available_skills

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_routed import DEFAULT_ROUTING, DEFAULT_HIDDEN, load_net, stable_seed


def build_v1_net(path: str, hidden: int, n_groups: int | None):
    """Load the ckpt at its pre-dtype shapes: obs chain v3→v4 only, modules
    rebuilt at the old widths, the v1 forward flag set."""
    sd = torch.load(path, map_location="cpu")
    sd = CombatPolicyNet.adapt_state_dict_for_obs_v3(sd)
    sd = CombatPolicyNet.adapt_state_dict_for_obs_v4(sd)
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
    net = CombatPolicyNet(hidden=hidden, n_head_groups=n_groups)
    net.skill_proj = nn.Linear(_SKILL_DIM_V1, 64)
    for lst in (net.skill_heads, net.entity_heads):
        for i in range(len(lst)):
            lst[i] = nn.Linear(64 + hidden + 32, 1)
    old_in = N_ENTITY_SLOTS * ENTITY_DIM + 4 + END_FEATURES_DIM + _SKILL_DIM_V1
    net.critic.net[0] = nn.Linear(old_in, net.critic.net[0].out_features)
    net._dtype_era_v1 = True
    net.load_state_dict(sd)   # strict — proves the old sd matches old shapes
    net.eval()
    return net


def make_actor(net, blind):
    def act(ot, env, actor):
        o = ot
        if blind:
            from distill_routed import blind_self_identity_t
            o = blind_self_identity_t({k: v.clone() for k, v in ot.items()})
        with torch.no_grad():
            el, s, e, g = net(o)
        s = apply_resource_mask(s, env.resources, env.ws, actor)
        e = apply_entity_mask(e, o, env.ws, actor)
        return list(pick_action(el[0], s[0], e[0], g[0],
                                ws=env.ws, agent_id=actor))
    return act


def play(arch, opp, seed, actor_fn, shadow_fn=None, disagree=None):
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
    ckpt = sys.argv[2] if len(sys.argv) > 2 else None
    gpo = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    if target == "student":
        from distill_routed import load_student
        path = ckpt or "models/pop_mon/pop_u0005.pt"
        net_new = load_student(path)
        net_v1 = build_v1_net(path, hidden=128, n_groups=1)
        arch_pool = ["battle_master", "evocation", "life", "assassin"]
        blind = True
    else:
        path = ckpt or DEFAULT_ROUTING[target]
        net_new = load_net(path, DEFAULT_HIDDEN[target])
        net_v1 = build_v1_net(path, DEFAULT_HIDDEN[target], n_groups=None)
        arch_pool = [target]
        blind = False

    act_new = make_actor(net_new, blind)
    act_v1 = make_actor(net_v1, blind)

    print(f"target={target} ckpt={path}  {gpo} games x {len(ARCHETYPE_LIST)} opps/arm")
    wins_v1 = wins_new = n = 0
    disagree = Counter()
    for arch in arch_pool:
        for opp in ARCHETYPE_LIST:
            base = stable_seed(f"diagdt_{arch}_{opp}")
            for i in range(gpo):
                wins_v1 += play(arch, opp, base + i, act_v1)
                wins_new += play(arch, opp, base + i, act_new,
                                 shadow_fn=act_v1, disagree=disagree)
                n += 1
    states = disagree.pop("__states__", 0)
    mismatches = sum(disagree.values())
    print(f"\nWR v1 = {wins_v1/n:.1%}   WR new = {wins_new/n:.1%}   (n={n}/arm)")
    print(f"action disagreement: {mismatches}/{states} states "
          f"({mismatches/max(1,states):.2%})")
    for (a, b), c in disagree.most_common(10):
        print(f"  new={a:<28s} v1={b:<28s} x{c}")


if __name__ == "__main__":
    main()
