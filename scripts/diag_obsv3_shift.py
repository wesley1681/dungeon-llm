"""Diagnose the berserker WR drop after the obs-v3 slot widening.

Hypothesis: forward()'s skill→entity attention softmaxes over ALL entity
slots without masking padding rows. A zero row through entity_mlp is the
bias vector (non-zero), so widening 6 → 10 slots grew the padding mass in
the softmax from 4 rows to 8 and shifted sk_ent_ctx → skill logits.

Two arms, SAME process, SAME seeds, SAME checkpoint weights:
  new : migrated net, v3 obs (current production path)
  old : un-migrated net built under monkeypatched legacy constants
        (ENTITY_DIM-6, 6 slots), fed a legacy VIEW of the v3 obs
        (rows [0,1,2,4,5,6], first old-dim columns) — bit-exact
        pre-migration inference; its entity picks are remapped to v3
        slots before env.step.
Reports per-arm WR plus per-state action agreement measured on identical
states (the new arm's trajectory states, both nets queried).

Usage: python scripts/diag_obsv3_shift.py <arch> [games_per_opp]
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
from trpg.rl.obs import N_V3_EXTRA, ENTITY_DIM, N_ENTITY_SLOTS, ENEMY_SLOT_START
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.engine.skill import available_skills

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_routed import DEFAULT_ROUTING, load_net, stable_seed

OLD_DIM   = M._ENTITY_DIM_V2   # frozen pre-v3 width (v4 broke the old
                               # `ENTITY_DIM - N_V3_EXTRA` arithmetic)
OLD_SLOTS = M.CombatPolicyNet._LEGACY_N_SLOTS
# legacy slot s -> v3 slot (self/allies keep index, enemies shift)
LEGACY_TO_V3 = [s if s <= M.CombatPolicyNet._LEGACY_N_ALLY
                else ENEMY_SLOT_START + (s - M.CombatPolicyNet._LEGACY_N_ALLY - 1)
                for s in range(OLD_SLOTS)]
V3_ROWS = LEGACY_TO_V3   # rows of the v3 entities matrix visible pre-v3


class _LegacyConsts:
    """Temporarily restore the pre-v3 obs constants inside model.py."""
    def __enter__(self):
        self.saved = (M.ENTITY_DIM, M.N_ENTITY_SLOTS)
        M.ENTITY_DIM, M.N_ENTITY_SLOTS = OLD_DIM, OLD_SLOTS
    def __exit__(self, *a):
        M.ENTITY_DIM, M.N_ENTITY_SLOTS = self.saved


def build_legacy_net(path: str) -> "M.CombatPolicyNet":
    sd = torch.load(path, map_location="cpu")
    # per-arch tiling only — NO v3 migration (we want the original shapes)
    mappings = {"skill_head": "skill_heads", "entity_head": "entity_heads",
                "grid_query_proj": "grid_query_projs"}
    from trpg.rl.obs import N_ARCHETYPES
    for ob, nb in mappings.items():
        for suf in ("weight", "bias"):
            k = f"{ob}.{suf}"
            if k in sd:
                v = sd.pop(k)
                for i in range(N_ARCHETYPES):
                    sd[f"{nb}.{i}.{suf}"] = v.clone()
    with _LegacyConsts():
        net = M.CombatPolicyNet(hidden=128)
    msd = net.state_dict()
    sd = {k: v for k, v in sd.items()
          if not (k in msd and v.shape != msd[k].shape)}
    net.load_state_dict(sd, strict=False)
    net.legacy_unmasked_attn = True   # genuine pre-migration softmax (pads in)
    net.eval()
    return net


def legacy_view(ot: dict) -> dict:
    out = dict(ot)
    out["entities"] = ot["entities"][:, V3_ROWS, :OLD_DIM]
    return out


def act_old(net_old, ot, env, actor):
    with _LegacyConsts():
        lot = legacy_view(ot)
        with torch.no_grad():
            el, s, e, g = net_old(lot)
        s = apply_resource_mask(s, env.resources, env.ws, actor)
        e = apply_entity_mask(e, lot, env.ws, actor)
        act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
    act[1] = LEGACY_TO_V3[act[1]] if act[1] < OLD_SLOTS else act[1]
    return act


def act_new(net_new, ot, env, actor):
    with torch.no_grad():
        el, s, e, g = net_new(ot)
    s = apply_resource_mask(s, env.resources, env.ws, actor)
    e = apply_entity_mask(e, ot, env.ws, actor)
    return list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))


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
                    return (sks[a[0]].skill_id if a[0] < len(sks) else "end") \
                        + ("" if a == [0, 0, 0] else f"/e{a[1]}")
                disagree[(nm(act), nm(alt))] += 1
            disagree["__states__"] += 1
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    return (not env.ws.characters[oid].is_alive()
            and env.ws.characters[aid].is_alive())


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else "berserker"
    gpo = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    path = DEFAULT_ROUTING[arch]
    net_new = load_net(path)
    net_old = build_legacy_net(path)

    print(f"arch={arch} ckpt={path}  {gpo} games x {len(ARCHETYPE_LIST)} opps/arm")
    wins_old = wins_new = n = 0
    disagree = Counter()
    for opp in ARCHETYPE_LIST:
        base = stable_seed(f"diagv3_{arch}_{opp}")
        for i in range(gpo):
            wins_old += play(arch, opp, base + i,
                             lambda ot, env, a: act_old(net_old, ot, env, a))
            wins_new += play(arch, opp, base + i,
                             lambda ot, env, a: act_new(net_new, ot, env, a),
                             shadow_fn=lambda ot, env, a: act_old(net_old, ot, env, a),
                             disagree=disagree)
            n += 1
    states = disagree.pop("__states__", 0)
    mismatches = sum(disagree.values())
    print(f"\nWR old-view = {wins_old/n:.1%}   WR new-view = {wins_new/n:.1%}   (n={n}/arm)")
    print(f"action disagreement: {mismatches}/{states} states ({mismatches/max(1,states):.1%})")
    for (a, b), c in disagree.most_common(12):
        print(f"  new={a:<28s} old={b:<28s} x{c}")


if __name__ == "__main__":
    main()
