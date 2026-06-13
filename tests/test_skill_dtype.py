"""Skill damage-type schema guard (SKILL_FEATURE_DIM 53→66, 2026-06-12i).

The dtype tail of every skill row is DATA (SkillFeatures.damage_types) — this
test cross-checks it against the ENGINE's actual damage packets, the same way
the catalog enumeration probe derives ground truth: build each reachable
skill's action in a live env and extract its packet types. Any future ability
whose declared dtype drifts from what execute_action would deal fails here,
loudly, instead of silently feeding the policy a wrong matchup signal.
"""
from __future__ import annotations

import numpy as np
import pytest

from trpg.rl.env_v2 import CombatEnvV2
from trpg.engine.skill import (
    available_skills, action_damage_types, SKILL_FEATURE_DIM,
    SKILL_DTYPE_START, WEAPON_DTYPE,
)
from trpg.engine.damage import DAMAGE_TYPES, DAMAGE_TYPE_INDEX
from trpg.engine.abilities import ABILITY_REGISTRY
from trpg.scenarios.archetypes import STANDARD_ARCHETYPES
from trpg.scenarios.monsters import register_monsters, MONSTER_DEFS

LEVELS = (1, 3, 5, 8, 11)

# Damage dealt by status machinery (auras, marks) — the cast action itself
# carries no packet, so the catalog pins the tick's type and we assert the
# pin against the engine literal it mirrors (combat.py apply_damage calls).
STATUS_DEALT_DTYPES = {
    "spirit_guardians": {"光耀"},   # 3d8 aura tick, combat.environment aura
    "hunters_mark":     {"穿刺"},   # +1d6 on-hit proc, _resolve_single_attack
    # Mixed case (Wave 2): the swallow ACTION deals the 穿刺 bite packet; the
    # 強酸 share is the swallowed status's 6d6 tick (STATUS_TICK_DAMAGE in
    # tick_aura_damage) — status-dealt, so pinned here like the aura ticks.
    "swallow":          {"穿刺", "強酸"},
    # Wave 3 swallow variants — same chain, kraken 12d6 / tarrasque 16d6 tick.
    "kraken_swallow":    {"穿刺", "強酸"},
    "tarrasque_swallow": {"穿刺", "強酸"},
}


def _iter_reachable():
    """(skill, carrier_char, action, env) for every reachable skill."""
    register_monsters()
    pools = [(a, LEVELS) for a in STANDARD_ARCHETYPES]
    pools += [(m, (MONSTER_DEFS[m].natural_level,)) for m in sorted(MONSTER_DEFS)]
    for arch, levels in pools:
        for lvl in levels:
            env = CombatEnvV2(seed=11, n_agents=1, n_opps=1)
            env.reset(agent_archs=[arch], opp_archs=["orc"],
                      level=lvl, opp_level=3)
            aid, oid = env.agent_ids[0], env.opp_ids[0]
            a, o = env.ws.characters[aid], env.ws.characters[oid]
            for sk in available_skills(a, env.ws):
                try:
                    act = sk.build_action(aid, oid, (o.position.x, o.position.y))
                except Exception:
                    act = None
                yield sk, a, act


def test_dtype_matches_engine_packets():
    """Declared dtype ⊆ engine packet set; damaging rows are never untyped."""
    checked = 0
    for sk, carrier, act in _iter_reachable():
        f = sk.features
        toks = {tok for tok, _ in f.iter_damage_types()}
        assert WEAPON_DTYPE not in toks, (
            f"{sk.skill_id}: unresolved @weapon after materialize "
            f"(carrier {carrier.name})")
        if sk.skill_id in STATUS_DEALT_DTYPES:
            assert toks == STATUS_DEALT_DTYPES[sk.skill_id], (
                f"{sk.skill_id}: pinned status-tick dtype drifted: {toks}")
            checked += 1
            continue
        packets = action_damage_types(act, carrier)
        if f.expected_damage > 0 and packets:
            assert toks, (
                f"{sk.skill_id}: deals {packets} but declares no damage_types "
                f"(carrier {carrier.name})")
            assert toks <= packets, (
                f"{sk.skill_id}: declares {toks}, engine deals {packets} "
                f"(carrier {carrier.name})")
            checked += 1
        elif not packets:
            # Non-damaging action — a declared dtype would be phantom data.
            assert f.expected_damage > 0 or not toks, (
                f"{sk.skill_id}: declares {toks} but neither the action nor "
                f"expected_damage shows damage")
    assert checked > 40  # the sweep actually covered the catalog


def test_dtype_shares_are_normalized():
    """Damaging rows: shares in (0,1], summing to ≈1 (soft one-hot is a
    damage-share distribution, so the resist dot product reads as EV mult-1)."""
    for sk, carrier, act in _iter_reachable():
        pairs = list(sk.features.iter_damage_types())
        if not pairs:
            continue
        total = sum(s for _, s in pairs)
        assert 0.999 <= total <= 1.001, (
            f"{sk.skill_id}: dtype shares sum to {total}")
        for tok, share in pairs:
            assert tok in DAMAGE_TYPE_INDEX, f"{sk.skill_id}: bad dtype {tok!r}"
            assert 0.0 < share <= 1.0


def test_vector_tail_layout():
    """as_vector: dtype tail sits at [SKILL_DTYPE_START:], aligned to
    DAMAGE_TYPES, and the legacy 53-dim prefix is unchanged by dtype data."""
    from trpg.engine.skill import SkillFeatures
    f = SkillFeatures(expected_damage=5.0)
    base = f.as_vector()
    assert base.shape == (SKILL_FEATURE_DIM,)
    assert SKILL_FEATURE_DIM == SKILL_DTYPE_START + len(DAMAGE_TYPES)
    assert not base[SKILL_DTYPE_START:].any()

    g = SkillFeatures(expected_damage=5.0,
                      damage_types=(("火", 0.75), ("毒", 0.25)))
    v = g.as_vector()
    np.testing.assert_array_equal(v[:SKILL_DTYPE_START], base[:SKILL_DTYPE_START])
    assert v[SKILL_DTYPE_START + DAMAGE_TYPES.index("火")] == pytest.approx(0.75)
    assert v[SKILL_DTYPE_START + DAMAGE_TYPES.index("毒")] == pytest.approx(0.25)
    assert v[SKILL_DTYPE_START:].sum() == pytest.approx(1.0)


def test_weapon_rider_packets_split_share():
    """A weapon with typed on_hit dice gets damage-share-weighted bits."""
    from trpg.engine.skill import from_weapon
    from trpg.engine.items import WEAPON_DEFS
    register_monsters()
    riders = [w for w in WEAPON_DEFS.values()
              if (getattr(w, "on_hit", None) or {}).get("damage_dice")
              and (w.on_hit or {}).get("damage_type")]
    assert riders, "expected at least one rider weapon (wyvern stinger)"
    env = CombatEnvV2(seed=11, n_agents=1, n_opps=1)
    env.reset(agent_archs=["champion"], opp_archs=["orc"], level=5, opp_level=3)
    char = env.ws.characters[env.agent_ids[0]]
    for w in riders:
        # Aggregate by token — base and rider may share a type (火焰之觸:
        # 火 base + 1d10 火 rider), which as_vector likewise accumulates.
        pairs: dict[str, float] = {}
        for tok, share in from_weapon(w, char).features.iter_damage_types():
            pairs[tok] = pairs.get(tok, 0.0) + share
        assert set(pairs) == {w.damage_type, w.on_hit["damage_type"]}
        assert sum(pairs.values()) == pytest.approx(1.0)
        assert all(s > 0 for s in pairs.values())
