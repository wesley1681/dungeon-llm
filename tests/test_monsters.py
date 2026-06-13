"""Wave 0 monster registry tests (MONSTER_CATALOG.md §5).

Covers the three Wave 0 contracts:
  1. registering monsters NEVER shifts obs dims (N_ARCHETYPES frozen at 12)
  2. statblock fidelity — engine-derived to-hit / DC / HP / AC reproduce MM
  3. the pipeline runs — every monster plays a full env fight via
     GenericMonsterPolicy without errors, and reads as all-zero archetype
     one-hot in the enemy obs rows (the monster condition)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

import trpg.rl.obs as obs_mod
from trpg.scenarios.monsters import (
    MONSTER_DEFS, register_monsters, make_monster,
)

register_monsters()

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.action import encode_action
from trpg.engine.combat_policy import (
    make_archetype_policy, GenericMonsterPolicy,
)


# ── 1. Registration safety ───────────────────────────────────────────────────

def test_register_keeps_obs_frozen():
    assert obs_mod.N_ARCHETYPES == 12
    for mid in MONSTER_DEFS:
        assert mid not in obs_mod.ARCHETYPE_OBS_LIST


def test_register_idempotent():
    ids1 = register_monsters()
    ids2 = register_monsters()
    assert ids1 == ids2 == tuple(MONSTER_DEFS)


def test_monster_policy_routed():
    for mid in MONSTER_DEFS:
        assert isinstance(make_archetype_policy(mid), GenericMonsterPolicy)


def test_standard_panels_exclude_monsters():
    """Regression: ARCHETYPE_LIST / ARCHETYPE_OBS_LIST snapshot the frozen
    STANDARD_ARCHETYPES constant, so they stay the standard 12 even though
    this test module registered monsters BEFORE importing trpg.rl.env_v2.
    (First caught live: calibrate_cr's '12-expert panel' silently became a
    23-member panel including the monsters themselves.)"""
    from trpg.rl.env_v2 import ARCHETYPE_LIST
    assert len(ARCHETYPE_LIST) == 12
    assert not set(MONSTER_DEFS) & set(ARCHETYPE_LIST)
    assert not set(MONSTER_DEFS) & set(obs_mod.ARCHETYPE_OBS_LIST)


# ── 2. Statblock fidelity vs 5e MM ───────────────────────────────────────────
# to-hit = stat_mod + proficiency; natural_level reproduces MM prof exactly.

@pytest.mark.parametrize("mid,stat,to_hit,hp,ac", [
    ("commoner",   "STR", +2, 4,   10),
    ("goblin",     "DEX", +4, 7,   15),    # finesse 彎刀 → DEX
    ("orc",        "STR", +5, 15,  13),
    ("ogre",       "STR", +6, 59,  11),
    ("owlbear",    "STR", +7, 59,  13),
    ("manticore",  "STR", +5, 68,  14),
    ("ettin",      "STR", +7, 85,  12),
    ("hill_giant", "STR", +8, 105, 13),
    # Wave 1
    ("kobold",         "DEX", +4,  5,   12),
    ("wolf",           "DEX", +4,  11,  13),   # finesse 狼牙
    ("dire_wolf",      "STR", +5,  37,  14),
    ("skeleton",       "DEX", +4,  13,  13),
    ("zombie",         "STR", +3,  22,  8),
    ("ghoul",          "DEX", +4,  22,  12),
    ("shadow",         "DEX", +4,  16,  12),
    ("wight",          "STR", +4,  45,  14),
    ("gargoyle",       "STR", +4,  52,  15),
    ("wyvern",         "STR", +7,  110, 13),
    ("fire_elemental", "DEX", +6,  102, 13),
    ("young_red_dragon", "STR", +10, 178, 18),
    # Wave 2
    ("troll",          "STR", +7,  84,  15),
    ("basilisk",       "STR", +5,  52,  15),
    ("behir",          "STR", +10, 168, 17),
    ("beholder",       "STR", +5,  180, 18),   # bite +5 = STR 0 + prof 5 (MM)
])
def test_martial_statblocks(mid, stat, to_hit, hp, ac):
    c = make_monster(mid)
    assert c.stats.modifier(stat) + c.proficiency_bonus == to_hit
    assert c.max_hp == hp
    assert c.ac == ac
    assert c.level == MONSTER_DEFS[mid].natural_level


@pytest.mark.parametrize("mid,dc,hp", [
    ("mage_npc", 14, 40),
    ("archmage", 17, 99),
])
def test_caster_statblocks(mid, dc, hp):
    c = make_monster(mid)
    save_dc = 8 + c.proficiency_bonus + c.stats.modifier(c.spellcasting_ability)
    assert save_dc == dc
    assert c.max_hp == hp
    assert c.spell_slots, "caster monster must come with slots"


def test_multiattack_counts():
    assert make_monster("owlbear").attacks_per_action == 2
    assert make_monster("manticore").attacks_per_action == 3
    assert make_monster("hill_giant").attacks_per_action == 2
    assert make_monster("goblin").attacks_per_action == 1


def test_monster_hp_ignores_level():
    """A monster's HP is its statblock — whatever level the env passes."""
    from trpg.scenarios.archetypes import make_character
    assert make_character("ogre", level=1).max_hp == 59
    assert make_character("ogre", level=8).max_hp == 59


# ── 3. Pipeline smoke: full env fights ───────────────────────────────────────

def _play_vs_monster(mid: str, seed: int) -> str:
    """battle_master expert (agent side) vs monster (env-driven opp side).
    Returns 'agent' / 'opp' / 'tie' and asserts the episode terminates."""
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    obs, _ = env.reset(
        agent_archs=["battle_master"], opp_archs=[mid],
        level=5, opp_level=MONSTER_DEFS[mid].natural_level)
    aid, oid = env.agent_ids[0], env.opp_ids[0]
    expert = make_archetype_policy("battle_master")
    done = False
    steps = 0
    while not done:
        actor = env.current_agent_id
        a = env.ws.characters[actor]
        dec = expert.decide(actor, a, env.ws, env.resources,
                            env.ws.combat.round_number)
        act = ([0, 0, 0] if (dec.action is None or dec.fled)
               else list(encode_action(dec.action, env.ws, actor)))
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
        steps += 1
        assert steps < 2000, f"{mid}: episode did not terminate"
    if not env.ws.characters[oid].is_alive():
        return "agent"
    if not env.ws.characters[aid].is_alive():
        return "opp"
    return "tie"


@pytest.mark.parametrize("mid", sorted(MONSTER_DEFS))
def test_full_fight_completes(mid):
    for seed in (0, 1):
        _play_vs_monster(mid, seed)


def test_big_monsters_can_win():
    """Engine sanity: a CR 5 brute must sometimes beat a L5 expert.
    (If the monster never lands anything, the assembly is broken.)"""
    wins = sum(_play_vs_monster("hill_giant", s) == "opp" for s in range(10))
    assert wins > 0


def test_monster_enemy_row_onehot_zero():
    env = CombatEnvV2(seed=3, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=["battle_master"], opp_archs=["ogre"],
                       level=5, opp_level=2)
    row = obs["entities"][obs_mod.ENEMY_SLOT_START]
    arch_bits = row[7:7 + obs_mod.N_ARCHETYPES]
    assert float(np.abs(arch_bits).sum()) == 0.0
    assert row[0] > 0          # hp fraction populated — the row is live


# ── Wave 1 primitive unit tests ──────────────────────────────────────────────

from unittest.mock import patch
from trpg.engine.combat import apply_damage, execute_action
from trpg.engine.status import Poisoned, Prone
from trpg.engine.skill import available_skills


def test_typed_resist_table():
    sk = make_monster("skeleton")
    base = sk.hp                                         # 13
    dealt = apply_damage(sk, 4, dtype="鈍擊")            # vulnerable ×2
    assert dealt == 8 and sk.hp == base - 8
    sk.hp = base
    assert apply_damage(sk, 10, dtype="毒") == 0         # immune
    assert sk.hp == base
    assert apply_damage(sk, 5, dtype="斬擊") == 5        # neutral


def test_typed_absorb_heals():
    fe = make_monster("fire_elemental")
    fe.damage_multipliers["火"] = -1.0   # absorb variant (Iron Golem pattern)
    fe.hp = 50
    dealt = apply_damage(fe, 20, dtype="火")
    assert dealt == 0 and fe.hp == 70


def test_undead_fortitude_save_keeps_one_hp():
    z = make_monster("zombie")
    z.is_npc = True            # monster death rules (env sets this for opps)
    z.hp = 5
    with patch("trpg.engine.combat.make_saving_throw", return_value=(True, 20)):
        apply_damage(z, 10, dtype="斬擊")
    assert z.hp == 1 and z.is_alive()
    # radiant ignores fortitude entirely
    z2 = make_monster("zombie")
    z2.is_npc = True
    z2.hp = 5
    with patch("trpg.engine.combat.make_saving_throw", return_value=(True, 20)):
        apply_damage(z2, 10, dtype="光耀")
    assert z2.hp == 0 and z2.is_dead()


def test_condition_immunity_blocks_status():
    z = make_monster("zombie")
    z.add_status(Poisoned(applied_round=0))
    assert not z.has_status("poisoned")
    z.add_status(Prone(applied_round=0))     # not immune to prone
    assert z.has_status("prone")


def _env_attack(attacker_mid, target_arch="battle_master", seed=0,
                force_save=None, force_hit=True):
    """Monster (opp side) executes one weapon ATTACK on the agent; returns
    (result, env). Saves optionally forced via patch."""
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    env.reset(agent_archs=[target_arch], opp_archs=[attacker_mid],
              level=5, opp_level=MONSTER_DEFS[attacker_mid].natural_level)
    aid, oid = env.agent_ids[0], env.opp_ids[0]
    atk, tgt = env.ws.characters[oid], env.ws.characters[aid]
    atk.position = tgt.position.__class__(tgt.position.x + 1.0, tgt.position.y)
    weapon = atk.weapons[0]
    sk = next(s for s in available_skills(atk, env.ws)
              if s.skill_id == f"weapon:{weapon.name}")
    action = sk.build_action(oid, aid, tgt.position)
    patches = []
    if force_hit:
        patches.append(patch("trpg.engine.combat.resolve_attack",
                             return_value=(True, 30)))
    if force_save is not None:
        patches.append(patch("trpg.engine.combat.make_saving_throw",
                             return_value=(force_save, 1 if not force_save else 20)))
    for p in patches:
        p.start()
    try:
        result = execute_action(action, env.ws)
    finally:
        for p in patches:
            p.stop()
    return result, env, aid


def test_weapon_rider_status_prone_on_failed_save():
    result, env, aid = _env_attack("wolf", force_save=False)
    assert result.get("rider_status_applied") == "prone"
    assert env.ws.characters[aid].has_status("prone")


def test_weapon_rider_status_resisted_on_save():
    result, env, aid = _env_attack("wolf", force_save=True)
    assert "rider_status_applied" not in result
    assert not env.ws.characters[aid].has_status("prone")


def test_weapon_rider_paralysis_is_bounded():
    """Ghoul claws: rider paralysis must carry save_each + round cap —
    NOT the permanent default of the Paralyzed class."""
    result, env, aid = _env_attack("ghoul", force_save=False)
    tgt = env.ws.characters[aid]
    fx = next(f for f in tgt.status_effects
              if getattr(f, "name", "") == "paralyzed")
    assert fx.save_each == "CON DC10"
    assert fx.rounds_remaining == 10


def _per_hit(result: dict) -> dict:
    """Multiattack monsters return a MULTI_ATTACK wrapper; rider keys live in
    the per-hit sub-results."""
    if result.get("type") == "MULTI_ATTACK":
        return next((h for h in result.get("attacks", []) if h.get("hit")), {})
    return result


def test_weapon_rider_extra_damage_save_half():
    r, env, aid = _env_attack("wyvern", force_save=False)
    assert _per_hit(r).get("rider_damage", 0) >= 7        # 7d6 failed save
    r2, env2, aid2 = _env_attack("wyvern", force_save=True)
    assert 3 <= _per_hit(r2).get("rider_damage", 0) <= 21  # halved


def test_weapon_rider_stat_drain():
    result, env, aid = _env_attack("shadow")
    tgt = env.ws.characters[aid]
    stat, loss = result["rider_stat_drain"]
    assert stat == "STR" and 1 <= loss <= 4
    assert tgt.stats.STR == 16 - loss     # battle_master STR 16


def test_weapon_rider_max_hp_drain():
    result, env, aid = _env_attack("wight")
    tgt = env.ws.characters[aid]
    total_drained = sum(h.get("rider_max_hp_drain", 0)
                        for h in result.get("attacks", [result]))
    assert total_drained > 0
    assert tgt.max_hp == 50 - total_drained   # L5 battle_master: 10 + 5*8


def test_pack_tactics_advantage_with_adjacent_ally():
    env = CombatEnvV2(seed=1, n_agents=1, n_opps=2)
    env.reset(agent_archs=["battle_master"], opp_archs=["wolf", "wolf"],
              level=5, opp_level=1)
    aid = env.agent_ids[0]
    w1, w2 = env.opp_ids
    tgt = env.ws.characters[aid]
    V = tgt.position.__class__
    env.ws.characters[w1].position = V(tgt.position.x + 1.0, tgt.position.y)
    env.ws.characters[w2].position = V(tgt.position.x - 1.0, tgt.position.y)
    atk = env.ws.characters[w1]
    sk = next(s for s in available_skills(atk, env.ws)
              if s.skill_id.startswith("weapon:"))
    result = execute_action(sk.build_action(w1, aid, tgt.position), env.ws)
    assert result.get("advantage_mode") == "advantage"
    # move the packmate away → advantage gone
    env.ws.characters[w2].position = V(tgt.position.x - 20.0, tgt.position.y)
    result2 = execute_action(sk.build_action(w1, aid, tgt.position), env.ws)
    assert result2.get("advantage_mode") != "advantage"


def test_breath_recharge_at_turn_start():
    from trpg.engine.status import tick_status_effects
    d = make_monster("young_red_dragon")
    assert d.ability_uses["fire_breath"] == 1
    d.ability_uses["fire_breath"] = 0
    with patch("trpg.engine.dice.roll", return_value=6):
        tick_status_effects(d, "self_turn_start", 2)
    assert d.ability_uses["fire_breath"] == 1
    d.ability_uses["fire_breath"] = 0
    with patch("trpg.engine.dice.roll", return_value=2):
        tick_status_effects(d, "self_turn_start", 3)
    assert d.ability_uses["fire_breath"] == 0


def test_ability_use_deducted_in_execute_action():
    """Regression for the centralised deduction: opponents/BC used to have
    unlimited limited-use abilities (measured 4-7 surges/game, max_uses=1)."""
    env = CombatEnvV2(seed=2, n_agents=1, n_opps=1)
    env.reset(agent_archs=["battle_master"], opp_archs=["young_red_dragon"],
              level=5, opp_level=10)
    oid = env.opp_ids[0]
    drg = env.ws.characters[oid]
    tgt = env.ws.characters[env.agent_ids[0]]
    drg.position = tgt.position.__class__(tgt.position.x + 3.0, tgt.position.y)
    # The dragon may already have opened with breath during reset (if it won
    # initiative the env auto-plays its turn) — restore the use for the test.
    drg.ability_uses["fire_breath"] = 1
    sk = next(s for s in available_skills(drg, env.ws)
              if s.skill_id == "fire_breath")
    r = execute_action(sk.build_action(oid, None, tgt.position), env.ws)
    assert r.get("type") != "ERROR"
    assert drg.ability_uses["fire_breath"] == 0
    # spent: no longer offered
    assert all(s.skill_id != "fire_breath"
               for s in available_skills(drg, env.ws))


def test_descriptor_resist_columns():
    from trpg.rl.obs import capability_descriptor, ENT_DESC_START
    from trpg.engine.damage import DAMAGE_TYPES
    from trpg.engine.skill import N_STATUS_SLOTS, N_SAVE_STATS
    sk = make_monster("skeleton")
    desc = capability_descriptor(sk)
    off = 9 + N_STATUS_SLOTS + N_SAVE_STATS
    assert desc[off + DAMAGE_TYPES.index("鈍擊")] == pytest.approx(1.0)   # 2.0-1
    assert desc[off + DAMAGE_TYPES.index("毒")] == pytest.approx(-1.0)    # 0.0-1
    assert desc[off + DAMAGE_TYPES.index("斬擊")] == 0.0                  # neutral


# ── Wave 2 primitive unit tests (MONSTER_CATALOG §3 C 級) ────────────────────

from trpg.engine.status import (
    Restrained, tick_status_effects, MODIFIER_CLASSES,
    ENGINE_ONLY_STATUS_CLASSES, INCAPACITATING_STATUSES,
)
from trpg.engine.combat import tick_aura_damage


def test_obs_status_vocabulary_frozen():
    """Checkpoint-compat tripwire: RL_STATUS_NAMES (an ENTITY_DIM component)
    must not grow. 'petrified' rides its pre-reserved STATUS_SLOTS name;
    'swallowed'/'slowed' stay engine-only until the next obs surgery."""
    assert obs_mod.N_RL_STATUS == 31
    assert "petrified" in obs_mod.RL_STATUS_NAMES
    assert "swallowed" not in obs_mod.RL_STATUS_NAMES
    assert "slowed" not in obs_mod.RL_STATUS_NAMES
    assert set(ENGINE_ONLY_STATUS_CLASSES) == {"swallowed", "slowed"}
    assert "petrified" in MODIFIER_CLASSES


# ── regeneration ──────────────────────────────────────────────────────────────

def test_regeneration_heals_at_turn_start():
    t = make_monster("troll")
    t.hp = 50
    tick_status_effects(t, "self_turn_start", 2)
    assert t.hp == 60


def test_regeneration_suppressed_by_blocking_types():
    for dtype in ("火", "強酸"):
        t = make_monster("troll")
        t.hp = 50
        apply_damage(t, 5, dtype=dtype)
        tick_status_effects(t, "self_turn_start", 2)
        assert t.hp == 45, dtype                      # no regen this turn
        tick_status_effects(t, "self_turn_start", 3)  # window cleared above
        assert t.hp == 55, dtype                      # regen resumes


def test_regeneration_not_suppressed_by_other_types():
    t = make_monster("troll")
    t.hp = 50
    apply_damage(t, 5, dtype="斬擊")
    tick_status_effects(t, "self_turn_start", 2)
    assert t.hp == 55


def test_regeneration_immune_zeroed_hit_does_not_suppress():
    """5e wording: 'takes fire damage' — a hit zeroed by immunity was not
    taken. The window records post-multiplier damage only."""
    t = make_monster("troll")
    t.damage_multipliers["火"] = 0.0
    t.hp = 50
    apply_damage(t, 30, dtype="火")
    tick_status_effects(t, "self_turn_start", 2)
    assert t.hp == 60


def test_regeneration_caps_and_death():
    t = make_monster("troll")
    tick_status_effects(t, "self_turn_start", 2)
    assert t.hp == t.max_hp           # no overheal
    t.hp = 0
    t.is_npc = True
    tick_status_effects(t, "self_turn_start", 3)
    assert t.hp == 0                  # monsters die at 0 — no 0-HP regen


def test_recent_damage_window_cleared_for_everyone():
    o = make_monster("orc")
    apply_damage(o, 3, dtype="火")
    assert "火" in o.recent_damage_types
    tick_status_effects(o, "self_turn_start", 2)
    assert not o.recent_damage_types


# ── swallow_grapple ───────────────────────────────────────────────────────────

def _behir_setup(target_arch="life", restrain=True):
    env = CombatEnvV2(seed=5, n_agents=1, n_opps=1)
    env.reset(agent_archs=[target_arch], opp_archs=["behir"],
              level=8, opp_level=11)
    aid, oid = env.agent_ids[0], env.opp_ids[0]
    ag, op = env.ws.characters[aid], env.ws.characters[oid]
    V = ag.position.__class__
    ag.position = V(5.0, 5.0)
    op.position = V(6.0, 5.0)
    ag.hp = ag.max_hp = 200          # survives the chain under test
    if restrain:
        ag.add_status(Restrained(applied_round=1))
    return env, aid, oid, ag, op


def _swallow_action(oid, aid, op, env):
    sk = next(s for s in available_skills(op, env.ws) if s.skill_id == "swallow")
    return sk.build_action(oid, aid, None)


def test_swallow_requires_restrained_target():
    env, aid, oid, ag, op = _behir_setup(restrain=False)
    r = execute_action(_swallow_action(oid, aid, op, env), env.ws)
    assert r["type"] == "ERROR" and "restrained" in r["message"]


def test_swallow_is_single_bite_and_applies_composite():
    env, aid, oid, ag, op = _behir_setup()
    assert op.attacks_per_action == 2     # multiattack monster...
    with patch("trpg.engine.combat.resolve_attack", return_value=(True, 30)):
        r = execute_action(_swallow_action(oid, aid, op, env), env.ws)
    assert r["type"] == "ATTACK"          # ...but swallow is ONE bite
    assert r["rider_status_applied"] == "swallowed"
    fx = next(f for f in ag.status_effects if getattr(f, "name", "") == "swallowed")
    assert fx.save_each == "STR DC16"
    assert fx.metadata["tick_damage_dice"] == "6d6"
    assert fx.metadata["ends_if_source_dead"] is True
    # restrained-equivalent semantics: no movement while swallowed
    assert any(m.on_speed_multiplier(ag) == 0.0 for m in ag.iter_modifiers())


def _swallowed_victim():
    env, aid, oid, ag, op = _behir_setup()
    with patch("trpg.engine.combat.resolve_attack", return_value=(True, 30)):
        execute_action(_swallow_action(oid, aid, op, env), env.ws)
    assert ag.has_status("swallowed")
    return env, aid, oid, ag, op


def test_swallow_blocked_when_already_swallowed():
    env, aid, oid, ag, op = _swallowed_victim()
    r = execute_action(_swallow_action(oid, aid, op, env), env.ws)
    assert r["type"] == "ERROR" and "swallowed" in r["message"]


def test_swallowed_tick_deals_typed_acid_from_source():
    env, aid, oid, ag, op = _swallowed_victim()
    hp0 = ag.hp
    events = tick_aura_damage(ag, env.ws, 3)
    tick = next(e for e in events if e["type"] == "STATUS_TICK_DAMAGE")
    assert tick["damage_type"] == "強酸" and 6 <= tick["damage"] <= 36
    assert ag.hp == hp0 - tick["damage"]
    assert tick["source_name"] == op.name


def test_swallowed_escape_by_save():
    env, aid, oid, ag, op = _swallowed_victim()
    # tick_status_effects imports make_saving_throw from combat at call time
    with patch("trpg.engine.combat.make_saving_throw", return_value=(True, 20)):
        tick_status_effects(ag, "self_turn_end", 3)
    assert not ag.has_status("swallowed")


def test_swallowed_released_when_source_dies():
    env, aid, oid, ag, op = _swallowed_victim()
    op.hp = 0
    events = tick_aura_damage(ag, env.ws, 3)
    assert not any(e["type"] == "STATUS_TICK_DAMAGE" for e in events)
    assert not ag.has_status("swallowed")


def test_constrict_dual_rider():
    """貝希爾緊勒: one hit = bludgeon base + slashing rider dice + restrain."""
    env, aid, oid, ag, op = _behir_setup(restrain=False)
    sk = next(s for s in available_skills(op, env.ws)
              if s.skill_id == "weapon:貝希爾緊勒")
    with patch("trpg.engine.combat.resolve_attack", return_value=(True, 30)):
        r = execute_action(sk.build_action(oid, aid, None), env.ws)
    hit = next(h for h in r["attacks"] if h.get("hit"))  # MULTI_ATTACK wrapper
    assert hit.get("rider_damage", 0) >= 8               # 2d10+6 slashing
    assert ag.has_status("restrained")
    fx = next(f for f in ag.status_effects if getattr(f, "name", "") == "restrained")
    assert fx.save_each == "STR DC16"


# ── petrify (two-stage escalation) ───────────────────────────────────────────

def _gaze(env, oid, aid, op):
    sk = next(s for s in available_skills(op, env.ws)
              if s.skill_id == "petrifying_gaze")
    return execute_action(sk.build_action(oid, aid, None), env.ws)


def _basilisk_setup():
    env = CombatEnvV2(seed=7, n_agents=1, n_opps=1)
    env.reset(agent_archs=["battle_master"], opp_archs=["basilisk"],
              level=5, opp_level=3)
    aid, oid = env.agent_ids[0], env.opp_ids[0]
    ag, op = env.ws.characters[aid], env.ws.characters[oid]
    ag.hp = ag.max_hp                      # undo any reset-turn damage
    for name in ("restrained", "petrified"):
        ag.remove_status(name)
    return env, aid, oid, ag, op


def test_gaze_two_stage_escalation():
    env, aid, oid, ag, op = _basilisk_setup()
    with patch("trpg.engine.combat.make_saving_throw", return_value=(False, 1)):
        r1 = _gaze(env, oid, aid, op)
    assert r1["target_results"][0]["status_applied"] == "restrained"
    assert ag.has_status("restrained") and not ag.has_status("petrified")
    with patch("trpg.engine.combat.make_saving_throw", return_value=(False, 1)):
        r2 = _gaze(env, oid, aid, op)
    assert r2["target_results"][0]["status_applied"] == "petrified"
    fx = next(f for f in ag.status_effects if getattr(f, "name", "") == "petrified")
    assert fx.rounds_remaining is None and fx.save_each == ""   # permanent


def test_gaze_save_resists():
    env, aid, oid, ag, op = _basilisk_setup()
    with patch("trpg.engine.combat.make_saving_throw", return_value=(True, 20)):
        r = _gaze(env, oid, aid, op)
    assert not ag.has_status("restrained")
    assert r["target_results"][0]["status_applied"] == ""


def test_gaze_no_escalation_when_stage1_immune():
    env, aid, oid, ag, op = _basilisk_setup()
    ag.condition_immunities.append("restrained")
    with patch("trpg.engine.combat.make_saving_throw", return_value=(False, 1)):
        _gaze(env, oid, aid, op)
        _gaze(env, oid, aid, op)
    assert not ag.has_status("restrained") and not ag.has_status("petrified")


def test_petrified_is_incapacitated_and_resistant():
    env, aid, oid, ag, op = _basilisk_setup()
    with patch("trpg.engine.combat.make_saving_throw", return_value=(False, 1)):
        _gaze(env, oid, aid, op)
        _gaze(env, oid, aid, op)
    assert ag.has_status("petrified")
    assert ag.is_incapacitated()
    assert "petrified" in INCAPACITATING_STATUSES
    # all-type resistance: 10 → 5
    hp0 = ag.hp
    assert apply_damage(ag, 10, dtype="斬擊") == 5
    assert ag.hp == hp0 - 5
    # cannot act
    sk = next(s for s in available_skills(ag, env.ws)
              if s.skill_id.startswith("weapon:"))
    r = execute_action(sk.build_action(aid, oid, None), env.ws)
    assert r["type"] == "ERROR" and "petrified" in r["message"]


def test_petrified_melee_not_auto_crit():
    """5e: auto-crit applies to paralyzed/unconscious, NOT petrified."""
    env, aid, oid, ag, op = _basilisk_setup()
    with patch("trpg.engine.combat.make_saving_throw", return_value=(False, 1)):
        _gaze(env, oid, aid, op)
        _gaze(env, oid, aid, op)
    sk = next(s for s in available_skills(op, env.ws)
              if s.skill_id.startswith("weapon:"))
    with patch("trpg.engine.combat.resolve_attack", return_value=(True, 30)):
        r = execute_action(sk.build_action(oid, aid, None), env.ws)
    assert not r.get("auto_crit")


# ── multi_ray_table ──────────────────────────────────────────────────────────

def _beholder_setup():
    env = CombatEnvV2(seed=9, n_agents=1, n_opps=1)
    env.reset(agent_archs=["battle_master"], opp_archs=["beholder"],
              level=8, opp_level=13)
    aid, oid = env.agent_ids[0], env.opp_ids[0]
    ag, op = env.ws.characters[aid], env.ws.characters[oid]
    ag.hp = ag.max_hp = 500
    for name in list(MODIFIER_CLASSES) + list(ENGINE_ONLY_STATUS_CLASSES):
        ag.remove_status(name)
    return env, aid, oid, ag, op


def test_eye_rays_three_distinct_at_enemies():
    env, aid, oid, ag, op = _beholder_setup()
    sk = next(s for s in available_skills(op, env.ws) if s.skill_id == "eye_rays")
    r = execute_action(sk.build_action(oid, aid, None), env.ws)
    assert r["type"] == "EYE_RAYS"
    assert r["save_dc"] == 16                       # 8 + prof5 + INT3 (MM)
    names = [x["ray_name"] for x in r["rays"]]
    assert len(names) == 3 and len(set(names)) == 3
    assert all(x["target_name"] == ag.name for x in r["rays"])


def test_eye_rays_error_when_no_target_in_range():
    env, aid, oid, ag, op = _beholder_setup()
    V = ag.position.__class__
    ag.position = V(0.0, 0.0)
    op.position = V(60.0, 0.0)                      # beyond 36m
    sk = next(s for s in available_skills(op, env.ws) if s.skill_id == "eye_rays")
    r = execute_action(sk.build_action(oid, aid, None), env.ws)
    assert r["type"] == "ERROR"


def _one_ray(env, oid, spec, n=1):
    return execute_action({
        "type": "EYE_RAYS", "caster": oid, "skill_id": "eye_rays",
        "n_rays": n, "range_m": 36.0, "dc_stat": "INT",
        "table": [spec], "consumes": ["action"],
    }, env.ws)


def test_ray_damage_save_half_and_none():
    env, aid, oid, ag, op = _beholder_setup()
    spec = {"name": "衰弱射線", "save_stat": "CON", "damage_dice": "8d8",
            "damage_type": "黯蝕", "save_half": True}
    with patch("trpg.engine.combat.make_saving_throw", return_value=(False, 1)):
        r = _one_ray(env, oid, spec)
    assert 8 <= r["rays"][0]["damage"] <= 64
    spec2 = {"name": "死亡射線", "save_stat": "DEX", "damage_dice": "10d10",
             "damage_type": "黯蝕"}                  # no save_half → negates
    with patch("trpg.engine.combat.make_saving_throw", return_value=(True, 20)):
        r2 = _one_ray(env, oid, spec2)
    assert r2["rays"][0]["damage"] == 0


def test_ray_status_and_petrify_escalation():
    env, aid, oid, ag, op = _beholder_setup()
    spec = {"name": "石化射線", "save_stat": "DEX", "status": "restrained",
            "rounds": 10, "save_each": True, "escalates_to": "petrified"}
    with patch("trpg.engine.combat.make_saving_throw", return_value=(False, 1)):
        r1 = _one_ray(env, oid, spec)
        r2 = _one_ray(env, oid, spec)
    assert r1["rays"][0]["status_applied"] == "restrained"
    assert r2["rays"][0]["status_applied"] == "petrified"
    assert ag.has_status("petrified")


def test_ray_slowed_applies_engine_only_status():
    env, aid, oid, ag, op = _beholder_setup()
    spec = {"name": "緩速射線", "save_stat": "DEX", "status": "slowed",
            "rounds": 10, "save_each": True}
    with patch("trpg.engine.combat.make_saving_throw", return_value=(False, 1)):
        _one_ray(env, oid, spec)
    assert ag.has_status("slowed")
    assert any(m.on_speed_multiplier(ag) == 0.5 for m in ag.iter_modifiers())


# ── LINE target semantics ────────────────────────────────────────────────────

def test_point_segment_distance():
    from trpg.engine.vec2 import point_segment_distance, Vec2
    a, b = Vec2(0, 0), Vec2(6, 0)
    assert point_segment_distance(Vec2(3, 0), a, b) == pytest.approx(0.0)
    assert point_segment_distance(Vec2(3, 2), a, b) == pytest.approx(2.0)
    assert point_segment_distance(Vec2(9, 0), a, b) == pytest.approx(3.0)
    assert point_segment_distance(Vec2(-4, 3), a, b) == pytest.approx(5.0)
    assert point_segment_distance(Vec2(1, 1), a, a) == pytest.approx((2) ** 0.5)


def _breath(env, oid, coord):
    op = env.ws.characters[oid]
    op.ability_uses["lightning_breath"] = 1
    sk = next(s for s in available_skills(op, env.ws)
              if s.skill_id == "lightning_breath")
    return execute_action(sk.build_action(oid, None, coord), env.ws)


def test_line_hits_collinear_and_misses_lateral():
    env, aid, oid, ag, op = _behir_setup(restrain=False)
    V = ag.position.__class__
    op.position = V(0.0, 5.0)
    ag.position = V(3.0, 5.0)                      # on the beam
    r = _breath(env, oid, (3.0, 5.0))
    assert [t["target_name"] for t in r["target_results"]] == [ag.name]
    # lateral offset > width/2 → not affected even though aim range is fine
    ag.position = V(3.0, 6.5)
    r2 = _breath(env, oid, (3.0, 5.0))
    assert r2["target_results"] == []
    # beyond line length → not affected (aim stays in range)
    ag.position = V(8.0, 5.0)
    r3 = _breath(env, oid, (5.9, 5.0))
    assert r3["target_results"] == []


def test_line_aim_needs_direction_and_range():
    env, aid, oid, ag, op = _behir_setup(restrain=False)
    V = ag.position.__class__
    op.position = V(0.0, 5.0)
    r = _breath(env, oid, (0.0, 5.0))              # aim at self: no direction
    assert r["type"] == "ERROR"
    r2 = _breath(env, oid, (30.0, 5.0))            # aim beyond 6m range
    assert r2["type"] == "ERROR"


def test_line_immunity_still_consumes_breath():
    """Breath vs a lightning-immune target: the SPELL result reports the
    pre-resistance roll (long-standing semantics), but no HP moves — and the
    use is still spent."""
    env, aid, oid, ag, op = _behir_setup(restrain=False)
    V = ag.position.__class__
    op.position = V(0.0, 5.0)
    ag.position = V(3.0, 5.0)
    ag.damage_multipliers["閃電"] = 0.0
    hp0 = ag.hp
    r = _breath(env, oid, (3.0, 5.0))
    assert [t["target_name"] for t in r["target_results"]] == [ag.name]
    assert ag.hp == hp0
    assert op.ability_uses["lightning_breath"] == 0


# ── GenericMonsterPolicy v1 (control casts + preconditions) ──────────────────

def _decide(mid, target_arch="battle_master", prep=None, lvl=5):
    env = CombatEnvV2(seed=13, n_agents=1, n_opps=1)
    env.reset(agent_archs=[target_arch], opp_archs=[mid],
              level=lvl, opp_level=MONSTER_DEFS[mid].natural_level)
    aid, oid = env.agent_ids[0], env.opp_ids[0]
    ag, op = env.ws.characters[aid], env.ws.characters[oid]
    V = ag.position.__class__
    ag.position = V(5.0, 5.0)
    op.position = V(6.0, 5.0)
    ag.hp = ag.max_hp
    op.hp = op.max_hp
    for name in list(MODIFIER_CLASSES) + list(ENGINE_ONLY_STATUS_CLASSES):
        ag.remove_status(name)
    if prep:
        prep(ag, op, env)
    dec = GenericMonsterPolicy().decide(
        oid, op, env.ws, {"action": 1, "bonus_action": 1, "movement": 9.0}, 2)
    return dec.action and dec.action.get("skill_id")


def test_policy_basilisk_gazes_until_petrified_then_bites():
    assert _decide("basilisk") == "petrifying_gaze"
    assert _decide("basilisk", prep=lambda ag, op, env:
                   ag.add_status(Restrained(applied_round=1))) == "petrifying_gaze"
    from trpg.engine.status import Petrified
    assert _decide("basilisk", prep=lambda ag, op, env:
                   ag.add_status(Petrified(applied_round=1))
                   ).startswith("weapon:")


def test_policy_skips_control_when_stage1_immune():
    assert _decide("basilisk", prep=lambda ag, op, env:
                   ag.condition_immunities.append("restrained")
                   ).startswith("weapon:")


def test_policy_damage_led_caster_skips_control():
    """mage_npc is damage-LED (fireball 28 ≫ dagger) — the control-first
    A/B measured it 18pp weaker when opening with hold_person, so the
    control step is gated to control-led kits (basilisk) only."""
    pick = _decide("mage_npc")
    assert pick != "hold_person" and pick is not None


def test_policy_behir_swallow_gated_on_restrained():
    pick_free = _decide("behir", target_arch="life", lvl=8)
    assert pick_free != "swallow"
    pick_restrained = _decide(
        "behir", target_arch="life", lvl=8,
        prep=lambda ag, op, env: (
            ag.add_status(Restrained(applied_round=1)),
            op.ability_uses.__setitem__("lightning_breath", 0)))
    assert pick_restrained == "swallow"


# ═════════════════════ Wave 3 — legendary suite ══════════════════════════════

from trpg.engine.combat import make_saving_throw
from trpg.engine.combat_policy import (
    run_legendary_actions, decide_legendary_action,
)
from trpg.engine.status import Paralyzed
from trpg.scenarios.archetypes import TRAIT_APPLIERS


# ── statblock fidelity (MM) ──────────────────────────────────────────────────

@pytest.mark.parametrize("mid,stat,to_hit,hp,ac", [
    ("adult_white_dragon", "STR", +11, 200, 18),
    ("adult_red_dragon",   "STR", +14, 256, 19),
    ("ancient_red_dragon", "STR", +17, 546, 22),
    ("kraken",             "STR", +17, 472, 18),
    ("balor",              "STR", +14, 262, 19),
    ("tarrasque",          "STR", +19, 676, 25),
])
def test_wave3_statblocks(mid, stat, to_hit, hp, ac):
    c = make_monster(mid)
    assert c.stats.modifier(stat) + c.proficiency_bonus == to_hit
    assert c.max_hp == hp and c.ac == ac


def test_lich_statblock():
    c = make_monster("lich")
    dc = 8 + c.proficiency_bonus + c.stats.modifier("INT")
    assert dc == 20 and c.max_hp == 135 and c.ac == 17
    # 麻痺之觸: finesse → DEX3 + prof7 = +10 (MM spell attack +12 的武器管線近似)
    touch = c.get_weapon("麻痺之觸")
    assert "精巧" in touch.properties and touch.on_hit["save_dc"] == 18


def test_wave3_trait_numbers():
    """Stat-panel-derived DCs reproduce the MM (the breath-weapon law again)."""
    fp = {m: make_monster(m).frightful_presence for m in
          ("adult_white_dragon", "adult_red_dragon", "ancient_red_dragon",
           "tarrasque")}
    assert [fp[m]["dc"] for m in fp] == [14, 19, 21, 17]
    balor = make_monster("balor")
    assert balor.death_throes["dc"] == 20                      # 8+6+CON6
    assert balor.death_throes["damage_dice"] == "20d6"
    assert balor.legendary_actions_max == 0                    # 非傳奇 (MM)
    assert make_monster("kraken").legendary_resistance_uses == 0  # MM: 海妖無 LR
    assert make_monster("beholder").legendary_actions_max == 3    # W3 retrofit


def test_wave3_obs_vocabulary_frozen():
    """Wave 3 adds ZERO new status names (frightened/prone/paralyzed/
    restrained/swallowed all pre-exist) and the legendary resource fields
    live on Character, not in obs — dims stay frozen with no surgery."""
    assert obs_mod.N_RL_STATUS == 31
    assert obs_mod.N_ARCHETYPES == 12


# ── legendary_actions ────────────────────────────────────────────────────────

def _legendary_setup(boss="adult_white_dragon", dist=4.0, n_agents=1):
    env = CombatEnvV2(seed=11, n_agents=n_agents, n_opps=1)
    env.reset(agent_archs=["battle_master"] * n_agents, opp_archs=[boss],
              level=8, opp_level=MONSTER_DEFS[boss].natural_level)
    aid, oid = env.agent_ids[0], env.opp_ids[0]
    ag, op = env.ws.characters[aid], env.ws.characters[oid]
    V = ag.position.__class__
    ag.position = V(5.0, 5.0)
    op.position = V(5.0 + dist, 5.0)
    ag.hp = ag.max_hp = 500            # survives whatever the boss does
    op.hp = op.max_hp
    op.legendary_actions_remaining = op.legendary_actions_max
    for name in list(MODIFIER_CLASSES) + list(ENGINE_ONLY_STATUS_CLASSES):
        ag.remove_status(name)
        op.remove_status(name)
    # env.reset() may have auto-played turns (boss wins initiative) — the
    # agent could already have rolled its frightful save there. Reset it.
    ag.frightful_immune_to.clear()
    return env, aid, oid, ag, op


def test_legendary_budget_regains_at_turn_start():
    env, aid, oid, ag, op = _legendary_setup()
    op.legendary_actions_remaining = 0
    tick_status_effects(op, "self_turn_start", 2)
    assert op.legendary_actions_remaining == 3


def test_legendary_fires_one_option_per_trigger():
    env, aid, oid, ag, op = _legendary_setup(dist=4.0)   # tail reach, wing 外
    with patch("trpg.engine.combat.resolve_attack", return_value=(True, 30)):
        events = run_legendary_actions(env.ws, aid, 2)
    assert len(events) == 1
    ev = events[0]
    assert ev["action"]["weapon"] == "龍尾"               # EV/cost 貪心
    assert ev["action"]["n_attacks"] == 1                 # 單揮，非多重攻擊
    assert ev["cost"] == 1 and ev["remaining"] == 2
    assert op.legendary_actions_remaining == 2
    assert ev["result"]["type"] == "ATTACK"


def test_legendary_not_triggered_by_own_turn():
    env, aid, oid, ag, op = _legendary_setup()
    assert run_legendary_actions(env.ws, oid, 2) == []
    assert op.legendary_actions_remaining == 3


def test_legendary_skipped_while_incapacitated():
    env, aid, oid, ag, op = _legendary_setup()
    op.condition_immunities.clear()                # 允許測試掛麻痺
    op.add_status(Paralyzed(applied_round=1))
    assert run_legendary_actions(env.ws, aid, 2) == []


def test_legendary_out_of_reach_declines():
    env, aid, oid, ag, op = _legendary_setup(dist=6.0)   # 尾 4.5 / 翼 3 皆不及
    assert run_legendary_actions(env.ws, aid, 2) == []
    assert op.legendary_actions_remaining == 3           # 沒花就不扣


def test_legendary_respects_budget():
    env, aid, oid, ag, op = _legendary_setup(dist=2.0)
    op.legendary_options = [{"ability": "wing_attack", "cost": 2}]
    op.legendary_actions_remaining = 1                   # 買不起翼擊
    assert run_legendary_actions(env.ws, aid, 2) == []


def test_legendary_wing_attack_is_self_nova_excluding_caster():
    env, aid, oid, ag, op = _legendary_setup(dist=2.0)
    op.legendary_options = [{"ability": "wing_attack", "cost": 2}]
    with patch("trpg.engine.combat.make_saving_throw", return_value=(False, 1)):
        events = run_legendary_actions(env.ws, aid, 2)
    assert len(events) == 1 and events[0]["cost"] == 2
    r = events[0]["result"]
    assert r["type"] == "SPELL" and r["spell_name"] == "龍翼拍擊"
    names = [t["target_name"] for t in r["target_results"]]
    assert ag.name in names and op.name not in names      # excludes_caster
    hit = next(t for t in r["target_results"] if t["target_name"] == ag.name)
    assert hit["damage"] > 0 and hit["status_applied"] == "prone"


def test_legendary_lich_cantrip_scales():
    env, aid, oid, ag, op = _legendary_setup(boss="lich", dist=10.0)
    op.legendary_options = [{"ability": "chill_touch", "cost": 1}]
    with patch("trpg.engine.combat.make_saving_throw", return_value=(False, 1)), \
         patch("trpg.engine.combat.roll", return_value=8):
        events = run_legendary_actions(env.ws, aid, 2)
    r = events[0]["result"]
    hit = next(t for t in r["target_results"] if t["target_name"] == ag.name)
    assert hit["damage"] == 32        # 1d8 ×4 at L21 (cantrip scaling)


def test_legendary_trait_validation_fails_loud():
    c = make_monster("orc")
    apply_trait = TRAIT_APPLIERS["legendary_actions"]
    with pytest.raises(ValueError):    # ability not granted
        apply_trait(c, 1, {"options": [{"ability": "wing_attack", "cost": 1}]})
    with pytest.raises(ValueError):    # unknown weapon
        apply_trait(c, 1, {"options": [{"weapon": "幽靈劍", "cost": 1}]})
    with pytest.raises(ValueError):    # zero cost
        apply_trait(c, 1, {"options": [{"weapon": "巨斧", "cost": 0}]})
    with pytest.raises(ValueError):    # leveled spell as legendary option
        c.known_abilities.append("fireball_ev")
        apply_trait(c, 1, {"options": [{"ability": "fireball_ev", "cost": 1}]})


# ── legendary_resistance ─────────────────────────────────────────────────────

def test_lr_burns_only_on_status_saves():
    c = make_monster("adult_red_dragon")
    ok, _ = make_saving_throw(c, "WIS", 99, fail_applies_status=True)
    assert ok and c.legendary_resistance_uses == 2        # 失敗→改判成功
    ok, _ = make_saving_throw(c, "DEX", 99)               # 純傷害豁免
    assert not ok and c.legendary_resistance_uses == 2    # 絕不為半傷燒次數


def test_lr_covers_auto_fail_saves():
    c = make_monster("adult_red_dragon")
    c.condition_immunities.clear()
    c.add_status(Paralyzed(applied_round=1))              # STR/DEX 自動失敗
    ok, _ = make_saving_throw(c, "DEX", 5, fail_applies_status=True)
    assert ok and c.legendary_resistance_uses == 2


def test_lr_pool_depletes():
    c = make_monster("adult_red_dragon")
    c.legendary_resistance_uses = 0
    ok, _ = make_saving_throw(c, "WIS", 99, fail_applies_status=True)
    assert not ok


def test_lr_escapes_save_each_status():
    """A restrained boss buys its action economy back through LR (the
    status.tick re-save passes fail_applies_status=True)."""
    c = make_monster("tarrasque")
    c.condition_immunities.clear()
    fx = Restrained(applied_round=1)
    fx.save_each = "STR DC99"                             # 擲骰必敗
    c.add_status(fx)
    tick_status_effects(c, "self_turn_end", 2)
    assert not c.has_status("restrained")
    assert c.legendary_resistance_uses == 2


# ── frightful_presence ───────────────────────────────────────────────────────

def _fp_setup(dist=10.0):
    return _legendary_setup(boss="adult_white_dragon", dist=dist)


def test_frightful_failed_save_applies_frightened():
    env, aid, oid, ag, op = _fp_setup()
    with patch("trpg.engine.combat.make_saving_throw",
               return_value=(False, 1)) as mst:
        events = tick_aura_damage(ag, env.ws, 2)
    fp = next(e for e in events if e["type"] == "FRIGHTFUL_PRESENCE")
    assert fp["save_dc"] == 14 and not fp["save_success"]
    assert ag.has_status("frightened")
    fx = next(f for f in ag.status_effects
              if getattr(f, "name", "") == "frightened")
    assert fx.save_each == "WIS DC14"


def test_frightful_success_grants_combat_immunity():
    env, aid, oid, ag, op = _fp_setup()
    with patch("trpg.engine.combat.make_saving_throw",
               return_value=(True, 20)) as mst:
        events = tick_aura_damage(ag, env.ws, 2)
        n_first = mst.call_count
    assert oid in ag.frightful_immune_to
    assert not ag.has_status("frightened")
    with patch("trpg.engine.combat.make_saving_throw",
               return_value=(True, 20)) as mst2:
        tick_aura_damage(ag, env.ws, 3)
    assert mst2.call_count == 0           # 免疫後不再擲骰
    assert n_first == 1


def test_frightful_skips_condition_immune_without_rolling():
    env, aid, oid, ag, op = _fp_setup()
    ag.condition_immunities.append("frightened")
    with patch("trpg.engine.combat.make_saving_throw",
               return_value=(False, 1)) as mst:
        events = tick_aura_damage(ag, env.ws, 2)
    assert mst.call_count == 0
    assert not any(e["type"] == "FRIGHTFUL_PRESENCE" for e in events)


def test_frightful_out_of_radius_no_check():
    env, aid, oid, ag, op = _fp_setup()
    op.frightful_presence["radius_m"] = 3.0               # 縮小到 3m 驗證半徑閘
    with patch("trpg.engine.combat.make_saving_throw",
               return_value=(False, 1)) as mst:
        tick_aura_damage(ag, env.ws, 2)
    assert mst.call_count == 0


def test_frightful_already_frightened_not_rechecked():
    env, aid, oid, ag, op = _fp_setup()
    from trpg.engine.status import Frightened
    ag.add_status(Frightened(applied_round=1))
    with patch("trpg.engine.combat.make_saving_throw",
               return_value=(False, 1)) as mst:
        tick_aura_damage(ag, env.ws, 2)
    assert mst.call_count == 0


# ── death_throes ─────────────────────────────────────────────────────────────

def _balor_setup():
    env = CombatEnvV2(seed=23, n_agents=2, n_opps=1)
    env.reset(agent_archs=["battle_master", "life"], opp_archs=["balor"],
              level=8, opp_level=19)
    a1, a2 = env.agent_ids
    oid = env.opp_ids[0]
    near, far, balor = (env.ws.characters[a1], env.ws.characters[a2],
                        env.ws.characters[oid])
    V = near.position.__class__
    balor.position = V(15.0, 15.0)
    near.position = V(17.0, 15.0)      # 2m — inside the 9m blast
    far.position = V(28.0, 15.0)       # 13m — outside
    near.hp = near.max_hp = 500
    far.hp = far.max_hp = 500
    return env, near, far, balor


def test_death_throes_explodes_on_death():
    env, near, far, balor = _balor_setup()
    with patch("trpg.engine.combat.make_saving_throw", return_value=(False, 1)), \
         patch("trpg.engine.combat.roll", return_value=100):
        apply_damage(balor, 999, dtype="光耀", world_state=env.ws)
    assert balor.hp == 0 and balor.death_throes is None    # spec popped
    assert near.hp == 400                                  # full 100 火
    assert far.hp == 500                                   # 半徑外無傷
    evs = [e for e in env.ws.pending_events if e["type"] == "DEATH_THROES"]
    assert len(evs) == 1 and evs[0]["target_name"] == near.name
    assert evs[0]["damage_type"] == "火"


def test_death_throes_save_halves():
    env, near, far, balor = _balor_setup()
    with patch("trpg.engine.combat.make_saving_throw", return_value=(True, 25)), \
         patch("trpg.engine.combat.roll", return_value=100):
        apply_damage(balor, 999, dtype="光耀", world_state=env.ws)
    assert near.hp == 450


def test_death_throes_needs_world_state():
    balor = make_monster("balor")
    balor.is_npc = True
    apply_damage(balor, 999, dtype="光耀")                  # ws 缺席：不爆不炸
    assert balor.hp == 0


def test_death_throes_does_not_refire():
    env, near, far, balor = _balor_setup()
    with patch("trpg.engine.combat.make_saving_throw", return_value=(False, 1)), \
         patch("trpg.engine.combat.roll", return_value=100):
        apply_damage(balor, 999, dtype="光耀", world_state=env.ws)
        env.ws.pending_events.clear()
        apply_damage(balor, 50, dtype="光耀", world_state=env.ws)
    assert not env.ws.pending_events                       # 已 pop，不重爆


# ── kraken lightning storm（distinct_rays=False 多落雷）──────────────────────

def test_lightning_storm_three_bolts_from_single_entry_table():
    env = CombatEnvV2(seed=31, n_agents=2, n_opps=1)
    env.reset(agent_archs=["battle_master", "life"], opp_archs=["kraken"],
              level=8, opp_level=23)
    oid = env.opp_ids[0]
    op = env.ws.characters[oid]
    names = {env.ws.characters[a].name for a in env.agent_ids}
    for a in env.agent_ids:
        c = env.ws.characters[a]
        c.hp = c.max_hp = 500
    sk = next(s for s in available_skills(op, env.ws)
              if s.skill_id == "lightning_storm")
    r = execute_action(sk.build_action(oid, None, env.ws.characters[
        env.agent_ids[0]].position), env.ws)
    assert r["type"] == "EYE_RAYS" and len(r["rays"]) == 3
    assert all(ray["ray_name"] == "落雷" for ray in r["rays"])
    assert all(ray["target_name"] in names for ray in r["rays"])
    assert r["save_dc"] == 22                              # 8+prof7+CON7


def test_beholder_legendary_single_ray():
    env = CombatEnvV2(seed=37, n_agents=1, n_opps=1)
    env.reset(agent_archs=["battle_master"], opp_archs=["beholder"],
              level=8, opp_level=13)
    aid, oid = env.agent_ids[0], env.opp_ids[0]
    op = env.ws.characters[oid]
    ag = env.ws.characters[aid]
    ag.hp = ag.max_hp = 500
    op.legendary_actions_remaining = 3
    events = run_legendary_actions(env.ws, aid, 2)
    assert len(events) == 1
    r = events[0]["result"]
    assert r["type"] == "EYE_RAYS" and len(r["rays"]) == 1
    assert events[0]["cost"] == 1


# ── Wave 3 swallow variants（共用 W2 機制鏈）─────────────────────────────────

@pytest.mark.parametrize("mid,skill,escape,tick", [
    ("kraken",    "kraken_swallow",    "STR DC18", "12d6"),
    ("tarrasque", "tarrasque_swallow", "STR DC20", "16d6"),
])
def test_wave3_swallow_variants(mid, skill, escape, tick):
    env = CombatEnvV2(seed=41, n_agents=1, n_opps=1)
    env.reset(agent_archs=["life"], opp_archs=[mid],
              level=8, opp_level=MONSTER_DEFS[mid].natural_level)
    aid, oid = env.agent_ids[0], env.opp_ids[0]
    ag, op = env.ws.characters[aid], env.ws.characters[oid]
    V = ag.position.__class__
    ag.position = V(5.0, 5.0)
    op.position = V(6.0, 5.0)
    ag.hp = ag.max_hp = 800
    for name in list(MODIFIER_CLASSES) + list(ENGINE_ONLY_STATUS_CLASSES):
        ag.remove_status(name)
    sk_action = next(s for s in available_skills(op, env.ws)
                     if s.skill_id == skill).build_action(oid, aid, None)
    r = execute_action(sk_action, env.ws)
    assert r["type"] == "ERROR"                       # 未束縛 → 擋下
    ag.add_status(Restrained(applied_round=1))
    with patch("trpg.engine.combat.resolve_attack", return_value=(True, 40)):
        r = execute_action(sk_action, env.ws)
    assert r["type"] == "ATTACK" and r["rider_status_applied"] == "swallowed"
    fx = next(f for f in ag.status_effects
              if getattr(f, "name", "") == "swallowed")
    assert fx.save_each == escape
    assert fx.metadata["tick_damage_dice"] == tick


# ── 1vN smoke：傳奇 Boss 在 3v1 env 全鏈可玩 ─────────────────────────────────

@pytest.mark.parametrize("mid", [
    "adult_white_dragon", "lich", "kraken", "tarrasque", "balor",
])
def test_wave3_boss_full_team_fight_smoke(mid):
    env = CombatEnvV2(seed=43, n_agents=3, n_opps=1)
    env.reset(agent_archs=["battle_master", "life", "evocation"],
              opp_archs=[mid], level=8,
              opp_level=MONSTER_DEFS[mid].natural_level)
    experts = {a: make_archetype_policy(arch)
               for a, arch in zip(env.agent_ids, env.agent_archs)}
    done, steps = False, 0
    while not done and steps < 400:
        actor = env.current_agent_id
        a = env.ws.characters[actor]
        dec = experts[actor].decide(actor, a, env.ws, env.resources,
                                    env.ws.combat.round_number)
        act = ([0, 0, 0] if (dec.action is None or dec.fled)
               else list(encode_action(dec.action, env.ws, actor)))
        _, _, term, trunc, _ = env.step(act)
        done = term or trunc
        steps += 1
    assert done                                        # 對局收斂、無例外
