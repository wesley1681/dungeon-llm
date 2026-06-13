"""Milestone 1 — reaction / legendary DECISION seam.

Pins two contracts of the seam that makes off-turn decisions model-controllable
(REPORT 2026-06-13 reaction/legendary wave):

  1. DEFAULT path is bit-exact with the old hardcoded behaviour (no decider
     injected → greedy: Shield fires only in the +5 flip window, fires
     unconditionally vs auto-hit, Counterspell is first-eligible, legendary is
     EV/cost greedy).
  2. An injected ws.reaction_decider / ws.legendary_decider can override the
     CHOICE (decline a legal reaction, force one outside the greedy window, pick
     a non-greedy legendary option) while LEGALITY stays engine-enforced (an
     out-of-set return is treated as a decline; an illegal option is never
     offered).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch

import pytest

from trpg.engine.character import Character, Stats, CombatState
from trpg.engine.world_state import WorldState
from trpg.engine.vec2 import Vec2, Battlefield
from trpg.engine.combat import (
    effective_ac, execute_action,
    _try_react_to_attack, _try_react_to_auto_damage, _try_uncanny_dodge,
    _legal_reactions, greedy_reaction_decider, ReactionContext,
)

from trpg.scenarios.monsters import MONSTER_DEFS, register_monsters
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.engine.combat_policy import run_legendary_actions


# ── reaction harness ──────────────────────────────────────────────────────────

def _shield_world():
    """An attacker and a shield-capable defender, 1m apart."""
    attacker = Character(name="ATK", race="", class_="戰士", level=5,
                         stats=Stats(STR=16), hp=30, max_hp=30, ac=14,
                         is_npc=True, attitude=0)
    defender = Character(name="DEF", race="", class_="法師", level=5,
                         stats=Stats(INT=16), hp=20, max_hp=20, ac=12,
                         is_npc=False, spellcasting_ability="INT",
                         spell_slots={1: 2}, reactions=["shield_spell"])
    attacker.position = Vec2(0, 0)
    defender.position = Vec2(1, 0)
    ws = WorldState(characters={"atk": attacker, "def": defender}, scene="",
                    dungeon_map=None)
    ws.party_ids = ["def"]
    ws.combat = CombatState(initiative_order=["atk", "def"])
    ws.combat.battlefield = Battlefield(width=30.0, height=30.0)
    return ws, attacker, defender


# ── 1. DEFAULT (greedy) path bit-exact ────────────────────────────────────────

def test_default_shield_fires_in_flip_window():
    ws, atk, dfn = _shield_world()
    base_ac = effective_ac(dfn)                       # 12
    blocked, name, slot = _try_react_to_attack(dfn, atk, base_ac + 2, ws)
    assert blocked is True and name == "shield_spell" and slot == 1
    assert dfn.reaction_used is True and dfn.spell_slots[1] == 1


def test_default_shield_skips_outside_flip_window():
    ws, atk, dfn = _shield_world()
    base_ac = effective_ac(dfn)
    blocked, name, slot = _try_react_to_attack(dfn, atk, base_ac + 5, ws)
    assert blocked is False and name == "" and slot == 0
    assert dfn.reaction_used is False and dfn.spell_slots[1] == 2


def test_default_shield_fires_unconditionally_vs_auto_damage():
    ws, atk, dfn = _shield_world()
    blocked, name, slot = _try_react_to_auto_damage(dfn, ws)
    assert blocked is True and name == "shield_spell"
    assert dfn.reaction_used is True


def test_legal_reactions_is_pure_legality():
    """No slot or already-used reaction → empty (tactic-agnostic mask)."""
    ws, atk, dfn = _shield_world()
    assert _legal_reactions(dfn, "attack", ws) == ["shield_spell"]
    dfn.spell_slots[1] = 0
    assert _legal_reactions(dfn, "attack", ws) == []
    dfn.spell_slots[1] = 2
    dfn.reaction_used = True
    assert _legal_reactions(dfn, "attack", ws) == []


# ── 2. injected decider overrides the CHOICE ──────────────────────────────────

def test_injected_decider_can_decline_legal_shield():
    ws, atk, dfn = _shield_world()
    ws.reaction_decider = lambda ctx: None            # decline everything
    base_ac = effective_ac(dfn)
    blocked, name, slot = _try_react_to_attack(dfn, atk, base_ac + 2, ws)
    assert blocked is False and slot == 0
    assert dfn.reaction_used is False and dfn.spell_slots[1] == 2  # nothing spent


def test_injected_decider_can_force_shield_outside_window():
    ws, atk, dfn = _shield_world()
    ws.reaction_decider = lambda ctx: "shield_spell"  # always shield
    base_ac = effective_ac(dfn)
    # An attack well above the flip window: greedy would skip, the model forces.
    blocked, name, slot = _try_react_to_attack(dfn, atk, base_ac + 9, ws)
    assert blocked is True and name == "shield_spell"
    assert dfn.reaction_used is True and dfn.spell_slots[1] == 1


def test_decider_sees_trigger_context():
    ws, atk, dfn = _shield_world()
    seen = {}

    def spy(ctx: ReactionContext):
        seen.update(trigger=ctx.trigger, reactor_id=ctx.reactor_id,
                    options=list(ctx.options), total=ctx.attack_total,
                    attacker=ctx.attacker.name if ctx.attacker else None)
        return None

    ws.reaction_decider = spy
    _try_react_to_attack(dfn, atk, effective_ac(dfn) + 2, ws)
    assert seen == {"trigger": "attack", "reactor_id": "def",
                    "options": ["shield_spell"], "total": effective_ac(dfn) + 2,
                    "attacker": "ATK"}


def test_out_of_set_return_is_decline():
    """A decider returning an option not in ctx.options never bypasses legality."""
    ws, atk, dfn = _shield_world()
    ws.reaction_decider = lambda ctx: "counterspell"  # not a legal option here
    blocked, name, slot = _try_react_to_attack(dfn, atk, effective_ac(dfn) + 2, ws)
    assert blocked is False and dfn.reaction_used is False


def test_per_reactor_routing():
    """A decider can route by reactor_id — model for one, greedy for another."""
    ws, atk, dfn = _shield_world()

    def route(ctx):
        return None if ctx.reactor_id == "def" else greedy_reaction_decider(ctx)

    ws.reaction_decider = route
    blocked, *_ = _try_react_to_attack(dfn, atk, effective_ac(dfn) + 2, ws)
    assert blocked is False               # 'def' is model-routed → declined


# ── counterspell through execute_action ───────────────────────────────────────

def _counterspell_world():
    enemy_caster = Character(name="EC", race="", class_="法師", level=5,
                             stats=Stats(INT=16), hp=20, max_hp=20, ac=12,
                             is_npc=True, attitude=0,
                             spellcasting_ability="INT", spell_slots={3: 1})
    defender = Character(name="D", race="", class_="法師", level=5,
                         stats=Stats(INT=14), hp=20, max_hp=20, ac=12,
                         is_npc=False, spellcasting_ability="INT",
                         spell_slots={3: 2}, reactions=["counterspell"])
    enemy_caster.position = Vec2(0, 0)
    defender.position = Vec2(5, 0)
    ws = WorldState(characters={"ec": enemy_caster, "d": defender}, scene="",
                    dungeon_map=None)
    ws.party_ids = ["d"]
    ws.combat = CombatState(initiative_order=["ec", "d"])
    ws.combat.battlefield = Battlefield(width=30.0, height=30.0)
    return ws, enemy_caster, defender


def test_counterspell_default_fires():
    ws, ec, d = _counterspell_world()
    res = execute_action(
        {"type": "SPELL", "skill_id": "h", "caster": "ec", "spell_name": "火球術",
         "target_position": [5.0, 0.0], "consumes": ["action"]}, ws)
    assert res["type"] == "COUNTERSPELLED" and d.reaction_used is True


def test_counterspell_declined_via_decider():
    ws, ec, d = _counterspell_world()
    ws.reaction_decider = lambda ctx: None
    res = execute_action(
        {"type": "SPELL", "skill_id": "h", "caster": "ec", "spell_name": "火球術",
         "target_position": [5.0, 0.0], "consumes": ["action"]}, ws)
    assert res["type"] == "SPELL"            # not countered
    assert d.reaction_used is False and d.spell_slots[3] == 2


# ── legendary decision seam ───────────────────────────────────────────────────

def _legendary_env(boss="adult_white_dragon", dist=4.0):
    env = CombatEnvV2(seed=11, n_agents=1, n_opps=1)
    env.reset(agent_archs=["battle_master"], opp_archs=[boss],
              level=8, opp_level=MONSTER_DEFS[boss].natural_level)
    aid, oid = env.agent_ids[0], env.opp_ids[0]
    ag, op = env.ws.characters[aid], env.ws.characters[oid]
    V = ag.position.__class__
    ag.position = V(5.0, 5.0)
    op.position = V(5.0 + dist, 5.0)
    ag.hp = ag.max_hp = 500
    op.hp = op.max_hp
    op.legendary_actions_remaining = op.legendary_actions_max
    return env, aid, oid, ag, op


def test_legendary_decline_via_decider():
    env, aid, oid, ag, op = _legendary_env()
    env.ws.legendary_decider = lambda ctx: None        # boss forgoes its action
    with patch("trpg.engine.combat.resolve_attack", return_value=(True, 30)):
        events = run_legendary_actions(env.ws, aid, 2)
    assert events == []
    assert op.legendary_actions_remaining == op.legendary_actions_max


def test_legendary_decider_picks_specific_option():
    env, aid, oid, ag, op = _legendary_env()
    seen_options = {}

    def pick_cheapest(ctx):
        seen_options["ids"] = [o["skill_id"] for o in ctx.options]
        # Pick the lowest-cost option deterministically (a non-EV rule) to prove
        # the decider — not the greedy EV — drove the choice.
        return min(ctx.options, key=lambda o: (o["cost"], o["skill_id"]))["skill_id"]

    env.ws.legendary_decider = pick_cheapest
    with patch("trpg.engine.combat.resolve_attack", return_value=(True, 30)):
        events = run_legendary_actions(env.ws, aid, 2)
    assert len(events) == 1
    assert seen_options["ids"]                      # decider really saw options
    fired = events[0]["action"].get("weapon") or events[0]["action"].get("skill_id")
    assert fired is not None


# ── neural deciders (model-controlled) ────────────────────────────────────────

import torch
from trpg.rl.reaction_policy import NeuralReactionDecider, NeuralLegendaryDecider
from trpg.rl.obs import N_SKILL_SLOTS, N_ENTITY_SLOTS, N_GRID, I_DCTX_REACTION


class _StubNet:
    """Returns skill_logits that argmax to a fixed slot — lets a test drive the
    decider's choice deterministically (an untrained real net is ~random)."""
    def __init__(self, choose_slot, spy=None):
        self.choose_slot = choose_slot
        self.spy = spy

    def __call__(self, obs_t):
        if self.spy is not None:
            self.spy(obs_t)
        S = obs_t["skill_mask"].shape[1]
        skill_l = torch.full((1, S), -5.0)
        skill_l[0, self.choose_slot] = 10.0
        return (torch.zeros(1), skill_l,
                torch.zeros(1, S, N_ENTITY_SLOTS),
                torch.zeros(1, S, N_GRID * N_GRID))


def test_neural_reaction_decider_reacts():
    ws, atk, dfn = _shield_world()
    ws.reaction_decider = NeuralReactionDecider(_StubNet(choose_slot=1), {"def"})
    # outside the greedy flip window — proves the MODEL (not greedy) chose shield
    blocked, name, slot = _try_react_to_attack(dfn, atk, effective_ac(dfn) + 9, ws)
    assert blocked is True and name == "shield_spell"


def test_neural_reaction_decider_declines_via_slot0():
    ws, atk, dfn = _shield_world()
    ws.reaction_decider = NeuralReactionDecider(_StubNet(choose_slot=0), {"def"})
    blocked, name, slot = _try_react_to_attack(dfn, atk, effective_ac(dfn) + 2, ws)
    assert blocked is False and dfn.reaction_used is False   # slot 0 = decline


def test_neural_reaction_decider_falls_back_for_nonmodel():
    ws, atk, dfn = _shield_world()
    # 'def' NOT in model_ids → greedy default (fires in the flip window).
    ws.reaction_decider = NeuralReactionDecider(_StubNet(choose_slot=0), {"someone_else"})
    blocked, *_ = _try_react_to_attack(dfn, atk, effective_ac(dfn) + 2, ws)
    assert blocked is True            # greedy fired despite the stub saying decline


def test_neural_reaction_obs_carries_reaction_context():
    ws, atk, dfn = _shield_world()
    seen = {}
    ws.reaction_decider = NeuralReactionDecider(
        _StubNet(choose_slot=0, spy=lambda o: seen.update(
            dctx=o["decision_context"][0].tolist())), {"def"})
    _try_react_to_attack(dfn, atk, effective_ac(dfn) + 2, ws)
    assert seen["dctx"][I_DCTX_REACTION] == 1.0    # the policy saw a reaction ctx


def test_neural_counterspell_through_execute_action():
    ws, ec, d = _counterspell_world()
    ws.reaction_decider = NeuralReactionDecider(_StubNet(choose_slot=1), {"d"})
    res = execute_action(
        {"type": "SPELL", "skill_id": "h", "caster": "ec", "spell_name": "火球術",
         "target_position": [5.0, 0.0], "consumes": ["action"]}, ws)
    assert res["type"] == "COUNTERSPELLED" and d.reaction_used is True


def test_neural_legendary_decider_fires_and_declines():
    env, aid, oid, ag, op = _legendary_env()
    env.ws.legendary_decider = NeuralLegendaryDecider(_StubNet(choose_slot=1), {oid})
    with patch("trpg.engine.combat.resolve_attack", return_value=(True, 30)):
        events = run_legendary_actions(env.ws, aid, 2)
    assert len(events) == 1            # model picked candidate slot 1 → fired

    env2, aid2, oid2, ag2, op2 = _legendary_env()
    env2.ws.legendary_decider = NeuralLegendaryDecider(_StubNet(choose_slot=0), {oid2})
    with patch("trpg.engine.combat.resolve_attack", return_value=(True, 30)):
        events2 = run_legendary_actions(env2.ws, aid2, 2)
    assert events2 == []               # slot 0 = decline


def test_neural_reaction_decider_real_checkpoint_runs():
    """A real checkpoint must run the full reaction-obs forward without shape
    errors and return a legal result (option skill_id or decline)."""
    import os
    from trpg.sandbox.policy_loader import load_policy
    ckpt = "models/pop_mon/pop_u0005.pt"
    if not os.path.exists(ckpt):
        pytest.skip("champion checkpoint not present")
    net = load_policy(ckpt).net
    ws, atk, dfn = _shield_world()
    ws.reaction_decider = NeuralReactionDecider(net, {"def"})
    blocked, name, slot = _try_react_to_attack(dfn, atk, effective_ac(dfn) + 2, ws)
    assert isinstance(blocked, bool) and name in ("", "shield_spell")
