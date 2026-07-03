import pytest
from trpg.engine.vec2 import Vec2
from trpg.engine.combat_policy import HeuristicCombatPolicy
from trpg.sandbox.setup import build_world_state
from trpg.sandbox.frontend import ScriptedFrontend
from trpg.sandbox.driver import run_combat


def _kill_opponent_actions(ws):
    """Scripted sequence: attack until dead, then end. We introspect the
    agent's first weapon to build a real action_dict via the engine's own
    builder — avoids hard-coding a weapon name that may not exist on the
    chosen archetype."""
    from trpg.engine.skill import available_skills
    actor = ws.characters["agent"]
    skills = available_skills(actor, ws)
    weapon_sk = next(s for s in skills if s.skill_id.startswith("weapon:"))
    actions = []
    for _ in range(50):
        actions.append(weapon_sk.build_action("agent", "opponent", None))
    actions.append(None)
    return actions


def test_run_combat_returns_outcome_dict():
    ws = build_world_state(
        agent_arch="berserker", opponent_arch="berserker", level=3,
        agent_pos=Vec2(15.0, 15.5), opp_pos=Vec2(15.0, 16.5),  # adjacent
        terrain="empty",
    )
    # max_rounds=2 means the user may be prompted twice; supply one None
    # per round so the iterator doesn't run short.
    fe = ScriptedFrontend([None, None])
    opp_pol = HeuristicCombatPolicy()
    out = run_combat(ws, fe, opp_pol, max_rounds=2)
    assert "outcome" in out
    assert out["outcome"] in {"win", "loss", "truncated"}
    assert "rounds" in out
    assert out["rounds"] >= 1


def test_run_combat_user_can_kill_opponent():
    # 活骰測試：種全域骰，否則依賴前面測試消耗的 RNG 流＝順序脆弱
    #（20 回合上限內殺不掉 L3 evocation 是機率事件，不種子會偶發翻紅）
    import random
    random.seed(7)
    ws = build_world_state(
        agent_arch="berserker", opponent_arch="evocation", level=3,
        agent_pos=Vec2(15.0, 15.5), opp_pos=Vec2(15.0, 16.0),  # in melee
        terrain="empty",
    )
    fe = ScriptedFrontend(_kill_opponent_actions(ws))
    # Opponent does nothing useful — they just end their turn
    class _NoopPolicy:
        def decide(self, *a, **kw):
            from trpg.engine.combat_policy import CombatDecision
            return CombatDecision(ended=True)
    out = run_combat(ws, fe, _NoopPolicy(), max_rounds=20)
    assert out["outcome"] == "win"
    assert not ws.characters["opponent"].is_alive()


def test_run_combat_truncates_at_max_rounds():
    ws = build_world_state(
        agent_arch="evocation", opponent_arch="evocation", level=5,
        agent_pos=Vec2(5.0, 5.0), opp_pos=Vec2(25.0, 25.0),
        terrain="empty",
    )
    # Both sides end their turn immediately — fight never resolves
    fe = ScriptedFrontend([None] * 30)

    class _NoopPolicy:
        def decide(self, *a, **kw):
            from trpg.engine.combat_policy import CombatDecision
            return CombatDecision(ended=True)
    out = run_combat(ws, fe, _NoopPolicy(), max_rounds=3)
    assert out["outcome"] == "truncated"
    assert out["rounds"] == 3
