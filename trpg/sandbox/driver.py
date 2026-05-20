"""Combat driver for the sparring sandbox.

Drives turns by hand using engine primitives — bypasses env_v2 because we
don't need obs/reward/MultiDiscrete wrappers. Loop structure mirrors
env_v2.step + _run_opponent_turn for tick ordering, but agent decisions
come from a Frontend instead of a decoded MultiDiscrete action.
"""
from __future__ import annotations

from ..engine.combat import (
    execute_action, consume_resources, MOVE_BUDGET_M, tick_terrain_damage,
)
from ..engine.status import tick_status_effects
from ..engine.abilities import CLASS_ABILITIES
from ..engine.skill import available_skills


_MAX_SUB_ACTIONS_PER_TURN = 5


def _fresh_resources() -> dict:
    return {"action": 1, "bonus_action": 1, "movement": MOVE_BUDGET_M}


def _run_one_turn(actor_id: str, ws, decider, frontend) -> None:
    """Run one combatant's full turn.

    decider(actor_id, actor, ws, resources, round_num) -> action_dict | None
      Returns None to voluntarily end the turn.
    """
    actor = ws.characters[actor_id]
    if not actor.is_alive():
        return
    actor.reaction_used = False
    round_num = ws.combat.round_number
    tick_status_effects(actor, "self_turn_start", round_num)
    tick_terrain_damage(actor, ws.combat.battlefield)

    resources = _fresh_resources()
    for _ in range(_MAX_SUB_ACTIONS_PER_TURN):
        if not actor.is_alive():
            break
        # Snapshot skill list before mutation (matches env_v2 step ordering)
        skills_before = available_skills(actor, ws)
        action = decider(actor_id, actor, ws, resources, round_num)
        if action is None:
            break
        result = execute_action(action, ws)
        skill_id = action.get("skill_id")
        if result.get("type") == "ERROR":
            frontend.announce(f"[{actor.name}] 動作失敗: {result.get('message', '')}")
            # Don't consume resources on ERROR — same as env_v2 step path
            continue
        consume_resources(resources, action, result)
        # Decrement ability uses on success (mirrors env_v2.step)
        if skill_id is not None:
            ab = CLASS_ABILITIES.get(skill_id)
            if ab is not None and ab.max_uses > 0:
                actor.ability_uses[skill_id] = (
                    actor.ability_uses.get(skill_id, ab.max_uses) - 1
                )
        frontend.announce(_format_result(actor, result))
        if (resources["action"] <= 0 and resources["bonus_action"] <= 0
                and resources["movement"] <= 1e-6):
            break
    tick_status_effects(actor, "self_turn_end", round_num)


def _format_result(actor, result: dict) -> str:
    """One-line summary of an action result for the announce log."""
    t = result.get("type", "?")
    if t == "ATTACK" or t == "MULTI_ATTACK":
        if result.get("hits"):
            dmg = sum(h.get("damage", 0) for h in result.get("hits", []))
            return f"[{actor.name}] {t}: {dmg} 傷"
        return f"[{actor.name}] {t}: 未命中"
    if t == "SPELL":
        name = result.get("spell_name", "?")
        targets = result.get("target_results", [])
        dmg = sum(tr.get("damage", 0) for tr in targets)
        if dmg:
            return f"[{actor.name}] {name} → {dmg} 傷 ({len(targets)} 個目標)"
        return f"[{actor.name}] {name}"
    if t == "MOVE":
        return f"[{actor.name}] 移動 {result.get('distance', 0):.1f}m"
    if t == "APPLY_MOD":
        return f"[{actor.name}] {result.get('modifier', '?')} 施加於 {len(result.get('targets_affected', []))} 個目標"
    if t == "HEAL" or t == "LAY_ON_HANDS":
        return f"[{actor.name}] 治療 {result.get('amount') or result.get('healed', 0)}"
    return f"[{actor.name}] {t}"


def run_combat(ws, frontend, opponent_policy, *,
               agent_id: str = "agent", opp_id: str = "opponent",
               max_rounds: int = 20) -> dict:
    """Drive a sparring match. Returns {"outcome", "rounds"}.

    outcome:
      "win"       — opponent died
      "loss"      — agent died
      "truncated" — both alive after max_rounds
    """
    def user_decider(_aid, actor, _ws, resources, _round):
        return frontend.prompt_action(ws, actor, resources)

    def opp_decider(aid, actor, _ws, resources, round_num):
        decision = opponent_policy.decide(aid, actor, ws, resources, round_num)
        return decision.action if not decision.ended else None

    rounds = 0
    while True:
        rounds += 1
        ws.combat.round_number = rounds
        frontend.render(ws, agent_id, opp_id)
        _run_one_turn(agent_id, ws, user_decider, frontend)
        if not ws.characters[opp_id].is_alive():
            return {"outcome": "win", "rounds": rounds}
        if not ws.characters[agent_id].is_alive():
            return {"outcome": "loss", "rounds": rounds}
        _run_one_turn(opp_id, ws, opp_decider, frontend)
        if not ws.characters[opp_id].is_alive():
            return {"outcome": "win", "rounds": rounds}
        if not ws.characters[agent_id].is_alive():
            return {"outcome": "loss", "rounds": rounds}
        if rounds >= max_rounds:
            return {"outcome": "truncated", "rounds": rounds}
