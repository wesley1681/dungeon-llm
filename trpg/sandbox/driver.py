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


_MAX_SUB_ACTIONS_PER_TURN = 5


def _fresh_resources() -> dict:
    return {"action": 1, "bonus_action": 1, "movement": MOVE_BUDGET_M}


def _run_one_turn(actor_id: str, ws, decider, frontend,
                   event_log=None) -> None:
    """Run one combatant's full turn.

    decider(actor_id, actor, ws, resources, round_num) -> action_dict | None
      Returns None to voluntarily end the turn.
    event_log(event_dict) -> None  (optional)
      Called for every sub-action attempt (incl. ERRORs and end-of-turn).
    """
    actor = ws.characters[actor_id]
    if not actor.is_alive():
        return
    actor.reaction_used = False
    round_num = ws.combat.round_number
    tick_status_effects(actor, "self_turn_start", round_num)
    tick_terrain_damage(actor, ws.combat.battlefield)

    resources = _fresh_resources()
    for sub_idx in range(_MAX_SUB_ACTIONS_PER_TURN):
        if not actor.is_alive():
            break
        resources_before = dict(resources)
        action = decider(actor_id, actor, ws, resources, round_num)
        if action is None:
            if event_log is not None:
                event_log(_build_event(round_num, sub_idx, actor, None, None,
                                        resources_before, resources, ws))
                frontend.announce(f"[{actor.name}] 結束回合")
            break
        result = execute_action(action, ws)
        skill_id = action.get("skill_id")
        if result.get("type") == "ERROR":
            for line in _format_result(actor, result):
                frontend.announce(line)
            if event_log is not None:
                event_log(_build_event(round_num, sub_idx, actor, action,
                                        result, resources_before, resources, ws))
            continue
        consume_resources(resources, action, result)
        if (action.get("type") == "MOVE"
                and (result.get("distance", 0) < 0.01
                     or resources["movement"] < 0.5)):
            resources["movement"] = 0.0
        if skill_id is not None:
            ab = CLASS_ABILITIES.get(skill_id)
            if ab is not None and ab.max_uses > 0:
                actor.ability_uses[skill_id] = (
                    actor.ability_uses.get(skill_id, ab.max_uses) - 1
                )
        for line in _format_result(actor, result):
            frontend.announce(line)
        if event_log is not None:
            event_log(_build_event(round_num, sub_idx, actor, action, result,
                                    resources_before, resources, ws))
        if (resources["action"] <= 0 and resources["bonus_action"] <= 0
                and resources["movement"] <= 1e-6):
            break
    tick_status_effects(actor, "self_turn_end", round_num)


def _format_result(actor, result: dict) -> list[str]:
    """Verbose multi-line breakdown of an action result.

    Everything visible at the table — hit/miss + dice + AC, save throws +
    DC, reactions fired (shield_spell, counterspell), per-target damage,
    status applied, AOE coords — gets surfaced so the player can audit
    every step. Returns a list of lines so the caller can announce each
    individually.
    """
    t = result.get("type", "?")
    name = actor.name
    lines: list[str] = []

    if t == "ATTACK":
        tgt = result.get("target_name", "?")
        roll = result.get("roll", 0)
        ac = result.get("target_ac", 0)
        weap = result.get("weapon_name", "?")
        mode = result.get("advantage_mode", "normal")
        mode_tag = {"advantage": " 優勢", "disadvantage": " 劣勢"}.get(mode, "")
        if result.get("hit"):
            dmg = result.get("damage", 0)
            extra = []
            if result.get("sneak_attack_damage"):
                extra.append(f"偷襲 +{result['sneak_attack_damage']}")
            if result.get("smite_damage"):
                extra.append(f"神聖斬擊 +{result['smite_damage']}")
            if result.get("hunters_mark_damage"):
                extra.append(f"獵人標記 +{result['hunters_mark_damage']}")
            extras = f"  ({', '.join(extra)})" if extra else ""
            lines.append(f"[{name}] 攻擊[{weap}] → {tgt}: 命中 {dmg}傷"
                         f"  擲骰{roll}{mode_tag} ≥ AC{ac}{extras}")
        else:
            tag = "反應擋下" if result.get("reaction") else ""
            if tag:
                lines.append(f"[{name}] 攻擊[{weap}] → {tgt}: 被 {result['reaction']} {tag}"
                             f"  擲骰{roll}{mode_tag}")
            else:
                lines.append(f"[{name}] 攻擊[{weap}] → {tgt}: 未命中"
                             f"  擲骰{roll}{mode_tag} < AC{ac}")
        return lines

    if t == "MULTI_ATTACK":
        hits = result.get("hits", [])
        total_dmg = sum(h.get("damage", 0) for h in hits)
        n_hit = sum(1 for h in hits if h.get("hit"))
        lines.append(f"[{name}] 多重攻擊 ({n_hit}/{len(hits)} 命中, 共 {total_dmg}傷)")
        for h in hits:
            tgt = h.get("target_name", "?")
            roll = h.get("roll", 0)
            ac = h.get("target_ac", 0)
            if h.get("hit"):
                lines.append(f"  └ → {tgt}: 命中 {h.get('damage', 0)}傷  擲骰{roll} ≥ AC{ac}")
            else:
                lines.append(f"  └ → {tgt}: 未命中  擲骰{roll} < AC{ac}")
        return lines

    if t == "SPELL":
        spell = result.get("spell_name", "?")
        slot = result.get("slot_level", 0)
        center = result.get("center_name", "?")
        save = result.get("save_stat", "")
        dc = result.get("save_dc", 0)
        slot_tag = f" {slot}環" if slot > 0 else " (戲法)"
        head = f"[{name}] 施法[{spell}{slot_tag}] 中心={center}"
        if save:
            head += f"  DC{dc} {save}豁免"
        lines.append(head)
        for tr in result.get("target_results", []):
            tgt = tr.get("target_name", "?")
            dmg = tr.get("damage", 0)
            hp = f"{tr.get('target_hp', '?')}/{tr.get('target_max_hp', '?')}"
            if "save_success" in tr:
                ok = "成功" if tr["save_success"] else "失敗"
                sroll = tr.get("save_roll", 0)
                bits = f"豁免{ok} (擲{sroll} vs DC{dc})"
                lines.append(f"  └ {tgt}: {bits} → {dmg}傷  HP {hp}")
            else:
                lines.append(f"  └ {tgt}: {dmg}傷  HP {hp}")
            if tr.get("status_applied"):
                lines.append(f"    施加狀態: {tr['status_applied']}")
        if result.get("countered"):
            lines.append(f"  ✗ 被 {result.get('counterspeller', '?')} 反制 "
                         f"(消耗 {result.get('slot_used', '?')}環)")
        return lines

    if t == "AUTO_DAMAGE":
        dmg_per = result.get("damage_per", "?")
        slot = result.get("slot_level", 0)
        slot_tag = f" {slot}環" if slot > 0 else ""
        lines.append(f"[{name}] 自動命中{slot_tag} ({dmg_per})")
        for tr in result.get("target_results", []):
            tgt = tr.get("target_name", "?")
            dmg = tr.get("damage", 0)
            hp = f"{tr.get('target_hp', '?')}/{tr.get('target_max_hp', '?')}"
            if tr.get("reaction"):
                slot_used = tr.get("reaction_slot", "?")
                lines.append(f"  └ {tgt}: 被 {tr['reaction']} 擋下"
                             f" (消耗 {slot_used}環)  HP {hp}")
            else:
                lines.append(f"  └ {tgt}: {dmg}傷  HP {hp}")
        return lines

    if t == "MOVE":
        f = result.get("from_pos")
        to = result.get("to_pos")
        dist = result.get("distance", 0)
        from_str = f"({f.x:.1f},{f.y:.1f})" if hasattr(f, "x") else "?"
        to_str = f"({to.x:.1f},{to.y:.1f})" if hasattr(to, "x") else "?"
        tag = " [瞬移]" if result.get("teleport") else ""
        lines.append(f"[{name}] 移動{tag} {from_str} → {to_str}  距離 {dist:.1f}m")
        for oa in result.get("opportunity_attacks", []):
            ao = oa.get("attacker", "?")
            if oa.get("hit"):
                lines.append(f"  └ {ao} 的機會攻擊命中: {oa.get('damage', 0)}傷")
            else:
                lines.append(f"  └ {ao} 的機會攻擊未命中")
        return lines

    if t == "APPLY_MOD":
        mod = result.get("modifier", "?")
        targets = result.get("targets_affected", [])
        slot = result.get("slot_level", 0)
        slot_tag = f" {slot}環" if slot > 0 else ""
        dur = result.get("duration_rounds")
        dur_tag = f"  持續 {dur} 回合" if dur else ""
        lines.append(f"[{name}] 施加[{mod}]{slot_tag} → {', '.join(targets) or '無'}{dur_tag}")
        return lines

    if t in ("HEAL", "LAY_ON_HANDS"):
        amt = result.get("amount", result.get("healed", 0))
        tgt = result.get("target_name", name)
        lines.append(f"[{name}] 治療 +{amt} → {tgt}")
        return lines

    if t == "ACTION_SURGE":
        lines.append(f"[{name}] 行動激增（多得一個 action）")
        return lines

    if t in ("HIDE", "DODGE", "DISENGAGE"):
        zh = {"HIDE": "躲藏", "DODGE": "閃避", "DISENGAGE": "脫離"}
        lines.append(f"[{name}] {zh[t]}")
        return lines

    if t == "ERROR":
        lines.append(f"[{name}] 動作失敗: {result.get('message', '')}")
        return lines

    # Fallback
    lines.append(f"[{name}] {t}")
    return lines


def _vec(v) -> list[float] | None:
    """Coerce Vec2-or-tuple-or-None to [x, y] for JSON serialisation."""
    if v is None:
        return None
    if hasattr(v, "x") and hasattr(v, "y"):
        return [float(v.x), float(v.y)]
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return [float(v[0]), float(v[1])]
    return None


def _build_event(round_num: int, sub_idx: int, actor, action: dict | None,
                  result: dict | None, resources_before: dict,
                  resources_after: dict, ws) -> dict:
    """Snapshot one sub-action as a flat dict suitable for JSONL logging.

    Captures everything needed to reason about "wait why did X happen":
    actor identity & position, action emitted, result returned (full —
    rolls, saves, reactions all in there), resource state both sides of
    the action, current HP for every combatant.
    """
    ev: dict = {
        "round":     round_num,
        "sub":       sub_idx,
        "actor":     actor.name,
        "actor_pos": _vec(actor.position),
        "actor_hp":  f"{actor.hp}/{actor.max_hp}",
        "resources_before": dict(resources_before),
        "resources_after":  dict(resources_after),
        "action":    None,
        "result":    None,
        "hp_after":  {c.name: f"{c.hp}/{c.max_hp}" for c in ws.characters.values()},
        "spell_slots": {c.name: dict(c.spell_slots) for c in ws.characters.values()
                         if c.spell_slots},
    }
    if action is not None:
        ev["action"] = {
            "type":           action.get("type"),
            "skill_id":       action.get("skill_id"),
            "spell_name":     action.get("spell_name"),
            "slot_level":     action.get("slot_level"),
            "target":         action.get("target") or action.get("character"),
            "target_position": _vec(action.get("target_position")),
            "consumes":       action.get("consumes"),
        }
    if result is not None:
        # Result dicts can have Vec2 inside — flatten for JSON serialisation.
        clean = {}
        for k, v in result.items():
            if k in ("from_pos", "to_pos", "target_position"):
                clean[k] = _vec(v)
            else:
                clean[k] = v
        ev["result"] = clean
    return ev


def run_combat(ws, frontend, opponent_policy, *,
               agent_id: str = "agent", opp_id: str = "opponent",
               max_rounds: int = 20, event_log=None) -> dict:
    """Drive a sparring match. Returns {"outcome", "rounds"}.

    outcome:
      "win"       — opponent died
      "loss"      — agent died
      "truncated" — both alive after max_rounds
    """
    def user_decider(_aid, actor, _ws, resources, _round):
        # Re-render before each user sub-action so the player sees the
        # post-action state (their own moves, opp moves between rounds, etc.)
        # without waiting for the next round's start-of-round render.
        frontend.render(ws, agent_id, opp_id)
        return frontend.prompt_action(ws, actor, resources)

    def opp_decider(aid, actor, _ws, resources, round_num):
        try:
            decision = opponent_policy.decide(aid, actor, ws, resources, round_num)
        except Exception as exc:
            frontend.announce(f"[模型錯誤] {exc} — 強制結束回合")
            return None
        return decision.action if not decision.ended else None

    rounds = 0
    while True:
        rounds += 1
        ws.combat.round_number = rounds
        # No render here — user_decider re-renders before each sub-action,
        # which covers the start-of-round display for the human side.
        _run_one_turn(agent_id, ws, user_decider, frontend, event_log=event_log)
        if not ws.characters[opp_id].is_alive():
            return {"outcome": "win", "rounds": rounds}
        if not ws.characters[agent_id].is_alive():
            return {"outcome": "loss", "rounds": rounds}
        _run_one_turn(opp_id, ws, opp_decider, frontend, event_log=event_log)
        if not ws.characters[opp_id].is_alive():
            return {"outcome": "win", "rounds": rounds}
        if not ws.characters[agent_id].is_alive():
            return {"outcome": "loss", "rounds": rounds}
        # env_v2 parity: round_end tick so rounds_remaining countdowns work
        for cid in ws.combat.initiative_order:
            c = ws.characters.get(cid)
            if c and c.is_alive():
                tick_status_effects(c, "round_end", rounds)
        if rounds >= max_rounds:
            return {"outcome": "truncated", "rounds": rounds}
