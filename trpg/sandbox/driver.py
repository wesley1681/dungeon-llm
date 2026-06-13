"""Combat driver for the sparring sandbox.

Drives turns by hand using engine primitives — bypasses env_v2 because we
don't need obs/reward/MultiDiscrete wrappers. Loop structure mirrors
env_v2.step + _run_opponent_turn for tick ordering, but agent decisions
come from a Frontend instead of a decoded MultiDiscrete action.
"""
from __future__ import annotations

from ..engine.combat import (
    execute_action, consume_resources, MOVE_BUDGET_M, tick_terrain_damage,
    tick_aura_damage,
)
from ..engine.combat_policy import run_legendary_actions
from ..engine.status import tick_status_effects


_MAX_SUB_ACTIONS_PER_TURN = 5


def _announce_pending_events(ws, frontend) -> None:
    """Drain world-level events queued by the engine outside action results
    (death throes detonations) and narrate them."""
    events = getattr(ws, "pending_events", None)
    if not events:
        return
    for ev in events:
        if ev.get("type") == "DEATH_THROES":
            ok = "豁免成功（半傷）" if ev["save_success"] else "豁免失敗"
            frontend.announce(
                f"💥 {ev['source_name']} 死亡爆炸（{ev['damage_dice']} "
                f"{ev['damage_type']}）→ {ev['target_name']}：{ok}，"
                f"{ev['damage']} 點傷害（HP {ev['target_hp']}）")
    events.clear()


def _run_legendary_phase(ended_id: str, ws, frontend) -> None:
    """Fire legendary actions at the end of `ended_id`'s turn and narrate."""
    for ev in run_legendary_actions(ws, ended_id, ws.combat.round_number):
        actor = ws.characters[ev["actor_id"]]
        frontend.announce(
            f"⚡ 傳奇行動（{ev['actor_name']}，花費 {ev['cost']}，"
            f"剩餘 {ev['remaining']}）")
        for line in _format_result(actor, ev["result"]):
            frontend.announce(line)
    _announce_pending_events(ws, frontend)


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
    actor.leveled_spell_cast_this_turn = False
    round_num = ws.combat.round_number
    tick_status_effects(actor, "self_turn_start", round_num)
    tick_terrain_damage(actor, ws.combat.battlefield)
    for ev in tick_aura_damage(actor, ws, round_num):
        et = ev.get("type")
        if et == "FRIGHTFUL_PRESENCE":
            out = ("豁免成功（本場免疫）" if ev["save_success"]
                   else "豁免失敗 → frightened")
            frontend.announce(
                f"[{actor.name}] 恐懼威壓（{ev['source_name']}，"
                f"WIS DC{ev['save_dc']}，擲 {ev['save_roll']}）：{out}")
        elif et == "AURA_DAMAGE":
            frontend.announce(
                f"[{actor.name}] 受 {ev['source_name']} 的"
                f"{ev['spell_name']}光環 {ev['damage']} 點傷害"
                f"（HP {ev['target_hp']}/{ev['target_max_hp']}）")
        elif et == "STATUS_TICK_DAMAGE":
            frontend.announce(
                f"[{actor.name}] {ev['status_name']} 持續傷害 "
                f"{ev['damage']}（{ev['damage_type']}，HP {ev['target_hp']}）")
    _announce_pending_events(ws, frontend)

    # Incapacitated (status.INCAPACITATING_STATUSES): skip the entire turn.
    if actor.is_incapacitated():
        from ..engine.status import INCAPACITATING_STATUSES
        cond = next(s for s in INCAPACITATING_STATUSES if actor.has_status(s))
        frontend.announce(f"[{actor.name}] 處於 {cond} 狀態，跳過回合")
        tick_status_effects(actor, "self_turn_end", round_num)
        _run_legendary_phase(actor_id, ws, frontend)
        return

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
        # Pre-execute resource gate: consume_resources floors silently at 0,
        # and execute_action doesn't know about per-turn budgets — so without
        # this check you could cast two action-cost spells in one turn by
        # picking them sequentially while bonus_action / movement still has
        # value. Refuse early instead.
        consumes = action.get("consumes") or []
        missing = []
        if "action" in consumes and resources.get("action", 0) <= 0:
            missing.append("action")
        if "bonus_action" in consumes and resources.get("bonus_action", 0) <= 0:
            missing.append("bonus_action")
        if missing:
            msg = f"資源不足: {','.join(missing)}"
            frontend.announce(f"[{actor.name}] 動作失敗: {msg}")
            if event_log is not None:
                event_log(_build_event(round_num, sub_idx, actor, action,
                                        {"type": "ERROR", "message": msg},
                                        resources_before, resources, ws))
            continue
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
        # Limited-use deduction now happens inside execute_action (shared
        # funnel) — a driver-level copy here would double-deduct.
        for line in _format_result(actor, result):
            frontend.announce(line)
        _announce_pending_events(ws, frontend)
        if event_log is not None:
            event_log(_build_event(round_num, sub_idx, actor, action, result,
                                    resources_before, resources, ws))
        if (resources["action"] <= 0 and resources["bonus_action"] <= 0
                and resources["movement"] <= 1e-6):
            break
    tick_status_effects(actor, "self_turn_end", round_num)
    _run_legendary_phase(actor_id, ws, frontend)


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
        # Engine key is "attacks" ("hits" was never emitted — every
        # multiattack line used to render as 0/0).
        hits = result.get("attacks", [])
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
        td = result.get("terrain_damage", 0)
        if td:
            hp = f"{result.get('target_hp', '?')}/{result.get('target_max_hp', '?')}"
            lines.append(f"  └ 途經危險地形: {td}傷  HP {hp}")
        for ev in result.get("path_events", []):
            kind = ev.get("kind")
            pos = ev.get("pos", ("?", "?"))
            zh = {"hit_wall": "撞牆停下",
                  "out_of_bounds": "撞邊界停下",
                  "budget_exhausted": "移動額度用盡"}.get(kind, kind)
            lines.append(f"  └ {zh} @ ({pos[0]:.1f},{pos[1]:.1f})")
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

    if t == "EYE_RAYS":
        dc = result.get("save_dc", 0)
        lines.append(f"[{name}] 眼魔射線 ×{len(result.get('rays', []))}"
                     f"  (DC{dc}, 共 {result.get('total_damage', 0)}傷)")
        for ray in result.get("rays", []):
            tgt = ray.get("target_name", "?")
            ok = "成功" if ray.get("save_success") else "失敗"
            bits = f"{ray.get('ray_name', '?')} → {tgt}: " \
                   f"{ray.get('save_stat', '?')}豁免{ok} (擲{ray.get('save_roll', 0)})"
            if ray.get("damage"):
                hp = f"{ray.get('target_hp', '?')}"
                bits += f" → {ray['damage']}傷  HP {hp}"
            if ray.get("status_applied"):
                bits += f" → 施加 {ray['status_applied']}"
            lines.append(f"  └ {bits}")
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
