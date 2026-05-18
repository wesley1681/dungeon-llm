"""Combat decision policies — what every combatant calls to pick its sub-action.

A CombatPolicy returns a structured action dict (the same shape execute_action
consumes) directly. No LLM is invoked in this layer. The LLM's only combat role
is post-hoc narration once the action has resolved.

Implementations included here:
  - EndTurnPolicy            trivial stub; ends the turn immediately
  - HeuristicCombatPolicy    rule-based baseline used as the default until a
                             trained RL policy plugs in; also doubles as the
                             scripted opponent for RL training.
  - HumanInputPolicy         reads structured command text from a UI callback
                             and parses it into an action dict via regex.
                             No LLM in the parsing path.

The RL controller will land as a separate subclass that wraps a trained model.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from .character import Character


@dataclass
class CombatDecision:
    action: dict | None = None   # execute_action-shape dict, or None to skip
    ended: bool = False          # turn ends after this sub-action
    fled: bool = False           # actor leaves combat entirely


class CombatPolicy:
    """Abstract policy. Given the actor's observation, return a CombatDecision."""

    def decide(self, actor_id: str, actor: Character, world_state,
               resources: dict, round_num: int) -> CombatDecision:
        raise NotImplementedError


class EndTurnPolicy(CombatPolicy):
    """Always ends the turn. Used as a placeholder for human-controlled PCs
    until a structured-action UI is wired up; once that exists, swap in a
    HumanInputPolicy that reads structured choices from the event queue."""

    def decide(self, actor_id, actor, world_state, resources, round_num):
        return CombatDecision(ended=True)


class HeuristicCombatPolicy(CombatPolicy):
    """Greedy melee baseline.

    Picks the nearest valid target and:
      - attacks if in weapon reach (and action available)
      - otherwise moves toward them (if movement available)
      - otherwise ends the turn
    Does not consider spells, dodging, or cover yet. Good enough as the default
    NPC opponent and as a scripted baseline to compare RL training runs against.
    """

    def decide(self, actor_id, actor, world_state, resources, round_num):
        ws = world_state
        is_party = ws.is_party_ally(actor_id)
        room = ws.dungeon_map.current_room if ws.dungeon_map else None
        room_ids = set(room.npc_ids) if room else set(ws.characters.keys())

        candidates: list[tuple[str, Character]] = []
        for cid, c in ws.characters.items():
            if cid == actor_id or not c.is_alive():
                continue
            other_party = ws.is_party_ally(cid)
            if is_party:
                if c.is_npc and c.attitude == 0 and cid in room_ids:
                    candidates.append((cid, c))
            else:
                if other_party:
                    candidates.append((cid, c))

        if not candidates:
            return CombatDecision(ended=True)

        target_id, target = min(
            candidates, key=lambda kv: actor.position.distance_to(kv[1].position)
        )

        weapon = actor.get_weapon() if actor.weapons else None
        if weapon is None:
            return CombatDecision(ended=True)

        d = actor.position.distance_to(target.position)
        reach = weapon.range_normal or 1.5

        if d <= reach + 1e-6 and resources.get("action", 0) > 0:
            return CombatDecision(action={
                "type":     "ATTACK",
                "attacker": actor_id,
                "target":   target_id,
                "weapon":   weapon.name,
                "consumes": ["action"],
            })

        # Only walk when actually out of reach — otherwise the MOVE handler
        # would return a 0m no-op (combat.py keeps a 1m gap from creature
        # targets), burning sub-actions on idle steps.
        if d > reach + 1e-6 and resources.get("movement", 0.0) > 1e-6:
            return CombatDecision(action={
                "type":        "MOVE",
                "character":   actor_id,
                "target":      target_id,
                "description": "靠近",
                "consumes":    ["movement"],
            })

        return CombatDecision(ended=True)


# ── Human input policy ───────────────────────────────────────────────────────

_END_WORDS       = {"end", "結束", "結束回合", "我這回合到這"}
_FLEE_WORDS      = {"flee", "逃跑", "<flee>"}
_DODGE_WORDS     = {"dodge", "閃避", "專注防禦"}
_HIDE_WORDS      = {"hide", "躲藏", "躲"}
_DISENGAGE_WORDS = {"disengage", "脫身", "安全撤退"}


def _resolve_id(key: str, world_state) -> str | None:
    """Match a UI-typed identifier against character id then display name."""
    if key in world_state.characters:
        return key
    for cid, c in world_state.characters.items():
        if c.name == key:
            return cid
    return None


def _parse_command(text: str, actor_id: str, world_state) -> dict:
    """Parse one line of human input into an engine action dict.

    Grammar (whitespace-separated, case-insensitive on keywords; targets and
    spell/weapon names are case-sensitive):

      attack <target>                ATTACK with default weapon
      attack <target> <weapon>       ATTACK with named weapon
      move <x> <y>                   MOVE to absolute (x, y)
      move +<dx> +<dy>               MOVE by delta
      move <target>                  MOVE toward creature
      spell <name> <x> <y>           SPELL at coord
      spell <name> <target>          SPELL on creature
      dodge / 閃避                   DODGE
      hide / 躲藏                    HIDE

    Raises ValueError with a user-facing message on parse failure.
    """
    parts = text.strip().split()
    if not parts:
        raise ValueError("空指令")
    cmd = parts[0].lower()

    if cmd in ("attack", "攻擊"):
        if len(parts) < 2:
            raise ValueError("用法：攻擊 <目標> [武器]")
        target_id = _resolve_id(parts[1], world_state)
        if target_id is None:
            raise ValueError(f"找不到目標：{parts[1]}")
        actor = world_state.characters.get(actor_id)
        weapon = parts[2] if len(parts) >= 3 else (
            actor.get_weapon().name if (actor and actor.weapons) else ""
        )
        return {
            "type":     "ATTACK",
            "attacker": actor_id,
            "target":   target_id,
            "weapon":   weapon,
            "consumes": ["action"],
        }

    if cmd in ("move", "移動"):
        if len(parts) == 2:
            target_id = _resolve_id(parts[1], world_state)
            if target_id is None:
                raise ValueError(f"用法：移動 <x> <y> 或 移動 <目標>（找不到 {parts[1]}）")
            return {
                "type":      "MOVE",
                "character": actor_id,
                "target":    target_id,
                "consumes":  ["movement"],
            }
        if len(parts) >= 3:
            ax, ay = parts[1], parts[2]
            is_delta = any(s.startswith(("+", "-")) for s in (ax, ay))
            try:
                x = float(ax)
                y = float(ay)
            except ValueError:
                raise ValueError(f"無效座標：{ax} {ay}")
            key = "delta" if is_delta else "target_position"
            return {
                "type":      "MOVE",
                "character": actor_id,
                key:         [x, y],
                "consumes":  ["movement"],
            }
        raise ValueError("用法：移動 <x> <y> 或 移動 <目標>")

    if cmd in ("spell", "施法"):
        if len(parts) < 3:
            raise ValueError("用法：施法 <咒名> <目標 或 x y>")
        spell_name = parts[1]
        if len(parts) >= 4:
            try:
                x = float(parts[2])
                y = float(parts[3])
                return {
                    "type":            "SPELL",
                    "caster":          actor_id,
                    "spell_name":      spell_name,
                    "target_position": [x, y],
                    "consumes":        ["action"],
                }
            except ValueError:
                pass   # not numeric — fall through to target-id interpretation
        target_id = _resolve_id(parts[2], world_state)
        if target_id is None:
            raise ValueError(f"找不到法術目標：{parts[2]}")
        return {
            "type":       "SPELL",
            "caster":     actor_id,
            "spell_name": spell_name,
            "target":     target_id,
            "consumes":   ["action"],
        }

    if cmd in _DODGE_WORDS:
        return {"type": "DODGE", "character": actor_id, "consumes": ["action"]}
    if cmd in _HIDE_WORDS:
        return {"type": "HIDE", "character": actor_id, "consumes": ["action"]}
    if cmd in _DISENGAGE_WORDS:
        return {"type": "DISENGAGE", "character": actor_id, "consumes": ["action"]}

    # ── 招式 <skill_id> [args] ────────────────────────────────────────────
    # Invoke a ClassAbility from the catalogue. Args interpretation depends
    # on the ability's target_type:
    #   SELF                — no args
    #   SINGLE_ENEMY/ALLY   — arg[0] is target id or display name
    #   POINT               — arg[0], arg[1] are x, y in metres
    #   MULTI_ENEMY/ALLY    — arg[0] is "id1,id2,..." (comma-separated)
    if cmd in ("招式", "ability", "skill"):
        from .abilities import CLASS_ABILITIES
        from .skill import TargetType
        actor = world_state.characters.get(actor_id)
        if len(parts) < 2:
            # No skill_id → list what this actor can invoke
            avail = []
            for sid in (actor.known_abilities if actor else []):
                ab = CLASS_ABILITIES.get(sid)
                if ab and not ab.is_reaction and ab.engine_ready:
                    avail.append(f"{sid}（{ab.display_name}）")
            tip = "、".join(avail) or "（無）"
            raise ValueError(f"用法：招式 <skill_id> [args]。你會的招式：{tip}")
        skill_id = parts[1]
        ab = CLASS_ABILITIES.get(skill_id)
        if ab is None:
            for a in CLASS_ABILITIES.values():
                if a.display_name == skill_id:
                    ab = a
                    skill_id = a.skill_id
                    break
        if ab is None:
            raise ValueError(f"未知技能：{skill_id}")
        if ab.is_reaction:
            raise ValueError(f"{skill_id} 是反應動作，引擎自動觸發，不能主動使用")
        if not ab.engine_ready or ab.builder is None:
            raise ValueError(f"{skill_id} 引擎尚未就緒")
        if actor is not None and actor.known_abilities and skill_id not in actor.known_abilities:
            raise ValueError(f"你不會 {skill_id}")
        # Use-count gate (skip for unlimited abilities)
        if actor is not None and ab.max_uses > 0:
            remaining = actor.ability_uses.get(skill_id, ab.max_uses)
            if remaining <= 0:
                raise ValueError(f"{ab.display_name} 今日次數已用盡（短休或長休後恢復）")

        tt = ab.features.target_type
        args = parts[2:]
        target_arg = None
        coord_arg = None

        if tt == TargetType.SELF:
            pass
        elif tt in (TargetType.SINGLE_ENEMY, TargetType.SINGLE_ALLY):
            if not args:
                raise ValueError(f"用法：招式 {skill_id} <目標>")
            target_arg = _resolve_id(args[0], world_state) or args[0]
        elif tt == TargetType.POINT:
            if len(args) < 2:
                raise ValueError(f"用法：招式 {skill_id} <x> <y>")
            try:
                coord_arg = (float(args[0]), float(args[1]))
            except ValueError:
                raise ValueError(f"無效座標：{args[0]} {args[1]}")
        elif tt in (TargetType.MULTI_ENEMY, TargetType.MULTI_ALLY):
            if not args:
                raise ValueError(f"用法：招式 {skill_id} <目標1,目標2,...>")
            tokens = [t.strip() for t in args[0].split(",") if t.strip()]
            resolved = [_resolve_id(tok, world_state) or tok for tok in tokens]
            target_arg = ",".join(resolved)

        action = ab.builder(actor_id, target_arg, coord_arg, char=actor)
        if action is None:
            raise ValueError(f"{skill_id}：builder 回傳 None（檢查 args 是否齊全）")
        # Deduct one use.
        if actor is not None and ab.max_uses > 0:
            actor.ability_uses[skill_id] = actor.ability_uses.get(skill_id, ab.max_uses) - 1
        return action

    if cmd in ("預言", "portent"):
        if len(parts) < 3:
            raise ValueError("用法：預言 <骰值> <目標>")
        try:
            die_val = int(parts[1])
        except ValueError:
            raise ValueError(f"無效的骰值：{parts[1]}")
        tgt = _resolve_id(parts[2], world_state) or parts[2]
        return {
            "type":      "PORTENT",
            "caster":    actor_id,
            "target":    tgt,
            "die_value": die_val,
            "consumes":  [],
        }

    raise ValueError(f"無法解析指令：{text}")


class HumanInputPolicy(CombatPolicy):
    """Bridges a UI input stream into the CombatPolicy interface.

    Construction takes two callbacks so the policy is decoupled from event
    types in game.py (no circular import):

      prompt_fn(actor, ctx) -> str | None
        emit a combat prompt to the UI, block until the user submits one
        line, return that line. None signals quit.
      error_fn(message) -> None
        relay a parse-error message back to the UI when input is malformed.
    """

    def __init__(self, char_id: str,
                 prompt_fn: Callable,
                 error_fn: Callable[[str], None]):
        self.char_id = char_id
        self.prompt_fn = prompt_fn
        self.error_fn = error_fn

    def decide(self, actor_id, actor, world_state, resources, round_num):
        from .combat import build_combat_context   # local: avoid import cycle
        ctx = build_combat_context(actor_id, actor, world_state, resources, round_num)
        while True:
            raw = self.prompt_fn(actor, ctx)
            if raw is None:
                return CombatDecision(ended=True)
            text = raw.strip()
            if not text:
                continue
            low = text.lower()
            if low in _END_WORDS:
                return CombatDecision(ended=True)
            if low in _FLEE_WORDS:
                return CombatDecision(fled=True)
            try:
                action = _parse_command(text, actor_id, world_state)
            except ValueError as e:
                self.error_fn(str(e))
                continue
            return CombatDecision(action=action)
