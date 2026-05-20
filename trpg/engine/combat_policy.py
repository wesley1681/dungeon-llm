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
  - _ArchetypeBase           shared utilities for scripted expert policies
  - BattleMasterPolicy / ChampionPolicy      Fighter archetypes
  - TotemBearPolicy / BerserkerPolicy        Barbarian archetypes
  - EvocationPolicy / DivinationPolicy       Wizard archetypes
  - LifeClericPolicy / WarClericPolicy       Cleric archetypes
  - AssassinPolicy / ArcaneTricksterPolicy   Rogue archetypes
  - DevotionPolicy / VengeancePolicy         Paladin archetypes
  - make_archetype_policy(archetype_id)      factory → correct expert policy

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

        from .skill import available_skills
        skills = available_skills(actor, ws)
        weapon_skill_id = f"weapon:{weapon.name}"

        if d <= reach + 1e-6 and resources.get("action", 0) > 0:
            sk = next((s for s in skills if s.skill_id == weapon_skill_id), None)
            if sk is not None:
                return CombatDecision(action=sk.build_action(actor_id, target_id, target.position))

        # Only walk when actually out of reach — otherwise the MOVE handler
        # would return a 0m no-op (combat.py keeps a 1m gap from creature
        # targets), burning sub-actions on idle steps.
        if d > reach + 1e-6 and resources.get("movement", 0.0) > 1e-6:
            move_sk = next((s for s in skills if s.skill_id == "move"), None)
            if move_sk is not None:
                return CombatDecision(action=move_sk.build_action(actor_id, target_id, None))

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

    from .skill import available_skills
    actor = world_state.characters.get(actor_id)
    skills = available_skills(actor, world_state) if actor else []

    def _find(skill_id: str):
        return next((s for s in skills if s.skill_id == skill_id), None)

    if cmd in ("attack", "攻擊"):
        if len(parts) < 2:
            raise ValueError("用法：攻擊 <目標> [武器]")
        target_id = _resolve_id(parts[1], world_state)
        if target_id is None:
            raise ValueError(f"找不到目標：{parts[1]}")
        weapon = parts[2] if len(parts) >= 3 else (
            actor.get_weapon().name if (actor and actor.weapons) else ""
        )
        sk = _find(f"weapon:{weapon}")
        if sk is None:
            raise ValueError(f"沒有可用武器：{weapon}")
        return sk.build_action(actor_id, target_id, None)

    if cmd in ("move", "移動"):
        move_sk = _find("move")
        if move_sk is None:
            raise ValueError("此時無法移動")
        if len(parts) == 2:
            target_id = _resolve_id(parts[1], world_state)
            if target_id is None:
                raise ValueError(f"用法：移動 <x> <y> 或 移動 <目標>（找不到 {parts[1]}）")
            return move_sk.build_action(actor_id, target_id, None)
        if len(parts) >= 3:
            ax, ay = parts[1], parts[2]
            is_delta = any(s.startswith(("+", "-")) for s in (ax, ay))
            try:
                x = float(ax)
                y = float(ay)
            except ValueError:
                raise ValueError(f"無效座標：{ax} {ay}")
            if is_delta:
                # Move skill's builder doesn't support relative deltas — only
                # absolute coords or target ids. Rebuild and swap the key.
                action = move_sk.build_action(actor_id, None, (x, y))
                if action is None:
                    raise ValueError("移動失敗")
                action.pop("target_position", None)
                action["delta"] = [x, y]
                return action
            return move_sk.build_action(actor_id, None, (x, y))
        raise ValueError("用法：移動 <x> <y> 或 移動 <目標>")

    if cmd in ("spell", "施法"):
        if len(parts) < 3:
            raise ValueError("用法：施法 <咒名> <目標 或 x y>")
        spell_name = parts[1]
        spell_sk = _find(f"spell:{spell_name}")
        if spell_sk is None:
            raise ValueError(f"沒有可用法術：{spell_name}（檢查法術位或集中限制）")
        if len(parts) >= 4:
            try:
                x = float(parts[2])
                y = float(parts[3])
                return spell_sk.build_action(actor_id, None, (x, y))
            except ValueError:
                pass   # not numeric — fall through to target-id interpretation
        target_id = _resolve_id(parts[2], world_state)
        if target_id is None:
            raise ValueError(f"找不到法術目標：{parts[2]}")
        return spell_sk.build_action(actor_id, target_id, None)

    if cmd in _DODGE_WORDS:
        sk = _find("dodge")
        return sk.build_action(actor_id, None, None) if sk else None
    if cmd in _HIDE_WORDS:
        sk = _find("hide")
        return sk.build_action(actor_id, None, None) if sk else None
    if cmd in _DISENGAGE_WORDS:
        sk = _find("disengage")
        return sk.build_action(actor_id, None, None) if sk else None

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

        action = ab.build_action(actor_id, target_arg, coord_arg, char=actor)
        if action is None:
            raise ValueError(f"{skill_id}：build_action 回傳 None（檢查 args 是否齊全）")
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
        # PORTENT is a player-only reaction with a die_value parameter that
        # doesn't fit the (actor, target, coord) Skill.builder signature. We
        # set skill_id explicitly here so the engine validator passes; this
        # is the one documented exception to "always go through Skill".
        return {
            "type":      "PORTENT",
            "skill_id":  "portent",
            "caster":    actor_id,
            "target":    tgt,
            "die_value": die_val,
            "consumes":  [],
        }

    raise ValueError(f"無法解析指令：{text}")


# ── Archetype expert policies (Stage A — scripted teachers for RL BC) ─────────
#
# Each policy follows its archetype's priority ordering:
#   1. Bonus-action setup (rage, vow, buff) if not yet used this turn
#   2. Close gap / position (movement)
#   3. Main action (attack, spell, heal, buff)
#   4. End turn
#
# Resources dict keys: "action" (int), "bonus_action" (int), "movement" (float).
# Policies use available_skills() to build actions so they naturally respect
# ability-use limits, min_level gates, and archetype filters — the same
# interface the RL model will observe.

class _ArchetypeBase(CombatPolicy):
    """Shared utility methods for all archetype expert policies."""

    # ── Target selection ──────────────────────────────────────────────────────

    def _enemy_candidates(self, actor_id: str, ws) -> list:
        is_party = ws.is_party_ally(actor_id)
        room = ws.dungeon_map.current_room if ws.dungeon_map else None
        room_ids = set(room.npc_ids) if room else set(ws.characters.keys())
        out = []
        for cid, c in ws.characters.items():
            if cid == actor_id or not c.is_alive():
                continue
            if is_party:
                if c.is_npc and c.attitude == 0 and cid in room_ids:
                    out.append((cid, c))
            else:
                if ws.is_party_ally(cid):
                    out.append((cid, c))
        return out

    def _nearest_enemy(self, actor, ws, actor_id) -> tuple:
        enemies = self._enemy_candidates(actor_id, ws)
        if not enemies:
            return None, None
        return min(enemies, key=lambda kv: actor.position.distance_to(kv[1].position))

    def _find_heal_target(self, actor_id: str, ws, threshold: float = 0.45) -> str | None:
        """Return char_id of the ally with lowest HP% below threshold, or None."""
        best_id, best_ratio = None, threshold
        for cid, c in ws.characters.items():
            if cid == actor_id or not c.is_alive() or not ws.is_party_ally(cid):
                continue
            ratio = c.hp / c.max_hp if c.max_hp else 1.0
            if ratio < best_ratio:
                best_ratio, best_id = ratio, cid
        return best_id

    # ── Action helpers ────────────────────────────────────────────────────────

    def _in_reach(self, actor, target) -> bool:
        weapon = actor.get_weapon() if actor.weapons else None
        reach = (weapon.range_normal if weapon else 1.5) or 1.5
        return actor.position.distance_to(target.position) <= reach + 1e-6

    def _use_skill(self, actor_id: str, actor, ws, skill_id: str,
                   target_id: str | None = None, coord=None) -> dict | None:
        """Build an action dict for skill_id via available_skills(), or None if unavailable."""
        from .skill import available_skills
        sk = next((s for s in available_skills(actor, ws) if s.skill_id == skill_id), None)
        if sk is None:
            return None
        t = ws.characters.get(target_id) if target_id else None
        c = coord or (t.position if t else actor.position)
        return sk.build_action(actor_id, target_id, c)

    def _attack(self, actor_id: str, actor, ws, target_id: str) -> dict | None:
        weapon = actor.get_weapon() if actor.weapons else None
        if not weapon:
            return None
        action = self._use_skill(actor_id, actor, ws,
                                 f"weapon:{weapon.name}", target_id=target_id)
        return action

    def _smite_attack(self, actor_id: str, actor, ws, target_id: str) -> dict | None:
        """Attack with the lowest available spell slot for divine smite."""
        weapon = actor.get_weapon() if actor.weapons else None
        if not weapon:
            return None
        for slot in (1, 2, 3, 4):
            if actor.spell_slots.get(slot, 0) > 0:
                action = self._use_skill(actor_id, actor, ws,
                                         f"weapon:{weapon.name}", target_id=target_id)
                if action is None:
                    return None
                action["divine_smite_slot"] = slot
                return action
        return self._attack(actor_id, actor, ws, target_id)

    def _move_to(self, actor_id: str, actor, ws, target_id: str) -> dict | None:
        """Build a MOVE action toward target_id, avoiding blocked terrain.

        `_use_skill("move", target_id=...)` builds an action whose
        target_position IS the target character's exact position — if the
        target is standing in a wall cell (player can do this via sandbox
        bug, or wall-targeted spells), engine returns ERROR. Driver retries
        the same MOVE 10x silently. Worse, BC training records the failed
        MOVE as a label — model would learn "blunder into walls".

        This helper computes a safe waypoint along the line of sight (capped
        at MOVE_BUDGET_M, walked back along the ray if it lands in a wall)
        and passes that explicit coord. Returns None if no cell along the
        ray is reachable.
        """
        from .combat import MOVE_BUDGET_M
        from .vec2 import Vec2
        bf = ws.combat.battlefield if ws.combat else None
        target_char = ws.characters.get(target_id) if target_id else None
        # No battlefield (e.g. abstract combat) — fall back to original behaviour
        if bf is None or target_char is None:
            return self._use_skill(actor_id, actor, ws, "move", target_id=target_id)

        CLOSE_GAP_M = 1.0   # mirrors engine.combat MOVE handler
        old_pos = actor.position
        delta = target_char.position - old_pos
        dist = delta.length()
        if dist <= CLOSE_GAP_M + 1e-6:
            return None   # already in/inside the close gap — nothing to do

        travel = min(dist - CLOSE_GAP_M, MOVE_BUDGET_M)
        direction = delta.normalized()

        # Try the full straight-line waypoint first; if blocked, walk back.
        try_steps = [travel] + [i * 0.5 for i in range(int(travel / 0.5), 0, -1)]
        for t in try_steps:
            try_pos = old_pos + direction * t
            if bf.in_bounds(try_pos) and not bf.is_blocked(try_pos):
                return self._use_skill(actor_id, actor, ws, "move", coord=Vec2(try_pos.x, try_pos.y))
        return None


# ── Fighters ──────────────────────────────────────────────────────────────────

class BattleMasterPolicy(_ArchetypeBase):
    """Fighter Battle Master: approach → attack with superiority maneuver → plain attack."""
    _MANEUVERS = ("menacing_attack", "trip_attack", "pushing_attack")

    def decide(self, actor_id, actor, ws, resources, round_num):
        tid, tgt = self._nearest_enemy(actor, ws, actor_id)
        if tid is None:
            return CombatDecision(ended=True)

        if not self._in_reach(actor, tgt) and resources.get("movement", 0) > 1e-6:
            return CombatDecision(action=self._move_to(actor_id, actor, ws, tid))

        if resources.get("action", 0) > 0 and self._in_reach(actor, tgt):
            for m in self._MANEUVERS:
                a = self._use_skill(actor_id, actor, ws, m, tid)
                if a:
                    return CombatDecision(action=a)
            a = self._attack(actor_id, actor, ws, tid)
            if a:
                return CombatDecision(action=a)

        return CombatDecision(ended=True)


class ChampionPolicy(_ArchetypeBase):
    """Fighter Champion: approach → attack (relies on passive improved crit range)."""

    def decide(self, actor_id, actor, ws, resources, round_num):
        tid, tgt = self._nearest_enemy(actor, ws, actor_id)
        if tid is None:
            return CombatDecision(ended=True)

        if not self._in_reach(actor, tgt) and resources.get("movement", 0) > 1e-6:
            return CombatDecision(action=self._move_to(actor_id, actor, ws, tid))

        if resources.get("action", 0) > 0 and self._in_reach(actor, tgt):
            a = self._attack(actor_id, actor, ws, tid)
            if a:
                return CombatDecision(action=a)

        return CombatDecision(ended=True)


# ── Barbarians ────────────────────────────────────────────────────────────────

class _BarbarianBase(_ArchetypeBase):
    """Rage (bonus) first turn → reckless attack every turn."""

    def decide(self, actor_id, actor, ws, resources, round_num):
        tid, tgt = self._nearest_enemy(actor, ws, actor_id)
        if tid is None:
            return CombatDecision(ended=True)

        # Rage via bonus action if not already raging
        if not actor.has_status("raging") and resources.get("bonus_action", 1) > 0:
            a = self._use_skill(actor_id, actor, ws, "rage", actor_id)
            if a:
                return CombatDecision(action=a)

        if not self._in_reach(actor, tgt) and resources.get("movement", 0) > 1e-6:
            return CombatDecision(action=self._move_to(actor_id, actor, ws, tid))

        if resources.get("action", 0) > 0 and self._in_reach(actor, tgt):
            a = (self._use_skill(actor_id, actor, ws, "reckless_attack", tid)
                 or self._attack(actor_id, actor, ws, tid))
            if a:
                return CombatDecision(action=a)

        return CombatDecision(ended=True)


class TotemBearPolicy(_BarbarianBase):
    pass   # Bear totem resistance is passive; base logic suffices


class BerserkerPolicy(_BarbarianBase):
    pass   # Frenzy is engine_todo; falls back to base barbarian


# ── Wizards ───────────────────────────────────────────────────────────────────

class _WizardBase(_ArchetypeBase):
    """Kite + spell priority. Misty-step escape if cornered in melee."""

    # Subclasses provide ordered list of skill_ids to try each turn
    def _spell_priority(self, actor, tgt, distance: float) -> list[str]:
        return ["magic_missile"]

    def decide(self, actor_id, actor, ws, resources, round_num):
        from .vec2 import Vec2
        tid, tgt = self._nearest_enemy(actor, ws, actor_id)
        if tid is None:
            return CombatDecision(ended=True)

        d = actor.position.distance_to(tgt.position)

        # Cornered in melee → misty step away (bonus action)
        if d <= 1.5 + 1e-6 and resources.get("bonus_action", 1) > 0:
            away = Vec2(actor.position.x + 9.0, actor.position.y)
            a = self._use_skill(actor_id, actor, ws, "misty_step", None, away)
            if a:
                return CombatDecision(action=a)

        # Cast best available spell
        if resources.get("action", 0) > 0:
            for spell_id in self._spell_priority(actor, tgt, d):
                a = self._use_skill(actor_id, actor, ws, spell_id, tid, tgt.position)
                if a:
                    return CombatDecision(action=a)

        # Close gap if target out of spell range
        if d > 18.0 and resources.get("movement", 0) > 1e-6:
            return CombatDecision(action=self._move_to(actor_id, actor, ws, tid))

        return CombatDecision(ended=True)


class EvocationPolicy(_WizardBase):
    """Evocation: fireball > hold_person > burning_hands (close) > magic_missile."""

    def _spell_priority(self, actor, tgt, distance: float) -> list[str]:
        spells = []
        if distance > 4.5:
            spells.append("fireball_ev")
        if not tgt.has_status("paralyzed"):
            spells.append("hold_person")
        if distance <= 4.5:
            spells.append("burning_hands_ev")
        spells.append("magic_missile")
        return spells


class DivinationPolicy(_WizardBase):
    """Divination: hold_person > web > fireball > magic_missile.
    Once target is already debuffed, switch directly to damage spells."""

    def _spell_priority(self, actor, tgt, distance: float) -> list[str]:
        already_debuffed = tgt.has_status("paralyzed") or tgt.has_status("restrained")
        spells = []
        if not already_debuffed:
            spells.append("hold_person")
            spells.append("web_div")
        if distance > 4.5:
            spells.append("fireball_div")
        spells.append("magic_missile")
        return spells


# ── Clerics ───────────────────────────────────────────────────────────────────

class _ClericBase(_ArchetypeBase):
    """Heal wounded ally → bless early → sacred flame / melee attack."""
    _HEAL_THRESHOLD = 0.4

    def decide(self, actor_id, actor, ws, resources, round_num):
        tid, tgt = self._nearest_enemy(actor, ws, actor_id)
        if tid is None:
            return CombatDecision(ended=True)

        if resources.get("action", 0) > 0:
            # Heal critically low ally
            heal_id = self._find_heal_target(actor_id, ws, self._HEAL_THRESHOLD)
            if heal_id:
                a = self._use_skill(actor_id, actor, ws, "cure_wounds", heal_id)
                if a:
                    return CombatDecision(action=a)

            # Bless early (round 1 or 2)
            if round_num <= 2 and not actor.has_status("blessed"):
                a = self._use_skill(actor_id, actor, ws, "bless", actor_id)
                if a:
                    return CombatDecision(action=a)

            # Sacred flame at range, or melee attack if adjacent
            if self._in_reach(actor, tgt):
                a = self._attack(actor_id, actor, ws, tid)
            else:
                a = self._use_skill(actor_id, actor, ws, "sacred_flame", tid)
            if a:
                return CombatDecision(action=a)

        if not self._in_reach(actor, tgt) and resources.get("movement", 0) > 1e-6:
            return CombatDecision(action=self._move_to(actor_id, actor, ws, tid))

        return CombatDecision(ended=True)


class LifeClericPolicy(_ClericBase):
    """Life Cleric: heal aggressively (threshold 50%), lay_on_hands before cure_wounds."""
    _HEAL_THRESHOLD = 0.5

    def decide(self, actor_id, actor, ws, resources, round_num):
        tid, tgt = self._nearest_enemy(actor, ws, actor_id)
        if tid is None:
            return CombatDecision(ended=True)

        if resources.get("action", 0) > 0:
            heal_id = self._find_heal_target(actor_id, ws, self._HEAL_THRESHOLD)
            if heal_id:
                a = (self._use_skill(actor_id, actor, ws, "lay_on_hands_ability", heal_id)
                     or self._use_skill(actor_id, actor, ws, "cure_wounds", heal_id))
                if a:
                    return CombatDecision(action=a)

            if round_num <= 1 and not actor.has_status("blessed"):
                a = self._use_skill(actor_id, actor, ws, "bless", actor_id)
                if a:
                    return CombatDecision(action=a)

            if self._in_reach(actor, tgt):
                a = self._attack(actor_id, actor, ws, tid)
            else:
                a = self._use_skill(actor_id, actor, ws, "sacred_flame", tid)
            if a:
                return CombatDecision(action=a)

        if not self._in_reach(actor, tgt) and resources.get("movement", 0) > 1e-6:
            return CombatDecision(action=self._move_to(actor_id, actor, ws, tid))

        return CombatDecision(ended=True)


class WarClericPolicy(_ClericBase):
    """War Cleric: aggressive (bless round 1, sacred flame primary, heal only near-death)."""
    _HEAL_THRESHOLD = 0.25


# ── Rogues ────────────────────────────────────────────────────────────────────

class _RogueBase(_ArchetypeBase):
    """Cunning action hide (bonus) → attack for sneak attack → dash to close."""
    _HIDE_SKILL     = "cunning_action_hide"
    _DASH_SKILL     = "cunning_action_dash"
    _DISENGAGE_SKILL = "cunning_action_disengage"

    def decide(self, actor_id, actor, ws, resources, round_num):
        tid, tgt = self._nearest_enemy(actor, ws, actor_id)
        if tid is None:
            return CombatDecision(ended=True)

        d = actor.position.distance_to(tgt.position)
        in_melee = d <= 1.5 + 1e-6

        # Cunning action: hide for sneak-attack advantage (bonus action)
        if in_melee and not actor.has_status("hidden") and resources.get("bonus_action", 1) > 0:
            a = self._use_skill(actor_id, actor, ws, self._HIDE_SKILL)
            if a:
                return CombatDecision(action=a)

        # Attack (sneak attack fires automatically when conditions met)
        if resources.get("action", 0) > 0 and in_melee:
            a = self._attack(actor_id, actor, ws, tid)
            if a:
                return CombatDecision(action=a)

        # Close gap: cunning dash (bonus) then walk
        if not in_melee:
            if resources.get("bonus_action", 1) > 0:
                a = self._use_skill(actor_id, actor, ws, self._DASH_SKILL)
                if a:
                    return CombatDecision(action=a)
            if resources.get("movement", 0) > 1e-6:
                return CombatDecision(action=self._move_to(actor_id, actor, ws, tid))

        return CombatDecision(ended=True)


class AssassinPolicy(_RogueBase):
    _HIDE_SKILL      = "cunning_action_hide"
    _DASH_SKILL      = "cunning_action_dash"
    _DISENGAGE_SKILL = "cunning_action_disengage"


class ArcaneTricksterPolicy(_RogueBase):
    """Arcane Trickster: hide (action) when out of melee, dash in, then sneak attack.

    No cunning-action hide for this archetype; hide costs the action, so
    hiding and attacking happen on alternating turns.
    """
    _HIDE_SKILL      = ""                        # no bonus-action hide
    _DASH_SKILL      = "cunning_action_dash_at"
    _DISENGAGE_SKILL = "cunning_action_disengage_at"

    def decide(self, actor_id, actor, ws, resources, round_num):
        tid, tgt = self._nearest_enemy(actor, ws, actor_id)
        if tid is None:
            return CombatDecision(ended=True)

        d = actor.position.distance_to(tgt.position)
        in_melee = d <= 1.5 + 1e-6

        # Hide (action) when out of melee and not yet hidden — get advantage
        if not in_melee and not actor.has_status("hidden") and resources.get("action", 0) > 0:
            a = self._use_skill(actor_id, actor, ws, "hide")
            if a:
                return CombatDecision(action=a)

        # Cunning dash (bonus) to close gap
        if not in_melee and resources.get("bonus_action", 1) > 0:
            a = self._use_skill(actor_id, actor, ws, self._DASH_SKILL)
            if a:
                return CombatDecision(action=a)

        if not in_melee and resources.get("movement", 0) > 1e-6:
            return CombatDecision(action=self._move_to(actor_id, actor, ws, tid))

        # Attack (sneak fires if hidden or target debuffed)
        if resources.get("action", 0) > 0 and in_melee:
            a = self._attack(actor_id, actor, ws, tid)
            if a:
                return CombatDecision(action=a)

        return CombatDecision(ended=True)


# ── Paladins ──────────────────────────────────────────────────────────────────

class DevotionPolicy(_ArchetypeBase):
    """Devotion: sacred weapon buff (action, round 1) → smite attacks."""

    def decide(self, actor_id, actor, ws, resources, round_num):
        tid, tgt = self._nearest_enemy(actor, ws, actor_id)
        if tid is None:
            return CombatDecision(ended=True)

        if not self._in_reach(actor, tgt) and resources.get("movement", 0) > 1e-6:
            return CombatDecision(action=self._move_to(actor_id, actor, ws, tid))

        if resources.get("action", 0) > 0:
            # Sacred weapon buff (1 action; lasts 10 rounds). 5e rules don't
            # restrict it to round 1 — open it anytime you're about to start
            # attacking and haven't buffed yet.
            if not actor.has_status("sacred_weapon_buff"):
                a = self._use_skill(actor_id, actor, ws, "sacred_weapon_dev", actor_id)
                if a:
                    return CombatDecision(action=a)

            # Self-heal if critically low
            if actor.hp / max(actor.max_hp, 1) < 0.25:
                a = self._use_skill(actor_id, actor, ws, "lay_on_hands_ability", actor_id)
                if a:
                    return CombatDecision(action=a)

            if self._in_reach(actor, tgt):
                a = self._smite_attack(actor_id, actor, ws, tid)
                if a:
                    return CombatDecision(action=a)

        return CombatDecision(ended=True)


class VengeancePolicy(_ArchetypeBase):
    """Vengeance: vow of enmity (bonus, round 1) → smite attacks."""

    def decide(self, actor_id, actor, ws, resources, round_num):
        tid, tgt = self._nearest_enemy(actor, ws, actor_id)
        if tid is None:
            return CombatDecision(ended=True)

        if not self._in_reach(actor, tgt) and resources.get("movement", 0) > 1e-6:
            return CombatDecision(action=self._move_to(actor_id, actor, ws, tid))

        # Vow of enmity on target (bonus action; lasts 10 rounds). 5e rules
        # don't restrict it to round 1 — open it any turn the target isn't
        # yet vowed and bonus action is free.
        if not tgt.has_status("vow_target") and resources.get("bonus_action", 1) > 0:
            a = self._use_skill(actor_id, actor, ws, "vow_of_enmity_ven", tid)
            if a:
                return CombatDecision(action=a)

        if resources.get("action", 0) > 0:
            if actor.hp / max(actor.max_hp, 1) < 0.25:
                a = self._use_skill(actor_id, actor, ws, "lay_on_hands_ability", actor_id)
                if a:
                    return CombatDecision(action=a)

            if self._in_reach(actor, tgt):
                a = self._smite_attack(actor_id, actor, ws, tid)
                if a:
                    return CombatDecision(action=a)

        return CombatDecision(ended=True)


# ── Registry and factory ──────────────────────────────────────────────────────

ARCHETYPE_POLICIES: dict[str, type] = {
    "battle_master":    BattleMasterPolicy,
    "champion":         ChampionPolicy,
    "totem_bear":       TotemBearPolicy,
    "berserker":        BerserkerPolicy,
    "evocation":        EvocationPolicy,
    "divination":       DivinationPolicy,
    "life":             LifeClericPolicy,
    "war":              WarClericPolicy,
    "assassin":         AssassinPolicy,
    "arcane_trickster": ArcaneTricksterPolicy,
    "devotion":         DevotionPolicy,
    "vengeance":        VengeancePolicy,
}


def make_archetype_policy(archetype_id: str) -> CombatPolicy:
    """Return the scripted expert policy for the given archetype.

    Falls back to HeuristicCombatPolicy if the archetype is not recognised
    (e.g. generic NPCs without a specific archetype).
    """
    return ARCHETYPE_POLICIES.get(archetype_id, HeuristicCombatPolicy)()


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
