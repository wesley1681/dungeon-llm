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
    # Invoke a Ability from the catalogue. Args interpretation depends
    # on the ability's target_type:
    #   SELF                — no args
    #   SINGLE_ENEMY/ALLY   — arg[0] is target id or display name
    #   POINT               — arg[0], arg[1] are x, y in metres
    #   MULTI_ENEMY/ALLY    — arg[0] is "id1,id2,..." (comma-separated)
    if cmd in ("招式", "ability", "skill"):
        from .abilities import ABILITY_REGISTRY
        from .skill import TargetType
        if len(parts) < 2:
            # No skill_id → list what this actor can invoke
            avail = []
            for sid in (actor.known_abilities if actor else []):
                ab = ABILITY_REGISTRY.get(sid)
                if ab and not ab.is_reaction and ab.engine_ready:
                    avail.append(f"{sid}（{ab.display_name}）")
            tip = "、".join(avail) or "（無）"
            raise ValueError(f"用法：招式 <skill_id> [args]。你會的招式：{tip}")
        skill_id = parts[1]
        ab = ABILITY_REGISTRY.get(skill_id)
        if ab is None:
            for a in ABILITY_REGISTRY.values():
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
        # Limited-use deduction happens inside execute_action (shared funnel,
        # success-only) — deducting again here at build time would double-count.
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

    def _weakest_enemy(self, actor, ws, actor_id) -> tuple:
        """Return the lowest-HP alive enemy (focus-fire target for team play)."""
        enemies = self._enemy_candidates(actor_id, ws)
        if not enemies:
            return None, None
        return min(enemies, key=lambda kv: kv[1].hp / max(1, kv[1].max_hp))

    def _find_heal_target(self, actor_id: str, ws, threshold: float = 0.45) -> str | None:
        """Return char_id of the ally (or self) with lowest HP% below threshold,
        or ``None`` if nobody needs healing.

        Self IS included in the search — without that the cleric never heals
        itself in 1v1, so a critically-wounded solo cleric ends up casting
        bless/sacred_flame and dying. Life cleric expert win rate was 22%
        on this codebase before the fix.

        Dying (0 HP, death saves) allies ARE included — HP% 0 makes them the
        top priority automatically, and healing them is the 5e pick-up (any
        amount of healing returns them to the fight).

        「盟友」以施法者的陣營判定（actor 相對）——原寫法用絕對的
        is_party_ally（=玩家隊），敵方牧師會掃到玩家隊的傷員、貼臉時真的
        會治療敵人，而自己的隊友永遠不被奶。
        """
        my_side = ws.is_party_ally(actor_id)
        best_id, best_ratio = None, threshold
        for cid, c in ws.characters.items():
            if c.is_dead() or ws.is_party_ally(cid) != my_side:
                continue
            ratio = c.hp / c.max_hp if c.max_hp else 1.0
            if ratio < best_ratio:
                best_ratio, best_id = ratio, cid
        return best_id

    def _attempt_pickup(self, actor_id: str, actor, ws, resources):
        """Pick up a DYING same-side ally with lay_on_hands: heal if within
        touch range, otherwise walk toward them. Returns a CombatDecision or
        None (nobody dying / no lay_on_hands resource / no path).

        Keyed purely on the dying game-state (hp 0, death saves) — not on any
        class or archetype. Classes without lay_on_hands get None from
        _use_skill and fall through.
        """
        my_side = ws.is_party_ally(actor_id)
        dying = [(cid, c) for cid, c in ws.characters.items()
                 if cid != actor_id and c.is_dying()
                 and ws.is_party_ally(cid) == my_side]
        if not dying:
            return None
        cid, c = min(dying, key=lambda kv: actor.position.distance_to(kv[1].position))
        a = self._use_skill(actor_id, actor, ws, "lay_on_hands_ability", cid)
        if a is None:
            return None
        if actor.position.distance_to(c.position) <= 1.5 + 1e-6:
            return CombatDecision(action=a)
        if resources.get("movement", 0) > 1e-6:
            mv = self._move_to(actor_id, actor, ws, cid)
            if mv:
                return CombatDecision(action=mv)
        return None

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
        # 5e: one leveled spell per turn. Don't even build the action if the
        # rule would force the engine to ERROR — that ERROR would otherwise
        # be recorded as a BC label, teaching the model the wrong move.
        if (getattr(sk.features, "cost_slot_level", 0) > 0
                and getattr(actor, "leveled_spell_cast_this_turn", False)):
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

class _FighterBase(_ArchetypeBase):
    """Shared fighter rotation: emergency heal → approach → attack → burst.

    Hooks for subclass differentiation:
      _MANEUVERS — ordered tuple of maneuver skill_ids tried before plain
                   attack on a normal action.
    """
    _MANEUVERS: tuple[str, ...] = ()
    _SECOND_WIND_THRESHOLD = 0.4   # heal when HP fraction <= this
    _ACTION_SURGE_THRESHOLD = 0.4  # only burst when we're not near death

    def decide(self, actor_id, actor, ws, resources, round_num):
        tid, tgt = self._nearest_enemy(actor, ws, actor_id)
        if tid is None:
            return CombatDecision(ended=True)

        # Second Wind (bonus): emergency self-heal when HP is low. Availability
        # (max_uses, remaining) is handled inside _use_skill via available_skills.
        if (resources.get("bonus_action", 0) > 0
                and actor.hp <= actor.max_hp * self._SECOND_WIND_THRESHOLD):
            a = self._use_skill(actor_id, actor, ws, "second_wind", actor_id)
            if a:
                return CombatDecision(action=a)

        # Approach if out of melee range.
        if not self._in_reach(actor, tgt) and resources.get("movement", 0) > 1e-6:
            return CombatDecision(action=self._move_to(actor_id, actor, ws, tid))

        # Main action: maneuvers preferred, then plain attack.
        if resources.get("action", 0) > 0 and self._in_reach(actor, tgt):
            for m in self._MANEUVERS:
                a = self._use_skill(actor_id, actor, ws, m, tid)
                if a:
                    return CombatDecision(action=a)
            a = self._attack(actor_id, actor, ws, tid)
            if a:
                return CombatDecision(action=a)

        # Burst with Action Surge after spending the main action — gives a
        # second action via the `grants` dict. Only fire while still in melee
        # + healthy enough to make the extra hit worthwhile.
        if (resources.get("action", 0) == 0
                and self._in_reach(actor, tgt)
                and actor.hp >= actor.max_hp * self._ACTION_SURGE_THRESHOLD):
            a = self._use_skill(actor_id, actor, ws, "action_surge", actor_id)
            if a:
                return CombatDecision(action=a)

        return CombatDecision(ended=True)


class BattleMasterPolicy(_FighterBase):
    """Fighter Battle Master: maneuver → plain attack → action_surge burst."""
    # All four maneuvers are engine-ready (pushing_attack wired up with the
    # forced-move mechanic); the superiority-die pool is each maneuver's max_uses.
    _MANEUVERS = ("menacing_attack", "trip_attack", "distracting_strike",
                  "pushing_attack")


class ChampionPolicy(_FighterBase):
    """Fighter Champion: plain attacks + action_surge burst (relies on
    passive improved_critical range)."""
    _MANEUVERS = ()


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

    @staticmethod
    def _away_from(actor_pos, threat_pos, distance_m: float, battlefield) -> "Vec2 | None":
        """Return a position `distance_m` away from threat, clamped to battlefield.

        Returns ``None`` if clamping squashes the retreat to <1m of effective
        motion — i.e. the actor is already at the battlefield edge and no
        meaningful escape exists in that direction. Callers MUST handle the
        None case by skipping the retreat / teleport entirely instead of
        emitting a "move to current cell" action. Previously this clamped
        silently and the wizard expert emitted MOVE-to-self ~85% of the
        time when cornered, polluting BC labels and teaching the policy
        that "move" means "stay put".
        """
        from .vec2 import Vec2
        import math
        dx = actor_pos.x - threat_pos.x
        dy = actor_pos.y - threat_pos.y
        mag = math.sqrt(dx * dx + dy * dy)
        if mag > 1e-6:
            nx, ny = dx / mag, dy / mag
        else:
            nx, ny = 1.0, 0.0
        raw = Vec2(actor_pos.x + nx * distance_m, actor_pos.y + ny * distance_m)
        if battlefield is None:
            return raw
        margin = 0.5
        clamped = Vec2(
            max(margin, min(battlefield.width  - margin, raw.x)),
            max(margin, min(battlefield.height - margin, raw.y)),
        )
        if (clamped - actor_pos).length() < 1.0:
            return None
        return clamped

    def decide(self, actor_id, actor, ws, resources, round_num):
        tid, tgt = self._nearest_enemy(actor, ws, actor_id)
        if tid is None:
            return CombatDecision(ended=True)

        d = actor.position.distance_to(tgt.position)
        bf = ws.combat.battlefield if ws.combat else None

        # Cornered (≤3m) → misty step away from enemy direction (bonus action).
        # Skip teleport if no real escape exists (away is None) — otherwise
        # the expert teleports to the SELF cell, polluting BC labels.
        if d <= 3.0 and resources.get("bonus_action", 1) > 0:
            away = self._away_from(actor.position, tgt.position, 9.0, bf)
            if away is not None:
                a = self._use_skill(actor_id, actor, ws, "misty_step", None, away)
                if a:
                    return CombatDecision(action=a)

        # Cast best available spell
        if resources.get("action", 0) > 0:
            for spell_id in self._spell_priority(actor, tgt, d):
                a = self._use_skill(actor_id, actor, ws, spell_id, tid, tgt.position)
                if a:
                    return CombatDecision(action=a)

        # After spending action: back away if enemy is closing in. Same
        # guard as misty_step above — None means the actor is already at
        # the battlefield edge, so emitting a move-to-edge equals MOVE-to-self.
        if d < 9.0 and resources.get("movement", 0) > 1e-6:
            retreat = self._away_from(actor.position, tgt.position, 9.0, bf)
            if retreat is not None:
                move_a = self._use_skill(actor_id, actor, ws, "move", None, retreat)
                if move_a:
                    return CombatDecision(action=move_a)

        # Close gap if target is out of spell range
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
    """Spirit guardians (round 1, L5+) → spiritual weapon (bonus) → heal /
    bless / sacred flame / melee. Variant ids let domains override the
    spiritual_weapon skill (life vs war) without forking the whole method.
    """
    _HEAL_THRESHOLD = 0.4
    _SPIRITUAL_WEAPON_CAST_ID = "spiritual_weapon_life"
    _SPIRITUAL_WEAPON_ATK_ID  = "spiritual_weapon_attack_life"

    def decide(self, actor_id, actor, ws, resources, round_num):
        tid, tgt = self._nearest_enemy(actor, ws, actor_id)
        if tid is None:
            return CombatDecision(ended=True)

        # Heal critically low / dying ally first — supersedes any other plan.
        # Out-of-range target: spend movement walking to it instead of letting
        # the engine bounce the cast with a range ERROR (which would waste the
        # whole turn re-attempting the same action).
        if resources.get("action", 0) > 0:
            heal_id = self._find_heal_target(actor_id, ws, self._HEAL_THRESHOLD)
            if heal_id:
                a = self._attempt_heal(actor_id, actor, ws, heal_id)
                if a:
                    rng = float(a.get("range_m", 1.5) or 1.5)
                    dist = actor.position.distance_to(
                        ws.characters[heal_id].position)
                    if dist <= rng + 1e-6:
                        return CombatDecision(action=a)
                    if resources.get("movement", 0) > 1e-6:
                        mv = self._move_to(actor_id, actor, ws, heal_id)
                        if mv:
                            return CombatDecision(action=mv)

        # Spirit Guardians (L5+ slot 3, concentration): biggest single-action
        # damage output cleric has. Cast on round 1 if we have the slot and
        # we're not already concentrating on something else.
        if (resources.get("action", 0) > 0 and round_num <= 2
                and not actor.has_status("spirit_guardians_active")
                and not actor.concentrating_on):
            a = self._use_skill(actor_id, actor, ws, "spirit_guardians", actor_id)
            if a:
                return CombatDecision(action=a)

        # Spiritual Weapon (bonus, slot 2): persistent bonus-action attacks.
        if (resources.get("bonus_action", 0) > 0
                and not actor.has_status("spiritual_weapon_active")):
            a = self._use_skill(actor_id, actor, ws,
                                self._SPIRITUAL_WEAPON_CAST_ID, actor_id)
            if a:
                return CombatDecision(action=a)

        # Spiritual Weapon follow-up attack (bonus): if active and bonus left.
        if (resources.get("bonus_action", 0) > 0
                and actor.has_status("spiritual_weapon_active")):
            a = self._use_skill(actor_id, actor, ws,
                                self._SPIRITUAL_WEAPON_ATK_ID, tid)
            if a:
                return CombatDecision(action=a)

        if resources.get("action", 0) > 0:
            # Bless early (round 1 or 2) if no spirit_guardians concentration.
            if (round_num <= 2 and not actor.has_status("blessed")
                    and not actor.concentrating_on):
                a = self._use_skill(actor_id, actor, ws, "bless", actor_id)
                if a:
                    return CombatDecision(action=a)

            # Sacred flame at range, or melee attack if adjacent.
            if self._in_reach(actor, tgt):
                a = self._attack(actor_id, actor, ws, tid)
            else:
                a = self._use_skill(actor_id, actor, ws, "sacred_flame", tid)
            if a:
                return CombatDecision(action=a)

        if not self._in_reach(actor, tgt) and resources.get("movement", 0) > 1e-6:
            return CombatDecision(action=self._move_to(actor_id, actor, ws, tid))

        return CombatDecision(ended=True)

    def _attempt_heal(self, actor_id, actor, ws, heal_id):
        """Pick the best heal skill for `heal_id`. Override per-domain."""
        return self._use_skill(actor_id, actor, ws, "cure_wounds", heal_id)


class LifeClericPolicy(_ClericBase):
    """Life Cleric: heal threshold 50%, lay_on_hands preferred before cure_wounds."""
    _HEAL_THRESHOLD = 0.5

    def _attempt_heal(self, actor_id, actor, ws, heal_id):
        return (self._use_skill(actor_id, actor, ws, "lay_on_hands_ability", heal_id)
                or self._use_skill(actor_id, actor, ws, "cure_wounds", heal_id))


class WarClericPolicy(_ClericBase):
    """War Cleric: heal threshold 25%, spiritual weapon variants point to war ids."""
    _HEAL_THRESHOLD = 0.25
    _SPIRITUAL_WEAPON_CAST_ID = "spiritual_weapon_war"
    _SPIRITUAL_WEAPON_ATK_ID  = "spiritual_weapon_attack_war"


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
        hidden = actor.has_status("hidden")
        in_melee = d <= 1.5 + 1e-6
        # Ranged weapon (shortbow / crossbow) — gated by range_type so a
        # loadout swap still picks it up. Without this branch Assassin
        # ignored its bow entirely at range and just dashed to melee,
        # eating attacks all the way in — measured 21% expert win rate.
        bow = next((w for w in actor.weapons if w.range_type == "遠程"), None)
        bow_normal = (bow.range_normal if bow else 0.0) or 0.0
        in_bow_range = bow is not None and d <= bow_normal + 1e-6

        # Bonus-action hide first (sets up sneak attack on the next action).
        if not hidden and resources.get("bonus_action", 1) > 0:
            a = self._use_skill(actor_id, actor, ws, self._HIDE_SKILL)
            if a:
                return CombatDecision(action=a)

        # Action: melee swing if adjacent, else ranged shot if hidden +
        # in bow range (Hidden + ranged = advantage = sneak-attack trigger).
        if resources.get("action", 0) > 0:
            if in_melee:
                a = self._attack(actor_id, actor, ws, tid)
                if a:
                    return CombatDecision(action=a)
            elif hidden and in_bow_range and bow is not None:
                a = self._use_skill(actor_id, actor, ws,
                                     f"weapon:{bow.name}", target_id=tid)
                if a:
                    return CombatDecision(action=a)

        # Close gap (if not in bow range, walk in). Skip Cunning Dash when
        # in_bow_range — bonus action was already spent on Hide above and
        # extra movement isn't needed when the bow can already reach.
        if not in_bow_range and resources.get("movement", 0) > 1e-6:
            return CombatDecision(action=self._move_to(actor_id, actor, ws, tid))

        return CombatDecision(ended=True)


class AssassinPolicy(_RogueBase):
    _HIDE_SKILL      = "cunning_action_hide"
    _DASH_SKILL      = "cunning_action_dash"
    _DISENGAGE_SKILL = "cunning_action_disengage"


class ArcaneTricksterPolicy(_RogueBase):
    """Arcane Trickster: bonus-action hide → attack with advantage → sneak attack.

    Uses ``cunning_action_hide`` (bonus action) — the archetype factory
    grants this from level 2. Strategy each turn:
      1. If not hidden and bonus available → hide (bonus).
      2. If in melee → swing shortsword (advantage from Hidden, sneak fires).
      3. If hidden + in shortbow normal range (not in melee) → shoot bow
         (advantage + sneak; avoids the in-melee ranged-disadvantage).
      4. Out of bow range → move toward target.
    Cunning dash is dropped from the rotation — sneak-attack damage from the
    ranged shot beats the extra movement, and bonus action is committed to
    Hide. Dash is still kept available for the model via the skill list.
    """
    _HIDE_SKILL      = "cunning_action_hide"
    _DASH_SKILL      = "cunning_action_dash"
    _DISENGAGE_SKILL = "cunning_action_disengage"

    def decide(self, actor_id, actor, ws, resources, round_num):
        tid, tgt = self._nearest_enemy(actor, ws, actor_id)
        if tid is None:
            return CombatDecision(ended=True)

        d = actor.position.distance_to(tgt.position)
        hidden = actor.has_status("hidden")
        in_melee = d <= 1.5 + 1e-6

        # Find ranged weapon (shortbow) by range_type, not by hardcoded name —
        # avoids breaking if the loadout swaps to crossbow / longbow.
        bow = next((w for w in actor.weapons if w.range_type == "遠程"), None)
        bow_normal = (bow.range_normal if bow else 0.0) or 0.0
        in_bow_range = bow is not None and d <= bow_normal + 1e-6

        # 1. Bonus-action hide first — sets up Hidden for the attack below.
        if not hidden and resources.get("bonus_action", 1) > 0:
            a = self._use_skill(actor_id, actor, ws, self._HIDE_SKILL)
            if a:
                return CombatDecision(action=a)

        slots_left = actor.spell_slots.get(1, 0)
        target_disabled = (tgt.has_status("asleep") or tgt.has_status("blinded")
                            or tgt.has_status("paralyzed"))

        # 2. Round-1 control: sleep on a healthy enemy (5e: 5d8 HP cap means
        #    sleep is best opening turn). Save 1 slot for shield reaction.
        if (round_num == 1 and resources.get("action", 0) > 0
                and slots_left >= 2 and not target_disabled
                and tgt.hp / max(1, tgt.max_hp) > 0.5):
            a = self._use_skill(actor_id, actor, ws, "sleep", target_id=tid)
            if a:
                return CombatDecision(action=a)

        # 3. Attack while we have an action.
        if resources.get("action", 0) > 0:
            if in_melee:
                a = self._attack(actor_id, actor, ws, tid)
                if a:
                    return CombatDecision(action=a)
            elif hidden and in_bow_range and bow is not None:
                a = self._use_skill(actor_id, actor, ws,
                                     f"weapon:{bow.name}", target_id=tid)
                if a:
                    return CombatDecision(action=a)
            # 4. Out of bow range: magic missile (auto-hit) as fallback offense.
            elif not in_bow_range and slots_left >= 2:
                a = self._use_skill(actor_id, actor, ws, "magic_missile",
                                     target_id=tid)
                if a:
                    return CombatDecision(action=a)

        # 5. Close gap with movement when out of bow range.
        if not in_bow_range and resources.get("movement", 0) > 1e-6:
            return CombatDecision(action=self._move_to(actor_id, actor, ws, tid))

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
            # Pick up a dying ally first (5e: lay on hands is THE pick-up tool
            # — any amount of healing returns them to the fight).
            d = self._attempt_pickup(actor_id, actor, ws, resources)
            if d is not None:
                return d

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
            d = self._attempt_pickup(actor_id, actor, ws, resources)
            if d is not None:
                return d

            if actor.hp / max(actor.max_hp, 1) < 0.25:
                a = self._use_skill(actor_id, actor, ws, "lay_on_hands_ability", actor_id)
                if a:
                    return CombatDecision(action=a)

            if self._in_reach(actor, tgt):
                a = self._smite_attack(actor_id, actor, ws, tid)
                if a:
                    return CombatDecision(action=a)

        return CombatDecision(ended=True)


# ── Generic monster controller ───────────────────────────────────────────────

class GenericMonsterPolicy(_ArchetypeBase):
    """Data-driven controller for monster identities (MONSTER_CATALOG Wave 0).

    Knows NO skill ids and NO archetype names — every decision reads
    SkillFeatures off available_skills(), so any future monster assembled in
    trpg/scenarios/monsters.py is playable without touching this class.

    Turn priority:
      1.  below 35% HP, self-heal with the biggest expected_healing skill
      1.5 control casts (v1, Wave 2) — control-LED kits only (best damage EV
          ≤ weapon EV, e.g. basilisk gaze): zero-damage single-enemy save
          spells until the target carries the terminal status or is immune.
          Damage-led casters skip this (mage_npc A/B: control-first cost
          18pp@L5) and play pure nuker as in v0.
      2.  highest-EV damaging option that is affordable, in range, has line
          of sight, satisfies target-state preconditions (swallow), and
          (for AoE/LINE) doesn't catch an ally; weapon EV scaled by
          attacks_per_action
      3.  if a damaging option exists but is out of range / LoS-blocked,
          close toward the nearest enemy — a ranged kit already in range
          never walks closer

    Still not modelled (v1): bonus-action dashes, kiting, AoE control casts.
    HeuristicCombatPolicy stays untouched — it is the historical RL
    comparison baseline.
    """

    _HEAL_THRESHOLD = 0.35

    @staticmethod
    def _affordable(f, resources) -> bool:
        if f.cost_action and resources.get("action", 0) <= 0:
            return False
        if f.cost_bonus and resources.get("bonus_action", 0) <= 0:
            return False
        if f.cost_reaction:
            return False
        return True

    def _damage_options(self, actor_id, actor, ws, skills, resources,
                        tgt, dist, has_los, n_atk):
        """Collect the damaging candidates usable from the current position
        (decide() step 2, verbatim — also reused by the legendary-action
        executor with a restricted skill list). Pure evaluation, no dice.

        Returns (usable_now, usable_later): usable_now = [(ev, skill)] gated
        on affordability / range / LoS / target-state preconditions / ally
        splash; usable_later = True when something affordable exists but is
        out of range or LoS-blocked (the "walk closer" signal).
        """
        from .skill import TargetType
        from .abilities import ABILITY_REGISTRY
        my_side = ws.is_party_ally(actor_id)
        usable_now: list[tuple[float, object]] = []
        usable_later = False   # affordable but out of range / no LoS
        for s in skills:
            f = s.features
            if f.expected_damage <= 0 or not self._affordable(f, resources):
                continue
            # Target-state preconditions (behir swallow: requires restrained,
            # blocked once swallowed) — same data the engine re-validates.
            ab = ABILITY_REGISTRY.get(s.skill_id)
            if ab is not None:
                if (ab.requires_target_status
                        and not tgt.has_status(ab.requires_target_status)):
                    continue
                if (ab.blocked_by_target_status
                        and tgt.has_status(ab.blocked_by_target_status)):
                    continue
            is_weapon = s.skill_id.startswith("weapon:")
            reach = f.range_m or 1.5
            melee = reach <= 2.0
            if dist > reach + 1e-6 or (not has_los and not melee):
                usable_later = True
                continue
            if f.aoe_radius_m > 0 and not getattr(actor, "sculpt_spells", False):
                if f.target_type == TargetType.LINE:
                    # LINE convention: range_m = line length, aoe_radius_m =
                    # half-width. Check allies along the actual beam segment.
                    from .vec2 import point_segment_distance
                    aim = tgt.position - actor.position
                    line_end = (actor.position
                                + aim.normalized() * (f.range_m or 1.5))
                    friendly_hit = any(
                        c.is_alive() and cid != actor_id
                        and ws.is_party_ally(cid) == my_side
                        and point_segment_distance(
                            c.position, actor.position, line_end)
                            <= f.aoe_radius_m + 1e-6
                        for cid, c in ws.characters.items())
                else:
                    friendly_hit = any(
                        c.is_alive() and cid != actor_id
                        and ws.is_party_ally(cid) == my_side
                        and tgt.position.distance_to(c.position)
                            <= f.aoe_radius_m + 1e-6
                        for cid, c in ws.characters.items())
                if friendly_hit:
                    continue
            ev = f.expected_damage * (n_atk if is_weapon else 1)
            usable_now.append((ev, s))
        return usable_now, usable_later

    def decide(self, actor_id, actor, ws, resources, round_num):
        from .skill import available_skills, TargetType
        tid, tgt = self._nearest_enemy(actor, ws, actor_id)
        if tid is None:
            return CombatDecision(ended=True)

        skills = available_skills(actor, ws)
        dist = actor.position.distance_to(tgt.position)
        bf = ws.combat.battlefield if ws.combat else None
        has_los = (bf is None
                   or bf.has_line_of_sight(actor.position, tgt.position))

        # 1. Emergency self-heal.
        if actor.hp <= actor.max_hp * self._HEAL_THRESHOLD:
            heals = [s for s in skills
                     if s.features.expected_healing > 0
                     and self._affordable(s.features, resources)]
            if heals:
                h = max(heals, key=lambda s: s.features.expected_healing)
                a = self._use_skill(actor_id, actor, ws, h.skill_id, actor_id)
                if a:
                    return CombatDecision(action=a)

        # 1.5 Control casts (GenericMonsterPolicy v1, Wave 2): zero-damage
        #     single-enemy save abilities riding the SPELL registry. ONLY for
        #     control-LED kits — best damage EV no better than the weapon
        #     (basilisk: gaze + a 10-EV bite). Damage-led casters measured
        #     WEAKER when opening with save-or-lock vs the scripted panel
        #     (mage_npc A/B, n=120/level/arm: expert WR +18pp@L5 / +7pp@L6
        #     with control-first — save_each escapes + lost nuke tempo), so
        #     they skip straight to damage. Cast while the target lacks the
        #     TERMINAL status (escalates_to, else the status itself) and is
        #     not condition-immune; once locked down, fall through to damage
        #     (petrified targets eat melee). Data-driven — features and the
        #     spell entry only, no skill-id or monster names.
        from .spells import SPELLS
        n_atk = getattr(actor, "attacks_per_action", 1) or 1
        best_dmg_ev = max(
            (s.features.expected_damage
             * (n_atk if s.skill_id.startswith("weapon:") else 1)
             for s in skills if s.features.expected_damage > 0), default=0.0)
        best_weapon_ev = max(
            (s.features.expected_damage * n_atk
             for s in skills if s.skill_id.startswith("weapon:")), default=0.0)
        control_led = best_dmg_ev <= best_weapon_ev + 1e-6
        if control_led:
            for s in skills:
                f = s.features
                if f.expected_damage > 0 or f.expected_healing > 0:
                    continue
                if f.target_type != TargetType.SINGLE_ENEMY:
                    continue
                if not self._affordable(f, resources):
                    continue
                sp = SPELLS.get(s.display_name)
                if sp is None or not sp.applies_status_on_fail:
                    continue
                stage1 = sp.applies_status_on_fail
                terminal = sp.escalates_to or stage1
                if tgt.has_status(terminal):
                    continue
                if (stage1 in tgt.condition_immunities
                        or terminal in tgt.condition_immunities):
                    continue
                if dist > (f.range_m or 1.5) + 1e-6 or not has_los:
                    continue
                a = self._use_skill(actor_id, actor, ws, s.skill_id, tid)
                if a:
                    return CombatDecision(action=a)

        # 2. Damage options, split by whether they can fire from here.
        usable_now, usable_later = self._damage_options(
            actor_id, actor, ws, skills, resources, tgt, dist, has_los, n_atk)

        if usable_now:
            _, best = max(usable_now, key=lambda kv: kv[0])
            tt = best.features.target_type
            if tt in (TargetType.POINT, TargetType.LINE, TargetType.CONE):
                a = self._use_skill(actor_id, actor, ws, best.skill_id,
                                    coord=tgt.position)
            else:
                a = self._use_skill(actor_id, actor, ws, best.skill_id, tid)
            if a:
                return CombatDecision(action=a)

        # 3. Close the gap only when that unlocks an attack (also the no-LoS
        #    recovery path). A ranged kit already in range never walks closer.
        if usable_later and resources.get("movement", 0.0) > 1e-6:
            mv = self._move_to(actor_id, actor, ws, tid)
            if mv:
                return CombatDecision(action=mv)

        return CombatDecision(ended=True)


# ── Legendary actions (Wave 3, MONSTER_CATALOG §3 D 級) ──────────────────────
#
# 5e: a legendary creature has a per-round budget (refilled at its own turn
# start — status.tick_status_effects) and may spend it on ONE option from its
# table at the end of ANOTHER creature's turn. Every combat driver calls
# run_legendary_actions(ws, ended_char_id, round_num) right after a turn's
# self_turn_end tick; fights without a legendary creature return [] without
# touching the dice stream (baseline-safe).
#
# Options are pure data on the character (TraitGrant "legendary_actions"):
#   {"ability": skill_id, "cost": n}  — a granted ability (wing attack, eye
#                                       ray, cantrip…), uses-gated through
#                                       available_skills like any other skill
#   {"weapon": name, "cost": n}       — a single swing with a natural weapon
#                                       (dragon tail), n_attacks forced to 1;
#                                       resolved via Character.get_weapon so
#                                       the weapon may live outside the
#                                       multiattack kit (behir-jaw pattern)
# Selection is EV-per-cost greedy over the options usable RIGHT NOW (range /
# LoS / preconditions via the same _damage_options gates as the normal turn);
# no movement is spent — an out-of-reach option simply doesn't fire (5e: you
# may always decline). Control value beyond expected damage is not modelled
# (greedy damage bias — documented approximation).

_LEGENDARY_BRAIN = GenericMonsterPolicy()


@dataclass
class LegendaryContext:
    """A pending legendary-action decision handed to a legendary decider.

    ``options`` is the LEGAL set the engine assembled — each entry is
    {"skill_id", "cost", "ev"} where ev is the greedy damage estimate. The
    decider returns one of those skill_ids, or None to forgo the legendary
    action this trigger. Legality (budget, range, LoS, preconditions) is already
    enforced; the decider only chooses whether and which to spend.
    """
    actor: object
    actor_id: str
    options: list           # [{"skill_id": str, "cost": int, "ev": float}, ...]
    world_state: object
    target_id: str
    round_num: int


def greedy_legendary_decider(ctx: "LegendaryContext") -> str | None:
    """Default legendary policy = the historical EV-per-cost greedy pick
    (bit-for-bit: same options in the same order → same argmax)."""
    if not ctx.options:
        return None
    best = max(ctx.options, key=lambda o: (o["ev"] / o["cost"], o["ev"]))
    return best["skill_id"]


def decide_legendary_action(actor_id: str, actor, ws,
                            round_num: int) -> tuple[dict | None, int]:
    """Pick ONE affordable legendary option for `actor`. Returns
    (action_dict, cost) or (None, 0) when nothing is usable.

    Splits LEGALITY (which options are affordable & in range — assembled here)
    from CHOICE (which to spend — delegated to ws.legendary_decider, default
    greedy EV/cost). The RL env injects a decider to let a model-controlled
    boss choose its legendary actions; scripted bosses keep the greedy default.
    """
    from .skill import available_skills, from_weapon, TargetType

    remaining = actor.legendary_actions_remaining
    options = [o for o in actor.legendary_options
               if int(o.get("cost", 1)) <= remaining]
    if not options:
        return None, 0
    tid, tgt = _LEGENDARY_BRAIN._nearest_enemy(actor, ws, actor_id)
    if tid is None:
        return None, 0
    dist = actor.position.distance_to(tgt.position)
    bf = ws.combat.battlefield if ws.combat else None
    has_los = (bf is None
               or bf.has_line_of_sight(actor.position, tgt.position))
    # Legendary actions sit outside the action/bonus economy — affordability
    # here means only "enough legendary budget", so feed a fresh wallet.
    resources = {"action": 1, "bonus_action": 1, "movement": 0.0}

    candidates: list[tuple[object, int]] = []   # (Skill, cost)
    ability_ids = {o["ability"] for o in options if "ability" in o}
    if ability_ids:
        for s in available_skills(actor, ws):
            if s.skill_id in ability_ids:
                cost = next(int(o["cost"]) for o in options
                            if o.get("ability") == s.skill_id)
                candidates.append((s, cost))
    for o in options:
        wname = o.get("weapon")
        if wname:
            candidates.append((from_weapon(actor.get_weapon(wname), actor),
                               int(o["cost"])))
    if not candidates:
        return None, 0

    usable, _ = _LEGENDARY_BRAIN._damage_options(
        actor_id, actor, ws, [s for s, _ in candidates], resources,
        tgt, dist, has_los, n_atk=1)   # legendary swings never multiattack
    if not usable:
        return None, 0
    cost_by_sid = {s.skill_id: c for s, c in candidates}
    skill_by_sid = {kv[1].skill_id: kv[1] for kv in usable}
    # Preserve `usable` order so the greedy default's argmax is bit-exact. Each
    # option carries its Skill so a model decider can observe the candidate;
    # greedy_legendary_decider ignores the extra field.
    decider_options = [{"skill_id": sk.skill_id,
                        "cost": cost_by_sid[sk.skill_id], "ev": ev, "skill": sk}
                       for ev, sk in usable]
    ctx = LegendaryContext(actor=actor, actor_id=actor_id,
                           options=decider_options, world_state=ws,
                           target_id=tid, round_num=round_num)
    hook = getattr(ws, "legendary_decider", None)
    chosen_sid = hook(ctx) if hook is not None else greedy_legendary_decider(ctx)
    if chosen_sid not in skill_by_sid:
        return None, 0          # decline (or an out-of-set return = decline)
    best = skill_by_sid[chosen_sid]
    cost = cost_by_sid[best.skill_id]

    if best.skill_id.startswith("weapon:"):
        action = best.builder(actor_id, tid, None)
        if action is not None:
            action["n_attacks"] = 1
            action["skill_id"] = best.skill_id
    elif best.features.target_type in (TargetType.POINT, TargetType.LINE,
                                       TargetType.CONE):
        action = _LEGENDARY_BRAIN._use_skill(actor_id, actor, ws,
                                             best.skill_id, coord=tgt.position)
    else:
        action = _LEGENDARY_BRAIN._use_skill(actor_id, actor, ws,
                                             best.skill_id, tid)
    if action is None:
        return None, 0
    return action, cost


@dataclass
class LairContext:
    """A pending lair-action decision handed to a lair decider. ``options`` is
    the legal set ({"skill_id", "ev"}); the decider returns one skill_id or None
    to forgo the lair action this round. Mirrors LegendaryContext so a model can
    drive lair actions via the same seam (ws.lair_decider)."""
    actor: object
    actor_id: str
    options: list           # [{"skill_id": str, "ev": float}, ...]
    world_state: object
    target_id: str
    round_num: int


def greedy_lair_decider(ctx: "LairContext") -> str | None:
    """Default lair policy = highest listed ev (stable: first on ties)."""
    if not ctx.options:
        return None
    return max(ctx.options, key=lambda o: o.get("ev", 0.0))["skill_id"]


def decide_lair_action(actor_id: str, actor, ws,
                       round_num: int) -> dict | None:
    """Build ONE lair action for `actor`, or None. Splits LEGALITY (the legal
    option set, assembled here) from CHOICE (ws.lair_decider, default greedy).
    Lair abilities are resolved straight from ABILITY_REGISTRY (a lair effect
    need not be in the creature's normal turn kit), targeting the nearest enemy
    — environmental AoE/control abilities use that as their point/target."""
    from .abilities import ABILITY_REGISTRY
    opts = [o for o in (actor.lair_options or []) if "ability" in o]
    if not opts:
        return None
    options = [{"skill_id": o["ability"], "ev": float(o.get("ev", 1.0))}
               for o in opts]
    tid, tgt = _LEGENDARY_BRAIN._nearest_enemy(actor, ws, actor_id)
    ctx = LairContext(actor=actor, actor_id=actor_id, options=options,
                      world_state=ws, target_id=tid or "", round_num=round_num)
    hook = getattr(ws, "lair_decider", None)
    chosen = hook(ctx) if hook is not None else greedy_lair_decider(ctx)
    if chosen is None:
        return None
    ab = ABILITY_REGISTRY.get(chosen)
    if ab is None or ab.builder is None:
        return None
    coord = (tgt.position.x, tgt.position.y) if tgt is not None else None
    return ab.build_action(actor_id, tid, coord, char=actor)


def run_lair_actions(world_state, round_num: int) -> list[dict]:
    """Fire each lair-capable creature's once-per-round environmental effect.
    Self-limited via ``lair_acted_round`` so it can be called from every
    per-turn legendary hook yet still fires only once per round (≈ initiative
    count 20). [] when no creature has lair_options (zero overhead)."""
    results: list[dict] = []
    if world_state is None or world_state.combat is None:
        return results
    from .combat import execute_action
    for cid, char in world_state.characters.items():
        if (not char.lair_options
                or char.lair_acted_round == round_num
                or not char.is_alive()
                or char.is_incapacitated()):
            continue
        action = decide_lair_action(cid, char, world_state, round_num)
        char.lair_acted_round = round_num   # one attempt/round, even if declined
        if action is None:
            continue
        result = execute_action(action, world_state)
        if result.get("type") == "ERROR":
            continue
        results.append({
            "type":       "LAIR_ACTION",
            "actor_id":   cid,
            "actor_name": char.name,
            "action":     action,
            "result":     result,
        })
    return results


def run_legendary_actions(world_state, ended_char_id: str,
                          round_num: int) -> list[dict]:
    """Fire legendary actions triggered by the end of `ended_char_id`'s turn,
    plus any once-per-round lair actions (piggybacked here so every driver that
    already calls this gets lair actions for free — see run_lair_actions).

    Each legendary creature (≠ the one whose turn ended) that is alive, not
    incapacitated, and has budget left takes at most ONE option (5e RAW).
    Returns display-ready event dicts; [] when no legendary creature exists.
    """
    results: list[dict] = []
    if world_state is None or world_state.combat is None:
        return results
    from .combat import execute_action
    for cid, char in world_state.characters.items():
        if (cid == ended_char_id
                or char.legendary_actions_remaining <= 0
                or not char.legendary_options
                or not char.is_alive()
                or char.is_incapacitated()):
            continue
        action, cost = decide_legendary_action(cid, char, world_state,
                                               round_num)
        if action is None:
            continue
        result = execute_action(action, world_state)
        if result.get("type") == "ERROR":
            continue
        char.legendary_actions_remaining -= cost
        results.append({
            "type":       "LEGENDARY_ACTION",
            "actor_id":   cid,
            "actor_name": char.name,
            "cost":       cost,
            "remaining":  char.legendary_actions_remaining,
            "action":     action,
            "result":     result,
        })
    results.extend(run_lair_actions(world_state, round_num))
    return results


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
