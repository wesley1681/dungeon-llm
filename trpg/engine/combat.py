from dataclasses import dataclass, field

from .dice import roll, roll_d20, combine_advantage
from .character import Character, CombatState
from .world_state import WorldState
from .vec2 import Vec2, Battlefield

# Movement budget per turn (D&D 5e default speed for medium humanoid = 30 ft ≈ 9 m).
MOVE_BUDGET_M = 9.0

# Default battlefield: 30m × 30m open space, no obstacles.
DEFAULT_BATTLEFIELD_W = 30.0
DEFAULT_BATTLEFIELD_H = 30.0

# Spawn convention (preserves 1D distances from the legacy layout):
#   - Party at x = PARTY_SPAWN_X, on the midline
#   - Melee enemies face the party at +1.5m
#   - Ranged enemies / casters at +6.0m
# Multiple combatants of the same kind stagger by ±1m in y.
_PARTY_SPAWN_X = 5.0
_MELEE_SPAWN_X = 6.5
_RANGED_SPAWN_X = 11.0


def _spawn_offsets(n: int) -> list[float]:
    """y-axis stagger for n combatants centred on 0.0. e.g. 3 → [-1, 0, 1]."""
    return [(i - (n - 1) / 2.0) for i in range(n)]


def setup_combat_positions(world_state: WorldState, combat: CombatState) -> None:
    """Place combatants on the 2D battlefield at combat start.

    Creates a default open battlefield on the combat state if one isn't
    already attached. Party spawns on the left, hostiles on the right;
    melee at 1.5m engagement distance, ranged at 6m back.
    """
    if combat.battlefield is None:
        combat.battlefield = Battlefield(width=DEFAULT_BATTLEFIELD_W,
                                         height=DEFAULT_BATTLEFIELD_H)
    mid_y = combat.battlefield.height / 2.0

    party_ids: list[str] = []
    melee_ids: list[str] = []
    ranged_ids: list[str] = []
    for cid in combat.initiative_order:
        char = world_state.characters.get(cid)
        if not char:
            continue
        if world_state.is_party_ally(cid):
            party_ids.append(cid)
            continue
        has_ranged = any(w.range_type == "遠程" for w in char.weapons)
        is_caster = bool(char.spells and char.spellcasting_ability)
        (ranged_ids if (has_ranged or is_caster) else melee_ids).append(cid)

    def place(ids: list[str], x: float) -> None:
        for cid, dy in zip(ids, _spawn_offsets(len(ids))):
            world_state.characters[cid].position = Vec2(x, mid_y + dy)

    place(party_ids, _PARTY_SPAWN_X)
    place(melee_ids, _MELEE_SPAWN_X)
    place(ranged_ids, _RANGED_SPAWN_X)


def roll_initiative(world_state: WorldState,
                    character_ids: list[str] | None = None) -> CombatState:
    """Roll initiative. If character_ids given, only include those characters.

    Also resets combatant positions to the standard battlefield layout.
    """
    chars = world_state.characters
    if character_ids is not None:
        chars = {k: v for k, v in chars.items() if k in character_ids}
    results = {
        cid: roll("1d20") + char.stats.modifier("DEX")
        for cid, char in chars.items()
        if char.is_alive()
    }
    order = sorted(results, key=lambda n: results[n], reverse=True)
    state = CombatState(initiative_order=order)
    setup_combat_positions(world_state, state)
    return state


def distance_m(a: Character, b: Character) -> float:
    """Euclidean distance between two combatants, in metres."""
    return a.position.distance_to(b.position)


def attack_range_check(attacker: Character, target: Character, weapon,
                       battlefield: Battlefield | None = None) -> tuple[bool, str, str]:
    """Decide whether `attacker` can hit `target` with `weapon` right now.

    Returns (in_range, reason, advantage_mode):
      - in_range: False means caller should reject the attack
      - reason: human-readable explanation when not in range
      - advantage_mode: "normal" / "disadvantage" — disadvantage applies for:
          ranged attack while in melee (target within 1.5m of attacker)
          long-range ranged shot (between range_normal and range_long)
    Ranged attacks also require line of sight to the target when a
    battlefield with obstacles is supplied.
    """
    if weapon is None:
        return False, "未指定武器", "normal"
    d = distance_m(attacker, target)
    if weapon.range_type == "近戰":
        reach = weapon.range_normal or 1.5
        if d > reach:
            return False, f"目標距離 {d:.1f}m，超出 {weapon.name} 伸手範圍 {reach:.1f}m", "normal"
        return True, "", "normal"
    # Ranged
    normal_r = weapon.range_normal or 0.0
    long_r   = weapon.range_long   or normal_r
    if d > long_r:
        return False, f"目標距離 {d:.1f}m，超出 {weapon.name} 最大射程 {long_r:.0f}m", "normal"
    if battlefield is not None and not battlefield.has_line_of_sight(
            attacker.position, target.position):
        return False, f"視線被遮擋，無法瞄準 {target.name}", "normal"
    # In-melee penalty: any enemy within 1.5m of attacker forces disadvantage on ranged
    if d <= 1.5:
        return True, "", "disadvantage"
    # Long-range disadvantage zone
    if d > normal_r:
        return True, "", "disadvantage"
    return True, "", "normal"


def resolve_attack(attacker: Character, target: Character,
                   weapon=None, mode: str = "normal") -> tuple[bool, int]:
    """Returns (hit, total_roll). mode: 'normal' / 'advantage' / 'disadvantage'.

    Iterates attacker + target Modifiers to apply hook-based mode adjustments
    (equipment passives, status effects like Dodging, etc.)."""
    if weapon is None:
        weapon = attacker.get_weapon()
    if "精巧" in (weapon.properties if weapon else []):
        stat_mod = max(attacker.stats.modifier("STR"), attacker.stats.modifier("DEX"))
    elif weapon and weapon.range_type == "遠程":
        stat_mod = attacker.stats.modifier("DEX")
    else:
        stat_mod = attacker.stats.modifier("STR")

    for m in attacker.iter_modifiers():
        mode = m.on_outgoing_attack(attacker, target, weapon, mode)
    for m in target.iter_modifiers():
        mode = m.on_incoming_attack(target, attacker, weapon, mode)

    attack_roll = roll_d20(mode)
    total = attack_roll + stat_mod + attacker.proficiency_bonus
    return total >= target.ac, total


def apply_damage(target: Character, amount, dtype: str = "untyped",
                 attacker=None) -> int:
    """Apply damage to target. `amount` is either a dice notation string
    (e.g. '1d6+2') or a pre-rolled int. Filters through target's Modifiers.

    If `target` is concentrating on a spell, a CON save vs DC max(10, dmg//2)
    fires; failure clears `target.concentrating_on`.

    Returns the actual damage dealt after modifiers.
    """
    if isinstance(amount, str):
        amount = roll(amount)
    for m in target.iter_modifiers():
        amount = m.on_incoming_damage(target, attacker, amount, dtype)
    amount = max(0, amount)
    target.hp = max(0, target.hp - amount)

    # Concentration check — 5e rule: when a concentrating creature takes
    # damage, it makes a CON save vs DC = max(10, ⌊damage/2⌋). Failure ends
    # the spell. We just clear the flag; engine consumers can react.
    if amount > 0 and target.concentrating_on:
        dc = max(10, amount // 2)
        success, _ = make_saving_throw(target, "CON", dc)
        if not success:
            target.concentrating_on = ""

    return amount


def apply_heal(target: Character, dice_notation: str) -> int:
    amount = roll(dice_notation)
    target.hp = min(target.max_hp, target.hp + amount)
    return amount


# Damage dice for dangerous terrain (lava, acid pool, spike pit). Per-cell
# variants would override this when scenarios pass their own.
DANGEROUS_TERRAIN_DAMAGE = "1d4"


def tick_terrain_damage(char: Character, battlefield) -> int:
    """If `char` is standing in dangerous terrain, apply damage. Returns the
    damage dealt (0 if not in dangerous terrain). Called at each character's
    self-turn-start phase by the combat loop."""
    if battlefield is None:
        return 0
    if not battlefield.is_dangerous(char.position):
        return 0
    return apply_damage(char, DANGEROUS_TERRAIN_DAMAGE, dtype="environment")


def make_saving_throw(character: Character, stat: str, dc: int) -> tuple[bool, int]:
    roll_result = roll("1d20")
    modifier = character.stats.modifier(stat)
    if stat in character.proficiencies:
        modifier += character.proficiency_bonus
    total = roll_result + modifier
    return total >= dc, total


def _lookup_char(key: str, world_state: WorldState):
    """Look up character by ID first, then by display name (arbiter may return either)."""
    char = world_state.characters.get(key)
    if char is None:
        char = next((c for c in world_state.characters.values() if c.name == key), None)
    return char


def execute_action(action: dict, world_state: WorldState) -> dict:
    """Execute a parsed action JSON from the arbiter. Returns a result summary dict."""
    t = action.get("type")

    # ── ATTACK ────────────────────────────────────────────────────────────────
    if t == "ATTACK":
        attacker = _lookup_char(action.get("attacker", ""), world_state)
        target   = _lookup_char(action.get("target",   ""), world_state)
        if not attacker or not target:
            return {"type": "ERROR", "message": "找不到攻擊者或目標"}
        if not target.is_alive():
            return {"type": "ERROR", "message": f"{target.name} 已倒下，無法攻擊"}

        weapon = attacker.get_weapon(action.get("weapon", ""))

        # Distance / range gate (with LoS if a battlefield is set)
        battlefield = world_state.combat.battlefield if world_state.combat else None
        in_range, reason, range_mode = attack_range_check(attacker, target, weapon, battlefield)
        if not in_range:
            return {"type": "ERROR", "message": reason}

        # Consume ammo for ranged weapons
        if weapon.ammo:
            if not attacker.has_ammo(weapon.ammo):
                return {"type": "ERROR", "message": f"{attacker.name} 沒有 {weapon.ammo} 了"}
            attacker.consume(weapon.ammo)

        # Ability modifier used for both attack and damage (finesse takes higher of STR/DEX)
        if "精巧" in weapon.properties:
            dmg_mod = max(attacker.stats.modifier("STR"), attacker.stats.modifier("DEX"))
        elif weapon.range_type == "遠程":
            dmg_mod = attacker.stats.modifier("DEX")
        else:
            dmg_mod = attacker.stats.modifier("STR")

        # Advantage / disadvantage compose: target dodging → disadvantage; range_mode also contributes
        target_dodging = target.has_status("dodging")
        mode = combine_advantage(range_mode, "disadvantage" if target_dodging else "normal")

        hit, roll_total = resolve_attack(attacker, target, weapon, mode=mode)
        result = {
            "type":          "ATTACK",
            "attacker_name": attacker.name,
            "target_name":   target.name,
            "weapon_name":   weapon.name,
            "roll":          roll_total,
            "target_ac":     target.ac,
            "hit":           hit,
            "advantage_mode": mode,
        }
        if hit:
            base_dmg = roll(weapon.damage_dice)
            raw = max(1, base_dmg + dmg_mod)
            for m in attacker.iter_modifiers():
                raw = m.on_outgoing_damage(attacker, target, raw, weapon.damage_type)
            damage = apply_damage(target, raw, dtype=weapon.damage_type, attacker=attacker)
            result.update({
                "damage":        damage,
                "damage_dice":   weapon.damage_dice,
                "damage_mod":    dmg_mod,
                "target_hp":     target.hp,
                "target_max_hp": target.max_hp,
                "target_alive":  target.is_alive(),
            })
        return result

    # ── AOE ───────────────────────────────────────────────────────────────────
    if t == "AOE":
        attacker   = _lookup_char(action.get("attacker", ""), world_state)
        target_ids = action.get("targets", [])
        item_name  = action.get("item", "")
        save_stat   = action.get("save_stat", "DEX").upper()
        save_dc     = int(action.get("save_dc", 13))
        half_on_save = action.get("half_on_save", True)

        # Damage dice live on the consumable itself — arbiter no longer copies
        # them into the action so we look them up here.
        item_obj = attacker.get_consumable(item_name) if (attacker and item_name) else None
        damage_dice = item_obj.effect_value if (item_obj and item_obj.effect_value) else "1d6"

        if item_name and attacker:
            if not attacker.consume(item_name):
                return {"type": "ERROR", "message": f"{attacker.name} 沒有「{item_name}」了"}

        target_results = []
        for tid in target_ids:
            target = _lookup_char(tid, world_state)
            if not target or not target.is_alive():
                continue
            success, save_roll = make_saving_throw(target, save_stat, save_dc)
            full_dmg   = roll(damage_dice)
            actual_dmg = full_dmg // 2 if (success and half_on_save) else full_dmg
            target.hp  = max(0, target.hp - actual_dmg)
            target_results.append({
                "target_name":  target.name,
                "save_roll":    save_roll,
                "save_success": success,
                "damage":       actual_dmg,
                "target_hp":    target.hp,
                "target_max_hp": target.max_hp,
                "target_alive": target.is_alive(),
            })

        remaining = None
        if item_name and attacker:
            c = attacker.get_consumable(item_name)
            remaining = c.quantity if c else 0

        return {
            "type":           "AOE",
            "attacker_name":  attacker.name if attacker else "未知",
            "item":           item_name,
            "damage_dice":    damage_dice,
            "save_stat":      save_stat,
            "save_dc":        save_dc,
            "target_results": target_results,
            "remaining":      remaining,
        }

    # ── USE_ITEM ──────────────────────────────────────────────────────────────
    if t == "USE_ITEM":
        char = _lookup_char(action.get("character", ""), world_state)
        if not char:
            return {"type": "ERROR", "message": "找不到角色"}
        item_name = action.get("item", "")
        c = char.get_consumable(item_name)
        if not c:
            return {"type": "ERROR", "message": f"找不到道具：{item_name}"}
        if c.quantity <= 0:
            return {"type": "ERROR", "message": f"{item_name} 已用完"}

        if c.effect_type == "heal":
            target_id = action.get("target", action.get("character", ""))
            target = _lookup_char(target_id, world_state) or char
            healed = apply_heal(target, c.effect_value)
            c.quantity -= 1
            return {
                "type":          "USE_ITEM",
                "item":          item_name,
                "character":     char.name,
                "target":        target.name,
                "healed":        healed,
                "target_hp":     target.hp,
                "target_max_hp": target.max_hp,
                "remaining":     c.quantity,
            }
        # Non-heal consumables (light, utility)
        c.quantity -= 1
        return {
            "type":      "USE_ITEM",
            "item":      item_name,
            "character": char.name,
            "remaining": c.quantity,
        }

    # ── ROLL ──────────────────────────────────────────────────────────────────
    if t == "ROLL":
        char = _lookup_char(action.get("character", ""), world_state)
        if not char:
            return {"type": "ERROR", "message": "找不到角色"}
        stat = action.get("stat", "DEX").upper()
        dc   = int(action.get("dc", 12))
        success, total = make_saving_throw(char, stat, dc)
        return {
            "type":           "ROLL",
            "character_name": char.name,
            "stat":           stat,
            "total":          total,
            "dc":             dc,
            "success":        success,
            "skill":          action.get("skill_description", ""),
        }

    # ── MOVE ──────────────────────────────────────────────────────────────────
    if t == "MOVE":
        char = _lookup_char(action.get("character", ""), world_state)
        if not char:
            return {"type": "ERROR", "message": "找不到角色"}
        # Reverse-lookup the canonical char_id so we can determine the actor's side
        char_id = next((cid for cid, c in world_state.characters.items() if c is char), None)
        old_pos: Vec2 = char.position
        battlefield = world_state.combat.battlefield if world_state.combat else None

        # Resolve destination from one of four formats, in priority order:
        #   1. target (creature_id) — move toward that character along the
        #      shortest line, capped at their exact position so you never
        #      overshoot ("我朝薩滿衝鋒")
        #   2. direction ("advance"/"retreat") — toward / away from enemy
        #      centroid, fixed distance
        #   3. target_position ([x, y]) — absolute coordinate
        #   4. delta ([dx, dy]) — raw 2D delta
        new_pos: Vec2 | None = None
        target_key = action.get("target")
        if target_key:
            target_char = _lookup_char(target_key, world_state)
            if not target_char:
                return {"type": "ERROR", "message": f"找不到移動目標：{target_key}"}
            dest = target_char.position
            delta = dest - old_pos
            if delta.length() <= MOVE_BUDGET_M + 1e-6:
                new_pos = dest
            else:
                new_pos = old_pos + delta.normalized() * MOVE_BUDGET_M
        elif "direction" in action:
            direction = str(action["direction"]).lower()
            distance = abs(float(action.get("distance", MOVE_BUDGET_M)))
            distance = min(distance, MOVE_BUDGET_M)
            actor_in_party = (char_id is not None and world_state.is_party_ally(char_id))
            opponents = []
            for oid, other in world_state.characters.items():
                if not other.is_alive():
                    continue
                other_in_party = world_state.is_party_ally(oid)
                if actor_in_party:
                    if other.is_npc and other.attitude == 0:
                        opponents.append(other)
                else:
                    if other_in_party:
                        opponents.append(other)
            if opponents:
                cx = sum(o.position.x for o in opponents) / len(opponents)
                cy = sum(o.position.y for o in opponents) / len(opponents)
                toward = (Vec2(cx, cy) - old_pos).normalized()
                if toward.length() < 1e-9:
                    toward = Vec2(1.0, 0.0)
            else:
                toward = Vec2(1.0, 0.0)   # no opponents → default forward = +x
            if direction == "advance":
                move_dir = toward
            elif direction == "retreat":
                move_dir = Vec2(-toward.x, -toward.y)
            else:
                return {"type": "ERROR", "message": f"未知 MOVE direction：{direction}"}
            new_pos = old_pos + move_dir * distance
        elif "target_position" in action:
            new_pos = Vec2.coerce(action["target_position"])
        elif "delta" in action:
            new_pos = old_pos + Vec2.coerce(action["delta"])
        else:
            return {"type": "ERROR", "message": "MOVE 需要 target、direction、target_position 或 delta"}

        # Clamp to movement budget, accounting for destination terrain cost.
        # Difficult / dangerous terrain doubles the effective movement spent.
        delta_vec = new_pos - old_pos
        phys_dist = delta_vec.length()
        mult = battlefield.terrain_multiplier(new_pos) if battlefield is not None else 1.0
        if mult == float("inf"):
            # Endpoint is BLOCKED — caller picked a wall as destination.
            return {"type": "ERROR",
                    "message": f"目的座標 ({new_pos.x:.1f}, {new_pos.y:.1f}) 被障礙物佔據"}
        effective_cost = phys_dist * mult
        if effective_cost > MOVE_BUDGET_M + 1e-6:
            # Rescale physical distance so the cost matches the budget.
            phys_dist = MOVE_BUDGET_M / mult
            new_pos = old_pos + delta_vec.normalized() * phys_dist
            mult = battlefield.terrain_multiplier(new_pos) if battlefield is not None else mult
            effective_cost = phys_dist * mult

        # Battlefield bounds gate (BLOCKED is already caught above)
        if battlefield is not None and not battlefield.in_bounds(new_pos):
            return {"type": "ERROR",
                    "message": f"目的座標 ({new_pos.x:.1f}, {new_pos.y:.1f}) 超出戰場邊界"}

        char.position = new_pos
        return {
            "type":              "MOVE",
            "character":         char.name,
            "from_pos":          old_pos,
            "to_pos":            new_pos,
            "distance":          effective_cost,   # what the movement budget pays
            "physical_distance": phys_dist,         # actual displacement (for UI)
            "terrain_mult":      mult,
            "description":       action.get("description", "移動"),
        }

    # ── DODGE ─────────────────────────────────────────────────────────────────
    if t == "DODGE":
        from .status import Dodging
        char = _lookup_char(action.get("character", ""), world_state)
        if not char:
            return {"type": "ERROR", "message": "找不到角色"}
        round_num = world_state.combat.round_number if world_state.combat else 0
        char.add_status(Dodging(applied_round=round_num))
        return {
            "type":      "DODGE",
            "character": char.name,
        }

    # ── HIDE ──────────────────────────────────────────────────────────────────
    if t == "HIDE":
        from .status import Hidden
        char = _lookup_char(action.get("character", ""), world_state)
        if not char:
            return {"type": "ERROR", "message": "找不到角色"}
        success, total = make_saving_throw(char, "DEX", 12)
        if success:
            round_num = world_state.combat.round_number if world_state.combat else 0
            char.add_status(Hidden(applied_round=round_num))
        return {
            "type":    "HIDE",
            "success": success,
            "total":   total,
        }

    # ── SPELL ─────────────────────────────────────────────────────────────────
    if t == "SPELL":
        from .spells import SPELLS

        caster = _lookup_char(action.get("caster", ""), world_state)
        if not caster:
            return {"type": "ERROR", "message": "找不到施法者"}
        spell_name = action.get("spell_name", "")
        spell = SPELLS.get(spell_name)
        if not spell:
            return {"type": "ERROR", "message": f"未知法術：{spell_name}"}
        if spell_name not in caster.spells:
            return {"type": "ERROR", "message": f"{caster.name} 不會「{spell_name}」"}
        if not caster.spellcasting_ability:
            return {"type": "ERROR", "message": f"{caster.name} 不是施法者"}

        # Resolve slot_level — explicit arg, or first available level >= spell.level
        requested = action.get("slot_level")
        if requested is None:
            slot_level = None
            for lvl in sorted(caster.spell_slots):
                if lvl >= spell.level and caster.spell_slots[lvl] > 0:
                    slot_level = lvl
                    break
            if slot_level is None:
                return {"type": "ERROR",
                        "message": f"{caster.name} 沒有可用的法術位施展「{spell_name}」"}
        else:
            slot_level = int(requested)
            if slot_level < spell.level:
                return {"type": "ERROR",
                        "message": f"「{spell_name}」需要 {spell.level} 環或以上的法術位"}
            if caster.spell_slots.get(slot_level, 0) <= 0:
                return {"type": "ERROR",
                        "message": f"{caster.name} 沒有 {slot_level} 環法術位"}

        # Resolve AOE center position. Two ways:
        #   1. target_position: explicit [x, y] (lets the caster place AOE
        #      between creatures, away from allies, etc.) — takes priority.
        #   2. target: creature id ("self" = caster), AOE centers on that body.
        target_pos_arg = action.get("target_position")
        if target_pos_arg is not None:
            center_pos = Vec2.coerce(target_pos_arg)
            center_name = f"座標 ({center_pos.x:.1f}, {center_pos.y:.1f})m"
        else:
            target_key = action.get("target", "")
            if target_key == "self":
                center_pos = caster.position
                center_name = caster.name
            else:
                target_char = _lookup_char(target_key, world_state)
                if not target_char:
                    return {"type": "ERROR", "message": f"找不到法術中心目標：{target_key}"}
                center_pos = target_char.position
                center_name = target_char.name

        # Range gate: caster ↔ center (euclidean) + LoS through battlefield obstacles
        dist_to_center = caster.position.distance_to(center_pos)
        if dist_to_center > spell.range_m:
            return {"type": "ERROR",
                    "message": f"目標距離 {dist_to_center:.1f}m，超出「{spell_name}」射程 {spell.range_m:.0f}m"}
        battlefield = world_state.combat.battlefield if world_state.combat else None
        if battlefield is not None and not battlefield.has_line_of_sight(
                caster.position, center_pos):
            return {"type": "ERROR",
                    "message": f"視線被遮擋，無法將「{spell_name}」投至 {center_name}"}

        # Save DC = 8 + prof_bonus + spellcasting_ability_modifier
        save_dc = 8 + caster.proficiency_bonus + caster.stats.modifier(caster.spellcasting_ability)

        # Find all alive characters within aoe_radius_m of center, scoped to current room.
        room = world_state.dungeon_map.current_room if world_state.dungeon_map else None
        if room is not None:
            scope_ids = set(room.npc_ids)
            for cid, c in world_state.characters.items():
                if world_state.is_party_ally(cid):
                    scope_ids.add(cid)
        else:
            scope_ids = set(world_state.characters.keys())

        affected_ids: list[str] = []
        for cid in scope_ids:
            c = world_state.characters.get(cid)
            if not c or not c.is_alive():
                continue
            if c.position.distance_to(center_pos) <= spell.aoe_radius_m + 1e-6:
                affected_ids.append(cid)

        # Roll saves, apply damage
        target_results = []
        for cid in affected_ids:
            target = world_state.characters[cid]
            success, save_roll = make_saving_throw(target, spell.save_ability, save_dc)
            full_dmg = roll(spell.damage_dice) if spell.damage_dice else 0
            actual_dmg = full_dmg // 2 if success else full_dmg
            if actual_dmg > 0:
                apply_damage(target, actual_dmg, dtype=spell.damage_type, attacker=caster)
            target_results.append({
                "target_name":   target.name,
                "save_roll":     save_roll,
                "save_success":  success,
                "damage":        actual_dmg,
                "target_hp":     target.hp,
                "target_max_hp": target.max_hp,
                "target_alive":  target.is_alive(),
            })

        # Decrement slot after successful cast
        caster.spell_slots[slot_level] = caster.spell_slots.get(slot_level, 0) - 1

        return {
            "type":          "SPELL",
            "caster_name":   caster.name,
            "spell_name":    spell_name,
            "slot_level":    slot_level,
            "center_name":   center_name,
            "save_stat":     spell.save_ability,
            "save_dc":       save_dc,
            "damage_dice":   spell.damage_dice,
            "damage_type":   spell.damage_type,
            "target_results": target_results,
        }

    return {"type": "ERROR", "message": f"未知行動類型：{t}"}


def consume_resources(resources: dict, action: dict, result: dict) -> None:
    """Decrement per-turn resource budget based on action.consumes declaration.

    Resource slots understood:
      'action'        — sets to 0 (one action per turn)
      'bonus_action'  — sets to 0
      'movement'      — subtracts result.get('distance', 0)
    Unknown slots are silently ignored (reserved for future expansion).
    """
    for slot in action.get("consumes", []):
        if slot == "action":
            resources["action"] = 0
        elif slot == "bonus_action":
            resources["bonus_action"] = 0
        elif slot == "movement":
            resources["movement"] = max(0.0, resources.get("movement", 0.0) - result.get("distance", 0))


def _format_intent(action: dict) -> str:
    """Render a one-line action label from a structured action dict.

    This replaces the old "fake-natural-language description" string the
    arbiter used to round-trip. Direct field rendering — no LLM, no synth.
    """
    t = action.get("type", "")
    if t == "ATTACK":
        target = action.get("target", "?")
        weapon = action.get("weapon", "")
        return f"攻擊 {target}（{weapon}）" if weapon else f"攻擊 {target}"
    if t == "MOVE":
        if "target" in action:
            return f"朝 {action['target']} 移動"
        if "target_position" in action:
            tp = action["target_position"]
            return f"移動至 ({tp[0]:.1f}, {tp[1]:.1f})"
        if "delta" in action:
            d = action["delta"]
            return f"移動 Δ({d[0]:+.1f}, {d[1]:+.1f})"
        if "direction" in action:
            return "前進" if action["direction"] == "advance" else "後退"
        return "移動"
    if t == "SPELL":
        name = action.get("spell_name", "?")
        if "target_position" in action:
            tp = action["target_position"]
            return f"施展 {name} → ({tp[0]:.1f}, {tp[1]:.1f})"
        if "target" in action:
            return f"施展 {name} → {action['target']}"
        return f"施展 {name}"
    if t == "DODGE":
        return "閃避"
    if t == "HIDE":
        return "躲藏"
    if t == "USE_ITEM":
        return f"使用 {action.get('item', '?')}"
    if t == "AOE":
        return f"投擲 {action.get('item', '?')}"
    if t == "ROLL":
        return f"檢定 {action.get('stat', '?')}"
    return t or "未知行動"


def format_result(action: dict, result: dict, actor_name: str = "") -> str:
    """Convert (action, execute_action result) into a text summary for the GM."""
    label = f"【{actor_name}】" if actor_name else "【玩家】"
    lines = [f"{label}{_format_intent(action)}"]
    t = result.get("type")

    if t == "ATTACK":
        hit_str = "命中" if result["hit"] else "未命中"
        mode = result.get("advantage_mode", "normal")
        mode_str = ""
        if mode == "advantage":
            mode_str = "（優勢）"
        elif mode == "disadvantage":
            mode_str = "（劣勢）"
        lines.append(
            f"使用 {result['weapon_name']}{mode_str}，攻擊骰 {result['roll']} vs AC {result['target_ac']}：{hit_str}"
        )
        if result["hit"]:
            alive    = "存活" if result.get("target_alive") else "倒下"
            dmg_mod  = result.get("damage_mod", 0)
            mod_str  = f"+{dmg_mod}" if dmg_mod > 0 else (str(dmg_mod) if dmg_mod < 0 else "")
            dice_str = f"{result['damage_dice']}{mod_str}"
            lines.append(
                f"造成 {result['damage']} 點傷害（{dice_str}），"
                f"{result['target_name']} HP {result['target_hp']}/{result['target_max_hp']}（{alive}）"
            )

    elif t == "AOE":
        lines.append(
            f"投擲 {result['item']}（{result['damage_dice']}，"
            f"{result['save_stat']} DC{result['save_dc']} 豁免半傷）"
        )
        for tr in result.get("target_results", []):
            save_str  = "豁免成功（半傷）" if tr["save_success"] else "豁免失敗"
            alive_str = "存活" if tr["target_alive"] else "倒下"
            lines.append(
                f"  {tr['target_name']}：{save_str}，受 {tr['damage']} 傷害，"
                f"HP {tr['target_hp']}/{tr['target_max_hp']}（{alive_str}）"
            )
        if result.get("remaining") is not None:
            lines.append(f"剩餘 {result['item']}：{result['remaining']} 個")

    elif t == "SPELL":
        lines.append(
            f"施展「{result['spell_name']}」（{result['slot_level']} 環，"
            f"{result['damage_dice']} {result['damage_type']}傷，"
            f"以 {result['center_name']} 為中心，"
            f"{result['save_stat']} DC{result['save_dc']} 豁免半傷）"
        )
        for tr in result.get("target_results", []):
            save_str  = "豁免成功（半傷）" if tr["save_success"] else "豁免失敗"
            alive_str = "存活" if tr["target_alive"] else "倒下"
            lines.append(
                f"  {tr['target_name']}：{save_str}，受 {tr['damage']} 傷害，"
                f"HP {tr['target_hp']}/{tr['target_max_hp']}（{alive_str}）"
            )

    elif t == "USE_ITEM":
        if result.get("healed") is not None:
            lines.append(
                f"使用 {result['item']}：{result['target']} 恢復 {result['healed']} HP"
                f"（{result['target_hp']}/{result['target_max_hp']}），剩餘 {result['remaining']} 個"
            )
        else:
            lines.append(f"使用 {result['item']}，剩餘 {result['remaining']} 個")

    elif t == "ROLL":
        outcome = "成功" if result["success"] else "失敗"
        skill = f"（{result['skill']}）" if result.get("skill") else ""
        lines.append(
            f"{result['character_name']} {result['stat']}{skill} 檢定 "
            f"{result['total']} vs DC {result['dc']}：{outcome}"
        )

    elif t == "MOVE":
        if "from_pos" in result and "to_pos" in result:
            fp, tp = result['from_pos'], result['to_pos']
            phys = result.get('physical_distance', result['distance'])
            mult = result.get('terrain_mult', 1.0)
            cost_note = f"，地形 ×{mult:.0f}（消耗 {result['distance']:.1f}m）" if mult > 1.0 + 1e-6 else ""
            lines.append(
                f"移動：{result['description']} "
                f"(({fp.x:.1f}, {fp.y:.1f})m → ({tp.x:.1f}, {tp.y:.1f})m, 走了 {phys:.1f}m{cost_note})"
            )
        else:
            lines.append(f"移動：{result['description']}")

    elif t == "DODGE":
        lines.append(f"{result['character']} 採取閃避姿態（下次被攻擊前，攻擊者擲劣勢）")

    elif t == "HIDE":
        outcome = "成功隱身" if result["success"] else "躲藏失敗"
        lines.append(f"躲藏 DEX 檢定 {result['total']}：{outcome}")

    elif t == "ERROR":
        lines.append(f"錯誤：{result['message']}")

    return "\n".join(lines)


# ── Unified combat context ────────────────────────────────────────────────────

@dataclass
class CombatContext:
    """Snapshot of combat state from a single actor's perspective.

    Built once per sub-action by GameSession and consumed by all three
    ActorController implementations. Strings are pre-formatted for direct
    inclusion in LLM prompts or human UI.
    """
    round_num: int
    actor_id: str
    actor_position: Vec2
    weapons_str: str         # "長劍（近戰 1.5m）、短弓（遠程 24m / 最大 96m）"
    allies_str: str          # "凱恩 HP 22/22，座標 (5.0, 15.0)m（距離你 0.0m）"
    enemies_str: str         # "哥布林 HP 7/7，座標 (6.5, 15.0)m（距離你 1.5m）"
    spells_str: str = ""     # "火球術（3 環，剩餘 1 個 3 環法術位）" — empty for non-casters
    enemies: dict = field(default_factory=dict)   # {cid: name} — alive valid attack targets
    allies: dict = field(default_factory=dict)    # {cid: name} — alive non-self friendlies in room
    resources: dict = field(default_factory=dict)


def _spells_str(char) -> str:
    if not char.spells or not char.spellcasting_ability:
        return ""
    from .spells import SPELLS
    parts = []
    for name in char.spells:
        spell = SPELLS.get(name)
        if not spell:
            continue
        available = [lvl for lvl in sorted(char.spell_slots)
                     if lvl >= spell.level and char.spell_slots[lvl] > 0]
        if available:
            slots_str = "、".join(f"{lvl} 環×{char.spell_slots[lvl]}" for lvl in available)
            slot_part = f"可用：{slots_str}"
        else:
            slot_part = "無可用法術位"

        effect_parts = [f"射程 {spell.range_m:.0f}m"]
        if spell.aoe_radius_m > 0:
            effect_parts.append(
                f"AOE 半徑 {spell.aoe_radius_m:.0f}m [敵我不分，含自己]"
            )
        if spell.damage_dice:
            effect_parts.append(
                f"{spell.save_ability} 豁免 {spell.damage_dice} {spell.damage_type}傷（半傷）"
            )
        effect_str = "，".join(effect_parts)

        parts.append(f"{spell.name}（{spell.level} 環，{effect_str}，{slot_part}）")
    return "、".join(parts)


def _weapons_str(char) -> str:
    parts = []
    for w in char.weapons:
        if w.range_type == "近戰":
            parts.append(f"{w.name}（近戰，伸手 {w.range_normal:.1f}m）")
        else:
            parts.append(
                f"{w.name}（遠程，正常 {w.range_normal:.0f}m / 最大 {w.range_long:.0f}m）"
            )
    return "、".join(parts) or "無武器（徒手）"


def _entry_for(other, viewer) -> str:
    d = other.position.distance_to(viewer.position)
    dodging = "（閃避中）" if other.has_status("dodging") else ""
    p = other.position
    return f"{other.name} HP {other.hp}/{other.max_hp}，座標 ({p.x:.1f}, {p.y:.1f})m（距離你 {d:.1f}m）{dodging}"


def build_combat_context(actor_id: str, actor, world_state,
                         resources: dict, round_num: int) -> CombatContext:
    """Build the combat-side view that any controller (LLM or human) consumes.

    Sides are decided by party membership: from the actor's POV, party members
    are allies and hostile NPCs (alive, in the same room, attitude == 0) are
    enemies. Room scoping applies if a dungeon_map is present.
    """
    ws = world_state
    is_party = ws.is_party_ally(actor_id)
    room = ws.dungeon_map.current_room if ws.dungeon_map else None
    room_ids = set(room.npc_ids) if room else set(ws.characters.keys())

    ally_parts: list[str] = []
    enemy_parts: list[str] = []
    enemies_dict: dict[str, str] = {}
    allies_dict: dict[str, str] = {}

    for oid, other in ws.characters.items():
        if oid == actor_id or not other.is_alive():
            continue
        other_in_party = ws.is_party_ally(oid)
        other_hostile = other.is_npc and other.attitude == 0 and oid in room_ids
        entry = _entry_for(other, actor)
        if is_party:
            if other_in_party:
                ally_parts.append(entry)
                allies_dict[oid] = other.name
            elif other_hostile:
                enemy_parts.append(entry)
                enemies_dict[oid] = other.name
        else:
            if other_hostile:
                ally_parts.append(entry)
                allies_dict[oid] = other.name
            elif other_in_party:
                enemy_parts.append(entry)
                enemies_dict[oid] = other.name

    return CombatContext(
        round_num=round_num,
        actor_id=actor_id,
        actor_position=actor.position,
        weapons_str=_weapons_str(actor),
        spells_str=_spells_str(actor),
        allies_str="、".join(ally_parts) or "無",
        enemies_str="、".join(enemy_parts) or "無",
        enemies=enemies_dict,
        allies=allies_dict,
        resources=dict(resources),
    )
