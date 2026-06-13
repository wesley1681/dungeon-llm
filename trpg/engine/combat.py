import math
import re as _re
from dataclasses import dataclass, field

from .dice import roll, roll_d20, combine_advantage
from .character import Character, CombatState
from .world_state import WorldState
from .vec2 import Vec2, Battlefield

# Movement budget per turn (D&D 5e default speed for medium humanoid = 30 ft ≈ 9 m).
MOVE_BUDGET_M = 9.0


def _roll_scaled_cantrip(dice_str: str, caster_level: int) -> int:
    """Roll cantrip damage with 5e level-based scaling.

    Calls roll('1dN') once per die so that mock side_effect sequences work
    correctly in tests (each individual die roll consumes one mock value).

    '1d8' at L1-4 → 1 call; at L5-10 → 2 calls; at L11-16 → 3 calls;
    at L17+ → 4 calls.  Any +N modifier on the base dice string is added once.
    """
    multiplier = 1
    if caster_level >= 17:
        multiplier = 4
    elif caster_level >= 11:
        multiplier = 3
    elif caster_level >= 5:
        multiplier = 2
    m = _re.match(r'^(\d+)(d\d+)([+-]\d+)?$', dice_str)
    if not m:
        # Unrecognised notation — fall back to a single roll() call.
        return roll(dice_str)
    base_count = int(m.group(1))
    die = m.group(2)          # e.g. "d8"
    modifier = int(m.group(3)) if m.group(3) else 0
    total = sum(roll(f"1{die}") for _ in range(base_count * multiplier))
    return total + modifier

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


def setup_combat_positions(world_state: WorldState, combat: CombatState,
                            *, rng=None, distance_m: float = 11.0) -> None:
    """Place combatants on the 2D battlefield at combat start.

    Creates a default open battlefield on the combat state if one isn't
    already attached.

    Two modes:
      - ``rng is None`` (legacy): axis-aligned along y = height/2. Party at
        x=5, melee enemies at x=6.5, ranged at x=11. Tests assert these.
      - ``rng`` provided: random party-center + random angle. All hostiles
        spawn at ``distance_m`` from the party centroid in the chosen
        direction. Y-stagger within clusters runs perpendicular to that axis.
        This breaks the absolute-coordinate spatial bias that prevented the
        RL grid head from learning relative positioning.
    """
    if combat.battlefield is None:
        combat.battlefield = Battlefield(width=DEFAULT_BATTLEFIELD_W,
                                         height=DEFAULT_BATTLEFIELD_H)
    bf = combat.battlefield

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
        is_caster = bool(char.spellcasting_ability)
        (ranged_ids if (has_ranged or is_caster) else melee_ids).append(cid)

    if rng is None:
        mid_y = bf.height / 2.0

        def place(ids: list[str], x: float) -> None:
            for cid, dy in zip(ids, _spawn_offsets(len(ids))):
                world_state.characters[cid].position = Vec2(x, mid_y + dy)

        place(party_ids, _PARTY_SPAWN_X)
        place(melee_ids, _MELEE_SPAWN_X)
        place(ranged_ids, _RANGED_SPAWN_X)
        return

    angle = rng.uniform(0.0, 2.0 * math.pi)
    ux, uy = math.cos(angle), math.sin(angle)
    perp_x, perp_y = -uy, ux

    cluster_half_span = max(
        (len(party_ids) - 1) / 2.0,
        (len(melee_ids) - 1) / 2.0,
        (len(ranged_ids) - 1) / 2.0,
        0.0,
    )
    margin = cluster_half_span + 1.0
    hostile_dx = distance_m * ux
    hostile_dy = distance_m * uy
    min_x = min(0.0, hostile_dx) - margin
    max_x = max(0.0, hostile_dx) + margin
    min_y = min(0.0, hostile_dy) - margin
    max_y = max(0.0, hostile_dy) + margin
    px = rng.uniform(-min_x, bf.width - max_x)
    py = rng.uniform(-min_y, bf.height - max_y)

    def place(ids: list[str], cx: float, cy: float) -> None:
        for cid, off in zip(ids, _spawn_offsets(len(ids))):
            x = cx + perp_x * off
            y = cy + perp_y * off
            world_state.characters[cid].position = Vec2(x, y)

    place(party_ids,  px, py)
    place(melee_ids,  px + hostile_dx, py + hostile_dy)
    place(ranged_ids, px + hostile_dx, py + hostile_dy)


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


def effective_ac(char: Character) -> int:
    """Compute character AC including all active Modifier hooks (Shield +5,
    etc.). Use this anywhere combat needs to compare an attack total to
    a target's defence — `char.ac` alone misses temporary buffs/debuffs."""
    ac = char.ac
    for m in char.iter_modifiers():
        ac = m.on_compute_ac(char, ac)
    return ac


def resolve_attack(attacker: Character, target: Character,
                   weapon=None, mode: str = "normal",
                   breakdown: dict | None = None) -> tuple[bool, int]:
    """Returns (hit, total_roll). mode: 'normal' / 'advantage' / 'disadvantage'.

    Iterates attacker + target Modifiers to apply hook-based mode adjustments
    (equipment passives, status effects like Dodging, etc.).

    Pass `breakdown` to receive the roll components as a side-channel: keys
    `d20`, `stat_mod_kind`, `stat_mod`, `prof`, `modifiers` (list of
    `(name, delta)`), and `total`. Used by the display layer to show e.g.
    `[d20(7)+STR(2)+prof(2)+blessed(3)]` so players can verify buffs landed.
    """
    if weapon is None:
        weapon = attacker.get_weapon()
    if "精巧" in (weapon.properties if weapon else []):
        if attacker.stats.modifier("DEX") > attacker.stats.modifier("STR"):
            stat_mod_kind = "DEX"
        else:
            stat_mod_kind = "STR"
        stat_mod = attacker.stats.modifier(stat_mod_kind)
    elif weapon and weapon.range_type == "遠程":
        stat_mod_kind = "DEX"
        stat_mod = attacker.stats.modifier("DEX")
    else:
        stat_mod_kind = "STR"
        stat_mod = attacker.stats.modifier("STR")

    for m in attacker.iter_modifiers():
        mode = m.on_outgoing_attack(attacker, target, weapon, mode)
    for m in target.iter_modifiers():
        mode = m.on_incoming_attack(target, attacker, weapon, mode)

    d20 = roll_d20(mode)
    total = d20 + stat_mod + attacker.proficiency_bonus
    # Numeric riders on the d20 total (e.g. Bless +1d4). Track each modifier's
    # delta so the display layer can attribute the bonus.
    mod_contribs: list[tuple[str, int]] = []
    for m in attacker.iter_modifiers():
        prev = total
        total = m.on_outgoing_attack_total(attacker, target, weapon, total)
        if total != prev:
            mod_contribs.append((getattr(m, "name", type(m).__name__), total - prev))

    if breakdown is not None:
        breakdown["d20"]           = d20
        breakdown["stat_mod_kind"] = stat_mod_kind
        breakdown["stat_mod"]      = stat_mod
        breakdown["prof"]          = attacker.proficiency_bonus
        breakdown["mode"]          = mode
        breakdown["modifiers"]     = mod_contribs
        breakdown["mode"]          = mode
        breakdown["total"]         = total

    return total >= effective_ac(target), total


def _end_concentration(caster_id: str, world_state) -> None:
    """Remove all status effects whose source_id matches caster_id from every
    character in the world. Called when a concentration spell ends so the
    effect (e.g. paralyzed from hold_person) is cleaned up immediately."""
    from .status import StatusEffect
    for char in world_state.characters.values():
        char.status_effects[:] = [
            fx for fx in char.status_effects
            if not (isinstance(fx, StatusEffect) and fx.source_id == caster_id)
        ]


def apply_damage(target: Character, amount, dtype: str = "untyped",
                 attacker=None, world_state=None, crit: bool = False) -> int:
    """Apply damage to target. `amount` is either a dice notation string
    (e.g. '1d6+2') or a pre-rolled int. Filters through target's Modifiers.

    If `target` is concentrating on a spell, a CON save vs DC max(10, dmg//2)
    fires; failure clears `target.concentrating_on` and, if `world_state` is
    provided, also removes any status effects sourced from that caster.

    `crit` — the damage came from a critical hit; matters only against a
    dying target (5e: counts as 2 death-save failures instead of 1).

    Returns the actual damage dealt after modifiers.
    """
    if isinstance(amount, str):
        amount = roll(amount)
    for m in target.iter_modifiers():
        amount = m.on_incoming_damage(target, attacker, amount, dtype)

    # Typed resistance table (Wave 1, data-driven via the damage_table trait):
    # 0.5 resist / 0.0 immune / 2.0 vulnerable, rounded down like 5e halving.
    # Negative multiplier = ABSORB (Iron Golem's fire): the hit heals instead.
    mult = target.damage_multipliers.get(dtype)
    if mult is not None:
        if mult < 0:
            if target.is_alive():
                target.hp = min(target.max_hp,
                                target.hp + int(amount * -mult))
            return 0
        amount = int(amount * mult)
    amount = max(0, amount)

    # Regeneration suppression window (Wave 2): record the type of any damage
    # that actually lands (post-resistance; an immunity-zeroed or absorbed hit
    # was not "taken" in 5e terms). Cleared at the target's own turn start.
    if amount > 0:
        target.recent_damage_types.add(dtype)

    # Damage while dying (PC at 0 HP): count as a death save failure directly.
    # A critical hit counts as 2 failures; damage >= max HP is instant death.
    # No HP change (already 0). Any damage knocks a stabilized PC back to
    # actively dying (resumes death saves).
    if target.is_dying():
        if amount > 0:
            failures = 3 if amount >= target.max_hp else (2 if crit else 1)
            target.death_saves["failures"] = (
                target.death_saves.get("failures", 0) + failures)
            target.death_saves.pop("stable", None)
        return amount

    # Undead Fortitude (Zombie): a hit that would drop the creature to 0
    # lets it make a CON save DC 5 + damage to stay at 1 HP instead —
    # unless the damage is radiant or from a critical hit (5e MM).
    if (target.undead_fortitude and amount >= target.hp and amount > 0
            and not crit and dtype != "光耀"):
        success, _ = make_saving_throw(target, "CON", 5 + amount)
        if success:
            target.hp = 1
            return amount

    overflow = amount - target.hp
    target.hp = max(0, target.hp - amount)
    # 5e 即死規則：剩餘傷害（溢出 0 HP 的部分）≥ 最大 HP → 直接死亡，
    # 不進入瀕死狀態。
    if (target.hp == 0 and not target.is_npc and amount > 0
            and overflow >= target.max_hp):
        target.death_saves["failures"] = 3

    # Death throes (Wave 3, Balor): an NPC carrying the spec explodes when it
    # dies — typed AoE damage, save for half, hits BOTH sides (5e: each
    # creature within range). The spec is popped first so chained explosions
    # (one death throes killing another carrier) can't re-trigger this one.
    # Needs world_state to find victims; every combat damage path passes it.
    if (target.hp == 0 and target.is_npc and amount > 0
            and target.death_throes and world_state is not None):
        spec, target.death_throes = target.death_throes, None
        events = getattr(world_state, "pending_events", None)
        if events is None:
            events = world_state.pending_events = []
        for vid, victim in list(world_state.characters.items()):
            if victim is target or victim.is_dead():
                continue
            if (victim.position.distance_to(target.position)
                    > spec["radius_m"] + 1e-6):
                continue
            ok, _ = make_saving_throw(victim, spec.get("save_stat", "DEX"),
                                      int(spec["dc"]), world_state=world_state)
            raw = roll(spec["damage_dice"])
            dmg = raw // 2 if ok else raw
            dealt = apply_damage(victim, dmg, dtype=spec["damage_type"],
                                 attacker=target, world_state=world_state)
            events.append({
                "type":         "DEATH_THROES",
                "source_name":  target.name,
                "target_name":  victim.name,
                "save_success": ok,
                "damage":       dealt,
                "damage_dice":  spec["damage_dice"],
                "damage_type":  spec["damage_type"],
                "target_hp":    victim.hp,
                "target_alive": victim.is_alive(),
            })

    # D&D 5e: any damage wakes a sleeping creature.
    if amount > 0 and target.has_status("asleep"):
        target.status_effects = [s for s in target.status_effects
                                  if getattr(s, "name", "") != "asleep"]

    if amount > 0 and target.concentrating_on:
        dc = max(10, amount // 2)
        success, _ = make_saving_throw(target, "CON", dc)
        if not success:
            target.concentrating_on = ""
            if world_state is not None:
                caster_id = next(
                    (cid for cid, c in world_state.characters.items() if c is target),
                    None,
                )
                if caster_id:
                    _end_concentration(caster_id, world_state)

    return amount


def heal_flat(target: Character, amount: int, *, hp_cap: int | None = None) -> int:
    """Restore a flat HP amount (clamped ≥0). A dying target is stabilised and
    its death saves reset (5e). ``hp_cap`` lets pool healers (Preserve Life)
    bound the result below max HP (its "can't exceed half max HP" rule); None
    caps at the target's max HP. Returns HP actually restored."""
    amount = max(0, int(amount))
    ceiling = target.max_hp if hp_cap is None else min(target.max_hp, hp_cap)
    if target.is_dying():
        target.reset_death_saves()
    before = target.hp
    target.hp = min(ceiling, target.hp + amount)
    return max(0, target.hp - before)


def apply_heal(target: Character, dice_notation: str) -> int:
    # A heal can never harm: a negative ability modifier can pull the rolled
    # total below zero (e.g. 1d4-2 from a caster with a dumped casting stat).
    # Returns the rolled amount (overheal included) for back-compat; heal_flat
    # does the clamp + death-save reset.
    amount = max(0, roll(dice_notation))
    heal_flat(target, amount)
    return amount


def roll_death_save(char: Character) -> dict:
    """Roll one death saving throw for a dying PC (5e PHB):
      nat 20  → regain 1 HP (back in the fight)
      nat 1   → 2 failures
      10-19   → 1 success; at 3 successes the character STABILIZES
                (stays at 0 HP, unconscious, no further saves)
      2-9     → 1 failure; at 3 failures the character dies

    A stabilized character skips the roll entirely. Damage taken while
    dying/stable is handled by apply_damage (failure(s) + un-stabilize).

    Returns {"d20", "outcome", "successes", "failures", "note"} where
    outcome ∈ {"revived","stable","skipped_stable","success","failure","dead"}.
    """
    saves = char.death_saves
    if saves.get("stable"):
        return {"d20": 0, "outcome": "skipped_stable",
                "successes": saves.get("successes", 0),
                "failures": saves.get("failures", 0),
                "note": f"{char.name} 已穩定（0 HP 昏迷）"}
    d20 = roll("1d20")
    if d20 == 20:
        char.hp = 1
        char.reset_death_saves()
        outcome, note = "revived", f"{char.name} 死亡豁免 d20={d20}：奇蹟回生（HP 1）"
    elif d20 == 1:
        saves["failures"] = saves.get("failures", 0) + 2
        outcome = "dead" if saves["failures"] >= 3 else "failure"
        note = (f"{char.name} 死亡豁免 d20={d20}：大失敗，累計失敗 {saves['failures']}"
                + ("——死亡" if outcome == "dead" else ""))
    elif d20 >= 10:
        saves["successes"] = saves.get("successes", 0) + 1
        if saves["successes"] >= 3:
            saves["stable"] = True
            outcome, note = "stable", f"{char.name} 死亡豁免 d20={d20}：成功（第3次）——穩定化"
        else:
            outcome = "success"
            note = f"{char.name} 死亡豁免 d20={d20}：成功，累計 {saves['successes']}/3"
    else:
        saves["failures"] = saves.get("failures", 0) + 1
        if saves["failures"] >= 3:
            outcome, note = "dead", f"{char.name} 死亡豁免 d20={d20}：第3次失敗——{char.name} 死亡"
        else:
            outcome = "failure"
            note = f"{char.name} 死亡豁免 d20={d20}：失敗，累計 {saves['failures']}/3"
    return {"d20": d20, "outcome": outcome,
            "successes": saves.get("successes", 0),
            "failures": saves.get("failures", 0), "note": note}


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


def tick_aura_damage(char: Character, world_state: WorldState,
                     round_num: int) -> list[dict]:
    """At `char`'s turn start, apply damage from any HOSTILE aura they stand in.

    Currently handles Spirit Guardians (4.5m radius, 3d8 radiant on failed WIS
    save, half on success). The status sits on the CASTER; this scans the
    world for any opposing caster with spirit_guardians_active within range.

    Returns a list of damage entry dicts (one per triggering aura) so the
    caller can log them. Empty list when no aura applies.
    """
    if world_state is None or world_state.combat is None:
        return []
    char_id = next(
        (cid for cid, c in world_state.characters.items() if c is char), None
    )
    if char_id is None or not char.is_alive():
        return []
    char_is_party = world_state.is_party_ally(char_id)
    events: list[dict] = []
    for caster_id, caster in world_state.characters.items():
        if caster_id == char_id or not caster.is_alive():
            continue
        if world_state.is_party_ally(caster_id) == char_is_party:
            continue
        has_sg = any(getattr(fx, "name", "") == "spirit_guardians_active"
                     for fx in caster.status_effects)
        if not has_sg:
            continue
        if caster.position.distance_to(char.position) > 4.5 + 1e-6:
            continue
        save_stat = caster.spellcasting_ability or "WIS"
        spell_mod = caster.stats.modifier(save_stat)
        save_dc = 8 + caster.proficiency_bonus + spell_mod
        success, save_roll = make_saving_throw(char, "WIS", save_dc)
        raw = roll("3d8")
        amount = raw // 2 if success else raw
        dealt = apply_damage(char, amount, dtype="光耀",
                             attacker=caster, world_state=world_state)
        events.append({
            "type":        "AURA_DAMAGE",
            "source_name": caster.name,
            "spell_name":  "靈體守護",
            "save_stat":   "WIS",
            "save_dc":     save_dc,
            "save_roll":   save_roll,
            "save_success": success,
            "damage":      dealt,
            "damage_dice": "3d8",
            "target_hp":   char.hp,
            "target_max_hp": char.max_hp,
            "target_alive": char.is_alive(),
        })
        if not char.is_alive():
            break

    # Frightful presence (Wave 3, adult dragons / tarrasque): checked at each
    # ENEMY's turn start — the passive-aura pattern (same chokepoint as spirit
    # guardians, zero per-driver wiring). One WIS save per turn while exposed:
    # success → immune to that source for the rest of the combat (5e "24h"),
    # failure → frightened with end-of-turn re-saves. Divergence from RAW
    # (documented): escaping via the re-save does NOT grant immunity — the
    # next turn-start check rolls again until the first clean success.
    if char.is_alive():
        char_id_fp = char_id
        for src_id, src in world_state.characters.items():
            fp = src.frightful_presence
            if (fp is None or src_id == char_id_fp or not src.is_alive()
                    or world_state.is_party_ally(src_id) == char_is_party):
                continue
            if (src_id in char.frightful_immune_to
                    or char.has_status("frightened")
                    or "frightened" in char.condition_immunities):
                continue
            if src.position.distance_to(char.position) > fp["radius_m"] + 1e-6:
                continue
            dc = int(fp["dc"])
            success, save_roll = make_saving_throw(
                char, "WIS", dc, world_state=world_state,
                fail_applies_status=True)
            applied = ""
            if success:
                char.frightful_immune_to.add(src_id)
            else:
                if apply_named_status(char, "frightened",
                                      round_num=round_num, source_id=src_id,
                                      rounds=int(fp.get("rounds", 10)),
                                      save_each=f"WIS DC{dc}"):
                    applied = "frightened"
            events.append({
                "type":         "FRIGHTFUL_PRESENCE",
                "source_name":  src.name,
                "save_dc":      dc,
                "save_roll":    save_roll,
                "save_success": success,
                "status_applied": applied,
                "immune_now":   success,
            })

    # Generic status-tick damage (Wave 2): any StatusEffect on `char` whose
    # metadata declares tick_damage_dice deals that typed damage at the
    # holder's turn start (swallowed acid, future ongoing burns). Effects with
    # metadata["ends_if_source_dead"] are released instead of ticking once
    # their source is gone (behir dies → victim is regurgitated).
    from .status import StatusEffect as _TickSE
    released: list = []
    for fx in list(char.status_effects):
        if not isinstance(fx, _TickSE):
            continue
        md = fx.metadata or {}
        dice = md.get("tick_damage_dice")
        if not dice:
            continue
        source = world_state.characters.get(fx.source_id)
        if md.get("ends_if_source_dead") and (source is None or not source.is_alive()):
            released.append(fx)
            continue
        if not char.is_alive():
            break
        dealt = apply_damage(char, roll(dice),
                             dtype=md.get("tick_damage_type", "untyped"),
                             attacker=source, world_state=world_state)
        events.append({
            "type":        "STATUS_TICK_DAMAGE",
            "status_name": fx.name,
            "source_name": source.name if source else "",
            "damage":      dealt,
            "damage_dice": dice,
            "damage_type": md.get("tick_damage_type", "untyped"),
            "target_hp":   char.hp,
            "target_max_hp": char.max_hp,
            "target_alive": char.is_alive(),
        })
    if released:
        # Identity comparison — StatusEffect is a dataclass with structural
        # equality; `in` would drop the wrong twin if duplicates ever appear.
        char.status_effects[:] = [fx for fx in char.status_effects
                                  if not any(fx is r for r in released)]
    return events


def _walk_path(old_pos: Vec2, intended_pos: Vec2, budget_m: float,
                battlefield) -> tuple[Vec2, float, float, list[dict]]:
    """Walk from old_pos toward intended_pos along a straight ray, respecting
    walls, dangerous terrain, and difficult terrain along the path.

    Returns (final_pos, phys_dist_travelled, effective_cost, terrain_events).
    `terrain_events` is a list of {kind, pos, ...} dicts describing things
    that happened along the path (entered_dangerous, hit_wall, etc.) so the
    caller can surface them in the result.

    Stops early if:
      - the next sample would enter a BLOCKED cell (wall) — final_pos is the
        last clear point along the ray
      - effective_cost would exceed budget_m — final_pos is the point at
        which budget runs out (accounts for terrain multipliers)

    If battlefield is None (no terrain), reduces to a budget-clamped straight
    line.
    """
    delta = intended_pos - old_pos
    max_dist = delta.length()
    if max_dist < 1e-9:
        return (old_pos, 0.0, 0.0, [])
    if battlefield is None:
        if max_dist <= budget_m + 1e-6:
            return (intended_pos, max_dist, max_dist, [])
        return (old_pos + delta.normalized() * budget_m,
                budget_m, budget_m, [])

    direction = delta.normalized()
    # Step at half the engine grid resolution so we never skip across a 1-cell
    # wall (e.g. a 1m-wide pillar with cells at 0.5m).
    step_m = max(0.1, battlefield.grid_resolution / 2.0)

    cur_pos = old_pos
    cur_dist = 0.0
    cur_cost = 0.0
    events: list[dict] = []
    visited_cells: set[tuple[int, int]] = set()

    n_max_steps = int(max_dist / step_m) + 2
    for i in range(1, n_max_steps + 1):
        next_dist = min(i * step_m, max_dist)
        next_pos = old_pos + direction * next_dist
        if not battlefield.in_bounds(next_pos):
            events.append({"kind": "out_of_bounds",
                           "pos": (next_pos.x, next_pos.y)})
            return (cur_pos, cur_dist, cur_cost, events)
        if battlefield.is_blocked(next_pos):
            events.append({"kind": "hit_wall",
                           "pos": (next_pos.x, next_pos.y)})
            return (cur_pos, cur_dist, cur_cost, events)

        mult = battlefield.terrain_multiplier(next_pos)
        seg_phys = next_dist - cur_dist
        seg_cost = seg_phys * mult
        if cur_cost + seg_cost > budget_m + 1e-6:
            # Budget runs out partway through this segment — pro-rate.
            remaining = budget_m - cur_cost
            if remaining > 1e-6 and mult > 0:
                allowed_phys = remaining / mult
                cur_pos = cur_pos + direction * allowed_phys
                cur_dist += allowed_phys
                cur_cost = budget_m
            events.append({"kind": "budget_exhausted",
                           "pos": (cur_pos.x, cur_pos.y)})
            return (cur_pos, cur_dist, cur_cost, events)

        # Damage on first entry into a dangerous cell — once per cell, not
        # once per sample. Sampling sub-cell means we'd otherwise double-roll.
        cell = battlefield._cell(next_pos)
        if cell not in visited_cells:
            visited_cells.add(cell)
            if battlefield.is_dangerous(next_pos):
                events.append({"kind": "entered_dangerous",
                               "pos": (next_pos.x, next_pos.y),
                               "dice": DANGEROUS_TERRAIN_DAMAGE})

        cur_pos = next_pos
        cur_dist = next_dist
        cur_cost += seg_cost
        if cur_dist >= max_dist - 1e-6:
            return (cur_pos, cur_dist, cur_cost, events)

    return (cur_pos, cur_dist, cur_cost, events)


def make_saving_throw(character: Character, stat: str, dc: int,
                      breakdown: dict | None = None,
                      world_state=None,
                      fail_applies_status: bool = False) -> tuple[bool, int]:
    """Roll a saving throw vs `dc`. Returns (success, total).

    `breakdown` side-channel: when supplied, populated with d20 / stat_mod /
    prof / modifier contributions so the display layer can show e.g.
    `[d20(11)+CON(1)+blessed(3)]`.

    `fail_applies_status` — failing this save leaves (or keeps) a status on
    the character. Legendary resistance (Wave 3) only spends itself on these:
    the mechanic exists to protect a boss's action economy, not to shave HP
    off damage-half saves (5e MM tactical guidance). Damage-only call sites
    leave the default False and can never burn a use."""

    def _legendary_override(total: int) -> tuple[bool, int] | None:
        """A failed save → choose to succeed instead (5e legendary resistance,
        per-combat pool). Applies to rolled failures AND auto-failures."""
        if not fail_applies_status or character.legendary_resistance_uses <= 0:
            return None
        character.legendary_resistance_uses -= 1
        if breakdown is not None:
            breakdown["legendary_resistance"] = True
            breakdown.setdefault("modifiers", []).append(
                ("legendary_resistance", 0))
        return True, total

    # Auto-fail check: some conditions (paralyzed, stunned) make specific saves
    # automatically fail regardless of the roll.
    # 5e 昏迷（瀕死 0 HP）：STR / DEX 豁免自動失敗。
    if character.is_dying() and stat in ("STR", "DEX"):
        if breakdown is not None:
            breakdown.update({"d20": 0, "stat": stat, "stat_mod": 0, "prof": 0,
                              "modifiers": [("昏迷", -999)], "total": -999})
        return False, -999
    for m in character.iter_modifiers():
        if m.on_auto_fail_save(character, stat):
            total = -999
            if breakdown is not None:
                breakdown["d20"] = 0
                breakdown["stat"] = stat
                breakdown["stat_mod"] = 0
                breakdown["prof"] = 0
                breakdown["modifiers"] = [(getattr(m, "name", type(m).__name__), -999)]
                breakdown["total"] = total
            lr = _legendary_override(total)
            if lr is not None:
                return lr
            return False, total
    if character.pending_portent is not None:
        d20 = character.pending_portent
        character.pending_portent = None
    else:
        d20 = roll("1d20")
    stat_mod = character.stats.modifier(stat)
    prof = character.proficiency_bonus if stat in character.proficiencies else 0
    modifier = stat_mod + prof
    mod_contribs: list[tuple[str, int]] = []
    for m in character.iter_modifiers():
        prev = modifier
        modifier = m.on_saving_throw(character, stat, modifier)
        if modifier != prev:
            mod_contribs.append((getattr(m, "name", type(m).__name__), modifier - prev))
    # Aura of Protection: any living ally with aura_of_protection_bonus > 0
    # within 3m adds their bonus to this character's saves. 「盟友」以受保護
    # 者的陣營判定——原寫法用絕對的 is_party_ally（=玩家隊），敵方聖騎士
    # 的光環保護不了自己人，玩家隊聖騎士的光環反而會加持站旁邊的敵人。
    if world_state is not None:
        char_id_ap = next((cid for cid, c in world_state.characters.items()
                           if c is character), None)
        char_side = (world_state.is_party_ally(char_id_ap)
                     if char_id_ap else True)
        for oid, other in world_state.characters.items():
            if other is character or not other.is_alive():
                continue
            if world_state.is_party_ally(oid) != char_side:
                continue
            aura = getattr(other, "aura_of_protection_bonus", 0)
            if aura > 0 and other.position.distance_to(character.position) <= 3.0 + 1e-6:
                prev = modifier
                modifier += aura
                mod_contribs.append(("aura_protection", modifier - prev))
                break  # only one paladin's aura applies
    total = d20 + modifier
    if breakdown is not None:
        breakdown["d20"]       = d20
        breakdown["stat"]      = stat
        breakdown["stat_mod"]  = stat_mod
        breakdown["prof"]      = prof
        breakdown["modifiers"] = mod_contribs
        breakdown["total"]     = total
    if total < dc:
        lr = _legendary_override(total)
        if lr is not None:
            return lr
    return total >= dc, total


def _lookup_char(key: str, world_state: WorldState):
    """Look up character by ID first, then by display name (arbiter may return either)."""
    char = world_state.characters.get(key)
    if char is None:
        char = next((c for c in world_state.characters.values() if c.name == key), None)
    return char


# ── Reaction system ──────────────────────────────────────────────────────────
#
# Reactions fire in response to events on *another* creature's turn. Each
# character has one reaction per round, tracked on Character.reaction_used and
# reset at self_turn_start. New reaction kinds register on Character.reactions
# and add a branch to _legal_reactions (legality) + greedy_reaction_decider
# (the default tactic). No per-name branch lives at the trigger sites anymore.
#
# DECISION SEAM (2026-06-13, for model-controlled reactions): the engine splits
# into (1) LEGALITY — which reactions the reactor *could* spend right now
# (budget / slots / known reactions); assembled by _legal_reactions — and
# (2) CHOICE — whether and which legal option to actually use; delegated to a
# ReactionDecider. The default greedy decider reproduces the historical
# hardcoded behaviour bit-for-bit; the RL env injects ws.reaction_decider to
# route a model-controlled creature's choice through its policy. Mechanics
# (consume slot, attach Shielded, halve damage) stay in the engine — the
# decider only returns a chosen option skill_id, or None to decline.


@dataclass
class ReactionContext:
    """A pending reaction decision handed to a ReactionDecider.

    ``options`` is the LEGAL set (already filtered for reaction budget / slots /
    known reactions) — the decider picks one of these skill_ids or returns None
    to decline. Payload fields are populated per ``trigger``:
      "attack"      — attacker, attack_total (the to-hit roll vs the reactor)
      "auto_damage" — (no roll; e.g. Magic Missile auto-hit)
      "uncanny"     — attacker (incoming melee hit, pre-damage)
      "spell"       — spell_caster, spell_level (an enemy is casting)
    """
    reactor: Character
    reactor_id: str
    trigger: str
    options: list
    world_state: WorldState
    attacker: Character | None = None
    attack_total: int = 0
    spell_caster: Character | None = None
    spell_level: int = 0


def _reactor_id(defender: Character, world_state: WorldState) -> str:
    if world_state is None:
        return ""
    return next((cid for cid, c in world_state.characters.items()
                 if c is defender), "")


def _legal_reactions(reactor: Character, trigger: str,
                     world_state: WorldState) -> list:
    """The reaction skill_ids ``reactor`` may legally spend for ``trigger`` RIGHT
    NOW (reaction unused, alive, knows the reaction, resources available). This
    is the hard mask — identity-/tactic-agnostic. The decider chooses among
    these; it can never widen the set."""
    if reactor.reaction_used or not reactor.is_alive():
        return []
    known = reactor.reactions or []
    out: list = []
    has_slot = any(reactor.spell_slots.get(lvl, 0) > 0 for lvl in range(1, 10))
    if trigger in ("attack", "auto_damage"):
        if "shield_spell" in known and has_slot:
            out.append("shield_spell")
    elif trigger == "uncanny":
        if "uncanny_dodge_rogue" in known:
            out.append("uncanny_dodge_rogue")
    elif trigger == "spell":
        if "counterspell" in known:
            out.append("counterspell")   # slot-level legality checked by caller
    return out


def greedy_reaction_decider(ctx: "ReactionContext") -> str | None:
    """Default reaction policy = the historical hardcoded behaviour, now
    expressed as a decider so it can be swapped for a model. Reproduces:
      - Shield vs an attack roll: spend ONLY when +5 AC flips this hit to a miss
        (else save the reaction).
      - Shield vs auto-hit (Magic Missile): always spend when legal.
      - Uncanny Dodge: always spend on a melee hit when legal.
      - Counterspell: always spend (the first-eligible scan lives in the caller).
    """
    if ctx.trigger == "attack" and "shield_spell" in ctx.options:
        base_ac = effective_ac(ctx.reactor)
        if base_ac <= ctx.attack_total < base_ac + 5:
            return "shield_spell"
        return None
    if ctx.trigger == "auto_damage" and "shield_spell" in ctx.options:
        return "shield_spell"
    if ctx.trigger == "uncanny" and "uncanny_dodge_rogue" in ctx.options:
        return "uncanny_dodge_rogue"
    if ctx.trigger == "spell" and "counterspell" in ctx.options:
        return "counterspell"
    return None


def decide_reaction(ctx: "ReactionContext") -> str | None:
    """Route a reaction decision through the world's injected decider (model-
    controlled) or fall back to the greedy default. Returns a chosen option
    skill_id (must be in ctx.options) or None to decline; an out-of-set return
    is treated as a decline (the legality mask is authoritative)."""
    hook = getattr(ctx.world_state, "reaction_decider", None) if ctx.world_state else None
    choice = hook(ctx) if hook is not None else greedy_reaction_decider(ctx)
    return choice if choice in ctx.options else None


def _fire_shield_spell(defender: Character, world_state: WorldState) -> int:
    """Cast Shield: consume reaction + lowest available slot, attach Shielded.
    Returns the slot level consumed."""
    from .status import Shielded
    defender.reaction_used = True
    consumed = 0
    for lvl in (1, 2, 3, 4, 5, 6, 7, 8, 9):
        if defender.spell_slots.get(lvl, 0) > 0:
            defender.spell_slots[lvl] -= 1
            consumed = lvl
            break
    round_num = world_state.combat.round_number if world_state.combat else 0
    defender.add_status(Shielded(applied_round=round_num))
    return consumed


def _try_react_to_attack(defender: Character, attacker: Character,
                          attack_total: int, world_state: WorldState
                          ) -> tuple[bool, str, int]:
    """Offer the defender a reaction to an incoming attack roll.

    Returns (blocked, reaction_name, slot_consumed):
      blocked       — True if the attack should be re-evaluated as a miss
      reaction_name — for narration (empty when nothing fired)
      slot_consumed — spell slot level used (0 if none)

    LEGALITY (has slot, reaction unused) is decided by _legal_reactions; the
    CHOICE (Shield's default rule = spend only when +5 AC flips this hit to a
    miss; a model may decide otherwise) is delegated to the reaction decider.
    """
    options = _legal_reactions(defender, "attack", world_state)
    if not options:
        return False, "", 0
    ctx = ReactionContext(
        reactor=defender, reactor_id=_reactor_id(defender, world_state),
        trigger="attack", options=options, world_state=world_state,
        attacker=attacker, attack_total=attack_total)
    if decide_reaction(ctx) == "shield_spell":
        slot = _fire_shield_spell(defender, world_state)
        return True, "shield_spell", slot
    return False, "", 0


def _try_uncanny_dodge(defender: Character,
                       world_state: WorldState | None = None) -> bool:
    """Rogue's L5 Uncanny Dodge: spend reaction to halve damage from one attack
    per turn. Returns True if the reaction fires (and consumes it). The choice
    routes through the reaction decider (default: always when legal)."""
    options = _legal_reactions(defender, "uncanny", world_state)
    if not options:
        return False
    ctx = ReactionContext(
        reactor=defender, reactor_id=_reactor_id(defender, world_state),
        trigger="uncanny", options=options, world_state=world_state)
    if decide_reaction(ctx) == "uncanny_dodge_rogue":
        defender.reaction_used = True
        return True
    return False


def _try_react_to_auto_damage(defender: Character, world_state: WorldState
                                ) -> tuple[bool, str, int]:
    """For Magic Missile and similar auto-hits, Shield blocks (5e rule: Shield's
    casting trigger explicitly names Magic Missile). Default = block whenever
    legal; a model may decline. Returns (blocked, reaction_name, slot_consumed)."""
    options = _legal_reactions(defender, "auto_damage", world_state)
    if not options:
        return False, "", 0
    ctx = ReactionContext(
        reactor=defender, reactor_id=_reactor_id(defender, world_state),
        trigger="auto_damage", options=options, world_state=world_state)
    if decide_reaction(ctx) == "shield_spell":
        slot = _fire_shield_spell(defender, world_state)
        return True, "shield_spell", slot
    return False, "", 0


def apply_named_status(target: Character, status_name: str, *, round_num: int,
                       source_id: str = "", rounds: int | None = None,
                       save_each: str = "") -> bool:
    """Instantiate + attach a status by registry name (MODIFIER_CLASSES ∪
    engine-only classes; unknown names get a generic StatusEffect shell, the
    long-standing SPELL-handler fallback). Returns True iff the target has the
    status afterwards (False = condition immunity dropped it). Shared by the
    SPELL handler and the eye-ray table executor — escalation logic calls this
    twice with different names instead of duplicating the build."""
    from .status import ALL_STATUS_CLASSES, StatusEffect
    cls = ALL_STATUS_CLASSES.get(status_name)
    if cls is not None:
        try:
            fx = cls(applied_round=round_num, source_id=source_id)
        except TypeError:
            fx = cls(applied_round=round_num)
        fx.save_each = save_each
        fx.rounds_remaining = rounds
    else:
        fx = StatusEffect(name=status_name, expires_on="never",
                          rounds_remaining=rounds, save_each=save_each,
                          applied_round=round_num, source_id=source_id)
    target.add_status(fx)
    return target.has_status(status_name)


def _apply_weapon_hit_rider(attacker: Character, target: Character,
                            rider: dict, world_state: WorldState,
                            result: dict, was_dying_before_hit: bool) -> None:
    """Execute a natural weapon's damage/drain rider (Weapon.on_hit).

    Composable spec fields (all optional, data-driven — no creature names):
      damage_dice / damage_type      extra typed damage on hit
      save_stat + save_dc            save against the extra damage
      save_half = True               save halves instead of negating
      drain_stat = ("STR", "1d4")    reduce an ability score (floor 1)
      drain_max_hp = True            reduce max HP by the rider damage dealt
                                     (wight life drain; heals can't undo it)
    Status-type riders are handled by the shared rider machinery in
    _resolve_single_attack, not here.
    """
    extra = 0
    if rider.get("damage_dice"):
        extra = roll(rider["damage_dice"])
        save_dc = rider.get("save_dc")
        if save_dc and rider.get("save_stat"):
            success, save_roll = make_saving_throw(
                target, rider["save_stat"], int(save_dc),
                world_state=world_state)
            result["rider_damage_save_roll"] = save_roll
            result["rider_damage_save_success"] = success
            if success:
                extra = extra // 2 if rider.get("save_half") else 0
        if extra > 0 and not was_dying_before_hit:
            dealt = apply_damage(target, extra,
                                 dtype=rider.get("damage_type", "untyped"),
                                 attacker=attacker, world_state=world_state)
            result["rider_damage"] = dealt
            result["damage"] = result.get("damage", 0) + dealt
            result["target_hp"] = target.hp
            result["target_alive"] = target.is_alive()
            extra = dealt

    if rider.get("drain_stat"):
        stat, dice = rider["drain_stat"]
        loss = roll(dice)
        cur = getattr(target.stats, stat)
        setattr(target.stats, stat, max(1, cur - loss))
        result["rider_stat_drain"] = (stat, loss)

    if rider.get("drain_max_hp"):
        # Drain scales with the rider's own damage (5e wight) or, when the
        # rider deals no damage itself, with the weapon hit recorded so far.
        drained = extra if extra > 0 else result.get("damage", 0)
        if drained > 0:
            target.max_hp = max(1, target.max_hp - drained)
            target.hp = min(target.hp, target.max_hp)
            result["rider_max_hp_drain"] = drained


def _apply_knockback(attacker: Character, target: Character,
                     distance: float, world_state: WorldState,
                     result: dict) -> None:
    """Forced linear push of ``target`` directly away from ``attacker`` by up
    to ``distance`` metres (Pushing Attack = 4.5m; reusable for any shove /
    fling). Steps outward and stops at the last cell before a battlefield edge
    or obstacle — a push never teleports a creature out of bounds or into a
    wall. Records the realised distance on ``result['knockback']``."""
    if distance <= 0:
        return
    direction = target.position - attacker.position
    if direction.length() < 1e-6:
        return   # co-located: no defined push direction
    direction = direction.normalized()
    bf = world_state.combat.battlefield if world_state.combat else None
    step = 0.5
    pos = target.position
    moved = 0.0
    while moved + step <= distance + 1e-9:
        cand = pos + direction * step
        if bf is not None and (not bf.in_bounds(cand) or bf.is_blocked(cand)):
            break
        pos = cand
        moved += step
    if moved > 0:
        target.position = pos
        result["knockback"] = {
            "distance": round(moved, 2),
            "to": [round(pos.x, 2), round(pos.y, 2)],
        }


def _try_attack_roll_boost(attacker: Character, roll_total: int,
                           target_ac: int, d20: int) -> dict | None:
    """Post-roll attack boosts: spend a resource AFTER seeing the roll to add
    to it (the mechanic the engine previously lacked — "擲骰後加值").

      precision_attack            +1d8 (Battle Master superiority die, 4 uses)
      channel_divinity_guided_strike  +10 flat (War Cleric, 1 use)

    Greedy default: spend only when the roll missed but the boost can flip it to
    a hit, and the resource is available; a natural 1 can't be saved (auto-miss).
    Returns {"skill_id","amount"} if spent, else None. Deducts the use here
    (these are passive proxies — engine_ready=False, no available_skills entry,
    so execute_action's own use-deduction never touches them).

    The CHOICE is greedy; routing it to a model is an integration-wave concern
    (decision-context obs is strict-append → no retrain) — see
    project_reaction_legendary: conditional-use has no 1v1 headroom."""
    if d20 == 1 or roll_total >= target_ac:
        return None
    gap = target_ac - roll_total
    known = attacker.known_abilities or []
    # Precision Attack first (renewable 4/short-rest); the die may or may not
    # flip — that's the maneuver's nature — but only gambled when it could.
    if "precision_attack" in known and 0 < gap <= 8:
        if attacker.ability_uses.get("precision_attack", 4) > 0:
            attacker.ability_uses["precision_attack"] = (
                attacker.ability_uses.get("precision_attack", 4) - 1)
            return {"skill_id": "precision_attack", "amount": roll("1d8")}
    # Guided Strike: deterministic +10, guaranteed flip for gap ≤ 10.
    if "channel_divinity_guided_strike" in known and 0 < gap <= 10:
        if attacker.ability_uses.get("channel_divinity_guided_strike", 1) > 0:
            attacker.ability_uses["channel_divinity_guided_strike"] = (
                attacker.ability_uses.get("channel_divinity_guided_strike", 1) - 1)
            return {"skill_id": "channel_divinity_guided_strike", "amount": 10}
    return None


def _ally_adjacent_to(world_state: WorldState, attacker: Character,
                      target: Character) -> bool:
    """Any living ally of `attacker` (its side, not the player's) within
    melee reach (1.5m) of `target`. Shared by Sneak Attack and Pack Tactics."""
    atk_id = next((cid for cid, c in world_state.characters.items()
                   if c is attacker), None)
    atk_side = world_state.is_party_ally(atk_id) if atk_id else True
    for oid, other in world_state.characters.items():
        if other is attacker or other is target:
            continue
        if world_state.is_party_ally(oid) == atk_side and other.is_alive():
            if other.position.distance_to(target.position) <= 1.5 + 1e-6:
                return True
    return False


def _resolve_single_attack(attacker: Character, target: Character, action: dict,
                            world_state: WorldState, dmg_mod: int,
                            mode: str) -> dict:
    """Resolve one weapon attack roll + damage. Returns an ATTACK-shaped result dict."""
    weapon = attacker.get_weapon(action.get("weapon", ""))
    # 5e 昏迷（瀕死）：對其攻擊有優勢。逐次判定（多重攻擊中目標可能
    # 在前一擊倒下，後續攻擊立即取得優勢）。
    was_dying_before_hit = target.is_dying()
    if was_dying_before_hit:
        mode = combine_advantage(mode, "advantage")
    # Pack Tactics (Wave 1 trait): advantage while an ally is adjacent to the
    # target. Re-checked per attack — allies can drop mid-multiattack.
    if attacker.pack_tactics and _ally_adjacent_to(world_state, attacker, target):
        mode = combine_advantage(mode, "advantage")
    attack_bd: dict = {}
    hit, roll_total = resolve_attack(attacker, target, weapon, mode=mode,
                                     breakdown=attack_bd)
    target_ac = effective_ac(target)

    # Post-roll attack boost (Precision Attack / Guided Strike): on a miss,
    # spend a resource to add to the roll and maybe flip it to a hit.
    roll_boost = None
    if not hit:
        roll_boost = _try_attack_roll_boost(
            attacker, roll_total, target_ac, attack_bd.get("d20", 0))
        if roll_boost:
            roll_total += roll_boost["amount"]
            attack_bd["roll_boost"] = roll_boost
            hit = roll_total >= target_ac

    reaction_blocked, reaction_name, reaction_slot = (False, "", 0)
    if hit:
        reaction_blocked, reaction_name, reaction_slot = _try_react_to_attack(
            target, attacker, roll_total, world_state
        )
        if reaction_blocked:
            target_ac = effective_ac(target)
            if roll_total < target_ac:
                hit = False

    result = {
        "type":           "ATTACK",
        "attacker_name":  attacker.name,
        "target_name":    target.name,
        "weapon_name":    weapon.name,
        "roll":           roll_total,
        "roll_breakdown": attack_bd,
        "target_ac":      target_ac,
        "hit":            hit,
        "advantage_mode": mode,
    }
    if reaction_blocked:
        result["reaction"]      = reaction_name
        result["reaction_slot"] = reaction_slot
    if roll_boost:
        result["roll_boost"] = roll_boost

    if hit:
        # 5e：對麻痺/沉睡/昏迷（瀕死）目標的近戰命中自動暴擊。
        auto_crit = (
            (target.has_status("paralyzed") or target.has_status("asleep")
             or target.is_dying())
            and distance_m(attacker, target) <= 1.5 + 1e-6
        )
        # Assassinate (Rogue L3): round 1 + attacker was hidden when this
        # attack started = treat as critical (proxy for "surprised target").
        round_num = world_state.combat.round_number if world_state.combat else 0
        if (not auto_crit and round_num == 1
                and "assassinate" in attacker.known_abilities
                and attacker.has_status("hidden")):
            auto_crit = True
        d20_result = attack_bd.get("d20", 20)
        is_crit = d20_result >= attacker.crit_range
        if auto_crit or is_crit:
            base_dmg = roll(weapon.damage_dice) + roll(weapon.damage_dice)
        else:
            base_dmg = roll(weapon.damage_dice)

        raw = max(1, base_dmg + dmg_mod)
        dmg_mod_contribs: list[tuple[str, int]] = []
        for m in attacker.iter_modifiers():
            prev = raw
            raw = m.on_outgoing_damage(attacker, target, raw, weapon.damage_type)
            if raw != prev:
                dmg_mod_contribs.append(
                    (getattr(m, "name", type(m).__name__), raw - prev)
                )
        # Rogue Uncanny Dodge (L5): defender spends reaction to halve damage.
        uncanny_dodge_fired = _try_uncanny_dodge(target, world_state)
        if uncanny_dodge_fired:
            raw = max(1, raw // 2)
        damage = apply_damage(target, raw, dtype=weapon.damage_type,
                              attacker=attacker, world_state=world_state,
                              crit=(auto_crit or is_crit))
        result.update({
            "damage":           damage,
            "damage_dice":      weapon.damage_dice,
            "damage_mod":       dmg_mod,
            "damage_base_roll": base_dmg,
            "damage_modifiers": dmg_mod_contribs,
            "target_hp":        target.hp,
            "target_max_hp":    target.max_hp,
            "target_alive":     target.is_alive(),
        })

        # Sneak Attack (5e PHB p.96): fires once per turn when one of the
        # following is true:
        #   - the attacker had advantage on the attack roll
        #     (covers hidden attacker, paladin vow target, reckless attacker, etc.)
        #   - an ally is adjacent to the target (within 1.5m) and the attacker
        #     did not have disadvantage
        #   - target is restrained / prone / poisoned (legacy trigger; kept for
        #     web/grapple cases where the engine doesn't grant attacker
        #     advantage automatically)
        sneak_dmg = 0
        if attacker.sneak_attack_dice:
            attacker_had_advantage = (attack_bd.get("mode") == "advantage")
            # 「攻擊者的盟友」以攻擊者的陣營判定（共用 helper，pack tactics 同源）。
            ally_adjacent = _ally_adjacent_to(world_state, attacker, target)
            attacker_had_disadvantage = (attack_bd.get("mode") == "disadvantage")
            target_disadvantaged = (
                target.has_status("restrained") or
                target.has_status("prone") or
                target.has_status("poisoned")
            )
            if (attacker_had_advantage
                    or (ally_adjacent and not attacker_had_disadvantage)
                    or target_disadvantaged):
                dice_parts = attacker.sneak_attack_dice.split("d")
                n_dice = int(dice_parts[0]) if len(dice_parts) >= 1 else 1
                die_size = dice_parts[1] if len(dice_parts) >= 2 else "6"
                sneak_dmg = sum(roll(f"1d{die_size}") for _ in range(n_dice))
                # 同一次命中只算一次死亡豁免失敗：目標在這次攻擊前就已瀕死
                # 時，武器傷害的 apply_damage 已記過失敗，偷襲傷害不再重複
                # 計（對 0 HP 目標也不改變 HP）。
                if not was_dying_before_hit:
                    apply_damage(target, sneak_dmg, dtype="穿刺",
                                 attacker=attacker, world_state=world_state)
                result["sneak_attack_damage"] = sneak_dmg
                result["damage"] = result.get("damage", 0) + sneak_dmg
                result["target_hp"] = target.hp
                result["target_alive"] = target.is_alive()

        # Hunter's Mark: +1d6 when the attacker is the declared source of the mark.
        from .status import StatusEffect as _SE3
        attacker_id_hm = next(
            (cid for cid, c in world_state.characters.items() if c is attacker),
            None,
        )
        for fx in target.status_effects:
            if (isinstance(fx, _SE3) and fx.name == "hunters_mark"
                    and fx.source_id == attacker_id_hm):
                hm_dmg = roll("1d6")
                apply_damage(target, hm_dmg, dtype="穿刺",
                             attacker=attacker, world_state=world_state)
                result["hunters_mark_damage"] = hm_dmg
                result["damage"] = result.get("damage", 0) + hm_dmg
                result["target_hp"] = target.hp
                result["target_alive"] = target.is_alive()
                break

        # Divine Smite: Paladin expends a spell slot on hit for bonus radiant
        # damage. 2d8 at slot 1, +1d8 per slot level above 1 (max 5d8 at slot 4+).
        if hit:
            smite_slot = action.get("divine_smite_slot", 0)
            if smite_slot > 0 and attacker.spell_slots.get(smite_slot, 0) > 0:
                attacker.spell_slots[smite_slot] -= 1
                smite_dice = min(5, 1 + smite_slot)   # 2d8 at L1, +1d8/slot
                smite_dmg = sum(roll("1d8") for _ in range(smite_dice))
                apply_damage(target, smite_dmg, dtype="光耀",
                             attacker=attacker, world_state=world_state)
                result["divine_smite_damage"] = smite_dmg
                result["damage"] = result.get("damage", 0) + smite_dmg
                result["target_hp"] = target.hp
                result["target_alive"] = target.is_alive()

        if auto_crit:
            result["auto_crit"] = True

        # On-hit riders come from two sources with the same machinery:
        #   - the ACTION (maneuvers: trip / menacing / hold-on-hit set
        #     rider_* fields when building the action dict)
        #   - the WEAPON definition (Wave 1 natural weapons: wolf bite →
        #     prone, ghoul claws → paralysis). Weapon riders also support
        #     extra typed damage and drains — see _apply_weapon_hit_rider.
        weapon_rider = getattr(weapon, "on_hit", None) or {}
        rider_status = action.get("rider_status", "") or weapon_rider.get("status", "")
        if rider_status and target.is_alive():
            rider_dc = action.get("rider_save_dc") or weapon_rider.get("save_dc")
            from .status import MODIFIER_CLASSES, StatusEffect
            round_num = world_state.combat.round_number if world_state.combat else 0
            attacker_id_str = next(
                (cid for cid, c in world_state.characters.items() if c is attacker),
                "",
            )

            def _build_rider_fx():
                from .status import ENGINE_ONLY_STATUS_CLASSES
                cls = (MODIFIER_CLASSES.get(rider_status)
                       or ENGINE_ONLY_STATUS_CLASSES.get(rider_status))
                if cls is not None:
                    try:
                        fx = cls(applied_round=round_num,
                                 source_id=attacker_id_str)
                    except TypeError:
                        fx = cls(applied_round=round_num)
                else:
                    fx = StatusEffect(
                        name=rider_status, expires_on="self_turn_end",
                        rounds_remaining=1, applied_round=round_num,
                        source_id=attacker_id_str,
                    )
                # Riders may bound the effect (ghoul paralysis: repeat CON
                # save each turn-end, hard cap in rounds) — without this a
                # rider-applied 'never'-expiring condition would be permanent.
                # Action-built riders (ability builders) use the same fields
                # with a rider_ prefix, mirroring the weapon spec.
                save_each = (action.get("rider_save_each")
                             or weapon_rider.get("save_each"))
                if save_each:
                    fx.save_each = save_each
                rounds = action.get("rider_rounds") or weapon_rider.get("rounds")
                if rounds:
                    fx.rounds_remaining = int(rounds)
                md = (action.get("rider_metadata")
                      or weapon_rider.get("metadata"))
                if md:
                    fx.metadata.update(md)
                return fx

            if rider_dc:
                # Save-on-fail rider (trip / menacing / hold-on-hit).
                rider_stat = (action.get("rider_save_stat")
                              or weapon_rider.get("save_stat", "STR"))
                rider_bd: dict = {}
                success, save_roll = make_saving_throw(
                    target, rider_stat, int(rider_dc), breakdown=rider_bd,
                    world_state=world_state, fail_applies_status=True,
                )
                result["rider_save_roll"]      = save_roll
                result["rider_save_stat"]      = rider_stat
                result["rider_save_success"]   = success
                result["rider_save_breakdown"] = rider_bd
                if not success:
                    target.add_status(_build_rider_fx())
                    result["rider_status_applied"] = rider_status
            else:
                # Unconditional on-hit rider (distracting strike).
                target.add_status(_build_rider_fx())
                result["rider_status_applied"] = rider_status

        # Weapon damage/drain riders (independent of the status rider above):
        # wyvern sting poison, fire elemental burn, shadow STR drain, wight
        # life drain. Skipped when the target died to the base hit — except
        # drains, which 5e applies as part of the same hit.
        if weapon_rider and (weapon_rider.get("damage_dice")
                             or weapon_rider.get("drain_stat")
                             or weapon_rider.get("drain_max_hp")):
            _apply_weapon_hit_rider(attacker, target, weapon_rider,
                                    world_state, result,
                                    was_dying_before_hit)

        # Action-level damage rider: a smite-style extra typed packet declared
        # by the ability builder, independent of the weapon (Wrathful Smite =
        # +1d6 精神 on hit). Unconditional on hit — any save-gated condition
        # rides the separate rider_status machinery above.
        if action.get("rider_damage_dice") and target.is_alive():
            _apply_weapon_hit_rider(
                attacker, target,
                {"damage_dice": action["rider_damage_dice"],
                 "damage_type": action.get("rider_damage_type", "untyped")},
                world_state, result, was_dying_before_hit)

        # Forced-move rider (Pushing Attack: shove the target up to 4.5m
        # straight back on hit). Optionally gated by a STR save (maneuver: fail
        # → pushed); with no push_save_dc the push is unconditional (monster
        # fling). Clamped to the battlefield by _apply_knockback.
        push_m = float(action.get("push_distance_m", 0) or 0)
        if push_m > 0 and not target.is_dead():
            pushed_ok = True
            push_dc = action.get("push_save_dc")
            if push_dc:
                stat = action.get("push_save_stat", "STR")
                success, save_roll = make_saving_throw(
                    target, stat, int(push_dc), world_state=world_state)
                result["push_save_roll"] = save_roll
                result["push_save_success"] = success
                pushed_ok = not success
            if pushed_ok:
                _apply_knockback(attacker, target, push_m, world_state, result)

    # Attacking reveals the attacker (5e PHB p.177): drop Hidden status.
    # Runs on both hit and miss — a swing-and-miss still gives your position
    # away. After this, multi-attack iterations and future turns won't see
    # Hidden's advantage hook.
    from .status import Hidden as _Hidden
    if any(isinstance(fx, _Hidden) for fx in attacker.status_effects):
        attacker.status_effects[:] = [
            fx for fx in attacker.status_effects if not isinstance(fx, _Hidden)
        ]

    return result


def _try_counterspell(caster: Character, spell_level: int,
                      world_state: WorldState) -> tuple[bool, str, int]:
    """Check if an opposing character can counter this spell with a reaction.

    Returns (countered, counterspeller_name, slot_used).
    Requires: "counterspell" in reactions, unused reaction, slot >= max(3, spell_level).
    Auto-succeeds for any spell level (simplified — full 5e needs ability check for >3).
    """
    caster_id = next(
        (cid for cid, c in world_state.characters.items() if c is caster), None
    )
    caster_in_party = world_state.is_party_ally(caster_id) if caster_id else False

    for oid, other in world_state.characters.items():
        if not other.is_alive() or other.reaction_used:
            continue
        other_in_party = world_state.is_party_ally(oid)
        if other_in_party == caster_in_party:
            continue   # same side — don't counter your own ally's spells
        if "counterspell" not in (other.reactions or []):
            continue
        min_slot = max(3, spell_level)
        available_slot = next(
            (lvl for lvl in sorted(other.spell_slots)
             if lvl >= min_slot and other.spell_slots[lvl] > 0),
            None,
        )
        if available_slot is None:
            continue
        # CHOICE: this opponent CAN counter (legal). Whether it does is the
        # decider's call (default greedy = always; a model may decline to save
        # the slot/reaction). The first opponent that elects to counter wins —
        # 5e RAW resolves one counterspell, and we keep the historical scan
        # order so the greedy default is bit-for-bit unchanged.
        ctx = ReactionContext(
            reactor=other, reactor_id=oid, trigger="spell",
            options=["counterspell"], world_state=world_state,
            spell_caster=caster, spell_level=spell_level)
        if decide_reaction(ctx) != "counterspell":
            continue
        other.reaction_used = True
        other.spell_slots[available_slot] -= 1
        return True, other.name, available_slot

    return False, "", 0


def _check_leveled_spell_limit(caster: Character, slot_level: int) -> dict | None:
    """5e: one leveled spell per turn. Returns ERROR dict if caster already
    cast a leveled spell this turn, else None. Cantrip (slot_level == 0) is
    always allowed. Caller is responsible for setting the flag on success
    via ``caster.leveled_spell_cast_this_turn = True``.
    """
    if slot_level <= 0:
        return None
    if getattr(caster, "leveled_spell_cast_this_turn", False):
        return {"type": "ERROR",
                "message": f"{caster.name} 本回合已施過有環法術 (5e 一回合限一個)"}
    return None


def _resolve_spell_attack_roll(caster: Character, target: Character,
                               damage_dice: str, damage_type: str,
                               world_state: WorldState,
                               add_spell_mod: bool = True) -> dict:
    """One spell attack roll vs AC + on-hit damage. Shared by the single-target
    SPELL_ATTACK handler and multi-ray spells (scorching ray), so the attack
    maths live in exactly one place. Aggregates advantage from both sides'
    modifiers (same rules as resolve_attack), applies the paralyzed/asleep/dying
    auto-crit-in-melee rule and Uncanny Dodge halving.

    Pure roll + damage: the caller owns range / LoS / slot gating and any
    on-hit status rider. Returns the per-attack fields the result dict needs."""
    spell_mod = caster.stats.modifier(caster.spellcasting_ability)
    attack_bonus = spell_mod + caster.proficiency_bonus
    mode = "normal"
    for m in caster.iter_modifiers():
        mode = m.on_outgoing_attack(caster, target, None, mode)
    for m in target.iter_modifiers():
        mode = m.on_incoming_attack(target, caster, None, mode)
    if target.is_dying():
        mode = combine_advantage(mode, "advantage")

    d = caster.position.distance_to(target.position)
    d20 = roll_d20(mode)
    roll_total = d20 + attack_bonus
    target_ac = effective_ac(target)
    is_crit_natural = d20 >= 20
    is_fumble = d20 == 1
    auto_crit_paralyzed = (
        (target.has_status("paralyzed") or target.has_status("asleep")
         or target.is_dying())
        and d <= 1.5 + 1e-6
    )
    hit = (not is_fumble) and (is_crit_natural or roll_total >= target_ac)
    is_crit = is_crit_natural or (hit and auto_crit_paralyzed)

    damage = 0
    if hit:
        base = roll(damage_dice)
        if is_crit:
            base += roll(damage_dice)
        raw = base + (spell_mod if add_spell_mod else 0)
        raw = max(1, raw)
        if _try_uncanny_dodge(target, world_state):
            raw = max(1, raw // 2)
        damage = apply_damage(target, raw, dtype=damage_type,
                              attacker=caster, world_state=world_state,
                              crit=is_crit)
    return {
        "d20":            d20,
        "attack_bonus":   attack_bonus,
        "roll_total":     roll_total,
        "target_ac":      target_ac,
        "hit":            hit,
        "crit":           is_crit,
        "advantage_mode": mode,
        "damage":         damage,
    }


def execute_action(action: dict, world_state: WorldState) -> dict:
    """Execute an action dict and, on success, consume one ability use.

    Limited-use deduction lives HERE (the one funnel every driver passes
    through) — it used to be copy-pasted per driver and was MISSING from the
    env opponent turn, BC demo collection and the v1 env, so opponents and
    demo experts had unlimited action_surge/rage/channel_divinity (measured:
    opp battle_master fired action_surge 4-7×/game with max_uses=1).
    Deducts only on success: an ERROR result costs nothing.
    """
    result = _execute_action_impl(action, world_state)
    if result.get("type") != "ERROR":
        sid = action.get("skill_id", "")
        if sid:
            from .abilities import ABILITY_REGISTRY
            ab = ABILITY_REGISTRY.get(sid)
            if ab is not None and ab.max_uses > 0:
                actor_key = (action.get("attacker") or action.get("caster")
                             or action.get("character"))
                actor = (_lookup_char(actor_key, world_state)
                         if isinstance(actor_key, str) else None)
                if actor is not None:
                    actor.ability_uses[sid] = (
                        actor.ability_uses.get(sid, ab.max_uses) - 1)
    return result


def _execute_action_impl(action: dict, world_state: WorldState) -> dict:
    """Execute a parsed action JSON from the arbiter. Returns a result summary dict.

    Every action dict must carry a ``skill_id`` identifying the Skill/Ability
    that produced it. Build through ``Skill.build_action`` or
    ``Ability.build_action`` rather than handwriting dicts so this is
    automatic. The check raises loudly to surface silent-bug paths.
    """
    if "skill_id" not in action:
        raise ValueError(
            f"action dict missing skill_id — build via Skill.build_action() "
            f"or Ability.build_action() instead of handwriting the dict. "
            f"Got: {action!r}"
        )
    t = action.get("type")

    # Incapacitating conditions (status.INCAPACITATING_STATUSES) — RAW 5e:
    # cannot take actions or reactions. Refuse the action up-front so a
    # policy or model that emits one anyway doesn't bypass the condition.
    actor_id = action.get("attacker") or action.get("caster") or action.get("character")
    if isinstance(actor_id, str):
        actor = world_state.characters.get(actor_id)
        if actor is not None and actor.is_incapacitated():
            from .status import INCAPACITATING_STATUSES
            cond = next(s for s in INCAPACITATING_STATUSES if actor.has_status(s))
            return {"type": "ERROR",
                    "message": f"{actor.name} 處於 {cond} 狀態，無法行動"}

    # ── ATTACK ────────────────────────────────────────────────────────────────
    if t == "ATTACK":
        attacker = _lookup_char(action.get("attacker", ""), world_state)
        target   = _lookup_char(action.get("target",   ""), world_state)
        if not attacker:
            return {"type": "ERROR", "message": "找不到攻擊者"}
        if not target:
            return {"type": "ERROR", "message": "找不到目標"}
        # 瀕死（0 HP 死亡豁免中）的角色仍可被攻擊（5e：對昏迷者攻擊有優勢、
        # 近戰命中自動暴擊、每次命中計死亡豁免失敗）；只有真正死亡才擋。
        if target.is_dead():
            return {"type": "ERROR", "message": f"{target.name} 已死亡"}

        # Target-state gates (data fields set by ability builders — behir
        # swallow needs a restrained target and must not re-swallow):
        rts = action.get("requires_target_status")
        if rts and not target.has_status(rts):
            return {"type": "ERROR",
                    "message": f"{target.name} 未處於 {rts} 狀態，無法使用此招"}
        bts = action.get("blocked_by_target_status")
        if bts and target.has_status(bts):
            return {"type": "ERROR",
                    "message": f"{target.name} 已處於 {bts} 狀態"}

        weapon = attacker.get_weapon(action.get("weapon", ""))
        battlefield = world_state.combat.battlefield if world_state.combat else None
        in_range, reason, range_mode = attack_range_check(attacker, target, weapon, battlefield)
        if not in_range:
            return {"type": "ERROR", "message": reason}

        # Charmed condition: cannot attack the charmer
        from .status import StatusEffect as _SE
        attacker_id = next(
            (cid for cid, c in world_state.characters.items() if c is attacker), None
        )
        for fx in target.status_effects:
            if isinstance(fx, _SE) and fx.name == "charmed" and fx.source_id == attacker_id:
                return {"type": "ERROR",
                        "message": f"{target.name} 被魅惑，無法攻擊 {attacker.name}"}

        if weapon.ammo:
            if not attacker.has_ammo(weapon.ammo):
                return {"type": "ERROR", "message": f"{attacker.name} 沒有 {weapon.ammo} 了"}
            attacker.consume(weapon.ammo)

        # Spell-slot-armed weapon strikes (Wrathful Smite: a L1 slot arms a
        # +1d6 精神 + frightened bonus-action strike). Charged on cast — a whiff
        # still spends the slot (5e: you cast the spell, then it rides your
        # hit). Distinct from divine_smite_slot, which is charged on hit and
        # scales. Optional concentration is registered so the existing break /
        # cleanup machinery ends both it and its rider-applied condition.
        atk_slot = int(action.get("spell_slot_cost", 0))
        if atk_slot > 0:
            if attacker.spell_slots.get(atk_slot, 0) <= 0:
                return {"type": "ERROR",
                        "message": f"{attacker.name} 沒有 {atk_slot} 環法術位"}
            err = _check_leveled_spell_limit(attacker, atk_slot)
            if err is not None:
                return err
            attacker.spell_slots[atk_slot] -= 1
            attacker.leveled_spell_cast_this_turn = True
            if action.get("concentration"):
                attacker_id_c = next(
                    (cid for cid, c in world_state.characters.items()
                     if c is attacker), None)
                if attacker_id_c:
                    _end_concentration(attacker_id_c, world_state)
                attacker.concentrating_on = action.get("spell_name", "smite")

        if "精巧" in weapon.properties:
            dmg_mod = max(attacker.stats.modifier("STR"), attacker.stats.modifier("DEX"))
        elif weapon.range_type == "遠程":
            dmg_mod = attacker.stats.modifier("DEX")
        else:
            dmg_mod = attacker.stats.modifier("STR")

        target_dodging = target.has_status("dodging")
        mode = combine_advantage(range_mode, "disadvantage" if target_dodging else "normal")
        if action.get("reckless") and weapon.range_type == "近戰":
            mode = combine_advantage(mode, "advantage")
            from .status import Reckless
            round_num = world_state.combat.round_number if world_state.combat else 0
            attacker.add_status(Reckless(applied_round=round_num))

        # Vow of Enmity: paladin who declared the vow attacks marked target with advantage.
        from .status import StatusEffect as _SE_VOW
        for fx in target.status_effects:
            if (isinstance(fx, _SE_VOW) and fx.name == "vow_target"
                    and fx.source_id == attacker_id):
                mode = combine_advantage(mode, "advantage")
                break

        # Abilities that ride the ATTACK pipeline but are a single strike by
        # rule (behir swallow = ONE bite) override the multiattack count.
        n_attacks = int(action.get("n_attacks") or attacker.attacks_per_action)
        attack_results = []
        for _ in range(n_attacks):
            # Extra Attack 可以繼續打瀕死目標（每次命中=死亡豁免失敗），
            # 直到目標真正死亡。
            if target.is_dead():
                break
            single = _resolve_single_attack(attacker, target, action,
                                            world_state, dmg_mod, mode)
            attack_results.append(single)

        if n_attacks == 1:
            return attack_results[0]

        total_damage = sum(r.get("damage", 0) for r in attack_results)
        return {
            "type":          "MULTI_ATTACK",
            "attacker_name": attacker.name,
            "target_name":   target.name,
            "attacks":       attack_results,
            "total_damage":  total_damage,
            "target_hp":     target.hp,
            "target_max_hp": target.max_hp,
            "target_alive":  target.is_alive(),
        }

    # ── EYE_RAYS ──────────────────────────────────────────────────────────────
    # multi_ray_table primitive (MONSTER_CATALOG §3 C 級, beholder): roll
    # `n_rays` DISTINCT entries off a data-driven effect table, each aimed at a
    # random eligible enemy. Each ray is an independent save vs
    # DC 8 + prof + dc_stat; entries deal typed damage (save halves or
    # negates) or apply a status (with optional two-stage escalation — the
    # petrification ray shares the basilisk-gaze semantics). The TABLE lives
    # on the ability definition; this handler stays creature-agnostic.
    if t == "EYE_RAYS":
        caster = _lookup_char(action.get("caster", ""), world_state)
        if not caster:
            return {"type": "ERROR", "message": "找不到施法者"}
        table = action.get("table") or []
        if not table:
            raise ValueError("EYE_RAYS action missing its effect table")
        caster_id = next(
            (cid for cid, c in world_state.characters.items() if c is caster), "")
        caster_side = world_state.is_party_ally(caster_id)
        rng_m = float(action.get("range_m", 36.0))
        battlefield = world_state.combat.battlefield if world_state.combat else None
        round_num = world_state.combat.round_number if world_state.combat else 0
        dc_stat = action.get("dc_stat") or "INT"
        save_dc = 8 + caster.proficiency_bonus + caster.stats.modifier(dc_stat)

        def _eligible() -> list[str]:
            out = []
            for cid, c in world_state.characters.items():
                # 瀕死者仍會被射線波及（同 AoE 慣例）；真死者排除。
                if cid == caster_id or c.is_dead():
                    continue
                if world_state.is_party_ally(cid) == caster_side:
                    continue
                if caster.position.distance_to(c.position) > rng_m + 1e-6:
                    continue
                if battlefield is not None and not battlefield.has_line_of_sight(
                        caster.position, c.position):
                    continue
                out.append(cid)
            return out

        if not _eligible():
            return {"type": "ERROR",
                    "message": f"{caster.name} 的射線範圍內沒有可見目標"}

        n_rays = int(action.get("n_rays", 3))
        # Beholder default (distinct_rays=True): roll per ray, reroll
        # duplicates — rays this turn are distinct (5e RAW). Kraken lightning
        # storm (distinct_rays=False): n identical bolts off a 1-entry table —
        # duplicates allowed, and a single-entry table consumes zero RNG.
        distinct = action.get("distinct_rays", True)
        picked: list[int] = []
        if len(table) == 1:
            picked = [0] * (1 if distinct else n_rays)
        elif distinct:
            while len(picked) < min(n_rays, len(table)):
                idx = roll(f"1d{len(table)}") - 1
                if idx not in picked:
                    picked.append(idx)
        else:
            picked = [roll(f"1d{len(table)}") - 1 for _ in range(n_rays)]

        ray_results = []
        for idx in picked:
            spec = table[idx]
            elig = _eligible()
            if not elig:
                break
            tid = elig[roll(f"1d{len(elig)}") - 1] if len(elig) > 1 else elig[0]
            target = world_state.characters[tid]
            save_stat = spec.get("save_stat", "DEX")
            success, save_roll = make_saving_throw(
                target, save_stat, save_dc, world_state=world_state,
                fail_applies_status=bool(spec.get("status")))
            entry = {
                "ray_name":     spec.get("name", f"ray_{idx}"),
                "target_name":  target.name,
                "save_stat":    save_stat,
                "save_dc":      save_dc,
                "save_roll":    save_roll,
                "save_success": success,
                "damage":       0,
                "status_applied": "",
            }
            if spec.get("damage_dice"):
                full = roll(spec["damage_dice"])
                dealt = full // 2 if (success and spec.get("save_half")) else (
                    0 if success else full)
                if dealt > 0:
                    dealt = apply_damage(target, dealt,
                                         dtype=spec.get("damage_type", "untyped"),
                                         attacker=caster, world_state=world_state)
                entry["damage"] = dealt
            elif spec.get("status") and not success:
                stage1 = spec["status"]
                terminal = spec.get("escalates_to")
                if terminal and target.has_status(stage1):
                    if apply_named_status(target, terminal, round_num=round_num,
                                          source_id=caster_id,
                                          rounds=None, save_each=""):
                        entry["status_applied"] = terminal
                else:
                    save_each = (f"{save_stat} DC{save_dc}"
                                 if spec.get("save_each") else "")
                    if apply_named_status(target, stage1, round_num=round_num,
                                          source_id=caster_id,
                                          rounds=spec.get("rounds"),
                                          save_each=save_each):
                        entry["status_applied"] = stage1
            entry["target_hp"] = target.hp
            entry["target_alive"] = target.is_alive()
            ray_results.append(entry)

        return {
            "type":         "EYE_RAYS",
            "caster_name":  caster.name,
            "save_dc":      save_dc,
            "rays":         ray_results,
            "total_damage": sum(r["damage"] for r in ray_results),
        }

    # ── SPELL_ATTACK ──────────────────────────────────────────────────────────
    # Single-target spell with attack roll vs AC (d20 + spell mod + prof bonus).
    # Used by spiritual_weapon (and future guiding_bolt). Distinct from SPELL
    # which routes through the save-based SPELLS registry.
    # Required action fields: caster, target, damage_dice (e.g. "1d8"),
    #   damage_type, range_m, spell_name.
    if t == "SPELL_ATTACK":
        caster = _lookup_char(action.get("caster", ""), world_state)
        target = _lookup_char(action.get("target", ""), world_state)
        if not caster:
            return {"type": "ERROR", "message": "找不到施法者"}
        if not target:
            return {"type": "ERROR", "message": "找不到目標"}
        if target.is_dead():
            return {"type": "ERROR", "message": f"{target.name} 已死亡"}
        if not caster.spellcasting_ability:
            return {"type": "ERROR", "message": f"{caster.name} 不是施法者"}

        range_m = float(action.get("range_m", 18.0))
        d = caster.position.distance_to(target.position)
        if d > range_m + 1e-6:
            return {"type": "ERROR",
                    "message": f"目標距離 {d:.1f}m，超出施法距離 {range_m:.0f}m"}
        battlefield = world_state.combat.battlefield if world_state.combat else None
        if battlefield is not None and not battlefield.has_line_of_sight(
                caster.position, target.position):
            return {"type": "ERROR",
                    "message": f"視線被遮擋，無法以法術攻擊 {target.name}"}

        # Optional spell-slot cost (guiding bolt = L1). Cantrip-tier spell
        # attacks (spiritual weapon) declare no slot_level and skip this.
        # The slot is spent on cast (5e RAW), hit or miss.
        slot_level = int(action.get("slot_level", 0))
        if slot_level > 0:
            if caster.spell_slots.get(slot_level, 0) <= 0:
                return {"type": "ERROR",
                        "message": f"{caster.name} 沒有 {slot_level} 環法術位"}
            err = _check_leveled_spell_limit(caster, slot_level)
            if err is not None:
                return err
            caster.spell_slots[slot_level] -= 1
            caster.leveled_spell_cast_this_turn = True

        damage_dice = action.get("damage_dice", "1d8")
        damage_type = action.get("damage_type", "force")
        spell_name = action.get("spell_name", "spell_attack")
        # Roll + damage live in the shared helper (also used by scorching ray).
        atk = _resolve_spell_attack_roll(
            caster, target, damage_dice, damage_type, world_state,
            add_spell_mod=action.get("add_spell_mod", True))
        hit = atk["hit"]
        damage = atk["damage"]

        # Optional on-hit status rider (guiding bolt → "distracted": the next
        # attacker against the target has advantage). Generic — any spell
        # attack can declare on_hit_status; the name is resolved through the
        # shared status registry, so it must already be an obs-visible /
        # engine-only status (no new MODIFIER_CLASSES name = no obs shift).
        status_applied = ""
        on_hit = action.get("on_hit_status")
        if hit and on_hit and target.is_alive():
            from .status import ALL_STATUS_CLASSES, StatusEffect
            round_num = world_state.combat.round_number if world_state.combat else 0
            caster_id = next(
                (cid for cid, c in world_state.characters.items() if c is caster), "")
            cls = ALL_STATUS_CLASSES.get(on_hit)
            if cls is not None:
                try:
                    fx = cls(applied_round=round_num, source_id=caster_id)
                except TypeError:
                    fx = cls(applied_round=round_num)
            else:
                fx = StatusEffect(name=on_hit, expires_on="round_end",
                                  rounds_remaining=1, applied_round=round_num,
                                  source_id=caster_id)
            target.add_status(fx)
            status_applied = on_hit

        return {
            "type":          "SPELL_ATTACK",
            "caster_name":   caster.name,
            "target_name":   target.name,
            "spell_name":    spell_name,
            "d20":           atk["d20"],
            "attack_bonus":  atk["attack_bonus"],
            "roll_total":    atk["roll_total"],
            "target_ac":     atk["target_ac"],
            "hit":           hit,
            "crit":          atk["crit"],
            "advantage_mode": atk["advantage_mode"],
            "damage":        damage,
            "damage_dice":   damage_dice,
            "damage_type":   damage_type,
            "slot_level":    slot_level,
            "status_applied": status_applied,
            "target_hp":     target.hp,
            "target_max_hp": target.max_hp,
            "target_alive":  target.is_alive(),
        }

    # ── MULTI_SPELL_ATTACK ──────────────────────────────────────────────────────
    # N independent spell attack rolls, each vs its own target's AC (Scorching
    # Ray = 3 rays × 2d6 fire). Reuses _resolve_spell_attack_roll per ray so the
    # attack maths match single-target SPELL_ATTACK exactly. One slot pays for
    # the whole casting; rays may pile on one target or split across several.
    if t == "MULTI_SPELL_ATTACK":
        caster = _lookup_char(action.get("caster", ""), world_state)
        if not caster:
            return {"type": "ERROR", "message": "找不到施法者"}
        if not caster.spellcasting_ability:
            return {"type": "ERROR", "message": f"{caster.name} 不是施法者"}
        ray_targets = action.get("ray_targets") or []
        if not ray_targets:
            return {"type": "ERROR", "message": "沒有指定射線目標"}

        slot_level = int(action.get("slot_level", 0))
        if slot_level > 0:
            if caster.spell_slots.get(slot_level, 0) <= 0:
                return {"type": "ERROR",
                        "message": f"{caster.name} 沒有 {slot_level} 環法術位"}
            err = _check_leveled_spell_limit(caster, slot_level)
            if err is not None:
                return err

        range_m = float(action.get("range_m", 27.0))
        battlefield = world_state.combat.battlefield if world_state.combat else None
        damage_dice = action.get("damage_dice", "2d6")
        damage_type = action.get("damage_type", "火")
        add_mod = bool(action.get("add_spell_mod", False))

        def _reachable(tid: str) -> bool:
            tc = _lookup_char(tid, world_state)
            if not tc or tc.is_dead():
                return False
            if caster.position.distance_to(tc.position) > range_m + 1e-6:
                return False
            if battlefield is not None and not battlefield.has_line_of_sight(
                    caster.position, tc.position):
                return False
            return True

        # Reject the whole casting if not a single ray can land — no slot is
        # spent on a cast that does nothing.
        if not any(_reachable(tid) for tid in ray_targets):
            return {"type": "ERROR",
                    "message": f"{caster.name} 的射線範圍內沒有可見目標"}

        if slot_level > 0:
            caster.spell_slots[slot_level] -= 1
            caster.leveled_spell_cast_this_turn = True

        ray_results = []
        for tid in ray_targets:
            target = _lookup_char(tid, world_state)
            # Re-check reachability per ray: an earlier ray may have dropped
            # the target, and you can't ray a corpse (mirrors EYE_RAYS).
            if not _reachable(tid):
                ray_results.append({
                    "target_name": target.name if target else tid,
                    "hit": False, "damage": 0, "fizzled": True})
                continue
            r = _resolve_spell_attack_roll(caster, target, damage_dice,
                                           damage_type, world_state,
                                           add_spell_mod=add_mod)
            r["target_name"]  = target.name
            r["target_hp"]    = target.hp
            r["target_alive"] = target.is_alive()
            ray_results.append(r)

        return {
            "type":         "MULTI_SPELL_ATTACK",
            "caster_name":  caster.name,
            "spell_name":   action.get("spell_name", "scorching_ray"),
            "damage_dice":  damage_dice,
            "damage_type":  damage_type,
            "slot_level":   slot_level,
            "rays":         ray_results,
            "total_damage": sum(r.get("damage", 0) for r in ray_results),
        }

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
            if not target or target.is_dead():
                continue
            success, save_roll = make_saving_throw(target, save_stat, save_dc)
            full_dmg   = roll(damage_dice)
            actual_dmg = full_dmg // 2 if (success and half_on_save) else full_dmg
            # 走 apply_damage：瀕死目標的 AoE 傷害也要計死亡豁免失敗。
            actual_dmg = apply_damage(target, actual_dmg, dtype="untyped",
                                      attacker=attacker, world_state=world_state)
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

    # ── HEAL ──────────────────────────────────────────────────────────────────
    # Dedicated healing action for spells / abilities (Cure Wounds, Second
    # Wind, Healing Word, etc.). Distinct from USE_ITEM which is for
    # consumables. Optionally consumes a spell slot when `slot_level` > 0.
    if t == "HEAL":
        caster = _lookup_char(action.get("caster", ""), world_state)
        target = _lookup_char(action.get("target", ""), world_state)
        if not caster or not target:
            return {"type": "ERROR", "message": "找不到治療者或目標"}
        # 5e：瀕死（0 HP 死亡豁免中）可被治療救起——apply_heal 會重置死亡
        # 豁免並從 0 回血。只有真正死亡（3 次失敗 / NPC 0 HP）無法治療。
        if target.is_dead():
            return {"type": "ERROR", "message": f"{target.name} 已死亡，無法治療"}

        slot_level = int(action.get("slot_level", 0))
        if slot_level > 0 and caster.spell_slots.get(slot_level, 0) <= 0:
            return {"type": "ERROR",
                    "message": f"{caster.name} 沒有 {slot_level} 環法術位"}
        err = _check_leveled_spell_limit(caster, slot_level)
        if err is not None:
            return err

        range_m = float(action.get("range_m", 1.5))
        d = caster.position.distance_to(target.position)
        if d > range_m + 1e-6:
            return {"type": "ERROR",
                    "message": f"目標距離 {d:.1f}m，超出治療範圍 {range_m:.1f}m"}

        dice = action.get("dice", "1d4")
        was_dying = target.is_dying()
        healed = apply_heal(target, dice)
        if slot_level > 0:
            caster.spell_slots[slot_level] -= 1
            caster.leveled_spell_cast_this_turn = True

        return {
            "type":          "HEAL",
            "caster_name":   caster.name,
            "target_name":   target.name,
            "amount":        healed,
            "dice":          dice,
            "slot_level":    slot_level,
            "target_hp":     target.hp,
            "target_max_hp": target.max_hp,
            "revived":       was_dying,   # 從瀕死被救起
        }

    # ── MULTI_HEAL ──────────────────────────────────────────────────────────────
    # Heal several allies at once. Two flavours via the action fields:
    #   dice="3d8+4"          per-target rolled heal (Mass Cure Wounds), one
    #                         spell slot for the whole casting.
    #   pool=N, cap_half=True a shared HP pool distributed to the most-wounded
    #                         allies, capped at half max HP each (Channel
    #                         Divinity: Preserve Life). No slot — its use is
    #                         deducted as an ability_use by execute_action.
    # Targets out of range / dead are skipped; range-gated like single HEAL.
    if t == "MULTI_HEAL":
        caster = _lookup_char(action.get("caster", ""), world_state)
        if not caster:
            return {"type": "ERROR", "message": "找不到治療者"}
        target_ids = action.get("targets") or []
        range_m = float(action.get("range_m", 9.0))

        slot_level = int(action.get("slot_level", 0))
        if slot_level > 0:
            if caster.spell_slots.get(slot_level, 0) <= 0:
                return {"type": "ERROR",
                        "message": f"{caster.name} 沒有 {slot_level} 環法術位"}
            err = _check_leveled_spell_limit(caster, slot_level)
            if err is not None:
                return err

        def _heal_reachable(tid: str):
            tc = _lookup_char(tid, world_state)
            if not tc or tc.is_dead():
                return None
            if caster.position.distance_to(tc.position) > range_m + 1e-6:
                return None
            return tc

        elig = [tc for tc in (_heal_reachable(tid) for tid in target_ids) if tc]
        if not elig:
            return {"type": "ERROR",
                    "message": f"{caster.name} 的治療範圍內沒有有效目標"}

        if slot_level > 0:
            caster.spell_slots[slot_level] -= 1
            caster.leveled_spell_cast_this_turn = True

        heal_results = []
        dice = action.get("dice")
        pool = int(action.get("pool", 0))
        cap_half = bool(action.get("cap_half", False))

        if dice:
            for tc in elig:
                was_dying = tc.is_dying()
                healed = apply_heal(tc, dice)
                heal_results.append({
                    "target_name": tc.name, "amount": healed,
                    "target_hp": tc.hp, "target_max_hp": tc.max_hp,
                    "revived": was_dying})
        else:
            # Pool distribution: most-wounded first, cap at half max HP each.
            remaining = pool
            order = sorted(elig, key=lambda c: c.max_hp - c.hp, reverse=True)
            for tc in order:
                if remaining <= 0:
                    break
                cap = tc.max_hp // 2 if cap_half else tc.max_hp
                room = max(0, cap - tc.hp)
                give = min(remaining, room)
                if give <= 0:
                    continue
                was_dying = tc.is_dying()
                healed = heal_flat(tc, give, hp_cap=cap)
                remaining -= healed
                heal_results.append({
                    "target_name": tc.name, "amount": healed,
                    "target_hp": tc.hp, "target_max_hp": tc.max_hp,
                    "revived": was_dying})

        return {
            "type":         "MULTI_HEAL",
            "caster_name":  caster.name,
            "spell_name":   action.get("spell_name", "mass_heal"),
            "slot_level":   slot_level,
            "targets":      heal_results,
            "total_healed": sum(r["amount"] for r in heal_results),
        }

    # ── SUMMON ────────────────────────────────────────────────────────────────
    # Add creatures to the fight mid-combat (conjurer minions, splitting
    # monsters). The summons take the SUMMONER's side — side is party_ids
    # membership, so obs.partition_entities picks them up live and truncates
    # past the slot budget exactly like any large monster pack (no new obs
    # dimension). They're inserted into the initiative order right after the
    # summoner so drivers iterating the live order step them. Conjured creatures
    # are is_npc=True (die at 0, no death saves) regardless of which side.
    if t == "SUMMON":
        summoner = _lookup_char(action.get("summoner", ""), world_state)
        if not summoner:
            return {"type": "ERROR", "message": "找不到召喚者"}
        monster_id = action.get("monster_id", "")
        from ..scenarios.monsters import make_monster, MONSTER_DEFS
        if monster_id not in MONSTER_DEFS:
            return {"type": "ERROR", "message": f"未知的召喚生物：{monster_id}"}
        count = max(1, int(action.get("count", 1)))
        summoner_id = next(
            (cid for cid, c in world_state.characters.items() if c is summoner), "")
        summoner_is_party = world_state.is_party_ally(summoner_id)
        bf = world_state.combat.battlefield if world_state.combat else None
        base = action.get("spawn_id_prefix") or f"summon_{monster_id}"

        created: list[str] = []
        for k in range(count):
            n = 0
            nid = f"{base}_{n}"
            while nid in world_state.characters:
                n += 1
                nid = f"{base}_{n}"
            creature = make_monster(monster_id)
            creature.is_npc = True
            # Fan the summons out beside the summoner, clamped to the field.
            offset = Vec2(1.5, (k - (count - 1) / 2.0) * 1.5)
            pos = summoner.position + offset
            if bf is not None:
                margin = 0.5
                pos = Vec2(max(margin, min(bf.width - margin, pos.x)),
                           max(margin, min(bf.height - margin, pos.y)))
            creature.position = pos
            world_state.characters[nid] = creature
            if summoner_is_party and nid not in world_state.party_ids:
                world_state.party_ids.append(nid)
            created.append(nid)

        # Splice the summons into initiative as a contiguous block right after
        # the summoner (preserving spawn order), so drivers iterating the live
        # order step them this round.
        cs = world_state.combat
        if cs is not None and created:
            fresh = [nid for nid in created if nid not in cs.initiative_order]
            at = (cs.initiative_order.index(summoner_id) + 1
                  if summoner_id in cs.initiative_order
                  else len(cs.initiative_order))
            cs.initiative_order[at:at] = fresh

        return {
            "type":          "SUMMON",
            "summoner_name": summoner.name,
            "monster_id":    monster_id,
            "summoned_ids":  created,
            "summoned_name": world_state.characters[created[0]].name if created else "",
            "count":         len(created),
        }

    # ── ACTION_SURGE ──────────────────────────────────────────────────────────
    # Grants the actor an extra action this turn. consume_resources reads the
    # `grants` dict on the result and adds back into the resource budget.
    if t == "ACTION_SURGE":
        char = _lookup_char(action.get("character", ""), world_state)
        if not char:
            return {"type": "ERROR", "message": "找不到角色"}
        return {
            "type":      "ACTION_SURGE",
            "character": char.name,
            "grants":    {"action": 1},
        }

    # ── AUTO_DAMAGE ───────────────────────────────────────────────────────────
    # Auto-hit multi-target damage (Magic Missile and friends). targets is a
    # list of {"id": <cid>, "darts": int} entries; each dart rolls
    # `damage_per` and applies to that target. Range-gated + LoS-gated like
    # SPELL. Consumes a slot if slot_level > 0.
    if t == "AUTO_DAMAGE":
        attacker = _lookup_char(action.get("attacker", ""), world_state)
        if not attacker:
            return {"type": "ERROR", "message": "找不到攻擊者"}
        slot_level = int(action.get("slot_level", 0))
        if slot_level > 0 and attacker.spell_slots.get(slot_level, 0) <= 0:
            return {"type": "ERROR",
                    "message": f"{attacker.name} 沒有 {slot_level} 環法術位"}
        err = _check_leveled_spell_limit(attacker, slot_level)
        if err is not None:
            return err

        range_m = float(action.get("range_m", 0.0))
        damage_per = action.get("damage_per", "1d4")
        dtype = action.get("damage_type", "untyped")
        battlefield = world_state.combat.battlefield if world_state.combat else None
        targets_spec = action.get("targets", [])

        if not targets_spec:
            raise RuntimeError(
                f"AUTO_DAMAGE built with empty `targets` list — "
                f"caller would have consumed slot {slot_level} for no effect "
                f"(attacker={attacker.name})"
            )

        target_results = []
        for spec in targets_spec:
            tid = spec.get("id")
            hits = max(1, int(spec.get("darts", 1)))
            target = _lookup_char(tid, world_state)
            if not target or target.is_dead():
                continue
            d = attacker.position.distance_to(target.position)
            if range_m > 0 and d > range_m:
                return {"type": "ERROR",
                        "message": f"{target.name} 距離 {d:.1f}m 超出射程 {range_m:.0f}m"}
            if battlefield is not None and not battlefield.has_line_of_sight(
                    attacker.position, target.position):
                return {"type": "ERROR",
                        "message": f"視線被遮擋，無法擊中 {target.name}"}
            # Shield blocks Magic Missile in full (5e rule)
            blocked, reaction_name, reaction_slot = _try_react_to_auto_damage(
                target, world_state
            )
            if blocked:
                target_results.append({
                    "target_name":   target.name,
                    "darts":         hits,
                    "damage":        0,
                    "reaction":      reaction_name,
                    "reaction_slot": reaction_slot,
                    "target_hp":     target.hp,
                    "target_max_hp": target.max_hp,
                    "target_alive":  target.is_alive(),
                })
                continue
            total_dmg = 0
            for _ in range(hits):
                rolled = roll(damage_per)
                total_dmg += apply_damage(target, rolled, dtype=dtype,
                                          attacker=attacker, world_state=world_state)
            target_results.append({
                "target_name":   target.name,
                "darts":         hits,
                "damage":        total_dmg,
                "target_hp":     target.hp,
                "target_max_hp": target.max_hp,
                "target_alive":  target.is_alive(),
            })

        if slot_level > 0:
            attacker.spell_slots[slot_level] -= 1
            attacker.leveled_spell_cast_this_turn = True

        return {
            "type":           "AUTO_DAMAGE",
            "attacker_name":  attacker.name,
            "damage_per":     damage_per,
            "damage_type":    dtype,
            "slot_level":     slot_level,
            "target_results": target_results,
        }

    # ── APPLY_MOD ─────────────────────────────────────────────────────────────
    # Apply a named Modifier (from status.MODIFIER_CLASSES) to one or more
    # targets. Generic path for Bless, Rage, Hunter's Mark, etc.
    if t == "APPLY_MOD":
        from .status import MODIFIER_CLASSES
        caster = _lookup_char(action.get("caster", ""), world_state)
        if not caster:
            return {"type": "ERROR", "message": "找不到施法者"}
        mod_name = action.get("modifier", "")
        mod_cls = MODIFIER_CLASSES.get(mod_name)
        if not mod_cls:
            return {"type": "ERROR", "message": f"未知 modifier：{mod_name}"}

        slot_level = int(action.get("slot_level", 0))
        if slot_level > 0 and caster.spell_slots.get(slot_level, 0) <= 0:
            return {"type": "ERROR",
                    "message": f"{caster.name} 沒有 {slot_level} 環法術位"}
        err = _check_leveled_spell_limit(caster, slot_level)
        if err is not None:
            return err

        target_ids = action.get("targets", [])
        max_targets = int(action.get("max_targets", len(target_ids) or 1))
        target_ids = target_ids[:max_targets]
        range_m = float(action.get("range_m", 0.0))

        if not target_ids:
            raise RuntimeError(
                f"APPLY_MOD built with empty `targets` list — caller would "
                f"have consumed slot {slot_level} + concentration for no effect "
                f"(caster={caster.name}, modifier={mod_name})"
            )

        targets = []
        for tid in target_ids:
            target = _lookup_char(tid, world_state)
            if not target or not target.is_alive():
                continue
            if range_m > 0:
                d = caster.position.distance_to(target.position)
                if d > range_m + 1e-6:
                    return {"type": "ERROR",
                            "message": f"{target.name} 距離 {d:.1f}m 超出 {range_m:.0f}m"}
            targets.append(target)

        round_num = world_state.combat.round_number if world_state.combat else 0
        try:
            sample = mod_cls(applied_round=round_num,
                             source_id=action.get("caster", ""))
        except TypeError:
            sample = mod_cls(applied_round=round_num)
        target_names = []
        caster_id = action.get("caster", "")
        mod_metadata = action.get("mod_metadata")
        for target in targets:
            try:
                fx = sample.__class__(applied_round=round_num, source_id=caster_id)
            except TypeError:
                fx = sample.__class__(applied_round=round_num)
            # Optional per-instance metadata (Bear Totem: tag this rage as
            # all-type resistant). The modifier class reads its own metadata —
            # no new status name, so obs dims are untouched.
            if mod_metadata:
                fx.metadata.update(mod_metadata)
            target.add_status(fx)
            target_names.append(target.name)

        if slot_level > 0:
            caster.spell_slots[slot_level] -= 1
            caster.leveled_spell_cast_this_turn = True
        spell_name = action.get("spell_name", mod_name)
        if action.get("requires_concentration"):
            if caster.concentrating_on:
                caster_id = next(
                    (cid for cid, c in world_state.characters.items() if c is caster), None
                )
                if caster_id:
                    _end_concentration(caster_id, world_state)
            caster.concentrating_on = spell_name

        return {
            "type":             "APPLY_MOD",
            "caster_name":      caster.name,
            "spell_name":       spell_name,
            "modifier":         mod_name,
            "targets_affected": target_names,
            "slot_level":       slot_level,
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

        # Teleport branch (e.g. Misty Step) — short-circuit before the
        # ground-movement logic. Teleport ignores LoS, terrain multipliers
        # and the standard movement budget; it has its own range, charged to
        # whatever resource the caller declared (typically bonus_action +
        # spell slot) and budgets are zero for the move slot itself.
        if action.get("teleport"):
            new_pos = Vec2.coerce(action.get("target_position", [old_pos.x, old_pos.y]))
            tp_range = float(action.get("range_m", 9.0))
            dist = old_pos.distance_to(new_pos)
            if dist > tp_range + 1e-6:
                return {"type": "ERROR",
                        "message": f"瞬移目標距離 {dist:.1f}m，超出射程 {tp_range:.0f}m"}
            if battlefield is not None:
                if not battlefield.in_bounds(new_pos):
                    return {"type": "ERROR",
                            "message": f"目的座標 ({new_pos.x:.1f}, {new_pos.y:.1f}) 超出戰場邊界"}
                if battlefield.is_blocked(new_pos):
                    return {"type": "ERROR",
                            "message": f"目的座標 ({new_pos.x:.1f}, {new_pos.y:.1f}) 被障礙物佔據"}
            # Consume spell slot if the teleport requires one (e.g. Misty Step = level 2).
            slot_level = int(action.get("slot_level", 0))
            if slot_level > 0:
                if char.spell_slots.get(slot_level, 0) <= 0:
                    return {"type": "ERROR",
                            "message": f"{char.name} 沒有 {slot_level} 環法術位"}
                err = _check_leveled_spell_limit(char, slot_level)
                if err is not None:
                    return err
                char.spell_slots[slot_level] -= 1
                char.leveled_spell_cast_this_turn = True
            char.position = new_pos
            return {
                "type":              "MOVE",
                "character":         char.name,
                "from_pos":          old_pos,
                "to_pos":            new_pos,
                "distance":          0.0,
                "physical_distance": dist,
                "terrain_mult":      1.0,
                "teleport":          True,
                "slot_level":        slot_level,
                "description":       action.get("description", "瞬移"),
            }

        # Some conditions (restrained, stunned) reduce movement to 0.
        speed_mult = 1.0
        for m in char.iter_modifiers():
            speed_mult *= m.on_speed_multiplier(char)
        if speed_mult <= 0.0:
            return {"type": "ERROR", "message": f"{char.name} 目前無法移動"}
        effective_budget = MOVE_BUDGET_M * speed_mult

        # Resolve destination from one of four formats, in priority order:
        #   1. target (creature_id) — move toward that character along the
        #      shortest line, capped at their exact position so you never
        #      overshoot ("我朝薩滿衝鋒")
        #   2. direction ("advance"/"retreat") — toward / away from enemy
        #      centroid, fixed distance
        #   3. target_position ([x, y]) — absolute coordinate
        #   4. delta ([dx, dy]) — raw 2D delta
        # Creature-target moves stop CLOSE_GAP_M short of the target so that
        # combatants don't pile onto the exact same cell. 1m leaves them
        # inside standard 1.5m melee reach without sharing coordinates —
        # which would otherwise inflate AOE / make positional rules ambiguous.
        CLOSE_GAP_M = 1.0
        new_pos: Vec2 | None = None
        target_key = action.get("target")
        if target_key:
            target_char = _lookup_char(target_key, world_state)
            if not target_char:
                return {"type": "ERROR", "message": f"找不到移動目標：{target_key}"}
            dest = target_char.position
            delta = dest - old_pos
            dist = delta.length()
            if dist <= CLOSE_GAP_M + 1e-6:
                # Already at/inside the gap — don't move at all.
                new_pos = old_pos
            else:
                travel = min(dist - CLOSE_GAP_M, effective_budget)
                new_pos = old_pos + delta.normalized() * travel
        elif "direction" in action:
            direction = str(action["direction"]).lower()
            distance = abs(float(action.get("distance", effective_budget)))
            distance = min(distance, effective_budget)
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

        # Path-trace: walk the ray from old_pos toward the intended new_pos,
        # stopping at walls, applying dangerous-terrain damage per cell
        # entered, and accumulating difficult-terrain cost along the way.
        # Pre-fix this was a teleport-on-ground that only sampled the
        # endpoint — so walking across lava without ending in lava dealt 0
        # damage, and my own LOS-removal for MOVE briefly let actors walk
        # through walls. Path-tracing closes both gaps.
        intended_pos = new_pos
        new_pos, phys_dist, effective_cost, path_events = _walk_path(
            old_pos, intended_pos, effective_budget, battlefield,
        )
        mult = (effective_cost / phys_dist) if phys_dist > 1e-9 else 1.0
        # Apply accumulated dangerous-terrain damage. Done after the walk
        # so dropping below 0 HP from terrain doesn't desync the path.
        terrain_damage = 0
        for ev in path_events:
            if ev.get("kind") == "entered_dangerous":
                rolled = roll(ev.get("dice", DANGEROUS_TERRAIN_DAMAGE))
                terrain_damage += apply_damage(
                    char, rolled, dtype="environment",
                    world_state=world_state,
                )

        # ── Opportunity attacks ───────────────────────────────────────────────
        # If a creature moves out of an enemy's melee reach (1.5m), that enemy
        # gets a free attack as a reaction (if reaction not already used).
        # Triggered only when the mover started in reach and ends out of reach;
        # the Disengage action (not yet implemented) would suppress this.
        oa_results: list[dict] = []
        disengaging = char.has_status("disengaging")
        if phys_dist > 1e-6 and not action.get("teleport") and not disengaging and world_state.combat:
            is_party = (char_id is not None and world_state.is_party_ally(char_id))
            for oid, other in world_state.characters.items():
                if oid == char_id or not other.is_alive():
                    continue
                other_in_party = world_state.is_party_ally(oid)
                if other_in_party == is_party:
                    continue   # same side
                if other.reaction_used:
                    continue
                oweapon = other.get_weapon() if other.weapons else None
                if oweapon is None:
                    continue
                reach = oweapon.range_normal or 1.5
                was_in_reach = old_pos.distance_to(other.position) <= reach + 1e-6
                now_out = new_pos.distance_to(other.position) > reach + 1e-6
                if was_in_reach and now_out:
                    other.reaction_used = True
                    oa_hit, oa_roll = resolve_attack(other, char, oweapon)
                    oa_dmg = 0
                    if oa_hit:
                        oa_base = roll(oweapon.damage_dice)
                        if "精巧" in oweapon.properties:
                            oa_mod = max(other.stats.modifier("STR"),
                                         other.stats.modifier("DEX"))
                        elif oweapon.range_type == "遠程":
                            oa_mod = other.stats.modifier("DEX")
                        else:
                            oa_mod = other.stats.modifier("STR")
                        oa_raw = max(1, oa_base + oa_mod)
                        oa_dmg = apply_damage(char, oa_raw,
                                              dtype=oweapon.damage_type,
                                              attacker=other,
                                              world_state=world_state)
                    oa_results.append({
                        "attacker": other.name, "target": char.name,
                        "weapon":   oweapon.name,
                        "roll":     oa_roll, "hit": oa_hit, "damage": oa_dmg,
                        "target_hp": char.hp, "target_max_hp": char.max_hp,
                    })

        char.position = new_pos
        result = {
            "type":              "MOVE",
            "character":         char.name,
            "from_pos":          old_pos,
            "to_pos":            new_pos,
            "distance":          effective_cost,
            "physical_distance": phys_dist,
            "terrain_mult":      mult,
            "description":       action.get("description", "移動"),
        }
        if terrain_damage:
            result["terrain_damage"] = terrain_damage
            result["target_hp"] = char.hp
            result["target_max_hp"] = char.max_hp
        if path_events:
            # Skip the noisy "entered_dangerous" entries (the rolled damage
            # is already summarised above) — surface only the structural
            # events that explain why the actor didn't reach the requested
            # destination.
            interesting = [e for e in path_events
                           if e.get("kind") in ("hit_wall", "out_of_bounds",
                                                 "budget_exhausted")]
            if interesting:
                result["path_events"] = interesting
        if oa_results:
            result["opportunity_attacks"] = oa_results
        return result

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

    # ── DISENGAGE ─────────────────────────────────────────────────────────────
    # Spend an action to safely leave melee. Until start of your next turn,
    # your movement won't provoke opportunity attacks.
    if t == "DISENGAGE":
        from .status import StatusEffect
        char = _lookup_char(action.get("character", ""), world_state)
        if not char:
            return {"type": "ERROR", "message": "找不到角色"}
        round_num = world_state.combat.round_number if world_state.combat else 0
        char.add_status(StatusEffect(
            name="disengaging", expires_on="self_turn_start",
            applied_round=round_num, kind="buff",
        ))
        return {"type": "DISENGAGE", "character": char.name}

    # ── LAY_ON_HANDS ──────────────────────────────────────────────────────────
    if t == "LAY_ON_HANDS":
        caster = _lookup_char(action.get("caster", ""), world_state)
        target = _lookup_char(action.get("target", ""), world_state)
        if not caster:
            return {"type": "ERROR", "message": "找不到施術者"}
        if not target:
            return {"type": "ERROR", "message": "找不到目標"}
        if target.is_dead():
            return {"type": "ERROR", "message": f"{target.name} 已死亡，無法治療"}
        amount = int(action.get("amount", 0))
        if amount <= 0:
            return {"type": "ERROR", "message": "治療量必須大於 0"}
        if caster.lay_on_hands_pool < amount:
            return {"type": "ERROR",
                    "message": f"{caster.name} 聖療之手資源不足（剩餘 {caster.lay_on_hands_pool} HP）"}
        d = caster.position.distance_to(target.position)
        if d > 1.5 + 1e-6:
            return {"type": "ERROR",
                    "message": f"聖療之手需要近身接觸（目前距離 {d:.1f}m）"}
        caster.lay_on_hands_pool -= amount
        was_dying = target.is_dying()
        if was_dying:
            target.reset_death_saves()
        target.hp = min(target.max_hp, target.hp + amount)
        return {
            "type":          "LAY_ON_HANDS",
            "caster":        caster.name,
            "target":        target.name,
            "healed":        amount,
            "pool_left":     caster.lay_on_hands_pool,
            "target_hp":     target.hp,
            "target_max_hp": target.max_hp,
            "revived":       was_dying,
        }

    # ── PORTENT ───────────────────────────────────────────────────────────────
    if t == "PORTENT":
        caster  = _lookup_char(action.get("caster",  ""), world_state)
        target  = _lookup_char(action.get("target",  ""), world_state)
        die_val = int(action.get("die_value", 0))
        if not caster:
            return {"type": "ERROR", "message": "找不到施法者"}
        if not target:
            return {"type": "ERROR", "message": "找不到目標"}
        if die_val not in caster.portent_dice:
            return {"type": "ERROR",
                    "message": f"預言骰 {die_val} 不在你的儲存列表 {caster.portent_dice} 中"}
        caster.portent_dice.remove(die_val)
        target.pending_portent = die_val
        return {
            "type":      "PORTENT",
            "caster":    caster.name,
            "target":    target.name,
            "die_value": die_val,
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
        # Natural abilities riding the SPELL pipeline (dragon breath) carry
        # their own DC stat (save_dc_ability) — no spellcasting required.
        if not caster.spellcasting_ability and not spell.save_dc_ability:
            return {"type": "ERROR", "message": f"{caster.name} 不是施法者"}

        # Resolve slot_level — cantrips (level 0) consume no slot, otherwise
        # explicit arg or first available level ≥ spell.level.
        if spell.level == 0:
            slot_level = 0
        else:
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

        # 5e: one leveled spell per turn (cantrips exempt). Reject early so
        # we don't consume a slot or roll any dice.
        err = _check_leveled_spell_limit(caster, slot_level)
        if err is not None:
            return err

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

        # Counterspell reaction: enemies with counterspell + slot can cancel.
        # Cantrips (spell.level == 0) cannot be counterspelled.
        if spell.level > 0:
            countered, cspeller, cslot = _try_counterspell(
                caster, int(slot_level), world_state
            )
            if countered:
                return {
                    "type":           "COUNTERSPELLED",
                    "caster_name":    caster.name,
                    "spell_name":     spell_name,
                    "counterspeller": cspeller,
                    "slot_used":      cslot,
                }

        # Save DC = 8 + prof_bonus + casting-stat modifier. The casting stat
        # is the spell's save_dc_ability override when set (breath weapons:
        # CON), else the caster's spellcasting ability.
        _dc_stat = spell.save_dc_ability or caster.spellcasting_ability
        save_dc = 8 + caster.proficiency_bonus + caster.stats.modifier(_dc_stat)

        # Resolve affected creatures.
        #   AOE spell (aoe_radius_m > 0): every alive creature within radius
        #     of the centre, scoped to the current room.
        #   LINE spell (aoe_shape == "line"): every alive creature within
        #     line_width_m/2 of the caster→endpoint segment (endpoint =
        #     line_length_m along the aim direction), per-target LoS (the
        #     bolt is stopped by full cover). Caster is never self-hit.
        #   Single-target spell (aoe_radius_m == 0): only the named creature.
        #     Coordinate-only casts (target_position with aoe=0) have no
        #     meaningful target — we raise RuntimeError below so the broken
        #     builder is identified at the call site, not silently no-op'd.
        is_line = spell.aoe_shape == "line"
        affected_ids: list[str] = []
        if spell.aoe_radius_m > 0 or is_line:
            room = world_state.dungeon_map.current_room if world_state.dungeon_map else None
            if room is not None:
                scope_ids = set(room.npc_ids)
                for cid, c in world_state.characters.items():
                    if world_state.is_party_ally(cid):
                        scope_ids.add(cid)
            else:
                scope_ids = set(world_state.characters.keys())
            if is_line:
                from .vec2 import point_segment_distance
                aim = center_pos - caster.position
                if aim.length() < 1e-6:
                    return {"type": "ERROR",
                            "message": f"「{spell_name}」需要一個方向（瞄準點不可為施法者自身）"}
                line_end = caster.position + aim.normalized() * spell.line_length_m
                half_w = spell.line_width_m / 2.0
            for cid in scope_ids:
                c = world_state.characters.get(cid)
                # 瀕死（昏迷倒地）者仍會被 AoE 波及（5e：傷害照算，計死亡
                # 豁免失敗；STR/DEX 豁免自動失敗）。只排除真正死亡的。
                if not c or c.is_dead():
                    continue
                if is_line:
                    if c is caster:
                        continue
                    if point_segment_distance(
                            c.position, caster.position, line_end) > half_w + 1e-6:
                        continue
                    if battlefield is not None and not battlefield.has_line_of_sight(
                            caster.position, c.position):
                        continue
                    affected_ids.append(cid)
                elif c.position.distance_to(center_pos) <= spell.aoe_radius_m + 1e-6:
                    # Self-centred novas (dragon wing attack, lich disrupt
                    # life): "each creature within X of the caster" never
                    # includes the caster. Ordinary point AoEs keep the
                    # historical behaviour (a fireball at your feet burns you).
                    if spell.excludes_caster and c is caster:
                        continue
                    affected_ids.append(cid)
        else:
            # Single-target: resolve the explicit creature target.
            target_key = action.get("target", "")
            if not target_key:
                # Contract violation: a non-AOE spell needs a creature target.
                # The builder is expected to set `target`; if we see only
                # target_position here, the resources/concentration would be
                # spent on nobody. Fail loud instead of silently no-opping.
                raise RuntimeError(
                    f"Non-AOE spell {spell_name!r} cast without `target` — "
                    f"builder violated contract "
                    f"(target_position={action.get('target_position')!r})"
                )
            if target_key != "self":
                target_char = _lookup_char(target_key, world_state)
                tid = next((cid for cid, c in world_state.characters.items()
                            if c is target_char), None) if target_char else None
                if tid and not target_char.is_dead():
                    affected_ids.append(tid)
            else:
                # Self-targeted (rare for damage spells) — caster is the single target.
                caster_id = next((cid for cid, c in world_state.characters.items()
                                  if c is caster), None)
                if caster_id:
                    affected_ids.append(caster_id)

        # Sculpt Spells (Evocation Wizard L2): the CASTER's own allies (same
        # party side) are excluded from AOE. Original code keyed on
        # `is_party_ally(cid)` which always means "PC party" — so an enemy
        # evoker's sculpt was protecting the PCs instead of their own allies.
        # Compare each target's side against the CASTER's side.
        if (spell.aoe_radius_m > 0 or is_line) and caster.sculpt_spells:
            caster_id = next(
                (cid for cid, c in world_state.characters.items() if c is caster),
                None,
            )
            if caster_id is not None:
                caster_in_party = world_state.is_party_ally(caster_id)
                affected_ids = [
                    cid for cid in affected_ids
                    if world_state.is_party_ally(cid) != caster_in_party
                ]

        # Roll saves, apply damage + on-fail status
        target_results = []
        round_num = world_state.combat.round_number if world_state.combat else 0
        for cid in affected_ids:
            target = world_state.characters[cid]
            save_bd: dict = {}
            success, save_roll = make_saving_throw(
                target, spell.save_ability, save_dc, breakdown=save_bd,
                world_state=world_state,
                fail_applies_status=bool(spell.applies_status_on_fail),
            )
            if spell.damage_dice:
                full_dmg = (
                    _roll_scaled_cantrip(spell.damage_dice, caster.level)
                    if spell.level == 0 and spell.scales_as_cantrip
                    else roll(spell.damage_dice)
                )
            else:
                full_dmg = 0
            if success:
                actual_dmg = 0 if spell.save_for_no_damage else full_dmg // 2
            else:
                actual_dmg = full_dmg
            # Evasion and similar modifiers can override save damage result.
            for _m in target.iter_modifiers():
                actual_dmg = _m.on_incoming_save_damage(
                    target, spell.save_ability, success, actual_dmg
                )
            if actual_dmg > 0:
                apply_damage(target, actual_dmg, dtype=spell.damage_type,
                             attacker=caster, world_state=world_state)
            status_applied = ""
            if not success and spell.applies_status_on_fail:
                stage1 = spell.applies_status_on_fail
                caster_id = action.get("caster", "")
                if spell.escalates_to and target.has_status(stage1):
                    # Two-stage escalation (basilisk gaze, beholder petrify
                    # ray): a failed save while stage-1 is already on the
                    # target applies the TERMINAL status — permanent, no
                    # retry save (combat-end cleanup is the only way out).
                    if apply_named_status(target, spell.escalates_to,
                                          round_num=round_num,
                                          source_id=caster_id,
                                          rounds=None, save_each=""):
                        status_applied = spell.escalates_to
                else:
                    save_each = (f"{spell.save_ability} DC{save_dc}"
                                 if spell.save_each_on_status else "")
                    if apply_named_status(target, stage1, round_num=round_num,
                                          source_id=caster_id,
                                          rounds=spell.status_rounds,
                                          save_each=save_each):
                        status_applied = stage1
            target_results.append({
                "target_name":    target.name,
                "save_roll":      save_roll,
                "save_breakdown": save_bd,
                "save_success":   success,
                "damage":         actual_dmg,
                "status_applied": status_applied,
                "target_hp":      target.hp,
                "target_max_hp":  target.max_hp,
                "target_alive":   target.is_alive(),
            })

        # Decrement slot after successful cast (cantrips skip this)
        if slot_level > 0:
            caster.spell_slots[slot_level] = caster.spell_slots.get(slot_level, 0) - 1
            caster.leveled_spell_cast_this_turn = True

        # Concentration: replace any prior concentration spell with this one,
        # cleaning up any effects from the old spell first.
        if spell.requires_concentration:
            if caster.concentrating_on:
                caster_id = next(
                    (cid for cid, c in world_state.characters.items() if c is caster), None
                )
                if caster_id:
                    _end_concentration(caster_id, world_state)
            caster.concentrating_on = spell_name

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
    """Decrement per-turn resource budget based on action.consumes declaration,
    then apply any `grants` produced by the action's result.

    Resource slots understood:
      'action'        — sets to 0 (one action per turn)
      'bonus_action'  — sets to 0
      'movement'      — subtracts result.get('distance', 0)
    Unknown slots are silently ignored (reserved for future expansion).

    The `grants` dict on the result lets actions REFUND or ADD resources —
    e.g. Action Surge returns {"action": 1} so the actor gets a second action.
    """
    for slot in action.get("consumes", []):
        if slot == "action":
            resources["action"] = 0
        elif slot == "bonus_action":
            resources["bonus_action"] = 0
        elif slot == "movement":
            resources["movement"] = max(0.0, resources.get("movement", 0.0) - result.get("distance", 0))
    for slot, amount in (result.get("grants") or {}).items():
        resources[slot] = resources.get(slot, 0) + amount


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
    if t == "MULTI_ATTACK":
        return f"多重攻擊 {action.get('target', '?')}（{action.get('weapon', '武器')}）"
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
    if t == "SPELL_ATTACK":
        name = action.get("spell_name", "?")
        return f"施展 {name} → {action.get('target', '?')}"
    if t == "DODGE":
        return "閃避"
    if t == "HIDE":
        return "躲藏"
    if t == "USE_ITEM":
        return f"使用 {action.get('item', '?')}"
    if t == "HEAL":
        return f"治療 {action.get('target', '?')}"
    if t == "ACTION_SURGE":
        return "動作激增（+1 動作）"
    if t == "AUTO_DAMAGE":
        targets = action.get("targets", [])
        if len(targets) == 1:
            return f"自動命中 → {targets[0].get('id', '?')}"
        return f"自動命中（{len(targets)} 個目標）"
    if t == "APPLY_MOD":
        targets = action.get("targets", [])
        spell = action.get("spell_name") or action.get("modifier", "?")
        if len(targets) == 1:
            return f"施展 {spell} → {targets[0]}"
        return f"施展 {spell} → {len(targets)} 個目標"
    if t == "LAY_ON_HANDS":
        return f"聖療之手（{action.get('amount', '?')} HP）→ {action.get('target', '?')}"
    if t == "AOE":
        return f"投擲 {action.get('item', '?')}"
    if t == "ROLL":
        return f"檢定 {action.get('stat', '?')}"
    return t or "未知行動"


def _format_roll_breakdown(bd: dict | None) -> str:
    """Render an attack/save breakdown as `[d20(7)+STR(2)+prof(2)+blessed(3)] `,
    or empty string when nothing interesting to show (no breakdown supplied).
    Trailing space is included so callers can drop it straight before `vs AC`.
    """
    if not bd:
        return ""
    parts: list[str] = [f"d20({bd['d20']})"]
    stat_kind = bd.get("stat_mod_kind") or bd.get("stat")
    if bd.get("stat_mod"):
        parts.append(f"{stat_kind}({bd['stat_mod']:+d})")
    if bd.get("prof"):
        parts.append(f"prof({bd['prof']:+d})")
    for name, delta in bd.get("modifiers", []) or []:
        parts.append(f"{name}({delta:+d})")
    return "[" + "+".join(parts).replace("+-", "-") + "] "


def _format_damage_breakdown(base_roll, dmg_mod: int,
                             mod_contribs: list | None) -> str:
    """Render `[d?(N)+STR(2)+raging(2)]` when status modifiers contributed,
    otherwise empty (the existing `(1d8+2)` notation is enough for the
    plain case)."""
    if not mod_contribs:
        return ""
    parts: list[str] = []
    if base_roll is not None:
        parts.append(f"dice({base_roll})")
    if dmg_mod:
        parts.append(f"mod({dmg_mod:+d})")
    for name, delta in mod_contribs:
        parts.append(f"{name}({delta:+d})")
    return " [" + "+".join(parts).replace("+-", "-") + "]"


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
            f"使用 {result['weapon_name']}{mode_str}，攻擊骰 {result['roll']} "
            f"{_format_roll_breakdown(result.get('roll_breakdown'))}"
            f"vs AC {result['target_ac']}：{hit_str}"
        )
        if result.get("reaction"):
            slot_str = f"（消耗 {result['reaction_slot']} 環）" if result.get("reaction_slot") else ""
            lines.append(f"  ↳ 反應：{result['reaction']}{slot_str} 觸發，AC 變為 {result['target_ac']}")
        if result["hit"]:
            alive    = "存活" if result.get("target_alive") else "倒下"
            dmg_mod  = result.get("damage_mod", 0)
            mod_str  = f"+{dmg_mod}" if dmg_mod > 0 else (str(dmg_mod) if dmg_mod < 0 else "")
            dice_str = f"{result['damage_dice']}{mod_str}"
            dmg_extras = _format_damage_breakdown(
                result.get("damage_base_roll"), dmg_mod, result.get("damage_modifiers"),
            )
            bonus_parts = []
            if result.get("sneak_attack_damage"):
                bonus_parts.append(f"偷襲+{result['sneak_attack_damage']}")
            if result.get("hunters_mark_damage"):
                bonus_parts.append(f"獵人印記+{result['hunters_mark_damage']}")
            if result.get("divine_smite_damage"):
                bonus_parts.append(f"神聖打擊+{result['divine_smite_damage']}")
            bonus_str = f" [{', '.join(bonus_parts)}]" if bonus_parts else ""
            lines.append(
                f"造成 {result['damage']} 點傷害（{dice_str}）{dmg_extras}{bonus_str}，"
                f"{result['target_name']} HP {result['target_hp']}/{result['target_max_hp']}（{alive}）"
            )
            if "rider_save_roll" in result:
                save_outcome = "豁免成功" if result["rider_save_success"] else "豁免失敗"
                rider_bd_str = _format_roll_breakdown(result.get("rider_save_breakdown"))
                rider_line = (
                    f"  附加：{result['rider_save_stat']} 豁免 {result['rider_save_roll']} "
                    f"{rider_bd_str}：{save_outcome}"
                )
                if result.get("rider_status_applied"):
                    rider_line += f"，獲得狀態 [{result['rider_status_applied']}]"
                lines.append(rider_line)

    elif t == "MULTI_ATTACK":
        for i, atk in enumerate(result.get("attacks", []), start=1):
            hit_str  = "命中" if atk["hit"] else "未命中"
            mode_str = {"advantage": "（優勢）", "disadvantage": "（劣勢）"}.get(
                atk.get("advantage_mode", "normal"), ""
            )
            bd_str = _format_roll_breakdown(atk.get("roll_breakdown"))
            line = (f"  [{i}] {atk['weapon_name']}{mode_str} "
                    f"攻擊骰 {atk['roll']} {bd_str}vs AC {atk['target_ac']}：{hit_str}")
            if atk.get("hit"):
                dmg_mod  = atk.get("damage_mod", 0)
                mod_str  = f"+{dmg_mod}" if dmg_mod > 0 else (str(dmg_mod) if dmg_mod < 0 else "")
                dmg_ext  = _format_damage_breakdown(
                    atk.get("damage_base_roll"), dmg_mod, atk.get("damage_modifiers")
                )
                bonus_parts = []
                if atk.get("sneak_attack_damage"):
                    bonus_parts.append(f"偷襲+{atk['sneak_attack_damage']}")
                if atk.get("hunters_mark_damage"):
                    bonus_parts.append(f"獵人印記+{atk['hunters_mark_damage']}")
                if atk.get("divine_smite_damage"):
                    bonus_parts.append(f"神聖打擊+{atk['divine_smite_damage']}")
                bonus_str = f" [{', '.join(bonus_parts)}]" if bonus_parts else ""
                line += f"，{atk['damage']} 傷（{atk['damage_dice']}{mod_str}）{dmg_ext}{bonus_str}"
            lines.append(line)
            if atk.get("rider_save_roll") is not None:
                save_ok = atk.get("rider_save_success")
                rider_stat = atk.get("rider_save_stat", "")
                rider_roll = atk.get("rider_save_roll")
                rider_bd = _format_roll_breakdown(atk.get("rider_save_breakdown"))
                outcome = "成功" if save_ok else "失敗"
                rider_line = f"    附加：{rider_stat} 豁免 {rider_roll} {rider_bd}：{outcome}"
                if atk.get("rider_status_applied"):
                    rider_line += f"，施加 [{atk['rider_status_applied']}]"
                lines.append(rider_line)
        alive_str = "存活" if result.get("target_alive") else "倒下"
        lines.append(
            f"  合計 {result['total_damage']} 傷，"
            f"{result['target_name']} HP {result['target_hp']}/{result['target_max_hp']}（{alive_str}）"
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
        slot_str = f"{result['slot_level']} 環" if result['slot_level'] > 0 else "戲法"
        dmg_str = f"{result['damage_dice']} {result['damage_type']}傷，" if result.get('damage_dice') else ""
        lines.append(
            f"施展「{result['spell_name']}」（{slot_str}，{dmg_str}"
            f"以 {result['center_name']} 為中心，"
            f"{result['save_stat']} DC{result['save_dc']} 豁免）"
        )
        for tr in result.get("target_results", []):
            save_str  = "豁免成功" if tr["save_success"] else "豁免失敗"
            save_bd_str = _format_roll_breakdown(tr.get("save_breakdown"))
            alive_str = "存活" if tr["target_alive"] else "倒下"
            extras = []
            if tr["damage"] > 0:
                extras.append(f"受 {tr['damage']} 傷害")
            if tr.get("status_applied"):
                extras.append(f"獲得狀態 [{tr['status_applied']}]")
            extras_str = "，".join(extras) or "無效"
            lines.append(
                f"  {tr['target_name']}：{save_str} {tr['save_roll']} {save_bd_str}"
                f"，{extras_str}，"
                f"HP {tr['target_hp']}/{tr['target_max_hp']}（{alive_str}）"
            )

    elif t == "HEAL":
        slot_part = f"，{result['slot_level']} 環" if result.get("slot_level", 0) > 0 else ""
        lines.append(
            f"{result['caster_name']} 治療 {result['target_name']}（{result['dice']}{slot_part}）：恢復 {result['amount']} HP "
            f"({result['target_hp']}/{result['target_max_hp']})"
        )

    elif t == "ACTION_SURGE":
        lines.append(f"{result['character']} 激增動作：本回合再獲得 1 個動作")

    elif t == "AUTO_DAMAGE":
        slot_str = f"{result['slot_level']} 環" if result['slot_level'] > 0 else "戲法"
        lines.append(
            f"自動命中（{slot_str}，每發 {result['damage_per']} {result['damage_type']}傷）"
        )
        for tr in result.get("target_results", []):
            alive_str = "存活" if tr["target_alive"] else "倒下"
            if tr.get("reaction"):
                lines.append(
                    f"  {tr['target_name']}：反應 {tr['reaction']} 阻擋 {tr['darts']} 發傷害，"
                    f"HP {tr['target_hp']}/{tr['target_max_hp']}（{alive_str}）"
                )
            else:
                lines.append(
                    f"  {tr['target_name']}：{tr['darts']} 發共 {tr['damage']} 傷，"
                    f"HP {tr['target_hp']}/{tr['target_max_hp']}（{alive_str}）"
                )

    elif t == "APPLY_MOD":
        slot_str = f"，{result['slot_level']} 環" if result['slot_level'] > 0 else ""
        affected = "、".join(result.get("targets_affected", [])) or "無人"
        lines.append(
            f"施展「{result['spell_name']}」（[{result['modifier']}] modifier{slot_str}）"
            f"→ 影響 {affected}"
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
        for oa in result.get("opportunity_attacks", []):
            hit_str = "命中" if oa["hit"] else "未命中"
            dmg_str = f"，造成 {oa['damage']} 傷害，HP {oa['target_hp']}/{oa['target_max_hp']}" if oa["hit"] else ""
            lines.append(
                f"  ↳ 藉機攻擊：{oa['attacker']} 使用 {oa['weapon']} 攻擊 {oa['target']}"
                f"（{oa['roll']} vs AC）：{hit_str}{dmg_str}"
            )

    elif t == "DODGE":
        lines.append(f"{result['character']} 採取閃避姿態（下次被攻擊前，攻擊者擲劣勢）")

    elif t == "DISENGAGE":
        lines.append(f"{result['character']} 脫身（本回合移動不會觸發藉機攻擊）")

    elif t == "COUNTERSPELLED":
        lines.append(
            f"【反制魔法】{result['counterspeller']} 消耗 {result['slot_used']} 環法術位，"
            f"反制了 {result['caster_name']} 的「{result['spell_name']}」！"
        )

    elif t == "LAY_ON_HANDS":
        lines.append(
            f"聖療之手：{result['caster']} 治療 {result['target']} {result['healed']} HP "
            f"（剩餘資源 {result['pool_left']}），"
            f"HP {result['target_hp']}/{result['target_max_hp']}"
        )

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
    """Build a human-readable summary of every spell available to the caster.

    Walks `known_abilities`: any ability whose display_name matches a SPELLS
    entry is a spell. The Spell object provides level/range/damage/save info
    for the formatted line."""
    if not char.spellcasting_ability:
        return ""
    from .spells import SPELLS
    from .abilities import ABILITY_REGISTRY
    seen: set[str] = set()
    parts = []
    for skill_id in (char.known_abilities or []):
        ab = ABILITY_REGISTRY.get(skill_id)
        if ab is None:
            continue
        name = ab.display_name
        if name in seen:
            continue
        spell = SPELLS.get(name)
        if not spell:
            continue
        seen.add(name)
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
