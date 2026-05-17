"""Structural smoke test for combat depth (distance + dodge + advantage/disadvantage).

No LLM calls — verifies engine plumbing:
  - roll_d20 modes (advantage / disadvantage / normal)
  - combine_advantage cancellation rules
  - Weapon range fields default & shortbow values
  - setup_combat_positions assigns by side + weapon class
  - attack_range_check melee out-of-range
  - attack_range_check ranged-in-melee → disadvantage
  - attack_range_check ranged at long range → disadvantage
  - execute_action MOVE clamps to MOVE_BUDGET_M
  - execute_action ATTACK rejects out-of-range
  - execute_action DODGE adds 'dodging' status
  - ATTACK against dodging target → mode == disadvantage
"""
import sys
import random
sys.path.insert(0, ".")

from trpg.scenarios.dungeon import build_world_state
from trpg.engine import combat
from trpg.engine.dice import roll_d20, combine_advantage


def main() -> int:
    # ── 1. roll_d20 modes ─────────────────────────────────────────────────────
    random.seed(0)
    rolls_norm = [roll_d20("normal") for _ in range(2000)]
    rolls_adv  = [roll_d20("advantage") for _ in range(2000)]
    rolls_dis  = [roll_d20("disadvantage") for _ in range(2000)]
    avg_norm = sum(rolls_norm) / len(rolls_norm)
    avg_adv  = sum(rolls_adv)  / len(rolls_adv)
    avg_dis  = sum(rolls_dis)  / len(rolls_dis)
    print(f"avg normal={avg_norm:.2f}  advantage={avg_adv:.2f}  disadvantage={avg_dis:.2f}")
    assert 9.5 < avg_norm < 11.5
    assert 13.0 < avg_adv  < 14.5   # expected ~13.83
    assert 7.0  < avg_dis  < 8.5    # expected ~7.17
    print("roll_d20 modes: OK")

    # ── 2. combine_advantage cancellation ─────────────────────────────────────
    assert combine_advantage("advantage", "disadvantage") == "normal"
    assert combine_advantage("advantage", "advantage") == "advantage"
    assert combine_advantage("disadvantage", "disadvantage") == "disadvantage"
    assert combine_advantage("normal", "normal") == "normal"
    assert combine_advantage("normal", "advantage") == "advantage"
    print("combine_advantage cancellation: OK")

    # ── 3. Setup ──────────────────────────────────────────────────────────────
    ws = build_world_state()
    # Force party into guard_room with goblins (where combat actually happens)
    ws.dungeon_map.current_room_id = "guard_room"
    thor = ws.characters["thor"]
    aria = ws.characters["aria"]
    g1 = ws.characters["goblin_1"]  # grunt — melee
    g2 = ws.characters["goblin_2"]  # grunt — melee
    g3 = ws.characters["goblin_3"]  # archer — ranged
    # Equip g3 with shortbow so setup_combat_positions sees ranged weapon
    from trpg.engine.items import WEAPON_DEFS, Consumable
    g3.weapons = [WEAPON_DEFS["短弓"]]
    g3.consumables.append(Consumable("箭", 20, "ammo", ""))

    state = combat.roll_initiative(ws, ["thor", "aria", "goblin_1", "goblin_2", "goblin_3"])
    print(f"\nInitial positions after roll_initiative:")
    print(f"  thor pos = {thor.position}")
    print(f"  aria pos = {aria.position}")
    print(f"  goblin_1 (melee) pos = {g1.position}")
    print(f"  goblin_2 (melee) pos = {g2.position}")
    print(f"  goblin_3 (archer) pos = {g3.position}")
    assert thor.position == 0.0
    assert aria.position == 0.0
    assert g1.position == 1.5
    assert g3.position == 6.0   # archer in back row
    print("setup_combat_positions: OK")

    # ── 4. Melee out-of-range check ───────────────────────────────────────────
    longsword = thor.get_weapon("長劍")
    in_range, reason, mode = combat.attack_range_check(thor, g3, longsword)
    print(f"\nthor longsword vs g3 (dist 6m): in_range={in_range}, reason={reason!r}")
    assert not in_range and "超出" in reason
    print("melee out-of-range rejected: OK")

    # Move thor to within reach of g3
    thor.position = 5.0
    in_range, reason, mode = combat.attack_range_check(thor, g3, longsword)
    print(f"thor at 5.0m vs g3 at 6.0m (dist 1m): in_range={in_range}")
    assert in_range and mode == "normal"

    thor.position = 0.0   # reset

    # ── 5. Ranged-in-melee → disadvantage ─────────────────────────────────────
    bow = WEAPON_DEFS["短弓"]
    # g3 has bow; melee enemy g1 at distance 1.5 - 6.0 = 4.5 away from g3. Not melee.
    in_range, reason, mode = combat.attack_range_check(g3, thor, bow)
    print(f"\ng3 (ranged, 6m) vs thor (0m, dist 6m) with bow: in_range={in_range}, mode={mode}")
    assert in_range and mode == "normal"   # within normal range, not in melee

    # Force ranged-in-melee: move thor adjacent to g3
    thor.position = 5.0
    in_range, reason, mode = combat.attack_range_check(g3, thor, bow)
    print(f"g3 (6m) vs thor (5m, dist 1m) with bow: in_range={in_range}, mode={mode}")
    assert in_range and mode == "disadvantage"
    thor.position = 0.0

    # ── 6. Ranged long-range disadvantage ─────────────────────────────────────
    aria.position = -25.0   # well outside normal range 24m for shortbow
    in_range, reason, mode = combat.attack_range_check(g3, aria, bow)
    print(f"g3 (6m) vs aria (-25m, dist 31m) with bow (normal 24/max 96): in_range={in_range}, mode={mode}")
    assert in_range and mode == "disadvantage"

    # Beyond max range
    aria.position = -200.0
    in_range, reason, mode = combat.attack_range_check(g3, aria, bow)
    print(f"g3 vs aria at dist 206m: in_range={in_range}, reason={reason}")
    assert not in_range
    aria.position = 0.0

    print("Ranged disadvantage & rejection: OK")

    # ── 7. MOVE action clamps to budget ────────────────────────────────────────
    action = {"type": "MOVE", "character": "thor", "delta_m": 15.0}
    result = combat.execute_action(action, ws)
    print(f"\nMOVE thor +15m (budget 9): from={result['from_pos']:.1f} to={result['to_pos']:.1f} dist={result['distance']:.1f}")
    assert result["distance"] == 9.0
    assert result["to_pos"] == 9.0
    # Reset
    thor.position = 0.0

    # ── 7b. MOVE target arrives exactly at the creature (no overshoot) ────────
    g3.position = 6.0
    action = {"type": "MOVE", "character": "thor", "target": "goblin_3"}
    result = combat.execute_action(action, ws)
    print(f"MOVE thor → g3 (at 6m): to={result['to_pos']:.1f} dist={result['distance']:.1f}")
    assert result["to_pos"] == 6.0, f"should arrive exactly at g3, got {result['to_pos']}"
    assert result["distance"] == 6.0
    thor.position = 0.0

    # ── 7c. MOVE target capped by budget when target is too far ───────────────
    g3.position = 25.0
    action = {"type": "MOVE", "character": "thor", "target": "goblin_3"}
    result = combat.execute_action(action, ws)
    print(f"MOVE thor → g3 (at 25m, far): to={result['to_pos']:.1f} dist={result['distance']:.1f}")
    assert result["to_pos"] == 9.0, f"should cap at budget 9m, got {result['to_pos']}"
    assert result["distance"] == 9.0
    thor.position = 0.0
    g3.position = 6.0   # reset

    # ── 8. Out-of-range ATTACK rejected ───────────────────────────────────────
    g3.position = 20.0   # move archer far back
    action = {"type": "ATTACK", "attacker": "thor", "target": "goblin_3", "weapon": "長劍"}
    result = combat.execute_action(action, ws)
    print(f"\nATTACK thor with longsword vs g3 at 20m: {result}")
    assert result["type"] == "ERROR" and "超出" in result["message"]
    g3.position = 6.0  # reset

    # ── 9. DODGE adds dodging status ──────────────────────────────────────────
    aria.status_effects.clear()
    action = {"type": "DODGE", "character": "aria"}
    result = combat.execute_action(action, ws)
    print(f"\nDODGE aria: {result}")
    assert result["type"] == "DODGE"
    assert aria.has_status("dodging")
    print("DODGE adds 'dodging' status: OK")

    # ── 10. ATTACK vs dodging target → disadvantage ───────────────────────────
    # g1 attacks aria (who is dodging) with melee. Need g1 in reach.
    g1.position = 1.5   # reset
    aria.position = 0.0
    from trpg.engine.status import Dodging
    aria.status_effects = [Dodging(applied_round=1)]   # ensure dodging
    g1_weapon = g1.get_weapon()         # whatever they have
    if g1_weapon.range_type != "近戰":
        g1.weapons = [WEAPON_DEFS["彎刀"]]
    # The attack range needs to be valid — g1 at 1.5, aria at 0, dist 1.5 — within reach for 1.5m weapon.
    action = {"type": "ATTACK", "attacker": "goblin_1", "target": "aria", "weapon": g1.weapons[0].name}
    # Force seed so we can see roll output
    random.seed(42)
    result = combat.execute_action(action, ws)
    print(f"\nATTACK g1 vs dodging aria: mode={result.get('advantage_mode')}, hit={result.get('hit')}, roll={result.get('roll')}")
    assert result["advantage_mode"] == "disadvantage"
    print("ATTACK vs dodging → disadvantage: OK")

    print("\n=== ALL COMBAT DEPTH TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
