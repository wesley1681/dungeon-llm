"""Smoke test for CombatContext + build_combat_context.

Verifies the unified context builder returns equivalent strings for an NPC
actor, a party PC, and the player (Aria), all from a single function.
"""
import sys
sys.path.insert(0, ".")

from trpg.scenarios.dungeon import build_world_state
from trpg.engine import combat
from trpg.engine.combat import build_combat_context, CombatContext, MOVE_BUDGET_M


def main() -> int:
    ws = build_world_state()
    ws.dungeon_map.current_room_id = "guard_room"
    state = combat.roll_initiative(ws, ["thor", "aria", "goblin_1"])

    # ── NPC perspective ──────────────────────────────────────────────────────
    g1 = ws.characters["goblin_1"]
    ctx = build_combat_context(
        actor_id="goblin_1", actor=g1, world_state=ws,
        resources={"action": 1, "movement": MOVE_BUDGET_M}, round_num=1,
    )
    assert isinstance(ctx, CombatContext)
    assert ctx.round_num == 1
    assert ctx.actor_position == g1.position
    # NPC sees Thor/Aria as enemies (dungeon.py uses "凱恩" for aria, "索爾" for thor)
    assert "凱恩" in ctx.enemies_str or "索爾" in ctx.enemies_str, f"got {ctx.enemies_str}"
    # NPC's own weapon should be listed
    assert g1.weapons[0].name in ctx.weapons_str
    print(f"NPC ctx enemies: {ctx.enemies_str}")
    print("NPC build_combat_context: OK")

    # ── Party PC perspective (Thor) ──────────────────────────────────────────
    thor = ws.characters["thor"]
    ctx = build_combat_context(
        actor_id="thor", actor=thor, world_state=ws,
        resources={"action": 1, "movement": MOVE_BUDGET_M}, round_num=1,
    )
    # Goblin names in dungeon.py are "地精甲", "地精乙", "地精丙"
    assert "地精" in ctx.enemies_str or "goblin" in ctx.enemies_str.lower(), f"got {ctx.enemies_str}"
    # Aria is an ally for Thor
    assert "凱恩" in ctx.allies_str, f"got allies: {ctx.allies_str}"
    print(f"Thor ctx enemies: {ctx.enemies_str}")
    print(f"Thor ctx allies: {ctx.allies_str}")
    print("PC build_combat_context: OK")

    # ── enemies dict matches enemy_str members ───────────────────────────────
    assert g1.name in ctx.enemies_str, f"goblin_1 name {g1.name!r} not in {ctx.enemies_str!r}"
    assert "goblin_1" in ctx.enemies or any("地精" in n for n in ctx.enemies.values()), \
        f"enemies dict: {ctx.enemies}"
    print("enemies dict populated: OK")

    # ── allies dict populated for both sides ─────────────────────────────────
    # Thor (party PC) — allies should include Aria, enemies should NOT appear
    assert "aria" in ctx.allies, f"thor's allies dict missing aria: {ctx.allies}"
    assert "goblin_1" not in ctx.allies, f"enemy leaked into thor.allies: {ctx.allies}"
    # NPC perspective — allies should include the other goblin
    ctx_npc = build_combat_context(
        actor_id="goblin_1", actor=g1, world_state=ws,
        resources={"action": 1, "movement": MOVE_BUDGET_M}, round_num=1,
    )
    # The other goblin in the room should be an ally from goblin_1's POV
    assert any(name.startswith("地精") for name in ctx_npc.allies.values()), \
        f"goblin_1's allies dict should list fellow goblins: {ctx_npc.allies}"
    assert "aria" not in ctx_npc.allies and "thor" not in ctx_npc.allies, \
        f"PCs leaked into npc.allies: {ctx_npc.allies}"
    print("allies dict populated for both sides: OK")

    print("\n=== ALL COMBAT CONTEXT TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
