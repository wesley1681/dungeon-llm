"""Smoke test for the Phase 1 spell system.

Built up incrementally across tasks in docs/superpowers/plans/2026-05-17-spell-system-phase-1.md.
No LLM calls; all assertions are deterministic with random.seed where dice are rolled.
"""
import sys
import random
sys.path.insert(0, ".")


def test_spell_dataclass_and_catalog() -> None:
    """Task 1: Spell dataclass exists, SPELLS catalog has 火球術 with expected fields."""
    from trpg.engine.spells import Spell, SPELLS

    fireball = SPELLS["火球術"]
    assert isinstance(fireball, Spell)
    assert fireball.name == "火球術"
    assert fireball.level == 3
    assert fireball.range_m == 45.0
    assert fireball.aoe_radius_m == 6.0
    assert fireball.attack_type == "save"
    assert fireball.save_ability == "DEX"
    assert fireball.damage_dice == "8d6"
    assert fireball.damage_type == "火"
    print("Spell dataclass + SPELLS catalog: OK")


def test_character_spell_fields() -> None:
    """Task 2: Character carries spells / spellcasting_ability / spell_slots."""
    from trpg.engine.character import Character, Stats

    # Default values — non-caster character
    plain = Character(
        name="平民", race="人類", class_="—", level=1,
        stats=Stats(), hp=4, max_hp=4, ac=10,
    )
    assert plain.spells == []
    assert plain.spellcasting_ability == ""
    assert plain.spell_slots == {}

    # Caster — explicit setup
    caster = Character(
        name="法師", race="人類", class_="法師", level=5,
        stats=Stats(INT=16), hp=20, max_hp=20, ac=12,
        spells=["火球術"],
        spell_slots={1: 4, 2: 3, 3: 2},
        spellcasting_ability="INT",
    )
    assert caster.spells == ["火球術"]
    assert caster.spellcasting_ability == "INT"
    assert caster.spell_slots[3] == 2
    print("Character spell fields: OK")


def test_spell_handler_save_type() -> None:
    """Task 3: execute_action SPELL — fireball centered on a goblin hits
    everyone within 6m, rolls DEX saves, applies half-on-save, decrements slot."""
    from trpg.scenarios.dungeon import build_world_state
    from trpg.engine import combat
    from trpg.engine.character import Character, Stats

    ws = build_world_state()
    ws.dungeon_map.current_room_id = "guard_room"
    # Inject a temporary caster into the room — we don't need the full shaman yet,
    # we only need a Character object with the right fields. Task 8 places the real shaman.
    shaman = Character(
        name="測試薩滿", race="地精", class_="薩滅", level=3,
        stats=Stats(WIS=16, CON=12, DEX=10),
        hp=18, max_hp=18, ac=12,
        spells=["火球術"],
        spell_slots={3: 1},
        spellcasting_ability="WIS",
        is_npc=True, attitude=0,
        position=10.0,
    )
    ws.characters["test_shaman"] = shaman
    ws.dungeon_map.current_room.npc_ids.append("test_shaman")

    g1 = ws.characters["goblin_1"]
    g2 = ws.characters["goblin_2"]
    thor = ws.characters["thor"]
    aria = ws.characters["aria"]
    # Set positions: shaman 10m away from party, g1 at 1.5m (in radius of party), g2 at 8m.
    g1.position = 1.5
    g2.position = 8.0   # within 6m of shaman at 10m (dist 2m), should be hit
    thor.position = 0.0
    aria.position = 0.0
    # Reset HP so damage assertions are reliable.
    thor.hp = thor.max_hp
    aria.hp = aria.max_hp
    g1.hp = g1.max_hp
    g2.hp = g2.max_hp

    # Center on goblin_2 → AOE covers anyone within 6m of position 8m: g2 (0m away),
    # g1 (6.5m away — out), thor/aria (8m — out), shaman (2m — in, friendly fire).
    random.seed(7)
    action = {
        "type": "SPELL",
        "caster": "test_shaman",
        "spell_name": "火球術",
        "target": "goblin_2",
        # slot_level omitted — engine should default to 3 (spell.level), only slot available
    }
    result = combat.execute_action(action, ws)

    assert result["type"] == "SPELL", f"got {result}"
    assert result["spell_name"] == "火球術"
    assert result["slot_level"] == 3
    assert result["save_stat"] == "DEX"
    # DC = 8 + prof_bonus(3//4*1+2=2) + WIS_mod(+3) = 13
    assert result["save_dc"] == 13, f"got {result['save_dc']}"

    affected_names = {tr["target_name"] for tr in result["target_results"]}
    assert "地精乙" in affected_names, f"g2 should be in radius: {affected_names}"
    assert "測試薩滿" in affected_names, f"shaman self-hit (friendly fire): {affected_names}"
    assert "地精甲" not in affected_names, f"g1 too far, should be excluded: {affected_names}"
    assert "索爾" not in affected_names, f"thor too far, should be excluded: {affected_names}"

    # Slot was consumed
    assert shaman.spell_slots[3] == 0, f"slot not decremented: {shaman.spell_slots}"

    # Damage was applied to g2 (HP went down)
    g2_after = [tr for tr in result["target_results"] if tr["target_name"] == "地精乙"][0]
    assert g2_after["damage"] > 0, f"g2 should take damage: {g2_after}"

    print("SPELL handler save-type: OK")


def test_spell_handler_no_slot_available() -> None:
    """Task 3: ERROR when no slot of sufficient level is free."""
    from trpg.scenarios.dungeon import build_world_state
    from trpg.engine import combat
    from trpg.engine.character import Character, Stats

    ws = build_world_state()
    ws.dungeon_map.current_room_id = "guard_room"
    # Caster has slots but all below the spell's level
    weak_caster = Character(
        name="弱法師", race="人類", class_="法師", level=1,
        stats=Stats(INT=14), hp=8, max_hp=8, ac=10,
        spells=["火球術"],
        spell_slots={1: 2, 2: 1},   # no level-3 slot
        spellcasting_ability="INT",
        is_npc=True, attitude=0,
        position=10.0,
    )
    ws.characters["weak_caster"] = weak_caster
    ws.dungeon_map.current_room.npc_ids.append("weak_caster")

    action = {
        "type": "SPELL",
        "caster": "weak_caster",
        "spell_name": "火球術",
        "target": "goblin_1",
    }
    result = combat.execute_action(action, ws)
    assert result["type"] == "ERROR", f"expected ERROR, got {result}"
    assert "法術位" in result["message"], f"message should mention slots: {result}"
    print("SPELL handler no-slot ERROR: OK")


def test_spell_handler_explicit_slot_level() -> None:
    """Task 3: When caster provides slot_level explicitly, engine uses that slot."""
    from trpg.scenarios.dungeon import build_world_state
    from trpg.engine import combat
    from trpg.engine.character import Character, Stats

    ws = build_world_state()
    ws.dungeon_map.current_room_id = "guard_room"
    caster = Character(
        name="多環法師", race="人類", class_="法師", level=5,
        stats=Stats(INT=16), hp=20, max_hp=20, ac=12,
        spells=["火球術"],
        spell_slots={3: 2, 4: 1, 5: 1},
        spellcasting_ability="INT",
        is_npc=True, attitude=0,
        position=10.0,
    )
    ws.characters["multi_caster"] = caster
    ws.dungeon_map.current_room.npc_ids.append("multi_caster")

    action = {
        "type": "SPELL",
        "caster": "multi_caster",
        "spell_name": "火球術",
        "target": "goblin_1",
        "slot_level": 5,
    }
    result = combat.execute_action(action, ws)
    assert result["type"] == "SPELL"
    assert result["slot_level"] == 5
    assert caster.spell_slots[5] == 0
    assert caster.spell_slots[3] == 2   # untouched
    print("SPELL handler explicit slot_level: OK")


def test_spell_handler_out_of_range() -> None:
    """Task 3: ERROR when caster-to-center distance exceeds spell.range_m."""
    from trpg.scenarios.dungeon import build_world_state
    from trpg.engine import combat
    from trpg.engine.character import Character, Stats

    ws = build_world_state()
    ws.dungeon_map.current_room_id = "guard_room"
    far_caster = Character(
        name="遠法師", race="人類", class_="法師", level=5,
        stats=Stats(INT=16), hp=20, max_hp=20, ac=12,
        spells=["火球術"],
        spell_slots={3: 1},
        spellcasting_ability="INT",
        is_npc=True, attitude=0,
        position=100.0,   # 100m from target at 1.5m → dist 98.5m > 45m range
    )
    ws.characters["far_caster"] = far_caster
    ws.dungeon_map.current_room.npc_ids.append("far_caster")
    ws.characters["goblin_1"].position = 1.5

    action = {
        "type": "SPELL",
        "caster": "far_caster",
        "spell_name": "火球術",
        "target": "goblin_1",
    }
    result = combat.execute_action(action, ws)
    assert result["type"] == "ERROR"
    assert "射程" in result["message"]
    # Slot must NOT be consumed on failed cast
    assert far_caster.spell_slots[3] == 1
    print("SPELL handler out-of-range ERROR: OK")


def test_format_result_spell() -> None:
    """Task 4: format_result renders SPELL results readably."""
    from trpg.engine.combat import format_result

    result = {
        "type":          "SPELL",
        "caster_name":   "測試薩滿",
        "spell_name":    "火球術",
        "slot_level":    3,
        "center_name":   "地精乙",
        "save_stat":     "DEX",
        "save_dc":       13,
        "damage_dice":   "8d6",
        "damage_type":   "fire",
        "target_results": [
            {"target_name": "地精乙", "save_roll": 8, "save_success": False,
             "damage": 24, "target_hp": 0, "target_max_hp": 7, "target_alive": False},
            {"target_name": "測試薩滿", "save_roll": 18, "save_success": True,
             "damage": 12, "target_hp": 6, "target_max_hp": 18, "target_alive": True},
        ],
    }
    text = format_result("我對地精乙施展火球術", result, actor_name="測試薩滿")
    assert "火球術" in text
    assert "3 環" in text or "3環" in text
    assert "DC13" in text or "DC 13" in text
    assert "地精乙" in text and "倒下" in text
    assert "測試薩滿" in text and "豁免成功" in text
    print("format_result SPELL: OK")


def test_combat_context_spells_str() -> None:
    """Task 5: build_combat_context populates spells_str for casters, '' for others."""
    from trpg.scenarios.dungeon import build_world_state
    from trpg.engine import combat
    from trpg.engine.combat import build_combat_context, MOVE_BUDGET_M
    from trpg.engine.character import Character, Stats

    ws = build_world_state()
    ws.dungeon_map.current_room_id = "guard_room"
    caster = Character(
        name="測試薩滿", race="地精", class_="薩滅", level=3,
        stats=Stats(WIS=16), hp=18, max_hp=18, ac=12,
        spells=["火球術"],
        spell_slots={3: 1},
        spellcasting_ability="WIS",
        is_npc=True, attitude=0,
        position=10.0,
    )
    ws.characters["caster_ctx"] = caster
    ws.dungeon_map.current_room.npc_ids.append("caster_ctx")
    combat.roll_initiative(ws, ["thor", "aria", "goblin_1", "caster_ctx"])

    # Caster context — spells_str should mention the spell name and slot count
    ctx = build_combat_context(
        actor_id="caster_ctx", actor=caster, world_state=ws,
        resources={"action": 1, "movement": MOVE_BUDGET_M}, round_num=1,
    )
    assert hasattr(ctx, "spells_str"), "CombatContext missing spells_str field"
    assert "火球術" in ctx.spells_str, f"got {ctx.spells_str!r}"
    assert "3 環" in ctx.spells_str or "3環" in ctx.spells_str, f"got {ctx.spells_str!r}"
    # Geometry + friendly-fire warning must reach the caster
    assert "射程 45m" in ctx.spells_str, f"missing range: {ctx.spells_str!r}"
    assert "半徑 6m" in ctx.spells_str, f"missing AOE radius: {ctx.spells_str!r}"
    assert "敵我不分" in ctx.spells_str, f"missing friendly-fire warning: {ctx.spells_str!r}"
    assert "含自己" in ctx.spells_str, f"missing self-hit warning: {ctx.spells_str!r}"
    assert "DEX 豁免" in ctx.spells_str, f"missing save info: {ctx.spells_str!r}"
    assert "8d6" in ctx.spells_str, f"missing damage dice: {ctx.spells_str!r}"
    print(f"caster spells_str: {ctx.spells_str}")

    # Non-caster (Thor) — empty spells_str
    thor = ws.characters["thor"]
    ctx_thor = build_combat_context(
        actor_id="thor", actor=thor, world_state=ws,
        resources={"action": 1, "movement": MOVE_BUDGET_M}, round_num=1,
    )
    assert ctx_thor.spells_str == "", f"non-caster should have empty spells_str, got {ctx_thor.spells_str!r}"
    print("CombatContext spells_str: OK")


def test_arbiter_spell_defaults() -> None:
    """Task 6: ArbiterAgent._CONSUMES_DEFAULT includes SPELL → ['action']."""
    from trpg.llm.arbiter import ArbiterAgent
    assert "SPELL" in ArbiterAgent._CONSUMES_DEFAULT, f"keys: {list(ArbiterAgent._CONSUMES_DEFAULT)}"
    assert ArbiterAgent._CONSUMES_DEFAULT["SPELL"] == ["action"]

    # _apply_defaults fills consumes when missing
    arb = ArbiterAgent(model="dummy")
    filled = arb._apply_defaults({
        "valid": True, "type": "SPELL", "caster": "x", "spell_name": "火球術", "target": "y",
    })
    assert filled.get("consumes") == ["action"], f"got {filled}"
    print("Arbiter _CONSUMES_DEFAULT SPELL: OK")


def test_arbiter_situation_includes_spells() -> None:
    """Task 6: parse() situation block lists available spells when caster has them."""
    from trpg.llm.arbiter import ArbiterAgent
    from trpg.scenarios.dungeon import build_world_state
    from trpg.engine.character import Character, Stats

    ws = build_world_state()
    caster = Character(
        name="測試薩滿", race="地精", class_="薩滅", level=3,
        stats=Stats(WIS=16), hp=18, max_hp=18, ac=12,
        spells=["火球術"], spell_slots={3: 1}, spellcasting_ability="WIS",
        is_npc=True, attitude=0,
    )

    # Use _build_situation if extracted, else inspect the messages debug file.
    # Easiest: monkey-patch complete_chat and inspect the messages sent.
    captured: list[dict] = []
    import trpg.llm.arbiter as arb_mod
    real_complete = arb_mod.complete_chat
    def fake_complete(base_url, model, messages, options, backend="ollama", timeout=60):
        captured.extend(messages)
        return '{"valid": false, "reason": "test"}'
    arb_mod.complete_chat = fake_complete
    try:
        arb = ArbiterAgent(model="dummy")
        arb.parse(
            player_action="我對地精甲施展火球術",
            actor_id="caster_x",
            actor_name="測試薩滿",
            available_targets={"goblin_1": "地精甲"},
            actor_char=caster,
        )
    finally:
        arb_mod.complete_chat = real_complete

    full_text = "\n".join(m["content"] for m in captured)
    assert "可用法術" in full_text, f"missing 可用法術 line; sent:\n{full_text[:600]}"
    assert "火球術" in full_text, f"missing spell name in situation block"
    print("Arbiter situation includes spells: OK")


def test_rulebook_has_spell_section() -> None:
    """Task 6: COMBAT_RULEBOOK contains a 法術 / SPELL example."""
    from trpg.llm.rulebook import COMBAT_RULEBOOK
    assert "法術" in COMBAT_RULEBOOK, "rulebook missing 法術 heading"
    assert '"type": "SPELL"' in COMBAT_RULEBOOK, "rulebook missing SPELL JSON example"
    assert '"slot_level"' in COMBAT_RULEBOOK, "rulebook should document optional slot_level"
    print("Rulebook SPELL section: OK")


def test_npc_controller_nudge_includes_spells() -> None:
    """Task 7: LLMNpcController._build_nudge mentions available spells + 施法 option for casters,
    and stays identical for non-casters."""
    from trpg.llm.controllers import LLMNpcController
    from trpg.engine.combat import CombatContext, MOVE_BUDGET_M

    # Caster ctx
    ctx_caster = CombatContext(
        round_num=2, actor_id="x", actor_position=10.0,
        weapons_str="無武器（徒手）",
        spells_str="火球術（3 環，可用：3 環×1）",
        allies_str="無", enemies_str="索爾 HP 31/31，位置 0.0m（距你 10.0m）",
        enemies={"thor": "索爾"},
        resources={"action": 1, "movement": MOVE_BUDGET_M},
    )

    class DummyAgent:
        combat_tactics = ""
        char_id = "x"

    class DummyChar:
        name = "薩滿"
        hp = 18
        max_hp = 18

    ctrl = LLMNpcController(agent=DummyAgent(), emit_event=lambda e: None)
    nudge = ctrl._build_nudge(DummyChar(), ctx_caster)
    assert "可用法術" in nudge, f"caster nudge missing 可用法術:\n{nudge}"
    assert "火球術" in nudge
    assert "施法" in nudge, f"caster nudge missing 施法 action option:\n{nudge}"

    # Non-caster ctx
    ctx_plain = CombatContext(
        round_num=2, actor_id="g", actor_position=1.5,
        weapons_str="彎刀（近戰，伸手 1.5m）",
        spells_str="",
        allies_str="無", enemies_str="索爾",
        enemies={"thor": "索爾"},
        resources={"action": 1, "movement": MOVE_BUDGET_M},
    )
    nudge_plain = ctrl._build_nudge(DummyChar(), ctx_plain)
    assert "可用法術" not in nudge_plain, f"non-caster nudge should not mention 可用法術:\n{nudge_plain}"
    assert "施法" not in nudge_plain, f"non-caster nudge should not list 施法 option:\n{nudge_plain}"
    print("LLMNpcController nudge spells: OK")


def test_player_controller_nudge_includes_spells() -> None:
    """Task 7: LLMPlayerController has parallel behavior — spells line + 施法 option."""
    from trpg.llm.controllers import LLMPlayerController
    from trpg.engine.combat import CombatContext, MOVE_BUDGET_M

    ctx = CombatContext(
        round_num=1, actor_id="x", actor_position=0.0,
        weapons_str="法杖",
        spells_str="火球術（3 環，可用：3 環×2）",
        allies_str="無", enemies_str="哥布林",
        enemies={"g": "哥布林"},
        resources={"action": 1, "movement": MOVE_BUDGET_M},
    )

    class DummyAgent:
        combat_tactics = ""

    class DummyChar:
        name = "PC法師"
        hp = 20
        max_hp = 20

    ctrl = LLMPlayerController(agent=DummyAgent(), emit_event=lambda e: None)
    nudge = ctrl._build_nudge(DummyChar(), ctx)
    assert "可用法術" in nudge
    assert "火球術" in nudge
    assert "施法" in nudge
    print("LLMPlayerController nudge spells: OK")


def test_goblin_shaman_scenario() -> None:
    """Task 8: build_world_state includes goblin_shaman with the expected loadout."""
    from trpg.scenarios.dungeon import build_world_state, build_npc_agents

    ws = build_world_state()
    assert "goblin_shaman" in ws.characters, f"missing goblin_shaman: {list(ws.characters)}"
    shaman = ws.characters["goblin_shaman"]
    assert shaman.is_npc and shaman.attitude == 0
    assert "火球術" in shaman.spells
    assert shaman.spellcasting_ability == "WIS"
    assert shaman.spell_slots.get(3, 0) >= 1, f"slots: {shaman.spell_slots}"
    # Save DC = 8 + prof_bonus + WIS_mod. With level 3 (+2) + WIS 16 (+3) = 13.
    assert shaman.proficiency_bonus == 2
    assert shaman.stats.modifier("WIS") == 3

    # Placed in boss_chamber
    assert "goblin_shaman" in ws.dungeon_map.rooms["boss_chamber"].npc_ids, \
        f"shaman not in boss_chamber.npc_ids: {ws.dungeon_map.rooms['boss_chamber'].npc_ids}"

    # NpcAgent registered
    agents = build_npc_agents(ws, model="dummy", base_url="http://x", backend="ollama")
    assert "goblin_shaman" in agents, f"agent registry missing shaman: {list(agents)}"
    print("Goblin shaman scenario data: OK")


def test_end_to_end_shaman_fireball() -> None:
    """Task 9: Full end-to-end — load the scenario as shipped, simulate shaman
    casting fireball on the party in boss_chamber. No LLM involved; we hand the
    engine the SPELL action dict directly (as if Arbiter had parsed it)."""
    from trpg.scenarios.dungeon import build_world_state
    from trpg.engine import combat
    from trpg.engine.combat import build_combat_context, MOVE_BUDGET_M

    ws = build_world_state()
    ws.dungeon_map.current_room_id = "boss_chamber"
    shaman = ws.characters["goblin_shaman"]
    thor = ws.characters["thor"]
    aria = ws.characters["aria"]
    # Place party + shaman for the cast: party at 0m, shaman at 10m.
    thor.position = 0.0
    aria.position = 0.0
    shaman.position = 10.0
    # Other room occupants are not the focus; they'll be in their default starting positions.
    initial_shaman_slot = shaman.spell_slots[3]

    # Build context as the controller would — sanity check the spell is exposed.
    ctx = build_combat_context(
        actor_id="goblin_shaman", actor=shaman, world_state=ws,
        resources={"action": 1, "movement": MOVE_BUDGET_M}, round_num=1,
    )
    assert "火球術" in ctx.spells_str

    # Cast fireball centered on Thor; slot_level omitted to test default.
    random.seed(101)
    result = combat.execute_action({
        "type": "SPELL",
        "caster": "goblin_shaman",
        "spell_name": "火球術",
        "target": "thor",
    }, ws)
    assert result["type"] == "SPELL"
    assert result["slot_level"] == 3
    # Both PCs are within 6m of Thor (position 0); they should both be in the result.
    affected_names = {tr["target_name"] for tr in result["target_results"]}
    assert {"索爾", "凱恩"}.issubset(affected_names), \
        f"both PCs should be in fireball radius: {affected_names}"

    # Slot was decremented exactly once
    assert shaman.spell_slots[3] == initial_shaman_slot - 1

    # Some party damage occurred (party can't have full HP on both PCs after fireball)
    assert thor.hp < thor.max_hp or aria.hp < aria.max_hp, \
        "fireball produced no damage on either PC — check dice roll path"

    # format_result renders without error
    summary = combat.format_result("我對你們投出火球", result, actor_name="地精薩滿沃克")
    assert "火球術" in summary
    assert "DC13" in summary or "DC 13" in summary

    # Cast again with no slot → ERROR; HP and slot dict must not change
    hp_thor_before = thor.hp
    hp_aria_before = aria.hp
    result2 = combat.execute_action({
        "type": "SPELL",
        "caster": "goblin_shaman",
        "spell_name": "火球術",
        "target": "thor",
    }, ws)
    assert result2["type"] == "ERROR"
    assert thor.hp == hp_thor_before
    assert aria.hp == hp_aria_before
    print("End-to-end shaman fireball: OK")


def test_caster_starts_in_back_row() -> None:
    """setup_combat_positions should treat spellcasters as back-row (≥6m),
    even when they only carry a melee weapon. Otherwise shaman starts at
    1.5m next to the party and any AOE catches itself."""
    from trpg.scenarios.dungeon import build_world_state
    from trpg.engine import combat

    ws = build_world_state()
    ws.dungeon_map.current_room_id = "boss_chamber"
    combat.roll_initiative(ws, ["thor", "aria", "goblin_shaman", "goblin_1", "goblin_3"])
    shaman = ws.characters["goblin_shaman"]
    g1 = ws.characters["goblin_1"]   # melee-only (彎刀), no spells
    g3 = ws.characters["goblin_3"]   # archer, has bow
    # Shaman carries only a melee staff but is a caster → must be back row.
    assert shaman.position == 6.0, f"caster shaman not in back row: {shaman.position}"
    # goblin_1 is melee-only, no spells → front row
    assert g1.position == 1.5, f"melee-only goblin not in front row: {g1.position}"
    # Archer has ranged bow → back row
    assert g3.position == 6.0, f"archer not in back row: {g3.position}"
    print("Caster back-row placement: OK")


def test_spell_handler_target_position() -> None:
    """SPELL action with target_position uses the coord as AOE center directly,
    overriding `target`. Lets the caster avoid friendly fire by placing the
    circle away from allies."""
    from trpg.scenarios.dungeon import build_world_state
    from trpg.engine import combat
    from trpg.engine.character import Character, Stats

    ws = build_world_state()
    ws.dungeon_map.current_room_id = "guard_room"
    caster = Character(
        name="精準法師", race="人類", class_="法師", level=5,
        stats=Stats(INT=16), hp=20, max_hp=20, ac=12,
        spells=["火球術"],
        spell_slots={3: 1},
        spellcasting_ability="INT",
        is_npc=True, attitude=0,
        position=0.0,   # standing with the party
    )
    ws.characters["aim_mage"] = caster
    ws.dungeon_map.current_room.npc_ids.append("aim_mage")
    # Position the targets: g1 at 10m, g2 at 12m, party (Thor, Aria) at 0m.
    ws.characters["goblin_1"].position = 10.0
    ws.characters["goblin_2"].position = 12.0
    ws.characters["thor"].position = 0.0
    ws.characters["aria"].position = 0.0
    ws.characters["thor"].hp = ws.characters["thor"].max_hp
    ws.characters["aria"].hp = ws.characters["aria"].max_hp

    # Drop AOE at 11m — catches g1 (dist 1m) and g2 (dist 1m), spares party.
    random.seed(42)
    result = combat.execute_action({
        "type": "SPELL",
        "caster": "aim_mage",
        "spell_name": "火球術",
        "target_position": 11.0,
    }, ws)
    assert result["type"] == "SPELL", f"got {result}"
    assert result["center_name"] == "位置 11.0m", f"got {result['center_name']!r}"
    affected = {tr["target_name"] for tr in result["target_results"]}
    assert "地精甲" in affected and "地精乙" in affected, f"missing goblins: {affected}"
    assert "索爾" not in affected, f"thor was caught at distance 11m: {affected}"
    assert "凱恩" not in affected, f"aria was caught at distance 11m: {affected}"
    assert "精準法師" not in affected, f"caster self-hit: {affected}"
    print("SPELL target_position: OK")


def test_spell_target_position_overrides_target() -> None:
    """If both `target` and `target_position` are given, target_position wins."""
    from trpg.scenarios.dungeon import build_world_state
    from trpg.engine import combat
    from trpg.engine.character import Character, Stats

    ws = build_world_state()
    ws.dungeon_map.current_room_id = "guard_room"
    caster = Character(
        name="優先法師", race="人類", class_="法師", level=5,
        stats=Stats(INT=16), hp=20, max_hp=20, ac=12,
        spells=["火球術"], spell_slots={3: 1}, spellcasting_ability="INT",
        is_npc=True, attitude=0, position=0.0,
    )
    ws.characters["pri_mage"] = caster
    ws.dungeon_map.current_room.npc_ids.append("pri_mage")
    ws.characters["goblin_1"].position = 1.5   # adjacent
    ws.characters["thor"].position = 0.0
    ws.characters["aria"].position = 0.0

    # target says goblin_1 (would center at 1.5m, hits party), target_position
    # says 30.0 (far away, hits nothing). target_position must win.
    random.seed(0)
    result = combat.execute_action({
        "type": "SPELL",
        "caster": "pri_mage",
        "spell_name": "火球術",
        "target": "goblin_1",
        "target_position": 30.0,
    }, ws)
    assert result["type"] == "SPELL"
    assert result["center_name"] == "位置 30.0m"
    affected = {tr["target_name"] for tr in result["target_results"]}
    assert "地精甲" not in affected, f"target_position should override: {affected}"
    print("SPELL target_position overrides target: OK")


def main() -> int:
    test_spell_dataclass_and_catalog()
    test_character_spell_fields()
    test_spell_handler_save_type()
    test_spell_handler_no_slot_available()
    test_spell_handler_explicit_slot_level()
    test_spell_handler_out_of_range()
    test_format_result_spell()
    test_combat_context_spells_str()
    test_arbiter_spell_defaults()
    test_arbiter_situation_includes_spells()
    test_rulebook_has_spell_section()
    test_npc_controller_nudge_includes_spells()
    test_player_controller_nudge_includes_spells()
    test_goblin_shaman_scenario()
    test_caster_starts_in_back_row()
    test_spell_handler_target_position()
    test_spell_target_position_overrides_target()
    test_end_to_end_shaman_fireball()
    print("\n=== ALL SPELL TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
