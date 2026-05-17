"""Regression test for the [TALK]+[TRAVEL] same-batch log-snapshot bug.

When TagAgent emits both [TALK: civilian] and [TRAVEL: north] in the same
response, the previous code logged BOTH system results AFTER all tags ran,
so `current_room` had already moved to guard_room and the goblins were
listed in `present` for the "開始與 老柯 對話" entry. The pending
conversation then ran in guard_room context, so every conversation line
was also visible to the goblins.

Fix: execute_all_tags(..., log_to_narrative=True) logs per-tag immediately,
and TRAVEL clears stale pending_conversation when the target NPC is no
longer in the new room.
"""
import sys
sys.path.insert(0, ".")

from trpg.scenarios.dungeon import build_world_state
from trpg.llm.tag_parser import execute_all_tags


def main() -> int:
    ws = build_world_state()
    assert ws.dungeon_map.current_room_id == "entrance"

    # Simulate the exact pathological case: tag agent emits TALK + TRAVEL in
    # one response. TALK fires first (sets pending_conversation while still in
    # entrance), then TRAVEL moves to guard_room.
    batch = "[TALK: 老柯] [TRAVEL: north]"
    ok, errors = execute_all_tags(batch, ws, log_to_narrative=True)
    assert not errors, f"unexpected errors: {errors}"
    assert len(ok) == 2, f"expected 2 ok results, got {ok}"

    # TRAVEL must have invalidated the pending TALK — 老柯 isn't in guard_room
    assert ws.pending_conversation == "", \
        f"pending_conversation should be cleared after move, got {ws.pending_conversation!r}"

    # Two narrative entries written, with DIFFERENT present snapshots
    sys_entries = [e for e in ws.narrative_log if e["speaker"] == "system"]
    assert len(sys_entries) == 2, f"expected 2 system entries, got {len(sys_entries)}"

    talk_entry, travel_entry = sys_entries
    assert "開始與 老柯 對話" in talk_entry["text"], talk_entry["text"]
    assert "移動至：守衛室" in travel_entry["text"], travel_entry["text"]

    # CORE ASSERTION: the TALK entry was snapshotted in entrance —
    # 老柯 was present, goblins were NOT.
    assert talk_entry["room_id"] == "entrance", \
        f"TALK entry should be tagged with entrance, got {talk_entry['room_id']}"
    assert "civilian" in talk_entry["present"], \
        f"老柯 should be in present for TALK entry, got {talk_entry['present']}"
    assert "goblin_1" not in talk_entry["present"], \
        f"地精甲 should NOT see TALK entry — got present={talk_entry['present']}"
    assert "goblin_2" not in talk_entry["present"], \
        f"地精乙 should NOT see TALK entry — got present={talk_entry['present']}"

    # TRAVEL entry was snapshotted AFTER move — goblins ARE present.
    assert travel_entry["room_id"] == "guard_room", \
        f"TRAVEL entry should be tagged with guard_room, got {travel_entry['room_id']}"
    assert "goblin_1" in travel_entry["present"], \
        f"地精甲 should see TRAVEL entry, got {travel_entry['present']}"

    # Goblin's filtered log: TALK invisible, TRAVEL visible
    g1_log = ws.log_for("goblin_1")
    texts = [e["text"] for e in g1_log]
    assert all("開始與 老柯 對話" not in t for t in texts), \
        f"地精甲 leaked the TALK system event: {texts}"
    assert any("移動至：守衛室" in t for t in texts), \
        f"地精甲 missing TRAVEL event: {texts}"

    # 老柯's filtered log: TALK visible, TRAVEL invisible
    civ_log = ws.log_for("civilian")
    civ_texts = [e["text"] for e in civ_log]
    assert any("開始與 老柯 對話" in t for t in civ_texts), \
        f"老柯 missing TALK event: {civ_texts}"
    assert all("移動至：守衛室" not in t for t in civ_texts), \
        f"老柯 should not see post-move TRAVEL event: {civ_texts}"

    print("=== PASSED ===")
    print(f"  TALK  entry: room={talk_entry['room_id']}  present={talk_entry['present']}")
    print(f"  TRAVEL entry: room={travel_entry['room_id']}  present={travel_entry['present']}")
    print(f"  pending_conversation cleared: {ws.pending_conversation!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
