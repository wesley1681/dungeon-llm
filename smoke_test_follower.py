"""Structural smoke test for the follower / party system.

No LLM calls — tests the data plumbing only:
  - RECRUIT tag gate rules (attitude / alive / not already in)
  - RECRUIT sets pending_join_decision on agent
  - NpcAgent parses <JOIN> / <DECLINE> markers
  - TRAVEL syncs follower to new room.npc_ids
  - _alive_side_b includes follower
  - _build_npc_combat_context: hostile sees follower as enemy; follower sees PCs+other followers as allies
  - ATTACK_NPC on follower removes from party
  - Dead follower cleanup after combat end
"""
import sys
sys.path.insert(0, ".")

from trpg.scenarios.dungeon import build_world_state, build_npc_agents
from trpg.llm.tag_parser import execute_all_tags, set_npc_agent_registry
from trpg.game import GameSession


def main() -> int:
    ws = build_world_state()
    npc_agents = build_npc_agents(ws, model="stub", base_url="http://x", backend="ollama")
    set_npc_agent_registry(npc_agents)
    civ = ws.characters["civilian"]
    civ_agent = npc_agents["civilian"]

    print(f"Start: party_ids = {ws.party_ids}")
    assert ws.party_ids == ["aria", "thor"]

    # ── 1. RECRUIT gate: attitude too low ─────────────────────────────────────
    civ.attitude = 1   # 戒備
    ok, errors = execute_all_tags("[RECRUIT: civilian]", ws)
    print(f"\nRecruit @ attitude=1: ok={ok} errors={errors}")
    assert ok and "不夠信任" in ok[0]
    assert not civ_agent.pending_join_decision

    # ── 2. RECRUIT gate: attitude ok ──────────────────────────────────────────
    civ.attitude = 3   # 友好
    ok, errors = execute_all_tags("[RECRUIT: civilian]", ws)
    print(f"Recruit @ attitude=3: ok={ok} errors={errors}")
    assert ok and "考慮" in ok[0]
    assert civ_agent.pending_join_decision

    # ── 3. NpcAgent system prompt now contains recruit section ────────────────
    sys_prompt = civ_agent._system()
    assert "對方邀你加入冒險" in sys_prompt
    assert "<JOIN>" in sys_prompt
    print("Recruit section injected into system prompt: OK")

    # ── 4. Marker parser: <JOIN> / <DECLINE> ──────────────────────────────────
    import re
    from trpg.llm.npc_agent import _RECRUIT_RE
    m = _RECRUIT_RE.search("好的，我跟你們走\n<JOIN>")
    assert m and m.group(1).lower() == "join"
    m = _RECRUIT_RE.search("我還是留下吧\n<DECLINE>")
    assert m and m.group(1).lower() == "decline"
    print("JOIN / DECLINE markers parse correctly: OK")

    # ── 5. Simulate JOIN: manually add to party ───────────────────────────────
    ws.party_ids.append("civilian")
    civ_agent.in_party = True
    civ_agent.pending_join_decision = False
    print(f"\nAfter join: party_ids = {ws.party_ids}")
    assert ws.is_party_ally("civilian")
    assert ws.is_party_ally("aria")
    assert not ws.is_party_ally("goblin_1")

    # ── 6. TRAVEL syncs follower to new room ──────────────────────────────────
    print(f"\nBefore TRAVEL: entry_corridor npc_ids = {ws.dungeon_map.rooms['entrance'].npc_ids}")
    # current_room is entrance; civilian is in npc_ids
    # Mark guard_room as cleared so we don't trigger combat
    ws.dungeon_map.rooms["guard_room"].cleared = True
    ok, errors = execute_all_tags("[TRAVEL: north]", ws)
    print(f"TRAVEL result: ok={ok} errors={errors}")
    print(f"After TRAVEL: current_room = {ws.dungeon_map.current_room.name}")
    print(f"  entrance npc_ids = {ws.dungeon_map.rooms['entrance'].npc_ids}")
    print(f"  guard_room npc_ids = {ws.dungeon_map.rooms['guard_room'].npc_ids}")
    assert "civilian" not in ws.dungeon_map.rooms["entrance"].npc_ids
    assert "civilian" in ws.dungeon_map.rooms["guard_room"].npc_ids

    # ── 7. Build a GameSession to test combat helpers ─────────────────────────
    class StubAgent:
        def __init__(self): self.model = "stub"
    session = GameSession(
        world_state=ws,
        gm=StubAgent(),
        tag_agent=StubAgent(),
        thor_agent=StubAgent(),
        arbiter=StubAgent(),
        npc_agents=npc_agents,
    )

    side_b = session._alive_side_b()
    print(f"\n_alive_side_b: {side_b}")
    assert "aria" in side_b and "thor" in side_b and "civilian" in side_b
    assert "goblin_1" not in side_b

    # ── 8. _build_npc_combat_context from follower's POV ──────────────────────
    # Move goblins into guard_room so they're "in current room" (already are by default)
    # civilian is in guard_room (we moved there)
    weapons, allies, enemies = session._build_npc_combat_context("civilian", civ)
    print(f"\n老柯 視角：allies={allies!r}, enemies={enemies!r}")
    assert "凱恩" in allies and "索爾" in allies
    # Goblins are hostile + in guard_room → should be enemies
    assert "哥布林" in enemies or "地精" in enemies  # goblin name appears

    # ── 9. _build_npc_combat_context from hostile goblin's POV ────────────────
    g1 = ws.characters["goblin_1"]
    weapons, allies, enemies = session._build_npc_combat_context("goblin_1", g1)
    print(f"goblin_1 視角：allies={allies!r}, enemies={enemies!r}")
    assert "老柯" in enemies   # follower now appears as enemy to hostile
    assert "凱恩" in enemies and "索爾" in enemies

    # ── 10. ATTACK_NPC removes follower from party ────────────────────────────
    # Need civilian to be in current room (already in guard_room)
    ok, errors = execute_all_tags("[ATTACK_NPC: civilian]", ws)
    print(f"\nATTACK_NPC civilian: ok={ok} errors={errors}")
    assert "civilian" not in ws.party_ids, f"civilian should be removed, got {ws.party_ids}"
    assert not civ_agent.in_party
    assert civ.attitude == 0

    # ── 11. Dead follower cleanup ─────────────────────────────────────────────
    # Re-add civilian to party, kill them, and run combat-end cleanup logic
    civ.attitude = 3
    ws.party_ids.append("civilian")
    civ_agent.in_party = True
    civ.hp = 0
    # Simulate combat-end cleanup
    for npc_id in list(ws.party_ids):
        c = ws.characters.get(npc_id)
        if c and c.is_npc and not c.is_alive():
            ws.party_ids.remove(npc_id)
            agent = npc_agents.get(npc_id)
            if agent:
                agent.in_party = False
    print(f"\nAfter dead-cleanup: party_ids = {ws.party_ids}")
    assert "civilian" not in ws.party_ids
    assert not civ_agent.in_party

    print("\n=== ALL FOLLOWER TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
