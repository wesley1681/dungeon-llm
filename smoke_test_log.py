"""Structural smoke test for the unified narrative_log refactor.

Does NOT call any LLM — just verifies:
  - WorldState.log_event / log_for / log_all
  - render_messages produces correct role assignment per identity
  - render_script produces a clean script string
  - Cross-room visibility (entries from room A not visible to chars in room B)
  - NpcAgent._system() composes the prompt correctly without LLM
"""
import sys
sys.path.insert(0, ".")

from trpg.scenarios.dungeon import build_world_state, build_npc_agents
from trpg.llm.log_render import render_messages, render_script


def main() -> int:
    ws = build_world_state()
    civ = ws.characters["civilian"]
    print(f"場景: {ws.scenario_name}")
    print(f"當前房間: {ws.dungeon_map.current_room.name}")
    print(f"當前房間 npc_ids: {ws.dungeon_map.current_room.npc_ids}")

    # ── 1. Log an event in current room ───────────────────────────────────────
    ws.log_event("gm", "你們進入了陰暗的入口走廊。")
    ws.log_event("thor", "我握緊長劍，警戒地環視四周。")
    ws.log_event("aria", "我看看有沒有可以撿的東西。")
    ws.log_event("system", "凱恩 INT 檢定 14 vs DC 12：成功")

    print(f"\nnarrative_log 共 {len(ws.narrative_log)} 條")
    for i, e in enumerate(ws.narrative_log):
        print(f"  {i}: speaker={e['speaker']!r}, room={e['room_id']!r}, present={e['present']}, text={e['text'][:30]!r}")

    # ── 2. Verify visibility ──────────────────────────────────────────────────
    print("\n── log_for(thor) ──")
    for e in ws.log_for("thor"):
        print(f"  {e['speaker']}: {e['text'][:40]}")

    print("\n── log_for(civilian) ──")
    for e in ws.log_for("civilian"):
        print(f"  {e['speaker']}: {e['text'][:40]}")

    print("\n── log_all (GM 視角) ──")
    for e in ws.log_all():
        print(f"  {e['speaker']}: {e['text'][:40]}")

    # 老柯 should be present (he's in entry_corridor) so he sees everything pushed so far
    assert all("civilian" in e["present"] for e in ws.log_for("civilian")), \
        "civilian should see entry_corridor entries"

    # ── 3. Move to a different room, push event there ─────────────────────────
    ws.dungeon_map.current_room_id = "guard_room"
    ws.log_event("gm", "守衛室裡有兩隻地精。")
    ws.log_event("system", "凱恩 攻擊 哥布林：12 vs AC 13 命中")

    # civilian (still in entry_corridor) should NOT see guard_room entries
    civ_log = ws.log_for("civilian")
    guard_room_entries = [e for e in ws.narrative_log if e["room_id"] == "guard_room"]
    civ_sees_guard = [e for e in civ_log if e["room_id"] == "guard_room"]
    print(f"\nguard_room 共寫入 {len(guard_room_entries)} 條，civilian 看到 {len(civ_sees_guard)} 條（應為 0）")
    assert len(civ_sees_guard) == 0, "civilian must not see guard_room entries"

    # Goblins in guard_room should see those entries
    g1_log = ws.log_for("goblin_1")
    g1_sees_guard = [e for e in g1_log if e["room_id"] == "guard_room"]
    print(f"goblin_1 看到 guard_room {len(g1_sees_guard)} 條（應為 2）")
    assert len(g1_sees_guard) == 2, "goblin_1 should see guard_room entries"

    # And goblins should NOT see entry_corridor entries (they weren't there)
    g1_sees_entry = [e for e in g1_log if e["room_id"] == "entry_corridor"]
    print(f"goblin_1 看到 entry_corridor {len(g1_sees_entry)} 條（應為 0）")
    assert len(g1_sees_entry) == 0, "goblin_1 should not see entry_corridor entries"

    # ── 4. render_messages from thor's POV ────────────────────────────────────
    msgs = render_messages(ws, "thor")
    print("\n── render_messages(thor) ──")
    for m in msgs:
        print(f"  [{m['role']}] {m['content'][:50]}")

    # thor's own utterances should be `assistant`, all others `user`
    for m in msgs:
        assert m["role"] in ("user", "assistant")
    self_msgs = [m for m in msgs if m["role"] == "assistant"]
    assert all("握緊長劍" in m["content"] for m in self_msgs), \
        "thor's own line should appear as assistant"

    # ── 5. render_script (NPC combat / DC agent uses this) ────────────────────
    print("\n── render_script(civilian) ──")
    print(render_script(ws, "civilian"))

    # ── 6. Build a real NpcAgent and check _system() composes ─────────────────
    # Use stub model/url — we won't call LLM, just inspect the prompt
    npc_agents = build_npc_agents(ws, model="stub", base_url="http://x", backend="ollama")
    civ_agent = npc_agents["civilian"]

    sys_prompt = civ_agent._system()
    print(f"\n── 老柯 _system() 長度 {len(sys_prompt)} 字 ──")
    assert "老柯" in sys_prompt
    assert "知識邊界" in sys_prompt
    print("  OK包含角色名 + 知識邊界")

    # Switch to a force_reveal mode and verify template shifts
    civ_agent.force_reveal = "intimidate"
    sys_force = civ_agent._system()
    assert "屈服" in sys_force, "intimidate force should include 屈服"
    print("  OKforce_reveal='intimidate' → 包含 '屈服'")
    civ_agent.force_reveal = ""

    # pending_reveal mode
    civ_agent.pending_reveal = ["你已經採回月光草，謝謝你！"]
    sys_pend = civ_agent._system()
    assert "親口告訴對方" in sys_pend
    assert "月光草" in sys_pend
    print("  OKpending_reveal 注入劇情指示")
    civ_agent.pending_reveal = []

    print("\n=== ALL PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
