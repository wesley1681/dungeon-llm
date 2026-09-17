"""Headless tests for the 3-stage conversation resolution pipeline.

Covers the two game.py appliers (_apply_dialogue_events / _apply_quest_events)
and the DialogueDirector parse+filter logic, all with stubbed LLM seams so no
network is touched.
"""
from trpg.scenarios.dungeon import build_world_state, build_npc_agents
from trpg.game import GameSession
from trpg.llm.controllers import ActorController, HumanController
from trpg.llm.dialogue_flow import DialogueDirector, CheckResult, CheckPlan


class _StubTagAgent:
    # Only the connection attrs DialogueDirector.from_agent reads (all absent →
    # None, which is fine — the director is inert unless a stage is called).
    def generate_tags(self, actions, on_chunk=None):
        return "無"


def _session():
    ws = build_world_state()
    npc_agents = build_npc_agents(ws, model="stub", base_url="stub", backend="stub")
    session = GameSession(
        world_state=ws, gm=object(), tag_agent=_StubTagAgent(),
        thor_agent=object(), npc_agents=npc_agents,
    )
    session.controllers["thor"] = ActorController()
    for cid in npc_agents:
        session.controllers[cid] = ActorController()
    assert isinstance(session.controllers["kaine"], HumanController)
    return session, ws, npc_agents


# ── NpcAgent structural: attitude self-marking removed, new channels added ─────

def test_npc_agent_lost_attitude_self_marking():
    import trpg.llm.npc_agent as na
    assert not hasattr(na, "_MARKER_RE"), "[+/-] attitude marker must be gone"
    assert not hasattr(na, "_FORCE_DETAILS")
    _, _, npc_agents = _session()
    civ = npc_agents["civilian"]
    assert hasattr(civ, "revealed") and civ.revealed == []
    assert hasattr(civ, "pending_directives") and civ.pending_directives == []
    assert not hasattr(civ, "force_reveal")
    assert not hasattr(civ, "_skip_marker")


# ── Stage-2 appliers ──────────────────────────────────────────────────────────

def test_attitude_event_changes_attitude():
    session, ws, npc = _session()
    civ = npc["civilian"]; civ_char = ws.characters["civilian"]
    civ_char.attitude = 2
    combat = session._apply_dialogue_events(
        [{"event": "attitude", "delta": -1}], civ, civ_char, None)
    assert combat is False
    assert civ_char.attitude == 1, "attitude event must move attitude"


def test_attitude_event_clamps():
    session, ws, npc = _session()
    civ = npc["civilian"]; civ_char = ws.characters["civilian"]
    civ_char.attitude = 4
    session._apply_dialogue_events([{"event": "attitude", "delta": 5}], civ, civ_char, None)
    assert civ_char.attitude == 4, "attitude must clamp at 4"


def test_reveal_requires_won_check_and_is_sticky():
    session, ws, npc = _session()
    civ = npc["civilian"]; civ_char = ws.characters["civilian"]

    # Not won → concession dropped, nothing revealed.
    session._apply_dialogue_events(
        [{"event": "reveal", "indices": [0]}], civ, civ_char,
        CheckResult("說服", success=False, total=5, dc=12, actor="kaine"))
    assert civ.revealed == []
    assert civ.pending_directives == []

    # Won → secret 0 moves into the sticky revealed set + a directive is queued.
    won = CheckResult("威嚇", success=True, total=18, dc=12, actor="kaine")
    session._apply_dialogue_events([{"event": "reveal", "indices": [0, 2]}], civ, civ_char, won)
    assert len(civ.revealed) == 2
    assert civ._secrets[0] in civ.revealed and civ._secrets[2] in civ.revealed
    assert any("情報" in d for d in civ.pending_directives)


def test_no_check_blocks_concessions():
    session, ws, npc = _session()
    civ = npc["civilian"]; civ_char = ws.characters["civilian"]
    session._apply_dialogue_events(
        [{"event": "reveal", "indices": [0]}, {"event": "suggest_join"}],
        civ, civ_char, None)   # result=None → no check ran
    assert civ.revealed == []
    assert civ.pending_join_decision is False


def test_suggest_join_opens_recruit_decision():
    session, ws, npc = _session()
    civ = npc["civilian"]; civ_char = ws.characters["civilian"]
    won = CheckResult("說服", success=True, total=16, dc=12, actor="kaine")
    session._apply_dialogue_events([{"event": "suggest_join"}], civ, civ_char, won)
    assert civ.pending_join_decision is True
    assert any("加入" in d for d in civ.pending_directives)


def test_attack_event_starts_combat():
    session, ws, npc = _session()
    civ = npc["civilian"]; civ_char = ws.characters["civilian"]
    combat = session._apply_dialogue_events([{"event": "attack"}], civ, civ_char, None)
    assert combat is True, "attack event must signal the caller to end the conversation"
    assert civ_char.attitude == 0, "attacked NPC turns hostile"
    assert ws.combat is not None and ws.combat.active


# ── Stage-3 quest applier ─────────────────────────────────────────────────────

def test_quest_accept_complete_turnin_lifecycle():
    session, ws, npc = _session()
    q = ws.quests["moonlight_grass"]
    assert q.status == "inactive"

    session._apply_quest_events([{"event": "quest_accept", "id": "moonlight_grass"}])
    assert q.status == "active"

    session._apply_quest_events([{"event": "quest_complete", "id": "moonlight_grass"}])
    assert q.status == "completed", "narrative completion path active→completed"

    session._apply_quest_events([{"event": "quest_turnin", "id": "moonlight_grass"}])
    assert q.status == "turned_in"


def test_quest_event_bad_id_ignored():
    session, ws, npc = _session()
    session._apply_quest_events([{"event": "quest_accept", "id": "does_not_exist"}])
    # nothing crashes; real quest untouched
    assert ws.quests["moonlight_grass"].status == "inactive"


# ── DialogueDirector parse + filter (stubbed _ask / DC) ───────────────────────

def _director(ws, ask_returns):
    """A director whose _ask returns canned text keyed by debug_name substring,
    and whose DC is fixed."""
    d = DialogueDirector("m", "u", "b", ws)
    d._dc_agent.estimate = lambda **kw: 13
    def fake_ask(system, user, num_predict, temperature, debug_name):
        for key, val in ask_returns.items():
            if key in debug_name:
                return val
        return ""
    d._ask = fake_ask
    return d


def test_decide_check_parses_and_defaults_actor():
    _, ws, npc = _session()
    d = _director(ws, {"check": '{"check": "說服", "actor": "kaine"}'})
    plan = d.decide_check("我求你告訴我", npc["civilian"], ws)
    assert plan.check == "說服" and plan.dc == 13 and plan.actor == "kaine"

    # 'none' → no check
    d2 = _director(ws, {"check": '{"check": "none", "actor": ""}'})
    assert d2.decide_check("你好啊", npc["civilian"], ws).check is None

    # bogus actor falls back to a valid PC
    d3 = _director(ws, {"check": '{"check": "威嚇", "actor": "ghost"}'})
    assert d3.decide_check("再不說砍了你", npc["civilian"], ws).actor == "kaine"


def test_arrange_events_filters_unknown_and_parses_array():
    _, ws, npc = _session()
    d = _director(ws, {"event":
        '亂講 [{"event":"reveal","indices":[0]},{"event":"bogus"},{"event":"attitude","delta":-1}] 尾巴'})
    res = CheckResult("威嚇", True, 18, 12, "kaine")
    evs = d.arrange_events("威脅他", res, npc["civilian"], ws)
    kinds = [e["event"] for e in evs]
    assert "reveal" in kinds and "attitude" in kinds
    assert "bogus" not in kinds, "unknown events must be filtered out"


def test_arrange_events_quest_accept_gated_by_offered():
    _, ws, npc = _session()
    q = ws.quests["moonlight_grass"]   # giver == civilian, inactive
    # Not offered yet → accept is out of scope → dropped even if the model emits it
    # (this is the structural guard against same-turn false-accept).
    q.offered = False
    d = _director(ws, {"event": '[{"event":"quest_accept","id":"moonlight_grass"}]'})
    assert d.arrange_events("好我接", None, npc["civilian"], ws) == []

    # Offered → accept is in scope; valid id passes, bogus id dropped.
    q.offered = True
    d2 = _director(ws, {"event":
        '[{"event":"quest_accept","id":"moonlight_grass"},{"event":"quest_accept","id":"nope"}]'})
    evs = d2.arrange_events("好我接", None, npc["civilian"], ws)
    assert [e["id"] for e in evs] == ["moonlight_grass"]


def test_arrange_events_tolerates_quest_id_key():
    # Regression: the model sometimes emits "quest_id" instead of "id"; the parser
    # must tolerate it and normalise to "id" (else a valid accept silently drops).
    _, ws, npc = _session()
    ws.quests["moonlight_grass"].offered = True
    d = _director(ws, {"event": '[{"event":"quest_accept","quest_id":"moonlight_grass"}]'})
    evs = d.arrange_events("我決定接下老柯的任務", None, npc["civilian"], ws)
    assert evs == [{"event": "quest_accept", "id": "moonlight_grass"}]


def test_arrange_events_turnin_gated_by_completed():
    _, ws, npc = _session()
    q = ws.quests["moonlight_grass"]   # giver == civilian
    d = _director(ws, {"event": '[{"event":"quest_turnin","id":"moonlight_grass"}]'})
    # Not completed → turn-in out of scope → dropped.
    q.status = "active"
    assert d.arrange_events("我來領獎", None, npc["civilian"], ws) == []
    # Completed → turn-in in scope.
    q.status = "completed"
    evs = d.arrange_events("我來領獎", None, npc["civilian"], ws)
    assert [e["id"] for e in evs] == ["moonlight_grass"]


def test_judge_completion_detects_and_validates_id():
    _, ws, npc = _session()
    ws.quests["moonlight_grass"].status = "active"
    d = _director(ws, {"complete":
        '[{"event":"quest_complete","id":"moonlight_grass"},{"event":"quest_complete","id":"nope"}]'})
    evs = d.judge_completion("你答應停手嗎", "好啦我不收保護費了", ws)
    assert [e["id"] for e in evs] == ["moonlight_grass"], "invalid id must be dropped"


def test_judge_completion_skips_when_no_active_quests():
    _, ws, npc = _session()
    # moonlight_grass is inactive; no active quests → empty scope → skip LLM call.
    called = {"n": 0}
    d = DialogueDirector("m", "u", "b", ws)
    def spy_ask(*a, **k):
        called["n"] += 1
        return "[]"
    d._ask = spy_ask
    assert d.judge_completion("嗨", "哼", ws) == [] and called["n"] == 0


# ── Integration: _run_conversation wires the 3 stages together ────────────────

class _StubNpcCtrl(ActorController):
    def take_npc_opening(self, char):
        return "你是誰？"
    def take_npc_response(self, char):
        return "好…我告訴你守衛室的事，也接下你的請求。"


class _StubDirector:
    """Canned stage outputs — a guaranteed-success check (dc=0), a reveal, and a
    stage-2 quest accept — so the full pipeline path is exercised deterministically."""
    def decide_check(self, player_text, npc, ws):
        return CheckPlan(check="說服", dc=0, actor="kaine")
    def arrange_events(self, player_text, result, npc, ws):
        return [{"event": "reveal", "indices": [0]},
                {"event": "quest_accept", "id": "moonlight_grass"}]
    def judge_completion(self, player_text, npc_response, ws):
        return []


def test_run_conversation_chains_all_stages():
    session, ws, npc = _session()
    session.controllers["civilian"] = _StubNpcCtrl()
    session.dialogue = _StubDirector()
    civ = npc["civilian"]

    # 凱恩 speaks once (triggers the pipeline), then leaves.
    session.submit_player_input("我求求你，把守衛室的情況告訴我")
    session.submit_player_input("離開")
    outcome = session._run_conversation("civilian")

    assert not outcome.game_over
    # Stage 1+2: dc=0 check always succeeds → reveal concession granted (sticky).
    assert civ._secrets[0] in civ.revealed
    # Stage 3: quest accepted from the NPC's reply.
    assert ws.quests["moonlight_grass"].status == "active"
    # Leaving still clears the trigger queue (no regression of the leave fix).
    assert session._tag_actions == []


class _OfferingNpcCtrl(ActorController):
    """The NPC voices the quest offer in its OPENING line (as 老柯 does at 戒備+)."""
    def take_npc_opening(self, char):
        return "求求你們，能幫我採三朵月光草嗎？只要拿回來，我的藥草都分你們！"
    def take_npc_response(self, char):
        return "太好了，謝謝你們！"


class _GateFaithfulDirector:
    """Mirrors the REAL arrange_events offered-gate so the regression actually bites:
    it emits quest_accept only when the quest is already marked offered."""
    def decide_check(self, player_text, npc, ws):
        return CheckPlan(check=None, dc=0, actor="kaine")
    def arrange_events(self, player_text, result, npc, ws):
        q = ws.quests["moonlight_grass"]
        if (q.giver_id == getattr(npc, "char_id", "")
                and q.status == "inactive" and q.offered):
            return [{"event": "quest_accept", "id": "moonlight_grass"}]
        return []
    def judge_completion(self, player_text, npc_response, ws):
        return []


def test_opening_offer_unlocks_first_turn_accept():
    # Regression: the offer is voiced in the NPC opening (take_npc_opening), which
    # runs BEFORE the conversation loop. If `offered` is marked only after in-loop
    # responses, the player's very first "I accept" lands one exchange too early and
    # silently drops. Marking offered right after the opening fixes it.
    session, ws, npc = _session()
    assert npc["civilian"].attitude >= 1          # 老柯 starts 戒備 → voices the offer
    q = ws.quests["moonlight_grass"]
    assert q.giver_id == "civilian" and q.status == "inactive" and not q.offered

    session.controllers["civilian"] = _OfferingNpcCtrl()
    session.dialogue = _GateFaithfulDirector()
    session.submit_player_input("好，我們接下你的委託")
    session.submit_player_input("離開")
    session._run_conversation("civilian")

    assert q.status == "active", (
        "a quest offered in the NPC opening must be acceptable on the player's "
        "first reply — offered must be marked right after the opening")


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-q"]))
