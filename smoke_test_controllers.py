"""Smoke test for ActorController abstraction.

Verifies:
  - ActorDecision dataclass shape
  - HumanController reads from input callback and detects end/quit
  - on_invalid_action defaults to None for non-Human controllers
"""
import sys
import queue
sys.path.insert(0, ".")

from trpg.llm.controllers import (
    ActorController, ActorDecision, HumanController,
)
from trpg.engine.combat import CombatContext
from trpg.engine.character import Character, Stats
from trpg.engine.items import WEAPON_DEFS


def make_char() -> Character:
    return Character(
        name="凱恩", race="人類", class_="盜賊", level=2,
        stats=Stats(STR=10, DEX=14, CON=12, INT=10, WIS=10, CHA=10),
        hp=15, max_hp=15, ac=13,
        weapons=[WEAPON_DEFS["短劍"]],
    )


def make_ctx() -> CombatContext:
    return CombatContext(
        round_num=1, actor_id="aria", actor_position=0.0,
        weapons_str="短劍", allies_str="無", enemies_str="哥布林",
        enemies={"goblin_1": "哥布林"}, resources={"action": 1, "movement": 9.0},
    )


def main() -> int:
    # ── 1. ActorDecision default ─────────────────────────────────────────────
    d = ActorDecision()
    assert d.description == "" and not d.ended and not d.fled and not d.quit
    print("ActorDecision defaults: OK")

    # ── 2. Base ActorController.on_invalid_action returns None ───────────────
    base = ActorController()
    assert base.on_invalid_action("reason", "suggestion") is None
    print("ActorController.on_invalid_action default None: OK")

    # ── 3. HumanController: normal input ─────────────────────────────────────
    q = queue.Queue()
    q.put("我衝向哥布林")
    emitted = []
    ctrl = HumanController(
        "aria",
        get_input=lambda: q.get(),
        emit_event=emitted.append,
        end_inputs={"結束", "end"},
    )
    char = make_char()
    decision = ctrl.take_sub_action(char=char, ctx=make_ctx())
    assert decision.description == "我衝向哥布林"
    assert not decision.ended and not decision.quit
    # Should have emitted at least one CombatPrompt event
    assert len(emitted) >= 1
    print("HumanController normal input: OK")

    # ── 4. HumanController: end keyword ──────────────────────────────────────
    q = queue.Queue()
    q.put("結束")
    ctrl = HumanController("aria", lambda: q.get(), emitted.append,
                           end_inputs={"結束", "end"})
    decision = ctrl.take_sub_action(char=make_char(), ctx=make_ctx())
    assert decision.ended and decision.description == ""
    print("HumanController end keyword: OK")

    # ── 5. HumanController: quit (None from get_input) ───────────────────────
    q = queue.Queue()
    q.put(None)
    ctrl = HumanController("aria", lambda: q.get(), emitted.append,
                           end_inputs={"結束"})
    decision = ctrl.take_sub_action(char=make_char(), ctx=make_ctx())
    assert decision.quit
    print("HumanController quit: OK")

    # ── 6. HumanController: on_invalid_action returns retry decision ─────────
    q = queue.Queue()
    q.put("換個說法")
    ctrl = HumanController("aria", lambda: q.get(), emitted.append,
                           end_inputs={"結束"})
    retry = ctrl.on_invalid_action("reason", "suggestion")
    assert retry is not None and retry.description == "換個說法"
    print("HumanController on_invalid_action retry: OK")

    print("\n=== ALL CONTROLLER TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
