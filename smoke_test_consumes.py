"""Smoke test for consumes-from-JSON refactor.

Verifies:
  - arbiter._extract_json fallback fills in `consumes` for known action types
  - consume_resources decrements the correct slots based on action.consumes
  - MOVE consumes movement equal to result.distance
"""
import sys
sys.path.insert(0, ".")

from trpg.llm.arbiter import ArbiterAgent
from trpg.engine.combat import consume_resources


def main() -> int:
    # ── 1. _extract_json fallback fills consumes for known types ─────────────
    arb = ArbiterAgent(model="dummy")

    # Missing consumes for ATTACK → fallback adds ["action"]
    result = arb._extract_json('{"valid": true, "type": "ATTACK", "attacker": "x"}', "")
    assert result.get("consumes") == ["action"], f"got {result.get('consumes')}"

    # MOVE → fallback adds ["movement"]
    result = arb._extract_json('{"valid": true, "type": "MOVE", "character": "x"}', "")
    assert result.get("consumes") == ["movement"]

    # DODGE → ["action"]
    result = arb._extract_json('{"valid": true, "type": "DODGE", "character": "x"}', "")
    assert result.get("consumes") == ["action"]

    # Explicit consumes wins over fallback
    result = arb._extract_json(
        '{"valid": true, "type": "ATTACK", "attacker": "x", "consumes": ["bonus_action"]}', ""
    )
    assert result.get("consumes") == ["bonus_action"]

    # Invalid action — no fallback (keep as-is)
    result = arb._extract_json('{"valid": false, "reason": "x"}', "")
    assert "consumes" not in result
    print("arbiter consumes fallback: OK")

    # ── 2. consume_resources decrements action slot ──────────────────────────
    resources = {"action": 1, "movement": 9.0}
    consume_resources(resources, {"consumes": ["action"]}, {})
    assert resources["action"] == 0
    assert resources["movement"] == 9.0

    # ── 3. consume_resources decrements movement by distance ─────────────────
    resources = {"action": 1, "movement": 9.0}
    consume_resources(resources, {"consumes": ["movement"]}, {"distance": 4.5})
    assert resources["movement"] == 4.5
    assert resources["action"] == 1

    # ── 4. Multi-slot consume ────────────────────────────────────────────────
    resources = {"action": 1, "bonus_action": 1, "movement": 9.0}
    consume_resources(resources, {"consumes": ["action", "bonus_action"]}, {})
    assert resources["action"] == 0
    assert resources["bonus_action"] == 0
    print("consume_resources: OK")

    print("\n=== ALL CONSUMES TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
