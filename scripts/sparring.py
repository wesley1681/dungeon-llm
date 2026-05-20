"""Interactive sparring sandbox — play vs a model checkpoint.

Usage:
    python scripts/sparring.py --model models/ppo_best.pt
    python scripts/sparring.py --model expert:berserker
    python scripts/sparring.py --model random

The wizard then asks for your archetype, custom skill edits, positions, terrain.
Enter on each prompt accepts the default.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import argparse
from trpg.engine.vec2 import Vec2
from trpg.scenarios.archetypes import ARCHETYPE_FACTORIES
from trpg.sandbox.setup import build_world_state, apply_loadout, list_catalog
from trpg.sandbox.terrain import TERRAIN_PRESETS
from trpg.sandbox.policy_loader import load_policy
from trpg.sandbox.frontend import ConsoleFrontend
from trpg.sandbox.driver import run_combat


def _ask(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}] > ").strip()
    return raw or default


def _ask_position(label: str, default: Vec2) -> Vec2:
    while True:
        raw = _ask(f"{label} 座標 (x y)", f"{default.x:.0f} {default.y:.0f}")
        parts = raw.split()
        if len(parts) != 2:
            print(f"  需要兩個數字 (x y)，收到: {raw!r}")
            continue
        try:
            return Vec2(float(parts[0]), float(parts[1]))
        except ValueError:
            print(f"  無效座標: {raw!r}")


def _ask_int(prompt: str, default: str) -> int:
    while True:
        raw = _ask(prompt, default)
        try:
            return int(raw)
        except ValueError:
            print(f"  需要整數，收到: {raw!r}")


def _ask_archetype(label: str, default: str) -> str:
    print(f"  可用職業: {', '.join(ARCHETYPE_FACTORIES.keys())}")
    while True:
        choice = _ask(f"{label} 職業", default)
        if choice in ARCHETYPE_FACTORIES:
            return choice
        print(f"  無效職業: {choice!r}")


def _ask_loadout_edits(label: str, char) -> None:
    """Optionally edit the character's skill loadout."""
    if _ask(f"自訂 {label} 技能組?", "n").lower() not in ("y", "yes"):
        return
    catalog = list_catalog()
    print("\n  目前技能:")
    print(f"    武器: {[w.name for w in char.weapons]}")
    print(f"    法術: {char.spells}")
    print(f"    特性: {char.known_abilities}")
    print("\n  指令: a <skill_id> 新增 / r <skill_id> 移除 / l 列出 catalog / d 完成")
    while True:
        raw = input("  > ").strip().split(maxsplit=1)
        if not raw:
            continue
        cmd = raw[0]
        if cmd == "d":
            return
        if cmd == "l":
            for kind, items in catalog.items():
                print(f"    {kind}: {', '.join(items[:20])}"
                      + (" …" if len(items) > 20 else ""))
            continue
        if cmd in ("a", "r") and len(raw) > 1:
            try:
                if cmd == "a":
                    apply_loadout(char, add=[raw[1]], remove=[])
                else:
                    apply_loadout(char, add=[], remove=[raw[1]])
                print(f"  ✓ {raw[0]} {raw[1]}")
            except ValueError as e:
                print(f"  ✗ {e}")
            continue
        print(f"  未知指令: {raw}")


def _ask_terrain(default: str) -> str:
    print(f"  可用地形: {', '.join(TERRAIN_PRESETS.keys())}")
    while True:
        t = _ask("地形", default)
        if t in TERRAIN_PRESETS:
            return t
        print(f"  無效地形: {t!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparring sandbox")
    parser.add_argument("--model", required=True,
                        help="path/to/checkpoint.pt | expert:<arch> | random")
    parser.add_argument("--max-rounds", type=int, default=20)
    args = parser.parse_args()

    print("=== Sparring Sandbox 設定 ===\n")
    agent_arch = _ask_archetype("你的", "evocation")
    opponent_arch = _ask_archetype("對手", "berserker")
    level = _ask_int("等級", "5")
    agent_pos = _ask_position("你的", Vec2(8.0, 15.0))
    opp_pos = _ask_position("對手", Vec2(22.0, 15.0))
    terrain = _ask_terrain("empty")

    ws = build_world_state(
        agent_arch=agent_arch, opponent_arch=opponent_arch, level=level,
        agent_pos=agent_pos, opp_pos=opp_pos, terrain=terrain,
    )
    _ask_loadout_edits("你的", ws.characters["agent"])
    _ask_loadout_edits("對手", ws.characters["opponent"])

    print(f"\n載入對手模型: {args.model}")
    opp_policy = load_policy(args.model)
    frontend = ConsoleFrontend()

    print("\n=== 戰鬥開始 ===\n")
    outcome = run_combat(ws, frontend, opp_policy, max_rounds=args.max_rounds)
    frontend.render(ws, "agent", "opponent")
    print(f"\n結果: {outcome['outcome']}  (rounds={outcome['rounds']})")


if __name__ == "__main__":
    main()
