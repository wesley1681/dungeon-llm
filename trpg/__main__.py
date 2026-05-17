"""統一入口。

用法：
    python -m trpg                  → 網頁版（預設）
    python -m trpg --mode web       → 網頁版
    python -m trpg --mode terminal  → 終端機版
    python -m trpg --mode claude    → Claude Code 測試模式（終端機 + CLAUDE_TEST=True）
"""
import argparse
import sys

sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(prog="trpg", description="TRPG LLM Engine")
    parser.add_argument(
        "--mode",
        choices=["web", "terminal", "claude"],
        default="web",
        help="啟動模式（預設：web）",
    )
    args = parser.parse_args()

    if args.mode == "web":
        from . import web
        web.main()
    elif args.mode == "terminal":
        from . import cli as game_main
        game_main.run_game()
    elif args.mode == "claude":
        from . import cli as game_main
        game_main.CLAUDE_TEST = True
        game_main.run_game()


if __name__ == "__main__":
    main()
