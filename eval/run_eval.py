"""Combat reasoning eval runner.

Scenario files in eval/scenarios/*.txt contain ONLY the state block
(persona + 戰鬥狀態 + 資源 + 盟友/敵人). The action menu and reasoning
instruction are loaded from eval/footers/ and appended at run time so
prompts can be tuned without touching scenarios.

Footers:
  single.txt           — single-action menu (used when scenario has spells)
  single_no_spells.txt — single-action menu (used when scenario has none)
  plan.txt             — full-turn plan instruction (mode=plan, all scenarios)

The single footer is auto-picked by checking whether "可用法術：" appears
in the scenario state.

Outputs to eval/results/<timestamp>_<mode>/:
  <scenario>.md   — N runs concatenated + manual-scoring section
  summary.md      — table across scenarios
"""
import argparse
import datetime
import pathlib
import sys
import time

# Ensure CJK output is readable when running on Windows (cp950 otherwise mangles it).
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from trpg.llm.backend import complete_chat  # noqa: E402

EVAL_DIR = pathlib.Path(__file__).resolve().parent
SCENARIOS_DIR = EVAL_DIR / "scenarios"
FOOTERS_DIR = EVAL_DIR / "footers"
RESULTS_DIR = EVAL_DIR / "results"

def _pick_footer(state: str, mode: str) -> pathlib.Path:
    if mode == "plan":
        return FOOTERS_DIR / "plan.txt"
    # single: spell-aware menu if scenario lists spells, else trimmed menu
    has_spells = "可用法術：" in state
    return FOOTERS_DIR / ("single.txt" if has_spells else "single_no_spells.txt")


def load_scenario(path: pathlib.Path, mode: str) -> str:
    """Read scenario state, append the appropriate footer."""
    state = path.read_text(encoding="utf-8").rstrip()
    footer = _pick_footer(state, mode).read_text(encoding="utf-8")
    return state + "\n\n" + footer


def run_one(prompt: str, model: str, base_url: str, options: dict) -> str:
    messages = [{"role": "user", "content": prompt}]
    return complete_chat(base_url, model, messages, options,
                         backend="ollama", timeout=180)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["single", "plan"], default="single",
                   help="single = pick one sub-action; plan = full turn plan")
    p.add_argument("--n", type=int, default=5, help="Runs per scenario")
    p.add_argument("--model", default="gemma4:26b")
    p.add_argument("--temp", type=float, default=0.85)
    p.add_argument("--num-predict", type=int, default=1024)
    p.add_argument("--base-url", default="http://localhost:11434")
    p.add_argument("--scenarios", default="",
                   help="Comma-separated prefix filter, e.g. 'low,medium_02'")
    p.add_argument("--dry-run", action="store_true",
                   help="Print assembled prompts; do not call the model")
    return p.parse_args()


def select_scenarios(filter_str: str) -> list[pathlib.Path]:
    all_files = sorted(SCENARIOS_DIR.glob("*.txt"))
    if not filter_str:
        return all_files
    wants = [w.strip() for w in filter_str.split(",") if w.strip()]
    return [s for s in all_files if any(s.stem.startswith(w) for w in wants)]


def build_scoring_block(n: int) -> list[str]:
    lines = ["## 你的評分（手填）", ""]
    for i in range(1, n + 1):
        lines.append(f"Run {i}: [ ] ✓  [ ] ✗   理由：")
    lines += ["", f"**命中：__/{n}**", "**整體觀察：**"]
    return lines


def main():
    args = parse_args()
    options = {"temperature": args.temp, "num_predict": args.num_predict}
    scenarios = select_scenarios(args.scenarios)
    if not scenarios:
        print("No scenarios matched.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        for s in scenarios:
            print(f"\n========== {s.stem} ({args.mode}) ==========\n")
            print(load_scenario(s, args.mode))
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = RESULTS_DIR / f"{timestamp}_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(scenarios)} scenarios × {args.n} runs "
          f"({args.mode} mode) → {out_dir}")
    print(f"Model: {args.model}  Temp: {args.temp}  "
          f"num_predict: {args.num_predict}\n")

    for scenario in scenarios:
        prompt = load_scenario(scenario, args.mode)
        lines = [
            f"# {scenario.stem}",
            f"model: {args.model} | temp: {args.temp} | "
            f"num_predict: {args.num_predict} | n: {args.n} | mode: {args.mode}",
            f"timestamp: {timestamp}",
            "",
        ]
        for i in range(1, args.n + 1):
            print(f"  {scenario.stem}  run {i}/{args.n} ... ",
                  end="", flush=True)
            t0 = time.time()
            try:
                resp = run_one(prompt, args.model, args.base_url, options)
            except Exception as e:
                resp = f"[ERROR: {e}]"
            print(f"{time.time() - t0:.1f}s")
            lines += [f"## Run {i}", "", resp, "", "---", ""]
        lines += build_scoring_block(args.n)
        (out_dir / f"{scenario.stem}.md").write_text(
            "\n".join(lines), encoding="utf-8")
        print(f"  → {scenario.stem}.md\n")

    summary = [
        "# Eval Summary",
        f"timestamp: {timestamp} | mode: {args.mode}",
        f"model: {args.model} | temp: {args.temp} | "
        f"num_predict: {args.num_predict} | n: {args.n}",
        "",
        "| Scenario | 命中 | 觀察 |",
        "|----------|------|------|",
    ]
    for scenario in scenarios:
        summary.append(f"| {scenario.stem} | __/{args.n} | |")
    (out_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"\nDone. Summary: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
