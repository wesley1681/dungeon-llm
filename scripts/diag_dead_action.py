"""死動作審計：模型會不會選到引擎 execute_action 直接拒(ERROR)的動作？

chill_touch BUG 的那一類——available_skills 提供了不可執行的動作、sampling mask
沒攔住、模型選了、該回合 action 白白浪費(agent 席)或整回合空轉(對手席)。已在
source 修掉「非施法者拿到職業法術」那一支；本審計用**經驗**證明這一類在任意
kit 上都關閉了(std12/synth/縫合/怪)，並把任何殘留的 (skill, 原因) 抓出來。

方法：monkeypatch env_v2.execute_action，只統計 **agent 席**(actor id 以 "agent"
起頭)所選動作 execute 回 ERROR 的次數與原因；對手席不計。模型駕駛 agent 席，
對手由 env 內部腳本驅動(稱職基準)。全域骰配對。

用法: python scripts/diag_dead_action.py models/unified/uni_v9.pt --games 2
"""
from __future__ import annotations
import sys, os, argparse, random
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import trpg.rl.env_v2 as env_v2

# ── monkeypatch execute_action to tally AGENT-seat dead actions ───────────────
_real_exec = env_v2.execute_action
_TALLY: Counter = Counter()       # (skill_id, message) -> count  (agent only)
_AGENT_EXEC = [0]                 # total agent execute calls (rate denominator)


def _wrap(action, ws):
    r = _real_exec(action, ws)
    actor_id = (action.get("attacker") or action.get("caster")
                or action.get("character"))
    if isinstance(actor_id, str) and actor_id.startswith("agent"):
        _AGENT_EXEC[0] += 1
        if isinstance(r, dict) and r.get("type") == "ERROR":
            _TALLY[(action.get("skill_id", "?"),
                    str(r.get("message", "?"))[:60])] += 1
    return r


env_v2.execute_action = _wrap

# import AFTER patch so run_combat's env.step resolves the wrapped name
from diag_degen_audit import run_combat, load           # noqa: E402
from eval_generalize import build_buckets, resolve_levels  # noqa: E402
from synth_identity import STANDARD_IDS                 # noqa: E402

OPP_PANEL = ["battle_master", "champion", "evocation", "vengeance"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--games", type=int, default=2,
                    help="games per (identity, opponent) pairing")
    ap.add_argument("--buckets", nargs="*",
                    default=["std12", "synth", "chimera_cls", "monster",
                             "chimera_mon"])
    ap.add_argument("--synth_n", type=int, default=12)
    ap.add_argument("--level", type=int, default=6)
    args = ap.parse_args()

    net = load(args.ckpt); net.eval()
    buckets = build_buckets(args.buckets, args.synth_n, 20260701)
    print(f"ckpt={args.ckpt}  games/pairing={args.games}  "
          f"opp_panel={OPP_PANEL}\n", flush=True)

    grand_err = grand_exec = 0
    for bname, idents in buckets.items():
        _TALLY.clear(); _AGENT_EXEC[0] = 0
        for ident in idents:
            m_lvl, o_lvl = resolve_levels(ident, args.level)
            for opp in OPP_PANEL:
                for k in range(args.games):
                    key = f"dead|{ident}|{opp}|{k}"
                    try:
                        run_combat(net, ident, [opp], m_lvl, o_lvl, key)
                    except Exception as ex:
                        print(f"  !! {ident} vs {opp}: {type(ex).__name__}: "
                              f"{str(ex)[:80]}")
        err = sum(_TALLY.values()); ex = _AGENT_EXEC[0]
        grand_err += err; grand_exec += ex
        rate = err / max(1, ex)
        print(f"== {bname:12s}  agent_execs={ex:6d}  DEAD={err:4d}  "
              f"({rate:.2%}) ==")
        for (sk, msg), c in _TALLY.most_common(12):
            print(f"     {c:4d}  {sk:20s} | {msg}")
    print(f"\nOVERALL agent_execs={grand_exec}  DEAD={grand_err}  "
          f"({grand_err/max(1,grand_exec):.2%})")


if __name__ == "__main__":
    main()
