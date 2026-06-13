"""Aggregate an eval_v2 result JSON: initiative bias, per-arch wr, loss structure."""
import sys, json
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "eval_results/v19_F.json"
d = json.load(open(path))
res = d["result"]
archs = list(res.keys())

all_wr, all_first, all_oppfirst = [], [], []
opp_hp_on_loss, agent_hp_on_win, rounds = [], [], []
for a in archs:
    for o in archs:
        c = res[a][o]
        all_wr.append(c["win_rate"])
        if "win_rate_agent_first" in c:
            all_first.append(c["win_rate_agent_first"])
            all_oppfirst.append(c["win_rate_opp_first"])
        if "avg_opp_hp_frac_on_loss" in c:
            opp_hp_on_loss.append(c["avg_opp_hp_frac_on_loss"])
        if "avg_agent_hp_frac_on_win" in c:
            agent_hp_on_win.append(c["avg_agent_hp_frac_on_win"])
        if "avg_rounds" in c:
            rounds.append(c["avg_rounds"])

print(f"=== {path} (label={d.get('label')}) ===")
print(f"OVERALL win_rate         : {np.mean(all_wr):.1%}")
if all_first:
    print(f"  when AGENT goes first  : {np.mean(all_first):.1%}")
    print(f"  when OPP   goes first  : {np.mean(all_oppfirst):.1%}")
    print(f"  initiative gap (first-opp): {np.mean(all_first)-np.mean(all_oppfirst):+.1%}")
print(f"avg opp HP frac on LOSS   : {np.mean(opp_hp_on_loss):.1%}  "
      f"(high => agent dies w/o damaging opp; low => close fight/timeout)")
print(f"avg agent HP frac on WIN  : {np.mean(agent_hp_on_win):.1%}")
print(f"avg rounds per game       : {np.mean(rounds):.1f}")

print("\nper-archetype win rate (as agent, averaged over 12 opps):")
for a in archs:
    wr = np.mean([res[a][o]["win_rate"] for o in archs])
    of = np.mean([res[a][o].get("win_rate_agent_first", res[a][o]["win_rate"]) for o in archs])
    oo = np.mean([res[a][o].get("win_rate_opp_first", res[a][o]["win_rate"]) for o in archs])
    print(f"  {a:18s} {wr:5.1%}   first={of:5.1%} oppfirst={oo:5.1%} gap={of-oo:+5.1%}")
