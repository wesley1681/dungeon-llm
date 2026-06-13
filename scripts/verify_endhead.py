"""VERIFY (with data) whether a class-blind end_head is a binding constraint.

The end_head reads only `end_features` (8 dims, NO archetype). If the SCRIPTED
EXPERT's end decision at a FIXED end_features bucket differs materially across
archetypes, then a function that maps end_features -> P(end) provably cannot fit
all classes: it must collapse to a blend, ending combo classes too early (or
wizards too late). That is the falsifiable test for "end_head archetype-blind is
a real binding constraint", per the two architecture/obs review subagents and
the project's data-before-surgery rule.

For each archetype we drive the EXPERT through env.step (the real action path),
and at every AGENT decision record:
   bucket = (action_remaining 0/1, bonus_remaining 0/1, movement>0 0/1,
             hp_bucket low/mid/high, enemy_alive)   <- coarsened end_features
   ended  = expert chose END this sub-action
Then per bucket we print P(end) per class and the spread across classes.
The critical bucket is action_SPENT (a=0) with bonus/movement left: do combo
classes keep going (chain) while others stop?
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from collections import defaultdict
import numpy as np
from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.obs import end_features
from trpg.engine.combat_policy import make_archetype_policy
from trpg.rl.action import encode_action

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def bucket(ef: np.ndarray) -> tuple:
    hp = ef[0]
    hp_b = 0 if hp < 0.34 else (1 if hp < 0.67 else 2)
    return (int(ef[1] > 0.5),          # action remaining
            int(ef[2] > 0.5),          # bonus remaining
            int(ef[3] > 0.05),         # movement remaining
            hp_b,                      # hp bucket
            int(ef[5] > 0.5))          # enemy alive


# bucket -> arch -> [n_end, n_total]
stats: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))

for arch in ARCHETYPE_LIST:
    expert = make_archetype_policy(arch)
    for opp in ARCHETYPE_LIST:
        base = hash(f"{arch}_{opp}") & 0xFFFFFF
        for g in range(GAMES):
            env = CombatEnvV2(seed=base + g, n_agents=1, n_opps=1)
            obs, _ = env.reset(agent_archs=[arch], opp_archs=[opp], level=5)
            aid = env.agent_ids[0]
            done = False
            while not done:
                actor = env.current_agent_id
                if actor == aid:
                    ef = end_features(env.ws, aid, env.resources)
                    b = bucket(ef)
                    a = env.ws.characters[aid]
                    dec = expert.decide(aid, a, env.ws, env.resources,
                                        env.ws.combat.round_number)
                    ended = (dec.action is None or dec.fled)
                    stats[b][arch][0] += int(ended)
                    stats[b][arch][1] += 1
                    if ended:
                        act = [0, 0, 0]
                    else:
                        act = list(encode_action(dec.action, env.ws, aid))
                    obs, _, term, trunc, _ = env.step(act)
                else:
                    obs, _, term, trunc, _ = env.step([0, 0, 0])
                done = term or trunc

# Report buckets seen by >=5 archetypes with >=20 samples each, sorted by spread.
print(f"=== expert end-rate by (end_features bucket, archetype), {GAMES} games/matchup ===")
print("bucket = (action, bonus, move, hp[0=low/1=mid/2=high], enemy_alive)\n")
rows = []
for b, per_arch in stats.items():
    valid = {a: v for a, v in per_arch.items() if v[1] >= 20}
    if len(valid) < 5:
        continue
    rates = {a: v[0] / v[1] for a, v in valid.items()}
    spread = max(rates.values()) - min(rates.values())
    rows.append((spread, b, rates, valid))

rows.sort(reverse=True)
for spread, b, rates, valid in rows[:12]:
    tot = sum(v[1] for v in valid.values())
    print(f"bucket {b}  spread={spread:.0%}  n={tot}")
    for a in sorted(rates, key=lambda x: rates[x]):
        print(f"    {a:18s} P(end)={rates[a]:5.0%}  (n={valid[a][1]})")
    print()

print("INTERPRETATION:")
print("  If high-spread buckets exist (combo classes P(end) far below wizards at")
print("  the SAME bucket), a class-blind end_head cannot represent both -> adding")
print("  archetype to end_features is a data-backed fix. If all classes agree")
print("  (low spread everywhere), end_head blindness is NOT the binding issue.")
