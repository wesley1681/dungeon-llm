"""Per-slot acceptance gates for the obs-v3 retrain wave, in one command.

Pipeline (mirrors the battle_master pilot):
  1. coarse pick + 1v1 guard      pick_team_snap.py <arch> 8 60 <snap_dir>
  2. big-sample slot judgement    eval_team_slot, >=1728 episodes/arm:
                                  best vs current-routing vs expert-in-slot
  3. level-sensitivity probe      probe_v3_features.py on the best snapshot

Prints a PASS/FAIL summary and the routing-table line to use if PASS.
Asym bucket check (eval_asym) is left as a separate step for swap candidates.

Usage: python scripts/gate_slot.py <arch> [snap_dir]
"""
from __future__ import annotations
import sys, os, re, subprocess, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

arch = sys.argv[1]
snap_dir = sys.argv[2] if len(sys.argv) > 2 else f"models/ppo_team_{arch}_v3"

PY = sys.executable

# ── 1. coarse pick + 1v1 guard ───────────────────────────────────────────────
print(f"=== gate_slot {arch} (snapshots: {snap_dir}) ===", flush=True)
pick_log = os.path.join(snap_dir, "pick.log")
pick_out = None
if os.path.exists(pick_log) and "--repick" not in sys.argv:
    cached = open(pick_log, encoding="utf-8").read()
    if "best:" in cached:
        print(f"(reusing existing {pick_log})")
        pick_out = cached
if pick_out is None:
    r = subprocess.run([PY, "scripts/pick_team_snap.py", arch, "8", "60", snap_dir],
                       capture_output=True, text=True, encoding="utf-8")
    pick_out = r.stdout
    with open(pick_log, "w", encoding="utf-8") as f:
        f.write(pick_out + ("\n[stderr]\n" + r.stderr if r.returncode else ""))
print(pick_out)
# pick prints e.g. "best: ppo_u0025 (models\ppo_team_x_v3\ppo_u0025.pt)  team WR=..."
m = (re.search(r"best:.*?\((\S+?\.pt)\)", pick_out)
     or re.search(r"best:\s+(\S+\.pt)", pick_out))
if not m:
    print("GATE FAIL — pick produced no best snapshot"); sys.exit(1)
best = m.group(1).replace("\\", "/")
guard_pass = "GUARD PASS" in pick_out

# ── 2. big-sample judgement (>=1728 eps/arm) ────────────────────────────────
from trpg.scenarios.archetypes import ARCHETYPE_ROLES
from eval_routed import DEFAULT_ROUTING, load_net
from train_team_ppo import eval_team_slot, ROLE_SLOT

slot = ROLE_SLOT[ARCHETYPE_ROLES[arch]]
buckets = [sorted(a for a, r0 in ARCHETYPE_ROLES.items() if r0 == "front"),
           sorted(a for a, r0 in ARCHETYPE_ROLES.items() if r0 == "striker"),
           sorted(a for a, r0 in ARCHETYPE_ROLES.items() if r0 == "support")]
n_comps = 1
for i, b in enumerate(buckets):
    if i != slot:
        n_comps *= len(b)
cells = n_comps * 3
games_big = math.ceil(1728 / cells)
print(f"big sample: {cells} cells x {games_big} games = {cells*games_big} eps/arm",
      flush=True)

cache = {}
mate_nets = {}
for a, path in DEFAULT_ROUTING.items():
    if a == arch:
        continue
    if path not in cache:
        cache[path] = load_net(path)
    mate_nets[a] = cache[path]

best_net = load_net(best)
cur_net = load_net(DEFAULT_ROUTING[arch])
wr_best = eval_team_slot(best_net, arch, slot, mate_nets, games=games_big)
print(f"  best       {best}  slot WR={wr_best:.1%}", flush=True)
wr_cur = eval_team_slot(cur_net, arch, slot, mate_nets, games=games_big)
print(f"  current    {DEFAULT_ROUTING[arch]}  slot WR={wr_cur:.1%}", flush=True)
wr_exp = eval_team_slot("expert", arch, slot, mate_nets, games=games_big)
print(f"  expert     slot WR={wr_exp:.1%}", flush=True)

n_eps = cells * games_big
sigma2 = 2 * 100 * math.sqrt(0.5 * 0.5 * 2 / n_eps)   # 2-sigma on a WR diff
d_cur = (wr_best - wr_cur) * 100
big_ok = d_cur >= -sigma2

with open(f"eval_results/{arch}_v3_slot_big.txt", "w", encoding="utf-8") as f:
    f.write(f"{os.path.basename(best):10s} slot WR = {wr_best:.1%}\n"
            f"current    slot WR = {wr_cur:.1%}\n"
            f"expert     slot WR = {wr_exp:.1%}\n"
            f"({n_eps} eps/arm; 2-sigma on diff +/-{sigma2:.1f}pp)\n")

# ── 3. level-sensitivity probe ───────────────────────────────────────────────
pr = subprocess.run([PY, "scripts/probe_v3_features.py", best, arch],
                    capture_output=True, text=True, encoding="utf-8")
print(pr.stdout[-1200:])
with open(f"eval_results/probe_v3_{arch}.txt", "w", encoding="utf-8") as f:
    f.write(pr.stdout)

# ── verdict ──────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"1v1 guard: {'PASS' if guard_pass else 'FAIL'}")
print(f"slot big-sample: best {wr_best:.1%} vs current {wr_cur:.1%} "
      f"(diff {d_cur:+.1f}pp, band ±{sigma2:.1f}) -> "
      f"{'OK' if big_ok else 'REGRESSION'}")
# bm-pilot precedent: a native-v3 snapshot is routed when it's AT PAR or
# better (within the 2-sigma band) — the wave's point is adopting the
# level-sensitive native-v3 nets wherever they don't regress.
do_nothing = (os.path.normpath(best)
              == os.path.normpath(DEFAULT_ROUTING[arch]))
if guard_pass and big_ok and not do_nothing:
    print(f"VERDICT: ROUTE  {arch} -> {best}")
elif guard_pass and big_ok:
    print(f"VERDICT: KEEP current routing (pick chose the do-nothing arm)")
else:
    print(f"VERDICT: DO NOT ROUTE")
