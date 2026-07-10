"""Phase 2b — fit the StrengthModel currency on scripted games and VALIDATE that
its form (per-atom rating + body-count g[k]) can reproduce the Phase 1 balance
surface it must serve. If a linear-in-logits additive model can't fit champion
L2–L8 + counts, no matchmaker built on it will balance — so this is the gate
before wiring it into training (Phase 3).

Checks:
  1. breakpoint    r[champ@L5] ≫ r[champ@L4]   (Extra Attack cliff is learned)
  2. action econ   g[2] < g[3]  and both > 0   (more bodies help beyond the sum)
  3. reproduction  model.win_prob vs freshly-measured score on the Phase 1
                   concentration cases (equal-Σlevel, must swing 0.11↔0.94)
  4. calibration   held-out: predicted P binned vs actual win rate
"""
from __future__ import annotations
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from trpg.rl.matchup import StrengthModel, atom_key
from exp_matchup_calib import run_scripted_game, matchup_score

CHAMP = "champion"
LEVELS = (2, 3, 4, 5, 6, 7, 8)


def champ_atoms(levels):
    return [atom_key(CHAMP, lv, False) for lv in levels]


def _one_game(a_lv, b_lv, seed, rng):
    """Play one champion matchup, seat-randomised. Return (a_atoms, b_atoms, y)
    with y from A's perspective (1 win / 0.5 draw / 0 loss)."""
    a = [CHAMP] * len(a_lv)
    b = [CHAMP] * len(b_lv)
    if rng.random() < 0.5:                    # A on agent seat
        r = run_scripted_game(a, b, a_lv, b_lv, seed)
        y = {"W": 1.0, "D": 0.5, "L": 0.0}[r]
    else:                                     # A on opp seat -> invert
        r = run_scripted_game(b, a, b_lv, a_lv, seed)
        y = {"W": 0.0, "D": 0.5, "L": 1.0}[r]
    return champ_atoms(a_lv), champ_atoms(b_lv), y


def gen_dataset(n_games, seed, max_team=3):
    rng = random.Random(seed)
    data = []
    for i in range(n_games):
        na, nb = rng.randint(1, max_team), rng.randint(1, max_team)
        a_lv = [rng.randint(2, 8) for _ in range(na)]
        b_lv = [rng.randint(2, 8) for _ in range(nb)]
        data.append(_one_game(a_lv, b_lv, seed + i * 7919 + 13, rng))
    return data


def fit(model, data, epochs, lr, seed):
    rng = random.Random(seed ^ 0xABCDEF)
    idx = list(range(len(data)))
    for _ in range(epochs):
        rng.shuffle(idx)
        for j in idx:
            a, b, y = data[j]
            model.update(a, b, y, lr=lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--val_n", type=int, default=120,
                    help="games per concentration case for fresh measurement")
    ap.add_argument("--max_team", type=int, default=3,
                    help="max bodies per side sampled in training (coverage)")
    ap.add_argument("--mm_games", type=int, default=24,
                    help="matchmaker end-to-end matchups to measure (check 5)")
    args = ap.parse_args()

    print(f"generating {args.games} scripted champion games (max_team={args.max_team}) ...")
    data = gen_dataset(args.games, args.seed, args.max_team)
    model = StrengthModel(k_max=max(4, args.max_team))
    fit(model, data, args.epochs, args.lr, args.seed)

    # center r for readability (gauge-free; predictions unchanged)
    lo = model.r[atom_key(CHAMP, 2, False)]
    for lv in LEVELS:
        model.r[atom_key(CHAMP, lv, False)] -= lo

    print("\n=== 學到的 per-atom rating（champion，已對 L2 置零）===")
    prev = None
    for lv in LEVELS:
        rv = model.r[atom_key(CHAMP, lv, False)]
        jump = "" if prev is None else f"  Δ={rv - prev:+.2f}"
        star = "  <<< 斷點" if prev is not None and (rv - prev) > 1.2 else ""
        print(f"  champ@L{lv}: r={rv:+.2f}{jump}{star}")
        prev = rv
    print("g[k] 動作經濟項:", [round(x, 2) for x in model.g])

    print("\n=== 檢1 斷點 / 檢2 動作經濟 ===")
    r4 = model.r[atom_key(CHAMP, 4, False)]
    r5 = model.r[atom_key(CHAMP, 5, False)]
    print(f"  r[L5]-r[L4] = {r5 - r4:+.2f}  (應顯著>其他相鄰級距=Extra Attack 懸崖)")
    print(f"  g[2]={model.g[2]:+.2f}  g[3]={model.g[3]:+.2f}  "
          f"(應 0<g2<g3 = 人多超線性)")

    print("\n=== 檢3 重現 Phase 1 集中度（模型預測 vs 現量）===")
    cases = [([3, 3], [6]), ([2, 2, 2], [6]), ([4, 4], [8]),
             ([3, 3, 3], [5, 4]), ([2, 2, 2, 2], [8]), ([4, 4], [5, 3])]
    print(f"{'A vs B':28s} {'模型P':>7s} {'現量':>7s} {'|Δ|':>6s}")
    tot = 0.0
    for a_lv, b_lv in cases:
        if len(a_lv) > model.k_max or len(b_lv) > model.k_max:
            model.k_max = max(len(a_lv), len(b_lv))          # allow 4-body eval
        p = model.win_prob(champ_atoms(a_lv), champ_atoms(b_lv))
        meas, *_ = matchup_score([CHAMP] * len(a_lv), a_lv,
                                 [CHAMP] * len(b_lv), b_lv, args.val_n,
                                 args.seed + sum(a_lv) * 31)
        d = abs(p - meas); tot += d
        print(f"{f'{a_lv} vs {b_lv}':28s} {p:7.2f} {meas:7.2f} {d:6.2f}")
    print(f"平均 |Δ| = {tot / len(cases):.3f}  (<0.10 = 模型形式足以承載平衡面)")

    print("\n=== 檢4 held-out 校準（fresh 局，預測 P 分箱 vs 實際）===")
    val = gen_dataset(600, args.seed + 999, args.max_team)
    bins = [[] for _ in range(5)]           # [0,.2),[.2,.4),...,[.8,1]
    for a, b, y in val:
        p = model.win_prob(a, b)
        bins[min(4, int(p * 5))].append(y)
    print(f"{'預測區間':12s} {'n':>4s} {'預測均P':>8s} {'實際勝率':>8s}")
    for i, bk in enumerate(bins):
        if bk:
            lo_, hi_ = i * 0.2, (i + 1) * 0.2
            print(f"[{lo_:.1f},{hi_:.1f})   {len(bk):>4d} "
                  f"{(i + 0.5) * 0.2:>8.2f} {sum(bk) / len(bk):>8.2f}")

    print("\n=== 檢5 端到端：matchmaker(target=0.5) 產局→實際量平衡 ===")
    from trpg.rl.matchup import propose_matchup
    rng = random.Random(args.seed + 4242)
    draw = lambda rr: (CHAMP, rr.randint(2, 8), False)
    fair = 0
    devs = []
    N = args.mm_games
    for i in range(N):
        a, b, p, _ = propose_matchup(model, rng, draw, sizes=(1, 2, 3),
                                     n_candidates=48, target=0.5,
                                     max_size_gap=1)
        a_lv = [s[1] for s in a]; b_lv = [s[1] for s in b]
        s, *_ = matchup_score([CHAMP] * len(a_lv), a_lv,
                              [CHAMP] * len(b_lv), b_lv, args.val_n,
                              args.seed + 5000 + i)
        fair += abs(s - 0.5) <= 0.15
        devs.append(abs(s - 0.5))
    print(f"  {N} 局 target=0.5：實際落 [0.35,0.65] 比例 = {fair}/{N} "
          f"= {fair / N:.0%}；平均 |score-0.5| = {sum(devs) / N:.3f}")
    print("  (裸相加基準下這些等和局會散到 0.11↔0.94；聚在 0.5=貨幣真能平衡)")

    out = Path("models") / "strength_champion_fit.json"
    out.parent.mkdir(exist_ok=True)
    model.save(out)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
