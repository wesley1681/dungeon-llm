"""Phase 1 — matchup balance calibration harness (scripted-vs-scripted).

GOAL (measure BEFORE building the online machine, per CLAUDE.md): quantify how
far a *naive sum of per-combatant strengths* is from actual balance, and extract
the shape of the action-economy correction — WITHOUT any model in the loop, so
the number is a property of the encounter, not of a policy.

Reference driver = per-entity SCRIPTED experts on BOTH seats (make_archetype_policy):
  - always available, architecture-agnostic;
  - apples-to-apples with how EQUIV_LEVEL_1V1 was originally calibrated (script
    panel), so it isolates action-economy from "model plays differently".
The model-specific correction is Phase 2's ONLINE job — that division is the point.

Incremental (data-driven) plan:
  Step 0  symmetric sanity   KvK identical units -> WR must be ~50% (harness unbiased)
  Step 1  level->strength    1v1 champ@La vs champ@Lb grid -> per-unit strength curve
  Step 2  concentration      equal-raw-sum asymmetric teams -> action-economy factor
          (built AFTER Step 1 gives r(L); not hardcoded here)

Metric = chess score from the AGENT seat: win + 0.5*draw (symmetric; 0.5 = a
400-turn non-terminal stall, rare for scripted teams). Draw/timeout counts printed
so a high stall rate can't masquerade as balance.
"""
from __future__ import annotations
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:                                          # Windows cp950 console can't emit ≈ etc.
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.action import encode_action
from trpg.engine.combat_policy import make_archetype_policy
from trpg.scenarios.monsters import register_monsters, MONSTER_DEFS, EQUIV_LEVEL_1V1

register_monsters()


def _scripted_driver(agent_archs):
    """Drive the agent seat with per-entity scripted experts (mirror of
    eval_gate._multi_expert_driver: one expert per seat index, correct kit)."""
    pols = [make_archetype_policy(a) for a in agent_archs]

    def drv(env, actor):
        aids = list(env.agent_ids)
        i = aids.index(actor) if actor in aids else 0
        pol = pols[min(i, len(pols) - 1)]
        a = env.ws.characters[actor]
        dec = pol.decide(actor, a, env.ws, env.resources,
                         env.ws.combat.round_number)
        if dec.action is None or getattr(dec, "fled", False):
            return [0, 0, 0]
        return list(encode_action(dec.action, env.ws, actor))

    return drv


def run_scripted_game(agent_archs, opp_archs, agent_levels, opp_levels, seed):
    """Both seats scripted. Returns 'W'/'L'/'D' from the AGENT seat.
    agent_levels/opp_levels are per-entity (len == seat size)."""
    random.seed(seed)                         # engine dice use global RNG
    env = CombatEnvV2(seed=seed ^ 0x5A5A5A,
                      n_agents=len(agent_archs), n_opps=len(opp_archs))
    env.reset(agent_archs=list(agent_archs), opp_archs=list(opp_archs),
              agent_levels=list(agent_levels), opp_levels=list(opp_levels),
              layout="open")
    driver = _scripted_driver(agent_archs)
    aids = list(env.agent_ids)
    done = False
    turns = 0
    while not done:
        actor = env.current_agent_id
        _, _, term, trunc, _ = env.step(driver(env, actor))
        done = term or trunc
        turns += 1
        if turns > 400:
            break
    team_alive = any(env.ws.characters[a].is_alive() for a in aids)
    opp_alive = any(env.ws.characters[o].is_alive() for o in env.opp_ids)
    if team_alive and not opp_alive:
        return "W"
    if opp_alive and not team_alive:
        return "L"
    return "D"


def score(agent_archs, opp_archs, agent_levels, opp_levels, n, base_seed):
    """Chess score (win + 0.5*draw) over n games, AGENT seat. Returns
    (score_frac, w, l, d)."""
    w = l = d = 0
    for g in range(n):
        r = run_scripted_game(agent_archs, opp_archs, agent_levels, opp_levels,
                              base_seed + g * 7919)
        w += r == "W"; l += r == "L"; d += r == "D"
    return (w + 0.5 * d) / n, w, l, d


def matchup_score(a_archs, a_levels, o_archs, o_levels, n, seed):
    """SEAT-UNBIASED score for team A (a_archs/a_levels) vs team B: half the
    games with A on the agent seat, half with A on the opp seat, averaged — this
    cancels any residual agent/opp execution-path asymmetry (measured ≤~3pp at
    3v3, within noise but neutralised by construction here). Returns A's
    (score, W, L, D)."""
    n1 = n // 2
    n2 = n - n1
    _, w1, l1, d1 = score(a_archs, o_archs, a_levels, o_levels, n1, seed)
    # A now sits on the OPP seat; score() reports from B's (agent) view → invert.
    _, w2, l2, d2 = score(o_archs, a_archs, o_levels, a_levels, n2, seed + 104729)
    a_w = w1 + l2
    a_l = l1 + w2
    a_d = d1 + d2
    return (a_w + 0.5 * a_d) / n, a_w, a_l, a_d


# ── Step 0: symmetric sanity — identical units KvK must land ~50% ──────────────

def step0_sanity(n, seed):
    print("=== Step 0: 對稱性 sanity（同單位 KvK，應 ≈50%；偏離=harness 有偏）===")
    print(f"{'配置':10s} {'score':>6s} {'W-L-D':>10s}")
    for k in (1, 2, 3):
        a = ["champion"] * k
        lv = [5] * k
        s, w, l, dd = score(a, a, lv, lv, n, seed + k)
        print(f"{f'{k}v{k}@L5':10s} {s:6.2f} {f'{w}-{l}-{dd}':>10s}")


# ── Step 1: level -> strength — 1v1 champion @ La vs @ Lb grid ─────────────────

def step1_level_curve(n, seed, levels):
    print("\n=== Step 1: 等級→強度（1v1 champion@La vs champion@Lb；seat-unbiased）===")
    print("行=La 列=Lb；每格 La 隊 chess score（>0.5=La 較強）")
    print("La\\Lb  " + "".join(f"{lb:>6d}" for lb in levels))
    for la in levels:
        row = f"{la:>5d}  "
        for lb in levels:
            s, *_ = matchup_score(["champion"], [la], ["champion"], [lb], n,
                                  seed + la * 100 + lb)
            row += f"{s:6.2f}"
        print(row)
    print("(對角應 ≈0.50、上三角>0.5＝等級單調；也讀『一級值多少 WR』)")


def step2_concentration(n, seed):
    """Does naive LEVEL-SUM predict balance across concentration? Build two sides
    with matched Σlevel but different body counts; if score deviates from 0.5,
    that deviation IS the action-economy correction the prior needs.
    All champions -> per-unit strength is a pure function of level (no cross-class
    or equiv-level confound)."""
    print("\n=== Step 2: 集中度（等 Σlevel、不同人數；champion；seat-unbiased）===")
    print("偏離 0.50 = 裸相加先驗的 action-economy 誤差（>0.5=多而弱佔優=需罰人數）")
    print(f"{'多而弱 (A)':22s} {'寡而強 (B)':14s} {'Σlv':>4s} {'A_score':>8s} {'W-L-D':>12s}")
    # (A_levels, B_levels): matched Σlevel, A = more bodies, B = fewer/stronger
    cases = [
        ([3, 3], [6]),          # Σ6:  2×L3   vs 1×L6
        ([2, 2, 2], [6]),       # Σ6:  3×L2   vs 1×L6
        ([4, 4], [8]),          # Σ8:  2×L4   vs 1×L8
        ([3, 3, 3], [5, 4]),    # Σ9:  3×L3   vs L5+L4
        ([2, 2, 2, 2], [8]),    # Σ8:  4×L2   vs 1×L8   (extreme)
        ([4, 4], [5, 3]),       # Σ8:  2×L4   vs L5+L3  (count-equal control-ish)
    ]
    for a_lv, b_lv in cases:
        a = ["champion"] * len(a_lv)
        b = ["champion"] * len(b_lv)
        s, w, l, dd = matchup_score(a, a_lv, b, b_lv, n, seed + sum(a_lv) * 31)
        tag_a = f"{len(a_lv)}×champ {a_lv}"
        tag_b = f"{len(b_lv)}×champ {b_lv}"
        print(f"{tag_a:22s} {tag_b:14s} {sum(a_lv):>4d} {s:>8.2f} "
              f"{f'{w}-{l}-{dd}':>12s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="games per cell (n≥48)")
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--step", choices=["0", "1", "2", "all"], default="all")
    args = ap.parse_args()
    levels = [2, 3, 4, 5, 6, 7, 8]
    if args.step in ("0", "all"):
        step0_sanity(args.n, args.seed)
    if args.step in ("1", "all"):
        step1_level_curve(args.n, args.seed, levels)
    if args.step in ("2", "all"):
        step2_concentration(args.n, args.seed)


if __name__ == "__main__":
    main()
