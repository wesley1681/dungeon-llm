"""Is the reward aligned with win/loss, and does it CREDIT control?

User's hypothesis: the dense HP-PBRS rewards immediate damage but control
(frighten/sleep/prone) does no immediate damage, so it earns ~0 the turn it's
used — its win-contribution (less FUTURE incoming) is delayed & noisy → the
learning signal under-credits it. Two clean measurements (BC champion that
DOES use control, 1v2, real env rewards):

PART 1 — episode-level alignment: mean TOTAL episode reward for WON vs LOST
  episodes. If won >> lost, the reward IS aligned with the outcome.

PART 2 — per-action immediate credit: classify each agent action by its EFFECT
  (general, no skill-name list):
    damage   — an enemy's total HP dropped this step
    control  — no HP drop, but an enemy's combat EFFECTIVENESS dropped
               (a disabling status landed: frighten/prone/sleep/stun…)
    heal     — the agent's HP rose
    other    — move / buff / nothing
  and report the mean IMMEDIATE step-reward per class. If control ≈ 0 (or <
  damage) while control episodes win more, the dense reward doesn't credit it.

Usage: python scripts/probe_reward_align.py --ckpt models/seed_kit/kit_s1000.pt
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import argparse, random
from collections import defaultdict
from zlib import crc32
import torch

from trpg.scenarios.monsters import register_monsters
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single
from synth_identity import STANDARD_IDS


def _opp_hp(env, oids):
    return sum(max(0, env.ws.characters[o].hp) for o in oids)


def _opp_eff(env, oids):
    """Sum of LIVING enemies' combat effectiveness (general signal)."""
    s = 0.0
    for o in oids:
        c = env.ws.characters[o]
        if c.is_alive():
            s += CombatEnvV2._combat_effectiveness(c)
    return s


def run(net, idents, games):
    # per-action-class: [sum_reward, n]
    cls = defaultdict(lambda: [0.0, 0])
    won_rew, lost_rew = [], []
    ctrl_win = [0, 0]     # [wins, n] for episodes that used >=1 control
    noctrl_win = [0, 0]
    for ident in idents:
        for gi in range(games):
            key = f"{ident}|{gi}"; k = crc32(key.encode()); random.seed(k)
            env = CombatEnvV2(seed=k ^ 0x5, n_agents=1, n_opps=2)
            opps = [STANDARD_IDS[(k + i) % len(STANDARD_IDS)] for i in range(2)]
            obs, _ = env.reset(agent_archs=[ident], opp_archs=opps, level=5,
                               opp_level=2, layout="open")
            aid = env.agent_ids[0]; oids = list(env.opp_ids)
            ep_rew = 0.0; n_ctrl = 0
            done = False
            while not done:
                actor = env.current_agent_id
                ob = blind_np_single(obs)
                ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
                with torch.no_grad():
                    el, s, e, g = net(ot)
                s = apply_resource_mask(s, env.resources, env.ws, actor)
                e = apply_entity_mask(e, ot, env.ws, actor)
                act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws,
                                       agent_id=actor))
                is_agent = (actor == aid)
                if is_agent:
                    hp0 = _opp_hp(env, oids); eff0 = _opp_eff(env, oids)
                    a_hp0 = max(0, env.ws.characters[aid].hp)
                obs, r, term, trunc, _ = env.step(act)
                done = term or trunc
                if is_agent:
                    ep_rew += r
                    hp1 = _opp_hp(env, oids); eff1 = _opp_eff(env, oids)
                    a_hp1 = max(0, env.ws.characters[aid].hp)
                    if hp1 < hp0 - 1e-6:
                        c = "damage"
                    elif eff1 < eff0 - 1e-6:
                        c = "control"; n_ctrl += 1
                    elif a_hp1 > a_hp0 + 1e-6:
                        c = "heal"
                    else:
                        c = "other"
                    cls[c][0] += r; cls[c][1] += 1
            won = (env.ws.characters[aid].is_alive()
                   and all(not env.ws.characters[o].is_alive() for o in oids))
            (won_rew if won else lost_rew).append(ep_rew)
            tgt = ctrl_win if n_ctrl > 0 else noctrl_win
            tgt[0] += int(won); tgt[1] += 1
    return cls, won_rew, lost_rew, ctrl_win, noctrl_win


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/seed_kit/kit_s1000.pt")
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--idents", nargs="*",
                   default=["battle_master", "arcane_trickster", "war"])
    args = p.parse_args()
    net = load_student(args.ckpt); net.eval()
    cls, won, lost, cw, nw = run(net, args.idents, args.games)
    import statistics as st
    print(f"ckpt={args.ckpt}  1v2 weak  idents={args.idents}\n")
    print("=== PART 1: episode reward aligned with outcome? ===")
    mw = st.mean(won) if won else float('nan')
    ml = st.mean(lost) if lost else float('nan')
    print(f"  WON  episodes: n={len(won):>3}  mean total reward = {mw:+.2f}")
    print(f"  LOST episodes: n={len(lost):>3}  mean total reward = {ml:+.2f}")
    print(f"  => won−lost reward gap = {mw - ml:+.2f}  "
          f"({'ALIGNED' if mw > ml else 'NOT aligned'})\n")
    print("=== PART 2: immediate step-reward by action effect ===")
    for c in ("damage", "control", "heal", "other"):
        tot, n = cls[c]
        if n:
            print(f"  {c:<8} n={n:>4}  mean immediate reward = {tot/n:+.3f}")
    print()
    print("=== control vs win (does using control correlate with winning?) ===")
    for nm, (w, n) in (("used-control", tuple(cw)), ("no-control", tuple(nw))):
        if n:
            print(f"  {nm:<13} WR={w/n:.0%}  (n={n})")


if __name__ == "__main__":
    main()
