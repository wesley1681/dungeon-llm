"""Model-as-MONSTER paired eval (扮演怪物 wave).

Measures how well a checkpoint PLAYS each monster, against the scripted
GenericMonsterPolicy playing the *same seat* — the first instrument of its
kind (every previous gate evaluated the net only on the class side).

Both arms run through the identical env orientation and action lattice:
  monster in the single agent seat, scripted class experts as opponents;
  the script arm drives the seat via encode_action (calibrate_boss.py
  convention — script coords get snapped to the RL grid, so both arms face
  the same action space and the comparison is play quality, not lattice).

Engine dice are GLOBAL RNG → before every episode we random.seed() with an
arm-independent key, so the two arms see identical streams until their
decisions diverge (fingerprint lesson, MONSTER_CATALOG §6.8). Same-process,
within-run comparison only.

Modes:
  1v1   every 1v1-viable monster (equiv ≤ 8) vs the 12 standard classes,
        class at round(equiv_level), monster at natural_level.
  boss  every party-banded boss (PARTY3_EQUIV_LEVEL ≤ 8) vs the calibration
        party (battle_master+life+evocation) at the two levels bracketing
        the measured party-equiv.

Telemetry: per-monster WR both arms + skill-usage mix (spots "never casts
gaze / never aims breath" failure modes), encode failures (action-space
coverage gaps), episode length.

Usage:
  python scripts/eval_monster_actor.py --ckpt models/seed_dtype1/seed_s1600.pt
         --mode both --games 4 --boss_games 12
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse
import math
import random
from collections import Counter
from zlib import crc32

import numpy as np
import torch

from trpg.scenarios.monsters import (MONSTER_DEFS, EQUIV_LEVEL_1V1,
                                     PARTY3_EQUIV_LEVEL, register_monsters,
                                     onev1_viable_monsters,
                                     party3_boss_monsters)

register_monsters()

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.action import encode_action
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.engine.skill import available_skills
from trpg.engine.combat_policy import make_archetype_policy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single
from synth_identity import STANDARD_IDS

PARTY = ("battle_master", "life", "evocation")   # calibration trio (§6.8)


def run_episode(mon: str, opp_archs: list[str], mon_lvl: int, opp_lvl: int,
                ep_key: str, net=None):
    """One episode with the monster in the agent seat. net=None → script arm.
    Returns (win, usage Counter, turns, encode_fails Counter)."""
    k = crc32(ep_key.encode())
    random.seed(k)                       # global engine dice — arm-independent
    env = CombatEnvV2(seed=k ^ 0x5A5A5A, n_agents=1, n_opps=len(opp_archs))
    obs, _ = env.reset(agent_archs=[mon], opp_archs=list(opp_archs),
                       level=mon_lvl, opp_level=opp_lvl)
    aid = env.agent_ids[0]
    script = None if net is not None else make_archetype_policy(mon)
    usage: Counter = Counter()
    fails: Counter = Counter()
    turns = 0
    done = False
    while not done:
        actor = env.current_agent_id
        ch = env.ws.characters[actor]
        if net is not None:
            ob = blind_np_single(obs)
            ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws,
                                   agent_id=actor))
        else:
            dec = script.decide(actor, ch, env.ws, env.resources,
                                env.ws.combat.round_number)
            if dec.action is None or getattr(dec, "fled", False):
                act = [0, 0, 0]
            else:
                try:
                    act = list(encode_action(dec.action, env.ws, actor))
                except Exception as ex:                       # coverage gap!
                    fails[f"{type(ex).__name__}:{str(ex)[:50]}"] += 1
                    act = [0, 0, 0]
        if act[0] > 0:
            sks = available_skills(ch, env.ws)
            if act[0] < len(sks):
                usage[sks[act[0]].skill_id] += 1
        turns += 1
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    won = (env.ws.characters[aid].is_alive()
           and not any(env.ws.characters[o].is_alive() for o in env.opp_ids))
    return won, usage, turns, fails


def fmt_usage(usage: Counter, turns: int, top: int = 3) -> str:
    if not turns or not usage:
        return "-"
    parts = []
    for sid, c in usage.most_common(top):
        name = sid.replace("weapon:", "")
        parts.append(f"{name}×{c / turns:.2f}")
    return " ".join(parts)


def paired_block(tag: str, mon: str, opp_sets: list[tuple[list[str], int, int]],
                 games: int, net):
    """Run model & script arms over every (opps, mon_lvl, opp_lvl) × games.
    Returns dict with per-arm wins/n/usage/turns/fails."""
    res = {arm: {"w": 0, "n": 0, "usage": Counter(), "turns": 0,
                 "fails": Counter()}
           for arm in ("model", "script")}
    for opps, m_lvl, o_lvl in opp_sets:
        for kk in range(games):
            key = f"{tag}|{mon}|{'+'.join(opps)}|{m_lvl}v{o_lvl}|{kk}"
            for arm, drv in (("model", net), ("script", None)):
                w, u, t, f = run_episode(mon, opps, m_lvl, o_lvl, key, drv)
                r = res[arm]
                r["w"] += int(w); r["n"] += 1
                r["usage"] += u; r["turns"] += t; r["fails"] += f
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/seed_dtype1/seed_s1600.pt")
    p.add_argument("--mode", choices=("1v1", "boss", "both"), default="both")
    p.add_argument("--games", type=int, default=4,
                   help="1v1: games per (monster, class-opponent)")
    p.add_argument("--boss_games", type=int, default=12,
                   help="boss: games per (boss, party level)")
    p.add_argument("--mons", nargs="*", default=None,
                   help="restrict to these monster ids")
    p.add_argument("--chimera", action="store_true",
                   help="register the eval-only chimera monsters "
                        "(zero-shot instruments) into the pools")
    args = p.parse_args()

    if args.chimera:
        from chimera_monsters import register_chimera_monsters
        print(f"chimera monsters registered: "
              f"{register_chimera_monsters()}")

    net = load_student(args.ckpt)
    net.eval()
    print(f"ckpt={args.ckpt}  (paired seeds, single process — within-run Δ only)")

    if args.mode in ("1v1", "both"):
        pool = [m for m in onev1_viable_monsters(8.0)
                if not args.mons or m in args.mons]
        print(f"\n== 1v1 monster seat: {len(pool)} monsters × "
              f"{len(STANDARD_IDS)} classes × {args.games} games/arm ==")
        print(f"{'monster':<16} {'equiv':>5} {'script%':>8} {'model%':>7} "
              f"{'Δpp':>6}  {'model-mix':<34} {'script-mix':<34}")
        tot = {"model": [0, 0], "script": [0, 0]}
        for mon in sorted(pool, key=lambda m: EQUIV_LEVEL_1V1[m]):
            cls_lvl = max(1, round(EQUIV_LEVEL_1V1[mon]))
            m_lvl = MONSTER_DEFS[mon].natural_level
            opp_sets = [([c], m_lvl, cls_lvl) for c in STANDARD_IDS]
            r = paired_block("ma1", mon, opp_sets, args.games, net)
            sw = r["script"]["w"] / r["script"]["n"]
            mw = r["model"]["w"] / r["model"]["n"]
            tot["model"][0] += r["model"]["w"]; tot["model"][1] += r["model"]["n"]
            tot["script"][0] += r["script"]["w"]; tot["script"][1] += r["script"]["n"]
            print(f"{mon:<16} {EQUIV_LEVEL_1V1[mon]:>5.1f} {sw:>8.1%} "
                  f"{mw:>7.1%} {(mw - sw) * 100:>+6.1f}  "
                  f"{fmt_usage(r['model']['usage'], r['model']['turns']):<34} "
                  f"{fmt_usage(r['script']['usage'], r['script']['turns']):<34}",
                  flush=True)
            for arm in ("model", "script"):
                if r[arm]["fails"]:
                    print(f"    !! {arm} encode/exec fails: "
                          f"{dict(r[arm]['fails'])}")
        sm = tot["script"][0] / max(1, tot["script"][1])
        mm = tot["model"][0] / max(1, tot["model"][1])
        print(f"{'MEAN':<16} {'':>5} {sm:>8.1%} {mm:>7.1%} "
              f"{(mm - sm) * 100:>+6.1f}   (n={tot['model'][1]}/arm)")

    if args.mode in ("boss", "both"):
        pool = [m for m in party3_boss_monsters(8.0)
                if not args.mons or m in args.mons]
        print(f"\n== 1v3 BOSS seat: party={'+'.join(PARTY)}, "
              f"{args.boss_games} games/(boss,level)/arm ==")
        print(f"{'boss':<16} {'pEq':>4} {'pL':>3} {'script%':>8} {'model%':>7} "
              f"{'Δpp':>6}  {'model-mix':<34} {'script-mix':<34}")
        tot = {"model": [0, 0], "script": [0, 0]}
        for mon in sorted(pool, key=lambda m: PARTY3_EQUIV_LEVEL[m]):
            eq = PARTY3_EQUIV_LEVEL[mon]
            m_lvl = MONSTER_DEFS[mon].natural_level
            for p_lvl in sorted({max(1, math.floor(eq)),
                                 min(8, math.ceil(eq))}):
                opp_sets = [(list(PARTY), m_lvl, p_lvl)]
                r = paired_block("mab", mon, opp_sets, args.boss_games, net)
                sw = r["script"]["w"] / r["script"]["n"]
                mw = r["model"]["w"] / r["model"]["n"]
                tot["model"][0] += r["model"]["w"]
                tot["model"][1] += r["model"]["n"]
                tot["script"][0] += r["script"]["w"]
                tot["script"][1] += r["script"]["n"]
                print(f"{mon:<16} {eq:>4.1f} {p_lvl:>3} {sw:>8.1%} {mw:>7.1%} "
                      f"{(mw - sw) * 100:>+6.1f}  "
                      f"{fmt_usage(r['model']['usage'], r['model']['turns']):<34} "
                      f"{fmt_usage(r['script']['usage'], r['script']['turns']):<34}",
                      flush=True)
                for arm in ("model", "script"):
                    if r[arm]["fails"]:
                        print(f"    !! {arm} encode/exec fails: "
                              f"{dict(r[arm]['fails'])}")
        sm = tot["script"][0] / max(1, tot["script"][1])
        mm = tot["model"][0] / max(1, tot["model"][1])
        print(f"{'MEAN':<16} {'':>4} {'':>3} {sm:>8.1%} {mm:>7.1%} "
              f"{(mm - sm) * 100:>+6.1f}   (n={tot['model'][1]}/arm)")


if __name__ == "__main__":
    main()
