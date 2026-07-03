"""Root-cause of the 1vN WR gap: is the model "over-moving" with WASTED moves
(moves that go nowhere useful) or legitimately APPROACHING (closing distance)?

The 1vN per-identity breakdown showed martial classes move 2-2.5x the scripted
expert and win less. "Moves a lot" is a symptom with two opposite causes:

  WASTED    chose MOVE but the move did NOT bring the agent closer to its
            nearest enemy (lateral / away / no-op). This is the project's own
            archetype-agnostic waste definition (WASTED_MOVE_COST geometry) and
            the structural pathology.
  PRODUCTIVE chose MOVE and got meaningfully closer to the nearest enemy (a
            real approach — correct play when out of reach).

We resolve which by capturing nearest-enemy distance BEFORE and AFTER each move
step: productive ⟺ distance dropped by > MOVE_EPS.

Per arm we report:
  move%        share of decisions that were MOVE
  waste%       share of decisions that were a WASTED move (didn't close)
  approach%    share of decisions that were a PRODUCTIVE move (closed distance)
  atk/mov/buf/end   overall action mix (same as diag_1vN)

If the model's waste%/miss-atk% are high AND the scripted expert's are ~0, the
over-move is a real structural pathology (reward/curriculum), not legitimate
kiting. The same instrument runs on the MONSTER seat too (model playing a
monster, and the dedicated monster-actor champion) — if the signature matches
the class seat, the pathology is cross-seat / structural, so any retrain to fix
it must re-examine the monster model as well, not just the 12 classes.

Paired global-RNG dice (seeded per episode) so arms face identical streams.

Usage:
  python scripts/diag_overmove.py --ckpt models/pop_los_final/pop_u0044.pt \
         --mon_ckpt models/mon_actor1/ma_u0032.pt --games 24
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import argparse, random
from collections import Counter
from zlib import crc32
import torch

from trpg.scenarios.monsters import (register_monsters, MONSTER_DEFS,
                                      EQUIV_LEVEL_1V1, onev1_viable_monsters)
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.action import decode_action, encode_action, ACTION_DIMS
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.engine.skill import available_skills
from trpg.engine.combat_policy import make_archetype_policy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single
from synth_identity import STANDARD_IDS

MOVE_EPS = 0.1   # a move must drop nearest-enemy distance by > this to count
                 # as productive (else it's lateral / away / no-op = wasted)
WASTED_THRESH = 0.5   # mirrors env_v2.WASTED_MOVE_THRESH: |Δdist| < this with
                      # LoS is what the env actually penalises (lateral). A move
                      # AWAY by >= this ESCAPES the penalty (the hole).


def _act_type(act, ws, aid):
    sk = available_skills(ws.characters[aid], ws)
    si = int(act[0])
    if si >= len(sk) or sk[si].skill_id == "end":
        return "end"
    ad = decode_action(act, ws, aid)
    if ad is None:
        return "end"
    t = str(ad.get("type", "")).upper()
    if t == "MOVE":
        return "mov"
    if "ATTACK" in t:
        return "atk"
    return "buf"


def _nearest_enemy_dist(env, actor, foe_ids):
    ch = env.ws.characters[actor]
    ds = [ch.position.distance_to(env.ws.characters[o].position)
          for o in foe_ids if env.ws.characters[o].is_alive()]
    return min(ds) if ds else None


def run_episode(agent_arch, opp_archs, a_lvl, opp_lvl, ep_key, net=None,
                layout="open"):
    """net=None -> scripted policy for `agent_arch` drives the agent seat."""
    k = crc32(ep_key.encode())
    random.seed(k)
    env = CombatEnvV2(seed=k ^ 0x5, n_agents=1, n_opps=len(opp_archs))
    obs, _ = env.reset(agent_archs=[agent_arch], opp_archs=list(opp_archs),
                       level=a_lvl, opp_level=opp_lvl, layout=layout)
    aid = env.agent_ids[0]
    oids = list(env.opp_ids)
    tot_max = sum(env.ws.characters[o].max_hp for o in oids)
    script = None if net is not None else make_archetype_policy(agent_arch)
    mix = Counter()
    n_dec = 0
    n_move = n_approach = n_lateral = n_away = 0   # move decomposition
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
                except Exception:
                    act = [0, 0, 0]
        is_agent = (actor == aid)
        t = _act_type(act, env.ws, aid) if is_agent else None
        d_before = _nearest_enemy_dist(env, aid, oids) if is_agent else None
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
        if is_agent:
            mix[t] += 1
            n_dec += 1
            if t == "mov":
                n_move += 1
                d_after = _nearest_enemy_dist(env, aid, oids)
                if d_before is None or d_after is None:
                    n_lateral += 1          # degenerate; count as non-productive
                else:
                    delta = d_before - d_after      # +closer / -farther
                    if delta >= WASTED_THRESH:
                        n_approach += 1             # productive (no env charge)
                    elif delta <= -WASTED_THRESH:
                        n_away += 1                 # farther: ESCAPES env penalty
                    else:
                        n_lateral += 1              # env charges -0.4 here
    alive = sum(env.ws.characters[o].is_alive() for o in oids)
    won = (env.ws.characters[aid].is_alive() and alive == 0)
    hp_left = sum(max(0, env.ws.characters[o].hp) for o in oids)
    dmg = 1.0 - hp_left / tot_max if tot_max else 0.0
    return dict(won=won, dmg=dmg, mix=mix, n_dec=n_dec, n_move=n_move,
                n_approach=n_approach, n_lateral=n_lateral, n_away=n_away)


def _agg(R):
    n = len(R)
    dec = sum(r["n_dec"] for r in R) or 1
    mv = sum(r["n_move"] for r in R) or 1
    M = Counter()
    for r in R:
        M += r["mix"]
    tot = sum(M.values()) or 1
    return dict(
        wr=sum(r["won"] for r in R) / n,
        dmg=sum(r["dmg"] for r in R) / n,
        move=sum(r["n_move"] for r in R) / dec,
        # of all MOVES: how they split
        m_appr=sum(r["n_approach"] for r in R) / mv,
        m_lat=sum(r["n_lateral"] for r in R) / mv,
        m_away=sum(r["n_away"] for r in R) / mv,
        # wasted share of ALL decisions (lateral+away)
        waste=sum(r["n_lateral"] + r["n_away"] for r in R) / dec,
        atk=M['atk']/tot, mov=M['mov']/tot, buf=M['buf']/tot, end=M['end']/tot)


def _row(label, s):
    return (f"  {label:<26} WR={s['wr']:>4.0%} dmg={s['dmg']:>4.0%} | "
            f"move={s['move']:>4.0%} WASTED={s['waste']:>4.0%} "
            f"[of moves: appr={s['m_appr']:>3.0%} lateral={s['m_lat']:>3.0%} "
            f"away={s['m_away']:>3.0%}] | atk={s['atk']:.0%} end={s['end']:.0%}")


def block(title, agent_archs, opp_pick, a_lvl, opp_lvl, n_opp, games,
          arms, per_ident=True):
    print(f"== {title} ==")
    for arm_name, net in arms:
        allR = []
        for ident in agent_archs:
            Ri = []
            for gi in range(games):
                key = f"{title}|{ident}|{gi}"
                k = crc32(key.encode())
                opps = opp_pick(k, n_opp, ident)
                Ri.append(run_episode(ident, opps, a_lvl, opp_lvl, key, net))
            allR += Ri
            if per_ident:
                print(_row(f"{arm_name}:{ident}", _agg(Ri)))
        print(_row(f"{arm_name} (ALL)", _agg(allR)))
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/pop_los_final/pop_u0044.pt")
    p.add_argument("--mon_ckpt", default="models/mon_actor1/ma_u0032.pt")
    p.add_argument("--games", type=int, default=24)
    p.add_argument("--seat", choices=("class", "monster", "both"),
                   default="both")
    p.add_argument("--idents", nargs="*",
                   default=["assassin", "champion", "war", "battle_master",
                            "evocation", "arcane_trickster"])
    p.add_argument("--mons", nargs="*", default=None)
    args = p.parse_args()

    net = load_student(args.ckpt); net.eval()
    print(f"class-ckpt={args.ckpt}")
    print("WASTED = a MOVE that did NOT close distance to nearest enemy "
          "(lateral/no-op = the pathology; casters kiting read as WASTED too).")
    print("approach = a MOVE that closed distance (correct when out of reach).\n")

    def std_opps(k, n, ident):
        return [STANDARD_IDS[(k + i) % len(STANDARD_IDS)] for i in range(n)]

    if args.seat in ("class", "both"):
        for n_opp in (1, 2):
            block(f"CLASS seat 1v{n_opp} weak (opp=std L2)", args.idents,
                  std_opps, 5, 2, n_opp, args.games,
                  [("model", net), ("expert", None)])

    if args.seat in ("monster", "both"):
        mon_net = load_student(args.mon_ckpt); mon_net.eval()
        print(f"monster-ckpt={args.mon_ckpt}\n")
        pool = (args.mons or
                sorted(onev1_viable_monsters(8.0),
                       key=lambda m: EQUIV_LEVEL_1V1[m])[:4])

        def class_opps(k, n, ident):
            return [STANDARD_IDS[(k + i) % len(STANDARD_IDS)] for i in range(n)]
        for mon in pool:
            cls_lvl = max(1, round(EQUIV_LEVEL_1V1[mon]))
            m_lvl = MONSTER_DEFS[mon].natural_level
            block(f"MONSTER seat: {mon} (L{m_lvl}) vs 1 std L{cls_lvl}",
                  [mon], class_opps, m_lvl, cls_lvl, 1, args.games,
                  [("genrl-model", net), ("mon-actor", mon_net),
                   ("script", None)], per_ident=False)


if __name__ == "__main__":
    main()
