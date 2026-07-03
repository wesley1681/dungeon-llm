"""1vN behaviour quality — is 0% WR "loses to numbers" or "plays badly"?

WR alone can't tell: a lone fighter vs 2-3 equals loses even when it plays
perfectly. So we compare the MODEL against a SCRIPTED EXPERT in the SAME lone
seat, same identities, paired global-RNG dice, and read BEHAVIOUR:

  WR        episodes the lone agent wipes all N opponents and lives
  kills     mean opponents killed before dying (0..N)
  dmg%      mean fraction of total enemy HP removed
  turns     mean agent turns survived
  atk/mov/buf/end   action-type mix over the agent's decisions
  focus%    of the agent's attacks, the share landing on the SINGLE most-
            targeted enemy (1vN's correct play is to focus one down, not split)

Two opponent tiers separate "competent but outnumbered" from "incompetent":
  equal  N standard experts at the agent's level (hard; both arms ~0% WR ->
         compare behaviour: model kills/dmg ≈ expert ⇒ competent-outnumbered)
  weak   N opponents at a much lower level (a winnable number — an L5 should
         beat 2-3 L2s; if the MODEL can't but the EXPERT can, that's a real gap)

Paired seeds + a seeded global engine RNG so the two arms face identical dice
(cross-arm noise is otherwise several pp — the seeded-std12 lesson).

Usage:
  python scripts/diag_1vN.py --ckpt models/pop_los_final/pop_u0044.pt --games 24
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

from trpg.scenarios.monsters import register_monsters
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.action import decode_action, encode_action
from trpg.engine.skill import available_skills
from trpg.engine.combat_policy import make_archetype_policy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single
from synth_identity import STANDARD_IDS


def _act_type(act, ws, aid):
    sk = available_skills(ws.characters[aid], ws)
    si = int(act[0])
    if si >= len(sk) or sk[si].skill_id == "end":
        return "end"
    ad = decode_action(act, ws, aid)
    if ad is None:
        return "end"
    t = ad.get("type", "")
    if t == "MOVE":
        return "mov"
    if "ATTACK" in str(t).upper() or "attack" in str(t).lower():
        return "atk"
    return "buf"


def run_episode(net, ident, n_opp, opp_level, ep_key):
    """net=None -> scripted-expert arm controls the lone agent."""
    k = crc32(ep_key.encode())
    random.seed(k)                                   # seed the global engine RNG
    env = CombatEnvV2(seed=k ^ 0x5, n_agents=1, n_opps=n_opp)
    opps = [STANDARD_IDS[(k + i) % len(STANDARD_IDS)] for i in range(n_opp)]
    obs, _ = env.reset(agent_archs=[ident], opp_archs=opps,
                       level=5, opp_level=opp_level, layout="open")
    aid = env.agent_ids[0]
    oids = list(env.opp_ids)
    tot_max = sum(env.ws.characters[o].max_hp for o in oids)
    script = None if net is not None else make_archetype_policy(ident)
    mix = Counter()
    tgt_counts = Counter()   # enemy_id -> attacks landed on it
    n_atk = 0
    turns = 0
    prev = None
    done = False
    while not done:
        actor = env.current_agent_id
        ch = env.ws.characters[actor]
        if actor == aid and actor != prev:
            turns += 1
        prev = actor
        if net is not None:
            ob = blind_np_single(obs)
            ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
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
        # classify + focus-fire bookkeeping (agent only)
        if actor == aid:
            mix[_act_type(act, env.ws, aid)] += 1
            ad = decode_action(act, env.ws, aid)
            if ad is not None and "ATTACK" in str(ad.get("type", "")).upper():
                tgt = ad.get("target") or ad.get("target_id") or ad.get("character")
                if tgt in oids:
                    tgt_counts[tgt] += 1; n_atk += 1
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    alive_opps = sum(env.ws.characters[o].is_alive() for o in oids)
    kills = n_opp - alive_opps
    hp_left = sum(max(0, env.ws.characters[o].hp) for o in oids)
    dmg_frac = 1.0 - hp_left / tot_max if tot_max else 0.0
    won = (env.ws.characters[aid].is_alive() and alive_opps == 0)
    focus = (max(tgt_counts.values()) / n_atk) if n_atk else float("nan")
    return {"won": won, "kills": kills, "dmg": dmg_frac, "turns": turns,
            "mix": mix, "focus": focus, "n_atk": n_atk}


def _stats(R, n_opp):
    n = len(R)
    wr = sum(r["won"] for r in R) / n
    kills = sum(r["kills"] for r in R) / n
    dmg = sum(r["dmg"] for r in R) / n
    focs = [r["focus"] for r in R if r["focus"] == r["focus"]]
    focus = sum(focs) / len(focs) if focs else float("nan")
    M = Counter()
    for r in R:
        M += r["mix"]
    tot = sum(M.values()) or 1
    return dict(wr=wr, kills=kills, dmg=dmg, focus=focus,
                atk=M['atk']/tot, mov=M['mov']/tot, buf=M['buf']/tot,
                end=M['end']/tot)


def _row(label, s, n_opp):
    return (f"  {label:<22} WR={s['wr']:>4.0%}  kills={s['kills']:>4.2f}/{n_opp}  "
            f"dmg={s['dmg']:>4.0%}  focus={s['focus']:>4.0%}  "
            f"| atk={s['atk']:.0%} mov={s['mov']:.0%} "
            f"buf={s['buf']:.0%} end={s['end']:.0%}")


def summarize(net, idents, n_opp, opp_level, games, tag, per_ident=False):
    allR = []
    per = {}
    for ident in idents:
        Ri = [run_episode(net, ident, n_opp, opp_level,
                          f"1v{n_opp}|{opp_level}|{ident}|{gi}")
              for gi in range(games)]
        per[ident] = _stats(Ri, n_opp)
        allR += Ri
    if per_ident:
        for ident in idents:
            print(_row(f"{tag}:{ident}", per[ident], n_opp), flush=True)
    print(_row(f"{tag} (ALL)", _stats(allR, n_opp), n_opp), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/pop_los_final/pop_u0044.pt")
    p.add_argument("--games", type=int, default=24)
    p.add_argument("--idents", nargs="*",
                   default=["assassin", "champion", "war", "battle_master",
                            "evocation", "arcane_trickster"])
    args = p.parse_args()
    net = load_student(args.ckpt); net.eval()
    print(f"ckpt={args.ckpt}  idents={len(args.idents)}  games/ident={args.games}")
    print("(equal = N experts @L5; weak = N @L2 — a winnable number for an L5)\n")
    for n_opp in (2, 3):
        for opp_level, tier in ((5, "equal"), (2, "weak")):
            # per-identity breakdown for the WINNABLE (weak) tiers — that's
            # where model < expert, so we want to see WHICH identity drives it.
            pid = (tier == "weak")
            print(f"== 1v{n_opp}  {tier} (opp L{opp_level}; opps=rotating "
                  f"standard classes) ==")
            summarize(net, args.idents, n_opp, opp_level, args.games,
                      "model", per_ident=pid)
            summarize(None, args.idents, n_opp, opp_level, args.games,
                      "expert", per_ident=pid)
            print()


if __name__ == "__main__":
    main()
