"""BC-seed a REGEN→MELEE behaviour into the champion, then verify causally.

Why (obs v5 goal, 2026-06-13): the v5 trait channel makes a creature's own
regeneration visible, but pure PPO won't EXPLORE into using it (the monster
pack-PPO run left flankON==flankABL exactly — the switch-wave lesson again:
「PPO 探索不到」). So we install the behaviour supervised (this script), exactly
as 12h/12j seeded the typed-resist switch, and verify with a causal ablation.

Behaviour (measured beneficial, probe_regen_skill.py): with regeneration a
dual-weapon bruiser should MELEE (high damage, the counter-hits heal back —
forced-melee 52% vs forced-ranged 30% @regen30); without it, KITE with the
ranged weapon. The warm champion kites regardless of regen (巨棒×0.00→0.03 flat
= blind to the channel).

Pipeline (mirrors seed_switch_bc):
  1. DEMOS — roll the warm net over hill_giant vs the calibration party;
     regeneration RANDOMISED on/off per episode (channel = the only systematic
     signal). At every agent state the ORACLE label is regen-conditioned:
       regen-on  → approach + MELEE weapon (smallest-reach damaging weapon)
       regen-off → attack with the RANGED weapon (largest-reach), from range
     built via the scripted move/weapon skills → encode_action, and EXECUTED
     (teacher forcing) so regen-on episodes follow melee trajectories.
  2. SEED — BC fine-tune, alternating regen-demo batches and anchor batches
     (switch_demos + warm self-anchor, 1:1) to protect the standard kits.
  3. VERIFY — verify_regen_causal.py: regen-on melee-rate ≫ regen-off, and
     ≫ regen-ABLATED (I_TRAIT_REGEN zeroed) = behaviour is driven by READING
     the channel (causal), plus WR-on > WR-ablated (the trait info necessarily
     helps).

Usage:
  python scripts/seed_regen_bc.py --states 6000 --steps 600 --out models/regen_v5
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse, random
from collections import Counter
from pathlib import Path
import numpy as np
import torch

from trpg.scenarios.monsters import (MONSTER_DEFS, register_monsters,
                                     onev1_viable_monsters)
register_monsters()

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.train_ppo import _sample_action
from trpg.rl.train_bc import bc_loss_step
from trpg.rl.obs import build_obs, migrate_entities_v3_to_v4
from trpg.rl.action import encode_action
from trpg.engine.skill import available_skills
from trpg.scenarios.archetypes import STANDARD_ARCHETYPES

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single, standard_probe, monster_opp_probe

# dual-weapon bruisers (melee + ranged). hill_giant trains; manticore held out
# IF it qualifies (checked at runtime). The rule learned must be "regen→melee",
# not "hill_giant→melee".
TRAIN_AGENT = "hill_giant"
PARTY = ("battle_master", "life", "evocation")
BLOCKED = ("火", "強酸")


def weapon_skills(ws, aid):
    """(melee_skill, ranged_skill) = smallest / largest reach damaging weapon."""
    a = ws.characters[aid]
    ws_sk = [sk for sk in available_skills(a, ws)
             if sk.skill_id.startswith("weapon:") and sk.features.expected_damage > 0]
    if len(ws_sk) < 2:
        return None, None
    ws_sk.sort(key=lambda s: s.features.range_m)
    return ws_sk[0], ws_sk[-1]


def regen_oracle(ws, aid):
    """Full regen-conditioned action: approach+melee if regenerating, else
    ranged poke. Returns an engine action dict (or None)."""
    a = ws.characters[aid]
    enemies = [(cid, c) for cid, c in ws.characters.items()
               if c.is_alive() and ws.is_party_ally(cid) != ws.is_party_ally(aid)]
    if not enemies:
        return None
    tid, tgt = min(enemies, key=lambda kv: a.position.distance_to(kv[1].position))
    melee, ranged = weapon_skills(ws, aid)
    if melee is None:
        return None
    regen_on = bool(getattr(a, "regeneration", None)
                    and (a.regeneration or {}).get("amount", 0) > 0)
    want = melee if regen_on else ranged
    d = a.position.distance_to(tgt.position)
    reach = want.features.range_m or 1.5
    skills = available_skills(a, ws)
    if d <= reach + 1e-6:
        return want.build_action(aid, tid, (tgt.position.x, tgt.position.y))
    move_sk = next((s for s in skills if s.skill_id == "move"), None)
    if move_sk is not None:
        return move_sk.build_action(aid, tid, None)
    return want.build_action(aid, tid, (tgt.position.x, tgt.position.y))


def collect_demos(net, n_states, seed, rng, opp_lvls=(3, 4, 5)):
    net.eval()
    O, A, T = [], [], []
    n_melee = n_ranged = ep = 0
    while len(A) < n_states:
        regen_on = rng.random() < 0.5
        o_lvl = rng.choice(opp_lvls)
        env = CombatEnvV2(seed=seed * 1_000_003 + ep, n_agents=1, n_opps=len(PARTY))
        env.reset(agent_archs=[TRAIN_AGENT], opp_archs=list(PARTY),
                  level=MONSTER_DEFS[TRAIN_AGENT].natural_level, opp_level=o_lvl)
        aid0 = env.agent_ids[0]
        env.ws.characters[aid0].regeneration = (
            {"amount": 30, "blocked_by": BLOCKED} if regen_on else None)
        obs = blind_np_single(build_obs(env.ws, env.current_agent_id, env.resources))
        done = False
        while not done:
            aid = env.current_agent_id
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                action, lp, val, skm, enm, grm = _sample_action(
                    net, ot, env.resources, env.ws, aid)
            exec_act = action.numpy().tolist()
            if aid == aid0:                       # only relabel the bruiser
                odict = regen_oracle(env.ws, aid)
                if odict is not None:
                    enc = encode_action(odict, env.ws, aid)
                    if enc[0] >= 0:
                        sks = available_skills(env.ws.characters[aid], env.ws)
                        tt = int(sks[enc[0]].features.target_type) \
                            if enc[0] < len(sks) else -1
                        O.append(obs); A.append(list(enc)); T.append(tt)
                        exec_act = list(enc)
                        sid = sks[enc[0]].skill_id if enc[0] < len(sks) else ""
                        if sid.startswith("weapon:"):
                            mel, ran = weapon_skills(env.ws, aid)
                            if mel and sid == mel.skill_id:
                                n_melee += 1
                            elif ran and sid == ran.skill_id:
                                n_ranged += 1
            obs2, _, term, trunc, _ = env.step(exec_act)
            done = term or trunc
            obs = blind_np_single(obs2) if not done else obs2
        ep += 1
        if ep % 200 == 0:
            print(f"  demos {len(A)}/{n_states} (mel {n_melee} ran {n_ranged} "
                  f"{ep} eps)", flush=True)
    obs_np = {k: np.stack([o[k] for o in O]) for k in O[0]}
    print(f"demos: {len(A)} pairs  melee-labels={n_melee} ranged-labels={n_ranged}",
          flush=True)
    return obs_np, np.array(A, np.int64), np.array(T, np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm", default="models/mon_actor1/ma_u0032.pt")
    p.add_argument("--switch_demos", default="models/seed_dtype1/switch_demos.npz")
    p.add_argument("--out", default="models/regen_v5")
    p.add_argument("--states", type=int, default=6000)
    p.add_argument("--self_states", type=int, default=4000)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "BLIND").write_text("regen->melee BC-seed (blind)\n")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    net = load_student(args.warm)
    print(f"warm={args.warm}", flush=True)

    print("collecting regen demos…", flush=True)
    d_obs, d_act, d_tt = collect_demos(net, args.states, args.seed * 7919 + 1, rng)

    # anchors: switch demos (migrate to v5) + a warm self-anchor on class seats
    sw = np.load(args.switch_demos)
    sw_obs = {k[4:]: sw[k] for k in sw.files if k.startswith("obs_")}
    sw_obs["entities"] = migrate_entities_v3_to_v4(sw_obs["entities"])
    sw_act, sw_tt = sw["actions"], sw["target_types"]
    print(f"switch anchor: {len(sw_act)} pairs", flush=True)
    print("collecting warm self-anchor (class seats)…", flush=True)
    sa_obs, sa_act, sa_tt = collect_self_anchor(net, args.self_states,
                                                args.seed * 104729 + 11, rng)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    anchors = [(sw_obs, sw_act, sw_tt), (sa_obs, sa_act, sa_tt)]
    print(f"BC-seed: {args.steps} steps, demo:anchor 1:1", flush=True)
    for step in range(1, args.steps + 1):
        net.train()
        sel = np.random.randint(0, len(d_act), size=args.batch)
        ob = {k: torch.from_numpy(v[sel]) for k, v in d_obs.items()}
        _, acc = bc_loss_step(net, ob, torch.from_numpy(d_act[sel]),
                              torch.from_numpy(d_tt[sel]), optim)
        src = anchors[step % 2]
        asel = np.random.randint(0, len(src[1]), size=args.batch)
        aob = {k: torch.from_numpy(v[asel]) for k, v in src[0].items()}
        bc_loss_step(net, aob, torch.from_numpy(src[1][asel]),
                     torch.from_numpy(src[2][asel]), optim)
        if step % 100 == 0:
            print(f"  step {step}/{args.steps} demo-skill-acc={acc.get('skill', float('nan')):.2f}",
                  flush=True)
            torch.save(net.state_dict(), out / f"regen_s{step:04d}.pt")
    torch.save(net.state_dict(), out / "regen_final.pt")
    net.eval()
    std, _ = standard_probe(net, games=3)
    mon, _ = monster_opp_probe(net, onev1_viable_monsters(8.0), games=1)
    print(f"done. std12={std:.1%} vs-mon={mon:.1%}  (saved {out}/regen_final.pt)",
          flush=True)


def collect_self_anchor(net, n_states, seed, rng):
    """Greedy self-distill on class seats (preserve class play)."""
    net.eval()
    O, A, T = [], [], []
    ep = 0
    while len(A) < n_states:
        a_lvl = rng.randint(5, 8)
        agent = rng.choice(list(STANDARD_ARCHETYPES))
        opp = rng.choice(list(STANDARD_ARCHETYPES))
        env = CombatEnvV2(seed=seed * 1_000_003 + ep, n_agents=1, n_opps=1)
        env.reset(agent_archs=[agent], opp_archs=[opp], level=a_lvl, opp_level=a_lvl)
        obs = blind_np_single(build_obs(env.ws, env.current_agent_id, env.resources))
        done = False
        while not done:
            aid = env.current_agent_id
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, aid)
            e = apply_entity_mask(e, ot, env.ws, aid)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=aid))
            sks = available_skills(env.ws.characters[aid], env.ws)
            tt = -1 if act[0] == 0 else (int(sks[act[0]].features.target_type)
                                         if act[0] < len(sks) else -1)
            O.append(obs); A.append(act); T.append(tt)
            obs2, _, term, trunc, _ = env.step(act)
            done = term or trunc
            obs = blind_np_single(obs2) if not done else obs2
        ep += 1
    obs_np = {k: np.stack([o[k] for o in O]) for k in O[0]}
    return obs_np, np.array(A, np.int64), np.array(T, np.int64)


if __name__ == "__main__":
    main()
