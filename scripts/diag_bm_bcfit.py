"""DECISIVE: can the blind net REPRESENT battle_master's scripted-expert policy
at all? COVSHIFT showed the shared BC student matches the bm expert only 37%
even on EXPERT-visited states (not covariate shift). Two hypotheses:

  INTERFERENCE  — the 12-class shared net sacrifices bm (hardest policy) to fit
                  the easier classes; a DEDICATED bm fit would reach high
                  agreement -> teacher-distill plan is viable.
  OBS-LIMIT     — the obs lacks what the bm script keys on; even a dedicated,
                  overfit single-class BC stalls at ~40% -> must fix the obs.

This trains a DEDICATED single-class bm BC (bm-only expert demos, NO self-
anchor, many epochs, all net capacity on bm) and reports bm skill-slot
agreement (expert-visited) + WR vs the expert every eval. If agreement climbs
to ~80%+ it's INTERFERENCE; if it stalls near the shared net's ~37% it's an
OBS-LIMIT. General: no skill-name targeting, raw CE on the expert's action.

Usage: python scripts/diag_bm_bcfit.py --ident battle_master --steps 1500
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import argparse, random
from zlib import crc32
import numpy as np
import torch

from trpg.scenarios.monsters import register_monsters
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.action import encode_action
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.obs import build_obs
from trpg.rl.train_bc import bc_loss_step
from trpg.engine.skill import available_skills
from trpg.engine.combat_policy import make_archetype_policy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single
from synth_identity import STANDARD_IDS


def _expert_enc(script, env, actor):
    ch = env.ws.characters[actor]
    dec = script.decide(actor, ch, env.ws, env.resources,
                        env.ws.combat.round_number)
    if dec.action is None or getattr(dec, "fled", False):
        return [0, 0, 0], -1
    try:
        enc = list(encode_action(dec.action, env.ws, actor))
    except Exception:
        return [0, 0, 0], -1
    sks = available_skills(ch, env.ws)
    tt = int(sks[enc[0]].features.target_type) if 0 <= enc[0] < len(sks) else -1
    return enc, tt


def collect_demos(ident, n_states, seed, rng):
    O, A, T = [], [], []
    ep = 0
    while len(A) < n_states:
        n_opp = rng.choice([1, 2, 2])
        opps = [rng.choice(STANDARD_IDS) for _ in range(n_opp)]
        env = CombatEnvV2(seed=seed * 1_000_003 + ep, n_agents=1, n_opps=n_opp)
        env.reset(agent_archs=[ident], opp_archs=opps, level=5,
                  opp_level=rng.randint(2, 5))
        aid = env.agent_ids[0]; script = make_archetype_policy(ident)
        done = False
        while not done:
            actor = env.current_agent_id
            obs = blind_np_single(build_obs(env.ws, actor, env.resources))
            enc, tt = _expert_enc(script, env, actor)
            if actor == aid:
                O.append(obs); A.append(enc); T.append(tt)
            _, _, term, trunc, _ = env.step(enc); done = term or trunc
        ep += 1
    obs_np = {k: np.stack([o[k] for o in O]) for k in O[0]}
    return obs_np, np.array(A, np.int64), np.array(T, np.int64)


def agreement_and_wr(net, ident, games):
    net.eval()
    agree = [0, 0]; wins = 0
    for gi in range(games):
        key = f"{ident}|2|{gi}"; k = crc32(key.encode()); random.seed(k)
        env = CombatEnvV2(seed=k ^ 0x5, n_agents=1, n_opps=2)
        opps = [STANDARD_IDS[(k + i) % len(STANDARD_IDS)] for i in range(2)]
        obs, _ = env.reset(agent_archs=[ident], opp_archs=opps, level=5,
                           opp_level=2, layout="open")
        aid = env.agent_ids[0]; oids = list(env.opp_ids)
        script = make_archetype_policy(ident)
        done = False
        while not done:
            actor = env.current_agent_id
            # expert-visited agreement: expert drives; compare student greedy
            e_enc, _ = _expert_enc(script, env, actor)
            if actor == aid:
                ob = blind_np_single(obs)
                ot = {kk: torch.from_numpy(v).unsqueeze(0)
                      for kk, v in ob.items()}
                with torch.no_grad():
                    el, s, e, g = net(ot)
                s = apply_resource_mask(s, env.resources, env.ws, actor)
                e = apply_entity_mask(e, ot, env.ws, actor)
                s_act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws,
                                         agent_id=actor))
                agree[1] += 1; agree[0] += int(s_act[0] == e_enc[0])
            obs, _, term, trunc, _ = env.step(e_enc); done = term or trunc
        if (env.ws.characters[aid].is_alive()
                and all(not env.ws.characters[o].is_alive() for o in oids)):
            wins += 1
    return agree[0] / max(1, agree[1]), wins / games


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm", default="models/pop_los_final/pop_u0044.pt")
    p.add_argument("--ident", default="battle_master")
    p.add_argument("--states", type=int, default=10000)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--eval_every", type=int, default=300)
    p.add_argument("--eval_games", type=int, default=40)
    p.add_argument("--out", default="")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    net = load_student(args.warm)
    print(f"DEDICATED single-class BC: {args.ident}  warm={args.warm}", flush=True)
    print("collecting bm-only expert demos...", flush=True)
    obs_np, A, T = collect_demos(args.ident, args.states, args.seed, rng)
    print(f"  {len(A)} demo states", flush=True)
    a0, wr0 = agreement_and_wr(net, args.ident, args.eval_games)
    print(f"baseline (warm): expert-visited agreement={a0:.0%}  WR={wr0:.0%}",
          flush=True)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    n = len(A)
    for step in range(1, args.steps + 1):
        sel = np.random.randint(0, n, size=args.batch)
        ob = {k: torch.from_numpy(v[sel]) for k, v in obs_np.items()}
        net.train()
        loss, acc = bc_loss_step(net, ob, torch.from_numpy(A[sel]),
                                 torch.from_numpy(T[sel]), optim)
        if step % args.eval_every == 0:
            ag, wr = agreement_and_wr(net, args.ident, args.eval_games)
            print(f"step {step:4d}  bc_skill_acc={acc.get('skill', float('nan')):.2f}"
                  f"  expert-visited agreement={ag:.0%}  WR(expert-drv)={wr:.0%}",
                  flush=True)
    if args.out:
        net.eval()
        torch.save(net.state_dict(), args.out)
        print(f"saved {args.out}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
