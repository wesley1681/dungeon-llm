"""DAgger for the blind student: fix the decisive RESOURCE-TIMING errors that
expert-rollout BC misses. Trace (diag_trace) showed the model wastes its own
early turns (second_wind at full HP, action_surge with no follow-up attack,
dodge) -> falls behind -> dies with enemies at ~24% HP. Those mistakes happen
on STUDENT-visited states the expert demos never contain, so vanilla BC (which
trains on EXPERT-visited states) never corrects them. DAgger does:

  each round:
    1. roll the STUDENT (sampled) over FOCUS classes 1v1+1v2, recording every
       agent-decision state;
    2. LABEL each with the SCRIPTED EXPERT's action at that exact state;
    3. add to the aggregated buffer; BC on it (+ a self-anchor batch to keep
       std12 / targeting / positioning).

No hardcoded rule, no skill-name oracle -> general: the expert label is the only
supervision, on the student's OWN state distribution. Reuses the existing
self_anchor.npz to protect general play.

Usage:
  python scripts/train_dagger.py --warm models/uni_bc/kit_s4000.pt \
      --self_anchor_in models/seed_kit/self_anchor.npz \
      --rounds 8 --roll_states 6000 --steps_per_round 400 --out_dir models/dagger
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import argparse, random
from pathlib import Path
import numpy as np
import torch

import math
from trpg.scenarios.monsters import (register_monsters, MONSTER_DEFS,
                                      EQUIV_LEVEL_1V1)
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

FOCUS = ["battle_master", "war", "arcane_trickster", "assassin", "champion"]


def collect_nonfocus_anchor(net, n_states, seed, rng):
    """Warm net's GREEDY actions on the NON-FOCUS classes only. The old
    self_anchor.npz mixed in the FOCUS classes too, which ANCHORS them to the
    warm net's flawed play and fights DAgger's correction. Restricting the
    anchor to non-focus classes preserves their (already strong) behaviour while
    leaving the focus classes free to be fixed."""
    pool = [a for a in STANDARD_IDS if a not in FOCUS]
    mons = sorted(MONSTER_DEFS)
    net.eval()
    O, A, T = [], [], []
    ep = 0
    while len(A) < n_states:
        agent = rng.choice(pool)
        if rng.random() < 0.5:
            mon = rng.choice(mons); eq = EQUIV_LEVEL_1V1[mon]
            a_lvl = int(min(8, max(3, round(eq)))) if math.isfinite(eq) else 8
            opp, o_lvl = mon, MONSTER_DEFS[mon].natural_level
        else:
            a_lvl = rng.randint(5, 8)
            opp, o_lvl = rng.choice(pool), None
            o_lvl = a_lvl
        env = CombatEnvV2(seed=seed * 1_000_003 + ep, n_agents=1, n_opps=1)
        env.reset(agent_archs=[agent], opp_archs=[opp], level=a_lvl,
                  opp_level=o_lvl)
        obs = blind_np_single(build_obs(env.ws, env.current_agent_id,
                                        env.resources))
        done = False
        while not done:
            aid = env.current_agent_id
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, aid)
            e = apply_entity_mask(e, ot, env.ws, aid)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws,
                                   agent_id=aid))
            tt = -1
            if act[0] != 0:
                sks = available_skills(env.ws.characters[aid], env.ws)
                tt = int(sks[act[0]].features.target_type) if act[0] < len(sks) else -1
            O.append(obs); A.append(act); T.append(tt)
            obs2, _, term, trunc, _ = env.step(act)
            done = term or trunc
            obs = blind_np_single(obs2) if not done else obs2
        ep += 1
    obs_np = {k: np.stack([o[k] for o in O]) for k in O[0]}
    return obs_np, np.array(A, np.int64), np.array(T, np.int64)


def _expert_label(script, env, actor):
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


def _student_action(net, env, actor, obs):
    """GREEDY student action = exactly the deployment behaviour (classic DAgger
    rolls the learned policy, beta=0). pick_action has no sampling mode."""
    ob = blind_np_single(obs)
    ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob.items()}
    with torch.no_grad():
        el, s, e, g = net(ot)
    s = apply_resource_mask(s, env.resources, env.ws, actor)
    e = apply_entity_mask(e, ot, env.ws, actor)
    a = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
    return a, ob


def collect_student_states(net, n_states, seed, rng):
    """Roll the STUDENT; record each agent state + the EXPERT's label there."""
    net.eval()
    O, A, T = [], [], []
    ep = 0
    while len(A) < n_states:
        ident = rng.choice(FOCUS)
        n_opp = rng.choice([1, 2, 2])
        opps = [rng.choice(STANDARD_IDS) for _ in range(n_opp)]
        env = CombatEnvV2(seed=seed * 1_000_003 + ep, n_agents=1, n_opps=n_opp)
        obs, _ = env.reset(agent_archs=[ident], opp_archs=opps, level=5,
                           opp_level=rng.randint(2, 5))
        aid = env.agent_ids[0]; script = make_archetype_policy(ident)
        done = False
        while not done:
            actor = env.current_agent_id
            s_act, ob = _student_action(net, env, actor, obs)
            if actor == aid:
                e_enc, tt = _expert_label(script, env, actor)
                O.append(ob); A.append(e_enc); T.append(tt)
            obs, _, term, trunc, _ = env.step(s_act); done = term or trunc
        ep += 1
    obs_np = {k: np.stack([o[k] for o in O]) for k in O[0]}
    return obs_np, np.array(A, np.int64), np.array(T, np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm", default="models/uni_bc/kit_s4000.pt")
    p.add_argument("--self_anchor_in", default="models/seed_kit/self_anchor.npz")
    p.add_argument("--out_dir", default="models/dagger")
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--roll_states", type=int, default=6000)
    p.add_argument("--steps_per_round", type=int, default=400)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--switch_boost", type=float, default=3.0)
    p.add_argument("--keep_rounds", type=int, default=3,
                   help="aggregate the last K rounds of student states")
    p.add_argument("--recollect_anchor", type=int, default=1,
                   help="1 = collect a NON-FOCUS-only anchor from warm net "
                        "(don't anchor the focus classes to flawed warm play); "
                        "0 = load --self_anchor_in as-is (mixed, legacy)")
    p.add_argument("--anchor_states", type=int, default=12000)
    p.add_argument("--switch_demos", default="",
                   help="optional switch_demos.npz to CO-TRAIN trait-switching "
                        "jointly with the class DAgger (avoids the sequential "
                        "catastrophic-forgetting that erodes fighters).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--focus", default="",
                   help="comma-separated identities to FOCUS (roll+expert-label); "
                        "everything else is anchored. Empty = module default "
                        "(resource-timing melee set).")
    p.add_argument("--freeze_casting", type=int, default=0,
                   help="1 = isolated surgery: train ONLY --keep_params, freeze "
                        "the rest (avoid eroding the tuned casting/targeting).")
    p.add_argument("--keep_params", default="end_head,grid_query",
                   help="comma substrings of param names to TRAIN when "
                        "--freeze_casting 1 (e.g. 'end_head' for heal-only fix).")
    args = p.parse_args()
    if args.focus:
        global FOCUS
        FOCUS = [x.strip() for x in args.focus.split(",") if x.strip()]
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    net = load_student(args.warm)
    if args.recollect_anchor:
        print("=== collecting NON-FOCUS anchor from warm net ===", flush=True)
        s_obs, s_act, s_tt = collect_nonfocus_anchor(
            net, args.anchor_states, args.seed * 104729 + 7, rng)
    else:
        z = np.load(args.self_anchor_in)
        s_act = z["actions"]; s_tt = z["target_types"]
        s_obs = {k[4:]: z[k] for k in z.files if k.startswith("obs_")}
    sw = None
    if args.switch_demos:
        z = np.load(args.switch_demos)
        sw_act = z["actions"]; sw_tt = z["target_types"]
        sw_obs = {k[4:]: z[k] for k in z.files if k.startswith("obs_")}
        sw = (sw_obs, sw_act, sw_tt)
        print(f"  CO-TRAIN switch demos: {len(sw_act)} states", flush=True)
    print(f"DAgger warm={args.warm}  anchor={len(s_act)} "
          f"(non-focus={bool(args.recollect_anchor)})  FOCUS={FOCUS}",
          flush=True)
    if args.freeze_casting:
        # ISOLATED surgery: train ONLY the named pathway, FREEZE everything else
        # (casting/targeting/encoder) so retraining cannot erode the already-
        # tuned policy — measured failure mode: full kite-DAgger dropped evo WR
        # 8→1. Reward-erosion law: only an isolated surgery adds capability
        # without eroding the rest. --keep_params picks the surface:
        #   "end_head,grid_query" = stop-decision + kite movement (default)
        #   "end_head"            = stop-decision ONLY (e.g. heal-waste fix:
        #                           end when reach=0 & full-HP, keep grid frozen)
        keep = tuple(x.strip() for x in args.keep_params.split(",") if x.strip())
        n_tr = n_fr = 0
        for name, p in net.named_parameters():
            if any(k in name for k in keep):
                p.requires_grad_(True); n_tr += p.numel()
            else:
                p.requires_grad_(False); n_fr += p.numel()
        print(f"  FREEZE_CASTING: train {n_tr} params (end_head+grid_query), "
              f"freeze {n_fr} (encoder+skill/entity heads)", flush=True)
        optim = torch.optim.Adam(
            [p for p in net.parameters() if p.requires_grad], lr=args.lr)
    else:
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    buf = []   # list of (obs_np, act, tt) per round
    for rd in range(1, args.rounds + 1):
        o_np, a_np, t_np = collect_student_states(
            net, args.roll_states, args.seed * 7919 + rd, rng)
        buf.append((o_np, a_np, t_np))
        buf = buf[-args.keep_rounds:]
        # aggregate buffer
        keys = buf[0][0].keys()
        agg_o = {k: np.concatenate([b[0][k] for b in buf], 0) for k in keys}
        agg_a = np.concatenate([b[1] for b in buf], 0)
        agg_t = np.concatenate([b[2] for b in buf], 0)
        net.train()
        for step in range(1, args.steps_per_round + 1):
            sel = np.random.randint(0, len(agg_a), size=args.batch)
            ob = {k: torch.from_numpy(v[sel]) for k, v in agg_o.items()}
            acts = torch.from_numpy(agg_a[sel])
            w = None
            if args.switch_boost > 0:
                with torch.no_grad():
                    _, sl, _, _ = net(ob)
                w = 1.0 + args.switch_boost * (sl.argmax(-1) != acts[:, 0]).float()
            bc_loss_step(net, ob, acts, torch.from_numpy(agg_t[sel]), optim,
                         skill_sample_w=w)
            sel = np.random.randint(0, len(s_act), size=args.batch)
            ob = {k: torch.from_numpy(v[sel]) for k, v in s_obs.items()}
            bc_loss_step(net, ob, torch.from_numpy(s_act[sel]),
                         torch.from_numpy(s_tt[sel]), optim)
            if sw is not None:
                sw_obs, sw_act, sw_tt = sw
                sel = np.random.randint(0, len(sw_act), size=args.batch)
                ob = {k: torch.from_numpy(v[sel]) for k, v in sw_obs.items()}
                swa = torch.from_numpy(sw_act[sel])
                w2 = None
                if args.switch_boost > 0:
                    with torch.no_grad():
                        _, sl2, _, _ = net(ob)
                    w2 = 1.0 + args.switch_boost * (sl2.argmax(-1) != swa[:, 0]).float()
                bc_loss_step(net, ob, swa, torch.from_numpy(sw_tt[sel]), optim,
                             skill_sample_w=w2)
        net.eval()
        torch.save(net.state_dict(), out_dir / f"dagger_r{rd:02d}.pt")
        print(f"round {rd}: rolled {len(a_np)} student states "
              f"(buf={len(agg_a)})  saved dagger_r{rd:02d}.pt", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
