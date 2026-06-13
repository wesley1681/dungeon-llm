"""In-context BC for a team-mode front slot: clone the EXPERT playing the
front seat of 3v3 template parties whose teammates are the FROZEN routed
model nets (deployment configuration) and whose opponents are expert teams.

This is the project's standard two-stage paradigm (scripted-expert demos ->
BC -> RL) applied at the team level — learning from expert play, not writing
expert rules into the model. Used for fronts whose team-PPO plateaued below
the expert-in-slot reference (berserker, devotion): PPO could not discover
the "walk in and stay engaged" behavior from these priors, so we clone it
from demonstrations gathered in the exact deployment context, then let eval
pick the best snapshot.

Usage:
  python scripts/bc_team_front.py --arch devotion --episodes 400 --epochs 6
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import (CombatPolicyNet, apply_resource_mask,
                           apply_entity_mask, pick_action)
from trpg.rl.action import encode_action
from trpg.rl.bc_collect import TT_END
from trpg.rl.train_bc import bc_loss_step
from trpg.rl.train_team import sample_comp
from trpg.engine.combat_policy import make_archetype_policy
from trpg.engine.skill import available_skills
from trpg.scenarios.archetypes import ARCHETYPE_ROLES

from eval_routed import DEFAULT_ROUTING, load_net
from train_team_ppo import eval_team_slot, ROLE_SLOT


def collect_episode(env_seed: int, arch: str, slot: int, mate_nets: dict,
                    rng: random.Random) -> list[tuple[dict, tuple, int]]:
    env = CombatEnvV2(seed=env_seed, n_agents=3, n_opps=3)
    comp = sample_comp(rng, fixed={slot: arch})
    opp_comp = sample_comp(rng)
    obs, _ = env.reset(agent_archs=comp, opp_archs=opp_comp)
    arch_of = dict(zip(env.agent_ids, env.agent_archs))
    front_id = env.agent_ids[slot]
    expert = make_archetype_policy(arch)
    pairs: list[tuple[dict, tuple, int]] = []
    done = False
    while not done:
        actor = env.current_agent_id
        if actor == front_id:
            ag = env.ws.characters[actor]
            dec = expert.decide(actor, ag, env.ws, env.resources,
                                env.ws.combat.round_number)
            if dec.action is None or dec.fled:
                act = (0, 0, 0)
                pairs.append((obs, act, TT_END))
            else:
                act = encode_action(dec.action, env.ws, actor)
                if act[0] < 0:
                    act = (0, 0, 0)      # skill vanished — treat as end, skip pair
                else:
                    skills = available_skills(ag, env.ws)
                    tt = int(skills[act[0]].features.target_type)
                    pairs.append((obs, tuple(act), tt))
            obs, _, term, trunc, info = env.step(list(act))
            # The engine rejected the expert's action (e.g. out-of-range
            # touch heal) — the action was wasted, so the label would teach
            # a no-op. Drop it.
            res = (info or {}).get("action_result") or {}
            if pairs and res.get("type") == "ERROR":
                pairs.pop()
        else:
            net = mate_nets[arch_of[actor]]
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            act = list(pick_action(el[0], s[0], e[0], g[0],
                                   ws=env.ws, agent_id=actor))
            obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    return pairs


def collect_v1_replay(net, arch: str, n_eps: int,
                      seed0: int = 50_000) -> list[tuple[dict, tuple, int]]:
    """1v1 SELF-replay pairs: the warm-start net plays 1v1 greedily and its
    own actions become labels. Mixed into the team-demo dataset this anchors
    the already-good 1v1 behaviour during fine-tuning — pure team demos were
    measured to catastrophically forget it (life: 43.5% -> 21.9% 1v1) because
    a support's team behaviour (hang back, heal) shares representations with
    its 1v1 behaviour (kite, self-sustain). The two state families are
    distinguishable from obs (ally rows zero in 1v1), so one net can keep
    both: clone yourself where you were good, clone the expert where you
    weren't."""
    from trpg.rl.env_v2 import ARCHETYPE_LIST
    pairs: list[tuple[dict, tuple, int]] = []
    per_opp = max(1, n_eps // len(ARCHETYPE_LIST))
    for k, opp in enumerate(ARCHETYPE_LIST):
        for i in range(per_opp):
            env = CombatEnvV2(seed=seed0 + k * 1000 + i, n_agents=1, n_opps=1)
            obs, _ = env.reset(agent_archs=[arch], opp_archs=[opp])
            done = False
            while not done:
                actor = env.current_agent_id
                ag = env.ws.characters[actor]
                ot = {k2: torch.from_numpy(v).unsqueeze(0)
                      for k2, v in obs.items()}
                with torch.no_grad():
                    el, s, e, g = net(ot)
                s = apply_resource_mask(s, env.resources, env.ws, actor)
                e = apply_entity_mask(e, ot, env.ws, actor)
                act = tuple(pick_action(el[0], s[0], e[0], g[0],
                                        ws=env.ws, agent_id=actor))
                if act[0] == 0:
                    pairs.append((obs, (0, 0, 0), TT_END))
                else:
                    skills = available_skills(ag, env.ws)
                    if act[0] < len(skills):
                        tt = int(skills[act[0]].features.target_type)
                        pairs.append((obs, act, tt))
                obs, _, term, trunc, info = env.step(list(act))
                res = (info or {}).get("action_result") or {}
                if pairs and res.get("type") == "ERROR":
                    pairs.pop()
                done = term or trunc
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, required=True)
    p.add_argument("--model", type=str, default=None,
                   help="checkpoint to fine-tune (default DEFAULT_ROUTING[arch])")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--replay_v1", type=int, default=0,
                   help="1v1 self-replay episodes mixed in to anchor 1v1 "
                        "behaviour (anti-forgetting; needed for support slots)")
    p.add_argument("--skill_weight", action="store_true",
                   help="identity-keyed inverse-frequency weight on the skill "
                        "CE (boosts rare skills, e.g. heals in support demos)")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--eval_games", type=int, default=8)
    args = p.parse_args()

    arch = args.arch
    slot = ROLE_SLOT[ARCHETYPE_ROLES[arch]]
    out_dir = Path(args.out_dir or f"models/bc_team_{arch}")
    out_dir.mkdir(parents=True, exist_ok=True)
    warm = args.model or DEFAULT_ROUTING[arch]
    print(f"arch={arch} slot={slot}  fine-tune from {warm}")

    cache: dict[str, CombatPolicyNet] = {}
    mate_nets: dict[str, CombatPolicyNet] = {}
    for a, path in DEFAULT_ROUTING.items():
        if a == arch:
            continue
        if path not in cache:
            cache[path] = load_net(path)
        mate_nets[a] = cache[path]

    rng = random.Random(args.seed)
    all_pairs: list[tuple[dict, tuple, int]] = []
    for ep in range(args.episodes):
        all_pairs.extend(collect_episode(10_000 + ep, arch, slot, mate_nets, rng))
        if (ep + 1) % 100 == 0:
            print(f"  collected {ep+1}/{args.episodes} episodes, "
                  f"{len(all_pairs)} pairs", flush=True)
    if args.replay_v1 > 0:
        n_team = len(all_pairs)
        replay_net = load_net(warm)
        all_pairs.extend(collect_v1_replay(replay_net, arch, args.replay_v1))
        print(f"  +1v1 self-replay: {len(all_pairs)-n_team} pairs "
              f"({args.replay_v1} eps)", flush=True)

    obs_keys = list(all_pairs[0][0].keys())
    obs_np = {k: np.stack([q[0][k] for q in all_pairs]) for k in obs_keys}
    actions_np = np.array([q[1] for q in all_pairs], dtype=np.int64)
    tts_np = np.array([q[2] for q in all_pairs], dtype=np.int64)
    n = len(all_pairs)
    n_end = int((actions_np[:, 0] == 0).sum())
    print(f"dataset: {n} pairs ({n_end} end, {n-n_end} act)")

    # Identity-keyed inverse-frequency skill weight (same recipe as
    # train_bc.train_bc): rare skills — for a support that's exactly the
    # heal casts, ~5% of act pairs — get up to ~7x weight so the clone
    # actually fires them. Without this the team-demo fine-tune kept the
    # heal rate at a third of the expert's.
    sample_w = np.ones(n, dtype=np.float32)
    if args.skill_weight:
        from collections import Counter
        act_pos = np.where(actions_np[:, 0] > 0)[0]
        chosen = obs_np["skills"][act_pos, actions_np[act_pos, 0]]
        keys = [k.tobytes() for k in np.round(chosen, 3)]
        cnt = Counter(keys)
        n_cls = max(1, len(cnt)); n_act = max(1, len(keys))
        id_w = {k: float(np.sqrt(n_act / (n_cls * c))) for k, c in cnt.items()}
        sample_w = np.zeros(n, dtype=np.float32)
        for j, i in enumerate(act_pos):
            sample_w[i] = id_w[keys[j]]
        print(f"  identity-weight: {len(cnt)} skills, range "
              f"[{min(id_w.values()):.2f}, {max(id_w.values()):.2f}]")

    net = load_net(warm)
    if net.attn_legacy.item() > 0:
        # BC fine-tune re-anchors behaviour to fresh demos, so train and ship
        # under the NATIVE v3 attention regime (presence mask), not the
        # migrated checkpoint's legacy-row emulation.
        net.attn_legacy.fill_(0)
        print("cleared attn_legacy: fine-tuning under native v3 attention")
    net.train()
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    idx = np.arange(n)
    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(idx)
        losses, accs = [], []
        for s0 in range(0, n, args.batch):
            sel = idx[s0:s0 + args.batch]
            obs_b = {k: torch.from_numpy(v[sel]) for k, v in obs_np.items()}
            act_b = torch.from_numpy(actions_np[sel])
            tt_b = torch.from_numpy(tts_np[sel])
            sw_b = torch.from_numpy(sample_w[sel])
            loss, acc = bc_loss_step(net, obs_b, act_b, tt_b, optim,
                                     skill_sample_w=sw_b)
            losses.append(float(loss))
            accs.append(acc)
        sk = np.nanmean([a["skill"] for a in accs])
        gr = np.nanmean([a["grid"] for a in accs])
        en = np.nanmean([a["end"] for a in accs])
        print(f"epoch {epoch}/{args.epochs} loss={np.mean(losses):.3f} "
              f"acc end={en:.2f} skill={sk:.2f} grid={gr:.2f}", flush=True)
        snap = out_dir / f"bc_e{epoch:02d}.pt"
        torch.save(net.state_dict(), snap)

    net.eval()
    wr = eval_team_slot(net, arch, slot, mate_nets, games=args.eval_games)
    ref = eval_team_slot("expert", arch, slot, mate_nets, games=args.eval_games)
    print(f"\nfinal team WR={wr:.1%}  (expert-in-slot {ref:.1%})")


if __name__ == "__main__":
    main()
