"""PPO fine-tune ONE archetype for 3v3 template-party play.

The trained archetype occupies its role's slot; teammates are frozen routed
checkpoints (deployment configuration); opponents are scripted expert teams.
Rewards: env defaults (team PBRS + wasted-move penalty). No new reward terms,
no per-class rewards.

Eval metric during training: team WR with the trained net in its slot + model
mates, vs expert opponent comps, on fixed seeds. The expert-in-slot reference
WR on the SAME seeds is printed at start — that's the bar to beat.

Usage:
  python scripts/train_team_ppo.py --arch battle_master --updates 60
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse
import itertools
import random
from pathlib import Path

import numpy as np
import torch

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import (CombatPolicyNet, apply_resource_mask,
                           apply_entity_mask, pick_action)
from trpg.rl.train_ppo import ppo_update
from trpg.rl.train_team import collect_team_rollout, sample_comp
from trpg.rl.action import encode_action
from trpg.engine.combat_policy import make_archetype_policy
from trpg.scenarios.archetypes import ARCHETYPE_ROLES

from eval_routed import DEFAULT_ROUTING, load_net, stable_seed

ROLE_SLOT = {"front": 0, "striker": 1, "support": 2}


def eval_team_slot(slot_driver, trained_arch: str, trained_slot: int,
                   mate_nets: dict, games: int = 4, n_opp: int = 3,
                   seed_tag: str = "team_eval") -> float:
    """Team WR with ``slot_driver`` in the trained slot, model mates, expert opps.

    slot_driver: CombatPolicyNet (greedy) or the string "expert".
    Comps: every template comp that has trained_arch in its slot; opponent
    comps drawn deterministically; fixed seeds — comparable across calls.
    """
    buckets = (
        sorted(a for a, r in ARCHETYPE_ROLES.items() if r == "front"),
        sorted(a for a, r in ARCHETYPE_ROLES.items() if r == "striker"),
        sorted(a for a, r in ARCHETYPE_ROLES.items() if r == "support"),
    )
    slots_pool = [([trained_arch] if i == trained_slot else list(b))
                  for i, b in enumerate(buckets)]
    comps = [list(c) for c in itertools.product(*slots_pool)]
    all_comps = [list(c) for c in itertools.product(*buckets)]
    rng = random.Random(20260610)
    opp_set = [all_comps[i] for i in rng.sample(range(len(all_comps)), n_opp)]

    is_expert = isinstance(slot_driver, str)
    wins = n = 0
    for comp in comps:
        for opp_comp in opp_set:
            base = stable_seed(f"{seed_tag}_{'+'.join(comp)}_{'+'.join(opp_comp)}")
            for g in range(games):
                env = CombatEnvV2(seed=base + g, n_agents=3, n_opps=3)
                obs, _ = env.reset(agent_archs=comp, opp_archs=opp_comp, level=5)
                arch_of = dict(zip(env.agent_ids, env.agent_archs))
                trained_id = env.agent_ids[trained_slot]
                expert_pol = (make_archetype_policy(trained_arch)
                              if is_expert else None)
                done = False
                while not done:
                    actor = env.current_agent_id
                    if actor == trained_id and is_expert:
                        ag = env.ws.characters[actor]
                        dec = expert_pol.decide(actor, ag, env.ws, env.resources,
                                                env.ws.combat.round_number)
                        act = ([0, 0, 0] if (dec.action is None or dec.fled)
                               else list(encode_action(dec.action, env.ws, actor)))
                    else:
                        net = (slot_driver if actor == trained_id
                               else mate_nets[arch_of[actor]])
                        ot = {k: torch.from_numpy(v).unsqueeze(0)
                              for k, v in obs.items()}
                        with torch.no_grad():
                            el, s, e, g_l = net(ot)
                        s = apply_resource_mask(s, env.resources, env.ws, actor)
                        e = apply_entity_mask(e, ot, env.ws, actor)
                        act = list(pick_action(el[0], s[0], e[0], g_l[0],
                                               ws=env.ws, agent_id=actor))
                    obs, _, term, trunc, _ = env.step(act)
                    done = term or trunc
                opps_dead = all(not env.ws.characters[oid].is_alive()
                                for oid in env.opp_ids)
                team_alive = any(env.ws.characters[aid].is_alive()
                                 for aid in env.agent_ids)
                n += 1
                wins += int(opps_dead and team_alive)
    return wins / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, required=True)
    p.add_argument("--model", type=str, default=None,
                   help="warm-start checkpoint (default: DEFAULT_ROUTING[arch])")
    p.add_argument("--updates", type=int, default=60)
    p.add_argument("--steps", type=int, default=2048,
                   help="learner transitions per update")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--value_warmup", type=int, default=3)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_games", type=int, default=4)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--asym", action="store_true",
                   help="sample a cross-side level gap per episode "
                        "(LEVEL_DELTA_TABLE, symmetric-majority)")
    args = p.parse_args()

    arch = args.arch
    role = ARCHETYPE_ROLES[arch]
    slot = ROLE_SLOT[role]
    out_dir = Path(args.out_dir or f"models/ppo_team_{arch}")
    out_dir.mkdir(parents=True, exist_ok=True)

    warm = args.model or DEFAULT_ROUTING[arch]
    print(f"arch={arch} role={role} slot={slot}  warm-start={warm}")

    net = load_net(warm)          # drops shape-mismatched (old critic) keys
    if net.attn_legacy.item() > 0:
        # Warm-started from a migrated pre-v3 checkpoint: train under the
        # NATIVE v3 attention regime (presence mask) so the product of this
        # run is a native v3 net. The warm-start eval below shows any dip.
        net.attn_legacy.fill_(0)
        print("cleared attn_legacy: training under native v3 attention")
    net.train()

    print("loading frozen mates from routing table...")
    cache: dict[str, CombatPolicyNet] = {}
    mate_nets: dict[str, CombatPolicyNet] = {}
    for a, path in DEFAULT_ROUTING.items():
        if a == arch:
            continue
        if path not in cache:
            cache[path] = load_net(path)
        mate_nets[a] = cache[path]

    optim = torch.optim.Adam([q for q in net.parameters() if q.requires_grad],
                             lr=args.lr)

    expert_ref = eval_team_slot("expert", arch, slot, mate_nets,
                                games=args.eval_games)
    net.eval()
    wr0 = eval_team_slot(net, arch, slot, mate_nets, games=args.eval_games)
    print(f"expert-in-slot reference WR = {expert_ref:.0%}")
    print(f"warm-start model WR        = {wr0:.0%}\n", flush=True)

    best_wr = wr0
    torch.save(net.state_dict(), out_dir / "ppo_best.pt")
    for update in range(1, args.updates + 1):
        batch = collect_team_rollout(net, mate_nets, arch, slot,
                                     n_steps=args.steps, seed=update,
                                     device="cpu", asym_levels=args.asym)
        info = ppo_update(net, batch, optim, n_epochs=args.epochs,
                          batch_size=args.batch, ent_coef=args.ent_coef,
                          device="cpu",
                          value_only=(update <= args.value_warmup))
        tag = " [warmup]" if update <= args.value_warmup else ""
        print(f"Update {update:3d}/{args.updates}{tag}  "
              f"policy={info['policy_loss']:+.4f}  "
              f"value={info['value_loss']:.4f}  "
              f"entropy={info['entropy']:.3f}", flush=True)

        if update % args.eval_every == 0:
            net.eval()
            wr = eval_team_slot(net, arch, slot, mate_nets,
                                games=args.eval_games)
            net.train()
            snap = out_dir / f"ppo_u{update:04d}.pt"
            torch.save(net.state_dict(), snap)
            marker = " *" if wr > best_wr else ""
            print(f"  => team WR={wr:.0%}  (expert ref {expert_ref:.0%}, "
                  f"warm-start {wr0:.0%}, best {best_wr:.0%}){marker}  "
                  f"snapshot={snap.name}", flush=True)
            if wr > best_wr:
                best_wr = wr
                torch.save(net.state_dict(), out_dir / "ppo_best.pt")

    print(f"\ndone. best team WR={best_wr:.0%} (expert ref {expert_ref:.0%})")


if __name__ == "__main__":
    main()
