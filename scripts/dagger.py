"""DAgger (Dataset Aggregation) on top of a BC checkpoint.

Loop:
  1. Load init model (default: bc_v15.pt)
  2. For each iter:
     a. Roll out the *model* in env, every step query the scripted expert
        for what it would do (relabel). Record (obs, expert_action) pairs.
     b. Append to growing dataset (aggregated across iters)
     c. Train a few epochs at low LR on the aggregated data
  3. Save final model

Why this fixes vengeance/devotion: BC only sees expert states. The model at
inference visits states the expert never visits (e.g. round 1 with no
vow_of_enmity active), where it has no useful training signal. DAgger
generates exactly those off-trajectory states and asks the expert what to
do, closing the distribution-shift gap.

Usage:
    python scripts/dagger.py --init models/bc_v15.pt \
        --iters 5 --episodes_per_arch 30 \
        --train_epochs 3 --lr 1e-4 \
        --out models/dagger_v1.pt
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from trpg.engine.combat_policy import make_archetype_policy
from trpg.engine.skill import available_skills
from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST, _AGENT_ID
from trpg.rl.obs import build_obs
from trpg.rl.action import encode_action
from trpg.rl.model import (
    CombatPolicyNet, apply_resource_mask, apply_entity_mask, pick_action,
)
from trpg.rl.bc_collect import TT_END
from trpg.rl.train_bc import bc_loss_step


def rollout_with_relabel(net, agent_arch: str, opp_arch: str, seed: int,
                          device: str) -> list[tuple[dict, tuple, int]]:
    """Model drives the env; expert is queried at every state for the label.

    Returns list of (obs, encoded_action_from_expert, target_type) pairs.
    """
    env = CombatEnvV2(seed=seed)
    obs, _ = env.reset(agent_arch=agent_arch, opponent_arch=opp_arch)
    expert = make_archetype_policy(agent_arch)
    pairs: list[tuple[dict, tuple, int]] = []
    done = False
    while not done:
        agent = env.ws.characters[_AGENT_ID]
        # Expert relabel: ask what it would do at this exact obs/state
        decision = expert.decide(
            _AGENT_ID, agent, env.ws, env.resources,
            env.ws.combat.round_number,
        )
        if decision.action is None or decision.fled:
            if not decision.fled:
                pairs.append((obs, (0, 0, 0), TT_END))
        else:
            enc = encode_action(decision.action, env.ws, _AGENT_ID)
            if enc[0] >= 0:
                skills = available_skills(agent, env.ws)
                tt = int(skills[enc[0]].features.target_type)
                pairs.append((obs, enc, tt))

        # Model drives: pick its own action, step env
        obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device)
                 for k, v in obs.items()}
        with torch.no_grad():
            end_l, s, e, g = net(obs_t)
        s = apply_resource_mask(s, env.resources, env.ws, _AGENT_ID)
        e = apply_entity_mask(e, obs_t)
        action = list(pick_action(end_l[0], s[0], e[0], g[0],
                                   ws=env.ws, agent_id=_AGENT_ID))
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    return pairs


def collect_iteration(net, episodes_per_arch: int, seed: int,
                       device: str) -> list[tuple[dict, tuple, int]]:
    rng = np.random.default_rng(seed)
    all_pairs: list[tuple[dict, tuple, int]] = []
    for arch in ARCHETYPE_LIST:
        for _ in range(episodes_per_arch):
            opp = ARCHETYPE_LIST[rng.integers(len(ARCHETYPE_LIST))]
            ep_seed = int(rng.integers(0, 2**31))
            all_pairs.extend(rollout_with_relabel(net, arch, opp, ep_seed, device))
    return all_pairs


def stack_dataset(pairs: list[tuple[dict, tuple, int]]) -> dict:
    obs_keys = list(pairs[0][0].keys())
    return {
        "obs":          {k: np.stack([p[0][k] for p in pairs], axis=0)
                          for k in obs_keys},
        "actions":      np.array([p[1] for p in pairs], dtype=np.int64),
        "target_types": np.array([p[2] for p in pairs], dtype=np.int64),
    }


def train_pass(net, ds: dict, epochs: int, batch: int, lr: float,
                device: str) -> None:
    actions = torch.from_numpy(ds["actions"]).long()
    target_types = torch.from_numpy(ds["target_types"]).long()
    n = actions.shape[0]
    idx = np.arange(n)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    for ep in range(epochs):
        np.random.shuffle(idx)
        ep_losses, ep_accs = [], {"end": [], "skill": [], "entity": [], "grid": []}
        for start in range(0, n, batch):
            sel = idx[start:start + batch]
            obs_b = {k: torch.from_numpy(v[sel]).to(device)
                     for k, v in ds["obs"].items()}
            loss, acc = bc_loss_step(net, obs_b,
                                      actions[sel].to(device),
                                      target_types[sel].to(device),
                                      optim)
            ep_losses.append(float(loss))
            for k in ep_accs:
                if acc[k] == acc[k]:
                    ep_accs[k].append(acc[k])
        print(f"   train epoch {ep+1}/{epochs}: loss={np.mean(ep_losses):.3f}  "
              f"end={np.mean(ep_accs['end']):.3f} "
              f"skill={np.mean(ep_accs['skill']):.3f} "
              f"entity={np.mean(ep_accs['entity']):.3f} "
              f"grid={np.mean(ep_accs['grid']):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="BC checkpoint to start from")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--episodes_per_arch", type=int, default=30)
    ap.add_argument("--train_epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Init:   {args.init}")

    net = CombatPolicyNet().to(device)
    net.load_state_dict(torch.load(args.init, map_location=device,
                                    weights_only=True))

    all_pairs: list[tuple[dict, tuple, int]] = []
    for it in range(args.iters):
        print(f"\n=== DAgger iter {it+1}/{args.iters} ===")
        t0 = time.time()
        net.eval()
        new_pairs = collect_iteration(net, args.episodes_per_arch,
                                       args.seed + it, device)
        all_pairs.extend(new_pairs)
        print(f"   collected {len(new_pairs):,} new pairs  "
              f"(total {len(all_pairs):,})  "
              f"in {time.time() - t0:.1f}s")
        ds = stack_dataset(all_pairs)
        train_pass(net, ds, args.train_epochs, args.batch, args.lr, device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), str(out_path))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
