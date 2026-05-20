"""PPO fine-tuning from a BC checkpoint.

Usage:
    python scripts/train_ppo.py [--model PATH] [--updates N] [--steps N]

Loads a BC-trained model and continues training with PPO.
Saves checkpoints by win-rate (low/mid/boss thresholds).
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
import torch
import numpy as np

from trpg.rl.model import CombatPolicyNet, apply_resource_mask, apply_entity_mask
from trpg.rl.train_ppo import collect_ppo_rollout, ppo_update
from trpg.rl.env_v2 import CombatEnvV2, _AGENT_ID


CHECKPOINT_THRESHOLDS = {"low": 0.40, "mid": 0.55, "boss": 0.70}


def qualify_candidate(candidate, pool, games_per_opp: int, threshold: float,
                       seed: int, device: str = "cpu") -> tuple[bool, float]:
    """Test ``candidate`` against every snapshot in ``pool``.

    Returns ``(qualified, win_rate)``. Win-rate is aggregated across all
    matches (games_per_opp × len(pool) total). Qualified iff win_rate >= threshold.
    """
    if not pool:
        return True, 1.0   # empty pool: candidate is seed, no test needed
    from trpg.rl.model import pick_action
    candidate.to(device).eval()
    wins = 0
    total = 0
    for opp_idx, opp_net in enumerate(pool):
        opp_net.to("cpu").eval()
        for g in range(games_per_opp):
            env = CombatEnvV2(seed=seed + opp_idx * games_per_opp + g)
            env.use_self_play_opponent(opp_net)
            obs, _ = env.reset()
            done = False
            while not done:
                obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device)
                         for k, v in obs.items()}
                with torch.no_grad():
                    end_l, s, e, gg = candidate(obs_t)
                s = apply_resource_mask(s, env.resources, env.ws, _AGENT_ID)
                e = apply_entity_mask(e, obs_t)
                action = list(pick_action(end_l[0], s[0], e[0], gg[0],
                                           ws=env.ws, agent_id=_AGENT_ID))
                obs, _, term, trunc, _ = env.step(action)
                done = term or trunc
            if not env.ws.characters["opponent"].is_alive():
                wins += 1
            total += 1
    wr = wins / total if total else 0.0
    return wr >= threshold, wr


def evaluate(net, n_episodes=100, device="cuda"):
    net.to(device).eval()
    env = CombatEnvV2(seed=88888)
    wins = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device)
                     for k, v in obs.items()}
            with torch.no_grad():
                end_l, s, e, g = net(obs_t)
            from trpg.rl.env_v2 import _AGENT_ID
            from trpg.rl.model import pick_action
            s = apply_resource_mask(s, env.resources, env.ws, _AGENT_ID)
            e = apply_entity_mask(e, obs_t)
            action = list(pick_action(end_l[0], s[0], e[0], g[0],
                                       ws=env.ws, agent_id=_AGENT_ID))
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
        # Only count as win if opponent actually died (truncation = draw, not win)
        if not env.ws.characters["opponent"].is_alive():
            wins += 1
    return wins / n_episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=str, default="models/bc_v1.pt",
                        help="BC checkpoint to warm-start from, "
                             "or 'scratch' for random init")
    parser.add_argument("--updates", type=int, default=100,
                        help="PPO update iterations")
    parser.add_argument("--value_warmup", type=int, default=0,
                        help="updates spent training only V(s) before the "
                             "policy gradient is enabled. Needed when starting "
                             "from a BC checkpoint — BC never trained value_head, "
                             "so its initial output is random and contaminates "
                             "the first batch of advantage estimates.")
    parser.add_argument("--kl_coef", type=float, default=0.0,
                        help="KL anchor strength. Loss adds kl_coef·KL(π‖π_ref) "
                             "where π_ref is the BC checkpoint we started from. "
                             "Prevents PPO from drifting arbitrarily far from BC "
                             "over many updates. Try 0.01-0.1.")
    parser.add_argument("--steps",   type=int, default=1024,
                        help="steps per rollout")
    parser.add_argument("--epochs",  type=int, default=4,
                        help="minibatch epochs per update")
    parser.add_argument("--batch",   type=int, default=256)
    parser.add_argument("--n_envs",   type=int, default=8,
                        help="parallel rollout workers (use CPU cores)")
    parser.add_argument("--ent_coef",     type=float, default=0.02,
                        help="entropy bonus coefficient")
    parser.add_argument("--curriculum",   type=int,   default=50,
                        help="first N updates use HeuristicCombatPolicy opponents (easier)")
    parser.add_argument("--lr",       type=float, default=3e-5,
                        help="use smaller LR than BC to avoid forgetting")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--self_play", action="store_true",
                        help="train against a pool of past snapshots instead of "
                             "expert ArchetypePolicy")
    parser.add_argument("--snapshot_every", type=int, default=10,
                        help="add a new snapshot to the opponent pool every N updates")
    parser.add_argument("--pool_size",     type=int, default=10,
                        help="max snapshots in the opponent pool (oldest dropped)")
    parser.add_argument("--qual_games", type=int, default=50,
                        help="qualification games per pool opponent for new snapshots")
    parser.add_argument("--qual_threshold", type=float, default=0.55,
                        help="min win-rate vs current pool to accept a new snapshot")
    parser.add_argument("--save_every_eval", action="store_true",
                        help="save a numbered snapshot ppo_uNNN.pt at every eval, "
                             "in addition to ppo_best.pt (preserves every stage)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    net = CombatPolicyNet().to(device)
    if args.model.lower() in ("scratch", "none", ""):
        print(f"Random init (no warm-start)  device={device}")
    else:
        net.load_state_dict(torch.load(args.model, map_location=device))
        print(f"Loaded: {args.model}  device={device}")

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    saved = set()

    # Initial eval — 100 episodes (std ≈ 5%, vs 7% at 50 eps)
    wr = evaluate(net, n_episodes=100, device=device)
    print(f"\nStart win-rate: {wr:.0%}")
    best_wr = wr
    best_path = out_dir / "ppo_best.pt"
    torch.save(net.state_dict(), str(best_path))
    print(f"  => baseline saved as best: {best_path} ({wr:.0%})")

    # Frozen reference policy for KL anchor — a copy of the loaded BC model.
    # Kept on the training device so the forward pass runs in lockstep with
    # the current policy. eval() + no_grad inside ppo_update prevents updates.
    reference_net = None
    if args.kl_coef > 0.0:
        from copy import deepcopy
        reference_net = deepcopy(net).to(device).eval()
        for p in reference_net.parameters():
            p.requires_grad_(False)
        print(f"  [kl-anchor] reference policy frozen, kl_coef={args.kl_coef}")
    # Required margin to overwrite best — 3% guards against sample noise so
    # the "best" checkpoint reflects a real improvement, not a lucky eval.
    best_margin = 0.03

    # Opponent pool — list of frozen snapshots, sampled per rollout.
    opponent_pool: list = []
    if args.self_play:
        from copy import deepcopy
        opponent_pool.append(deepcopy(net).to("cpu").eval())
        print(f"  [pool] initial snapshot added  "
              f"(pool size limit {args.pool_size}, add every {args.snapshot_every} updates)")

    import random as _py_random
    rng = _py_random.Random(0)

    for update in range(1, args.updates + 1):
        use_heuristic = (update <= args.curriculum)
        # Once past curriculum, choose between expert ArchetypePolicy (default)
        # or self-play. Heuristic curriculum still uses HeuristicCombatPolicy.
        if update == args.curriculum + 1:
            mode = "pool self-play" if args.self_play else "expert opponents"
            print(f"  [curriculum] switching to {mode} at update {update}")
        # Qualification check: every snapshot_every updates, the current model
        # plays games_per_opp games against EACH pool member. Only added if its
        # aggregate win-rate beats the threshold — keeps the pool monotonically
        # stronger and discourages sideways drift.
        if (args.self_play and update > args.curriculum
                and (update - args.curriculum - 1) % args.snapshot_every == 0
                and update > args.curriculum + 1):
            from copy import deepcopy
            candidate = deepcopy(net).to("cpu").eval()
            qualified, wr = qualify_candidate(
                candidate, opponent_pool,
                games_per_opp=args.qual_games,
                threshold=args.qual_threshold,
                seed=10_000 + update, device="cpu",
            )
            n_games = args.qual_games * len(opponent_pool)
            if qualified:
                opponent_pool.append(candidate)
                while len(opponent_pool) > args.pool_size:
                    opponent_pool.pop(0)
                print(f"  [pool] candidate ACCEPTED at update {update} "
                      f"({wr:.0%} over {n_games} games, pool size {len(opponent_pool)})")
            else:
                print(f"  [pool] candidate rejected at update {update} "
                      f"({wr:.0%} over {n_games} games < {args.qual_threshold:.0%})")
        active_opponent = None
        if args.self_play and not use_heuristic and opponent_pool:
            active_opponent = rng.choice(opponent_pool)
        batch = collect_ppo_rollout(net, n_steps=args.steps,
                                    n_envs=args.n_envs,
                                    seed=update, device=device,
                                    use_heuristic_opponent=use_heuristic,
                                    opponent_net=active_opponent)
        is_warmup = update <= args.value_warmup
        info = ppo_update(net, batch, optim,
                          n_epochs=args.epochs, batch_size=args.batch,
                          ent_coef=args.ent_coef,
                          device=device,
                          value_only=is_warmup,
                          reference_net=reference_net,
                          kl_coef=args.kl_coef)

        tag = " [warmup]" if is_warmup else ""
        kl_tag = f"  kl={info.get('kl', 0):.4f}" if args.kl_coef > 0 else ""
        print(f"Update {update:4d}/{args.updates}{tag}  "
              f"policy={info['policy_loss']:+.4f}  "
              f"value={info['value_loss']:.4f}  "
              f"entropy={info['entropy']:.3f}{kl_tag}")

        if update % args.eval_every == 0:
            wr = evaluate(net, n_episodes=100, device=device)
            print(f"  => win-rate: {wr:.0%}  (best so far: {best_wr:.0%})")

            if args.save_every_eval:
                snap_path = out_dir / f"ppo_u{update:04d}.pt"
                torch.save(net.state_dict(), str(snap_path))
                print(f"  => snapshot: {snap_path} ({wr:.0%})")

            if wr >= best_wr + best_margin:
                best_wr = wr
                torch.save(net.state_dict(), str(best_path))
                print(f"  => NEW BEST: {best_path} ({wr:.0%})")

            for label, threshold in CHECKPOINT_THRESHOLDS.items():
                if wr >= threshold and label not in saved:
                    path = out_dir / f"ppo_{label}.pt"
                    torch.save(net.state_dict(), str(path))
                    saved.add(label)
                    print(f"  => checkpoint saved: {path}")

    # Final save
    torch.save(net.state_dict(), str(out_dir / "ppo_final.pt"))
    print(f"\nFinal model: {out_dir}/ppo_final.pt")
    print(f"Best model:  {best_path}  (win-rate {best_wr:.0%})")


if __name__ == "__main__":
    main()
