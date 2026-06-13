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
from trpg.rl.env_v2 import CombatEnvV2


CHECKPOINT_THRESHOLDS = {"low": 0.40, "mid": 0.55, "boss": 0.70}


def save_training_state(path, net, optim, update, best_wr, saved,
                          opponent_pool):
    """Persist full training state for --resume.

    Saves model + optimizer + scheduler bookkeeping + opponent pool so a
    later run can pick up exactly where this one left off. Without the
    optim state, Adam momentum resets and the first few resumed updates
    are noisy; without the opponent pool, self-play loses the diverse
    snapshots that took many updates to qualify.
    """
    state = {
        "model":         net.state_dict(),
        "optim":         optim.state_dict(),
        "update":        update,
        "best_wr":       best_wr,
        "saved":         sorted(saved),
        "opponent_pool": [snap.state_dict() for snap in opponent_pool],
    }
    torch.save(state, str(path))


def load_training_state(path, net, optim, device):
    """Restore state dict written by save_training_state. Returns a dict
    with the non-tensor bookkeeping fields needed by the caller."""
    state = torch.load(path, map_location=device)
    # strict=False so we can resume from an older training_state.pt that
    # predates new params (e.g. FiLM); the new params keep their identity
    # init. Same for opponent_pool snapshots.
    net.load_state_dict(state["model"], strict=False)
    optim.load_state_dict(state["optim"])
    pool = []
    for sd in state.get("opponent_pool", []):
        snap = CombatPolicyNet()
        snap.load_state_dict(sd, strict=False)
        snap.to("cpu").eval()
        pool.append(snap)
    return {
        "update":        int(state["update"]),
        "best_wr":       float(state["best_wr"]),
        "saved":         set(state["saved"]),
        "opponent_pool": pool,
    }


_QUAL_SIZES = (1, 2, 3)  # balanced configs only — no asymmetric matchups in qual


def qualify_candidate(candidate, pool, games_per_opp: int, threshold: float,
                       seed: int, device: str = "cpu",
                       n_agents: int | None = None,
                       n_opps: int | None = None) -> tuple[bool, float]:
    """Test ``candidate`` against every snapshot in ``pool``.

    Returns ``(qualified, win_rate)``. Win-rate is aggregated across all
    matches (games_per_opp × len(pool) total). Qualified iff win_rate >= threshold.

    Qualification always uses balanced configs (1v1, 2v2, 3v3, cycling by game
    index) so every candidate faces identical conditions regardless of the
    training n_agents/n_opps setting.
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
            size = _QUAL_SIZES[g % len(_QUAL_SIZES)]
            env = CombatEnvV2(seed=seed + opp_idx * games_per_opp + g,
                              n_agents=size, n_opps=size)
            env.use_self_play_opponent(opp_net)
            obs, _ = env.reset()
            done = False
            while not done:
                agent_id = env.current_agent_id
                obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device)
                         for k, v in obs.items()}
                with torch.no_grad():
                    end_l, s, e, gg = candidate(obs_t)
                s = apply_resource_mask(s, env.resources, env.ws, agent_id)
                e = apply_entity_mask(e, obs_t)
                action = list(pick_action(end_l[0], s[0], e[0], gg[0],
                                          ws=env.ws, agent_id=agent_id))
                obs, _, term, trunc, _ = env.step(action)
                done = term or trunc
            if (all(not env.ws.characters[oid].is_alive() for oid in env.opp_ids)
                    and any(env.ws.characters[aid].is_alive() for aid in env.agent_ids)):
                wins += 1
            total += 1
    wr = wins / total if total else 0.0
    return wr >= threshold, wr


def evaluate(net, n_episodes=100, device="cuda",
             n_agents: int | None = None, n_opps: int | None = None,
             agent_arch: str | None = None):
    """Evaluate win-rate over n_episodes.

    n_agents / n_opps: fixed team sizes; None = sample from weighted distribution
    (same as training). Expert archetype policies play as opponents.
    ``agent_arch``: pin the agent to one archetype (specialist eval).
    """
    from trpg.rl.model import pick_action
    net.to(device).eval()
    env = CombatEnvV2(seed=88888, n_agents=n_agents, n_opps=n_opps)
    wins = 0
    for _ in range(n_episodes):
        obs, _ = env.reset(agent_archs=[agent_arch] if agent_arch else None)
        done = False
        while not done:
            agent_id = env.current_agent_id
            obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device)
                     for k, v in obs.items()}
            with torch.no_grad():
                end_l, s, e, g = net(obs_t)
            s = apply_resource_mask(s, env.resources, env.ws, agent_id)
            e = apply_entity_mask(e, obs_t)
            action = list(pick_action(end_l[0], s[0], e[0], g[0],
                                      ws=env.ws, agent_id=agent_id))
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
        # Win = ALL opponents dead AND the agent side survives. Requiring the
        # agent alive matches eval_v2's win condition; without it a mutual kill
        # (agent trades its life to drop the opp) counts as a win, which steers
        # checkpoint selection toward suicidal/over-extending policies that
        # eval_v2 scores as losses (measured: champion/assassin ppo_best picked
        # this way eval_v2-worse than the BC init).
        opps_dead = all(not env.ws.characters[oid].is_alive() for oid in env.opp_ids)
        agents_alive = any(env.ws.characters[aid].is_alive() for aid in env.agent_ids)
        if opps_dead and agents_alive:
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
    parser.add_argument("--n_envs",  type=int, default=8,
                        help="parallel rollout workers (use CPU cores)")
    parser.add_argument("--ent_coef",    type=float, default=0.02,
                        help="entropy bonus coefficient")
    parser.add_argument("--curriculum",  type=int,   default=50,
                        help="first N updates use HeuristicCombatPolicy opponents (easier)")
    parser.add_argument("--lr",      type=float, default=3e-5,
                        help="use smaller LR than BC to avoid forgetting")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--self_play", action="store_true",
                        help="train against a pool of past snapshots instead of "
                             "expert ArchetypePolicy")
    parser.add_argument("--snapshot_every", type=int, default=10,
                        help="add a new snapshot to the opponent pool every N updates")
    parser.add_argument("--pool_size",    type=int, default=10,
                        help="max snapshots in the opponent pool (oldest dropped)")
    parser.add_argument("--qual_games", type=int, default=50,
                        help="qualification games per pool opponent for new snapshots")
    parser.add_argument("--qual_threshold", type=float, default=0.55,
                        help="min win-rate vs current pool to accept a new snapshot")
    parser.add_argument("--save_every_eval", action="store_true",
                        help="save a numbered snapshot ppo_uNNN.pt at every eval, "
                             "in addition to ppo_best.pt (preserves every stage)")
    parser.add_argument("--n_agents", type=int, default=None,
                        help="fixed number of model-controlled agents (1-3). "
                             "Omit to sample from weighted distribution each episode.")
    parser.add_argument("--n_opps",   type=int, default=None,
                        help="fixed number of opponents (1-3). "
                             "Omit to sample from weighted distribution each episode.")
    parser.add_argument("--agent_arch", type=str, default=None,
                        help="pin the agent to ONE archetype (specialist training/eval). "
                             "Omit to sample archetypes each episode.")
    parser.add_argument("--expert_mix", type=float, default=0.0,
                        help="Fraction (0..1) of parallel rollout workers that "
                             "face scripted archetype experts instead of the "
                             "self-play snapshot. 0=pure self-play, 0.5=half "
                             "self-play half expert. Forces OOD opponents into "
                             "the pool so policies (esp. ranged kiters) can't "
                             "converge to static Nash equilibria.")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze encoder params (skill_proj, skill_encoder, "
                             "entity_mlp, res_mlp, trunk). Only PPO-train the "
                             "heads. Use to test whether encoder drift is what "
                             "kills individual archetypes' policies over long PPO.")
    parser.add_argument("--pcgrad", action="store_true",
                        help="Apply PCGrad surgery: per-archetype policy "
                             "gradients are projected away from each other's "
                             "negative-cosine components before optimiser step. "
                             "Directly attacks the ~55%% cross-class gradient "
                             "conflict measured on shared trunk/heads under v19. "
                             "~5x slower training (12 backwards per minibatch).")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to training_state.pt from a previous run. "
                             "Restores model, optimizer, update counter, "
                             "best_wr, threshold-saved flags, and opponent pool. "
                             "--updates is interpreted as TOTAL target, so "
                             "--resume from update 500 with --updates 2000 runs "
                             "1500 more. --model is still required (used only "
                             "if --resume file is missing) but is otherwise "
                             "ignored.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    net = CombatPolicyNet().to(device)
    if args.model.lower() in ("scratch", "none", ""):
        print(f"Random init (no warm-start)  device={device}")
    else:
        # strict=False so a BC checkpoint that pre-dates FiLM (or any other
        # post-BC additions) still loads — missing params keep their init.
        sd = torch.load(args.model, map_location=device)
        sd = CombatPolicyNet.adapt_state_dict_for_perarch(sd)
        # Drop keys whose shape no longer matches (e.g. the critic input grew
        # when we gave it the skill-pool feature). strict=False ignores
        # missing/extra keys but still ERRORS on a shape mismatch, so filter
        # first. The dropped params (random in a BC checkpoint anyway) keep
        # their fresh init and train from scratch under PPO.
        model_sd = net.state_dict()
        dropped = [k for k, v in sd.items()
                   if k in model_sd and v.shape != model_sd[k].shape]
        for k in dropped:
            del sd[k]
        net.load_state_dict(sd, strict=False)
        print(f"Loaded: {args.model}  device={device}"
              + (f"  (reinit {len(dropped)} reshaped: {dropped})" if dropped else ""))

    if args.freeze_encoder:
        encoder_prefixes = ("skill_proj", "skill_encoder", "entity_mlp",
                            "res_mlp", "trunk")
        n_frozen = 0
        for name, p in net.named_parameters():
            if name.startswith(encoder_prefixes):
                p.requires_grad_(False)
                n_frozen += p.numel()
        n_total = sum(p.numel() for p in net.parameters())
        print(f"  [freeze] encoder frozen: {n_frozen}/{n_total} params "
              f"({n_frozen/n_total:.0%}) not trained")

    optim = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad],
        lr=args.lr,
    )
    saved = set()
    best_path  = out_dir / "ppo_best.pt"
    state_path = out_dir / "training_state.pt"

    if args.n_agents is not None:
        mode_tag = f" [{args.n_agents}v{args.n_opps or args.n_agents}]"
    else:
        mode_tag = " [weighted]"

    # Opponent pool — list of frozen snapshots, sampled per rollout.
    opponent_pool: list = []

    start_update = 1
    best_wr = 0.0

    if args.resume and Path(args.resume).exists():
        print(f"\nResuming from: {args.resume}")
        restored = load_training_state(args.resume, net, optim, device)
        start_update   = restored["update"] + 1
        best_wr        = restored["best_wr"]
        saved          = restored["saved"]
        opponent_pool  = restored["opponent_pool"]
        print(f"  resumed at update {restored['update']}/{args.updates}, "
              f"best_wr={best_wr:.0%}, "
              f"pool_size={len(opponent_pool)}, "
              f"saved_thresholds={sorted(saved)}")
        if start_update > args.updates:
            print(f"  [resume] start_update {start_update} > target {args.updates}; nothing to do.")
            return
    else:
        if args.resume:
            print(f"\n--resume {args.resume} not found; starting fresh.")
        # Fresh-start init: initial eval + best baseline + pool seed
        wr = evaluate(net, n_episodes=100, device=device,
                      n_agents=args.n_agents, n_opps=args.n_opps,
                      agent_arch=args.agent_arch)
        print(f"\nStart win-rate{mode_tag}: {wr:.0%}")
        best_wr = wr
        torch.save(net.state_dict(), str(best_path))
        print(f"  => baseline saved as best: {best_path} ({wr:.0%})")
        if args.self_play:
            from copy import deepcopy
            opponent_pool.append(deepcopy(net).to("cpu").eval())
            print(f"  [pool] initial snapshot added  "
                  f"(pool size limit {args.pool_size}, add every {args.snapshot_every} updates)")

    # Frozen reference policy for KL anchor — a copy of the loaded model.
    reference_net = None
    if args.kl_coef > 0.0:
        from copy import deepcopy
        reference_net = deepcopy(net).to(device).eval()
        for p in reference_net.parameters():
            p.requires_grad_(False)
        print(f"  [kl-anchor] reference policy frozen, kl_coef={args.kl_coef}")
    # Required margin to overwrite best — 3% guards against sample noise.
    best_margin = 0.03

    import random as _py_random
    rng = _py_random.Random(0)

    for update in range(start_update, args.updates + 1):
        use_heuristic = (update <= args.curriculum)
        if update == args.curriculum + 1:
            mode = "pool self-play" if args.self_play else "expert opponents"
            print(f"  [curriculum] switching to {mode} at update {update}")
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
                n_agents=args.n_agents, n_opps=args.n_opps,
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
                                    opponent_net=active_opponent,
                                    n_agents=args.n_agents,
                                    n_opps=args.n_opps,
                                    expert_mix=args.expert_mix,
                                    agent_arch=args.agent_arch)
        is_warmup = update <= args.value_warmup
        info = ppo_update(net, batch, optim,
                          n_epochs=args.epochs, batch_size=args.batch,
                          ent_coef=args.ent_coef,
                          device=device,
                          value_only=is_warmup,
                          reference_net=reference_net,
                          kl_coef=args.kl_coef,
                          pcgrad=args.pcgrad)

        tag = " [warmup]" if is_warmup else ""
        kl_tag = f"  kl={info.get('kl', 0):.4f}" if args.kl_coef > 0 else ""
        print(f"Update {update:4d}/{args.updates}{tag}  "
              f"policy={info['policy_loss']:+.4f}  "
              f"value={info['value_loss']:.4f}  "
              f"entropy={info['entropy']:.3f}{kl_tag}")

        if update % args.eval_every == 0:
            wr = evaluate(net, n_episodes=100, device=device,
                          n_agents=args.n_agents, n_opps=args.n_opps,
                          agent_arch=args.agent_arch)
            print(f"  => win-rate{mode_tag}: {wr:.0%}  (best so far: {best_wr:.0%})")

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

            # Persist resumable training state at every eval point.
            save_training_state(state_path, net, optim, update, best_wr,
                                 saved, opponent_pool)

    # Final save (model weights + resumable state)
    torch.save(net.state_dict(), str(out_dir / "ppo_final.pt"))
    save_training_state(state_path, net, optim, args.updates, best_wr,
                         saved, opponent_pool)
    print(f"\nFinal model: {out_dir}/ppo_final.pt")
    print(f"Best model:  {best_path}  (win-rate {best_wr:.0%})")
    print(f"Resume state: {state_path}")


if __name__ == "__main__":
    main()
