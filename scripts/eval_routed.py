"""Archetype-routed ensemble: each archetype uses a specific model checkpoint.

For each game, read the agent's archetype (one-hot at entities[0, 7:7+N_ARCH])
and forward through THAT archetype's assigned model. Picks the model that was
empirically best for that class from prior single-model evals.
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import argparse
from pathlib import Path
import numpy as np
import torch

from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.model import (CombatPolicyNet, apply_resource_mask,
                            apply_entity_mask, pick_action)


def stable_seed(s: str) -> int:
    """Process-independent seed from a string.

    Python's builtin hash() is randomized per process (PYTHONHASHSEED), so
    seeds derived from it are NOT reproducible across runs — two eval runs
    would silently play different games. crc32 is stable everywhere.
    """
    import zlib
    return zlib.crc32(s.encode()) & 0xFFFFFF


def load_net(path: str, hidden: int = 128) -> CombatPolicyNet:
    net = CombatPolicyNet(hidden=hidden)
    sd = torch.load(path, map_location="cpu")
    sd = CombatPolicyNet.adapt_state_dict_for_perarch(sd)
    # Drop shape-mismatched keys (critic input dim varies across model versions;
    # inference never uses the critic).
    msd = net.state_dict()
    sd = {k: v for k, v in sd.items()
          if not (k in msd and v.shape != msd[k].shape)}
    net.load_state_dict(sd, strict=False)
    net.eval()
    return net


# Classes where the model OVER-EXTENDS (keeps moving after its action is spent,
# taking extra damage). For these, ending the turn once the action is gone is
# measured better. Kiters (wizards) are NOT here — they NEED to move after
# casting. This is a per-class inference fix, no retraining. See verify_combo.py.
NOEXT_ARCHS: set = set()   # net-neutral in 80-game ensemble eval; crude heuristic


def play_one_routed(arch_to_net, agent_arch, opp_arch, seed):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[agent_arch], opp_archs=[opp_arch], level=5)
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    net = arch_to_net[agent_arch]
    noext = agent_arch in NOEXT_ARCHS
    done = False
    while not done:
        actor_id = env.current_agent_id
        if (noext and actor_id == aid
                and env.resources.get("action", 0) <= 0):
            obs, _, term, trunc, _ = env.step([0, 0, 0])
            done = term or trunc
            continue
        obs_t = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
        with torch.no_grad():
            end_l, s, e, g = net(obs_t)
        s = apply_resource_mask(s, env.resources, env.ws, actor_id)
        e = apply_entity_mask(e, obs_t, env.ws, actor_id)
        action = list(pick_action(end_l[0], s[0], e[0], g[0],
                                   ws=env.ws, agent_id=actor_id))
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    return (not env.ws.characters[oid].is_alive()
             and env.ws.characters[aid].is_alive())


# Best NEURAL model per archetype — NEW-OBS era (2026-06-10, archetype-aware
# end_features). All old 8-dim-end_features checkpoints are incompatible. Six
# classes use the new generalist (ppo_endhead_gen); the 5 combo/rogue classes
# the generalist can't max use focused specialists trained with the end_head +
# win-cond + mutual-kill-reward + joint-ratio + critic-skill-pool fixes; BM
# specifically needed the critic-skill-pool (v5) to reach parity; vengeance is
# already best at the BC clone. Numbers = eval_class WR vs fair expert(env.step).
# Front-line classes were re-tuned for 3v3 template parties (ppo_team_* runs,
# 2026-06-10): same architecture, fine-tuned with frozen routed teammates vs
# expert opponent teams under the general rewards (team PBRS + wasted-move).
# Replacements beat the old checkpoint on the team metric (pick_team_snap,
# 192 eps) and pass the 1v1 guard, so one table serves both modes.
DEFAULT_ROUTING = {
    "battle_master":    "models/ppo_team_bm_v3/ppo_u0025.pt",        # NATIVE-v3 asym slot-PPO (obs v3 wave pilot): slot 48.6=47.7 (1728/arm); 1v1 54=56 guard PASS; asym buckets par; critic level-sensitive (V spread 0->1.04, probe_v3_features)
    "champion":         "models/ppo_team_champion_v3/ppo_u0010.pt",   # NATIVE-v3 asym slot-PPO (wave): slot 48.3=48.7 (1728/arm) par-swap; 1v1 guard PASS; critic V spread 2.29
    "totem_bear":       "models/ppo_team_totem_bear_v3/ppo_u0015.pt", # NATIVE-v3 asym slot-PPO (wave): slot 66.0=65.3 (1728/arm); 1v1 guard PASS; critic V spread 2.97
    "berserker":        "models/bc_team_berserker/bc_e01.pt",         # team 63 (exp-slot 69); 1v1 59=61 — in-context BC
    "evocation":        "models/ppo_team_evocation_v3/ppo_u0040.pt", # NATIVE-v3 asym slot-PPO (wave, split from ppo_endhead_gen): slot 75.5 > 70.6 (+4.9pp, 1728/arm) real gain; 1v1 guard PASS; critic V spread 3.67
    "divination":       "models/ppo_team_divination_v3/ppo_u0025.pt", # NATIVE-v3 asym slot-PPO (wave): slot 59.6=60.9 (1728/arm) par-swap; 1v1 guard 41=43 PASS; critic V spread 2.70 (probe)
    "life":             "models/bc_team_life4/bc_e08.pt",          # coop BC: heals 0.61/g 53% ally; 1v1 39>32 exp; team-slot -5.9pp (accepted for cooperation goal)
    "war":              "models/ppo_team_war_v3/ppo_u0030.pt",      # NATIVE-v3 asym slot-PPO (wave, split from ppo_endhead_gen): slot 59.0=57.6 (1728/arm); 1v1 guard PASS; critic V spread 4.36
    "assassin":         "models/ppo_team_assassin_v3/ppo_u0035.pt", # NATIVE-v3 asym slot-PPO (wave): slot 40.0 > 35.6 (+4.5pp, 1728/arm) — real gain, ties expert-in-slot 40.2; 1v1 guard PASS; critic V spread 5.17
    "arcane_trickster": "models/ppo_team_arcane_trickster_v3/ppo_u0040.pt", # NATIVE-v3 asym slot-PPO (wave): slot 41.4 > 34.1 (+7.3pp, 1728/arm) — biggest wave gain, beats expert-in-slot 33.4; 1v1 guard PASS; critic V spread 9.31
    "devotion":         "models/bc_team_devotion/bc_e03.pt",          # team 46 (exp-slot 48); 1v1 34=35 — in-context BC
    "vengeance":        "models/ppo_team_vengeance_v3/ppo_u0015.pt",  # NATIVE-v3 asym slot-PPO (wave): slot 44.9=44.3 (1728/arm) par-swap; 1v1 guard PASS; critic V spread 1.41
}

DEFAULT_HIDDEN = {a: 128 for a in (
    "battle_master", "champion", "totem_bear", "berserker", "evocation",
    "divination", "life", "war", "assassin", "arcane_trickster",
    "devotion", "vengeance")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    print("Loading per-arch models:")
    arch_to_net = {}
    cache = {}
    for arch, path in DEFAULT_ROUTING.items():
        if path not in cache:
            cache[path] = load_net(path, DEFAULT_HIDDEN[arch])
            print(f"  loaded {path} (h={DEFAULT_HIDDEN[arch]})")
        arch_to_net[arch] = cache[path]
        print(f"    {arch:20s} -> {path}")

    archs = list(ARCHETYPE_LIST)
    result = {}
    total = len(archs) ** 2
    done = 0
    for a in archs:
        result[a] = {}
        for o in archs:
            base = hash(f"{a}_{o}") & 0xFFFFFF
            wins = sum(1 for i in range(args.games)
                       if play_one_routed(arch_to_net, a, o, base + i))
            result[a][o] = {"win_rate": wins/args.games, "n": args.games}
            done += 1
            if done % 24 == 0:
                print(f"  {done}/{total}")

    print(f"\n=== ROUTED ENSEMBLE ===")
    print(f"{'arch':20s}  {'wr':>5s}")
    for a in archs:
        wr = np.mean([result[a][o]["win_rate"] for o in archs])
        print(f"{a:20s}  {wr:>4.0%}")
    overall = np.mean([result[a][o]["win_rate"] for a in archs for o in archs])
    print(f"{'OVERALL':20s}  {overall:>4.0%}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"label": "routed", "routing": DEFAULT_ROUTING,
                       "games_per_matchup": args.games, "result": result}, f, indent=2)


if __name__ == "__main__":
    main()
