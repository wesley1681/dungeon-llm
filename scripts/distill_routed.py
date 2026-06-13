"""Distill the 12-slot routing table into ONE shared-head net.

Teachers = the routed checkpoints (eval_routed.DEFAULT_ROUTING), each playing
its own archetype greedily; their decisions become BC labels. The student is
a CombatPolicyNet with n_head_groups=1 — no per-archetype head copies, no
identity routing. This is the feasibility experiment for collapsing the
routing table to a single file: gradient conflict was a PPO pathology
(moving targets fighting over shared weights); distillation targets are
fixed, so one net holding all 12 policies is a capacity question, which this
measures directly.

Two students from the same dataset:
  A (default)  identity inputs intact (self one-hot + end_features arch tail
               still visible). Tests pure capacity.
  B (--blind)  self-identity channels ZEROED in every sample (entities row 0
               arch one-hot + end_features cols 8..) — the net can only know
               itself through its skill list, resources, and v3 stat panel.
               Capability conditioning. Chimeras/monsters (all-zero one-hot)
               are in-distribution for this student by construction.

Usage:
  python scripts/distill_routed.py --collect            # build + cache dataset
  python scripts/distill_routed.py --train --out_dir models/distill_a
  python scripts/distill_routed.py --train --blind --out_dir models/distill_b
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

from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.model import (CombatPolicyNet, apply_resource_mask,
                           apply_entity_mask, pick_action)
from trpg.rl.bc_collect import TT_END
from trpg.rl.train_bc import bc_loss_step
from trpg.rl.train_team import sample_comp
from trpg.rl.obs import N_ARCHETYPES
from trpg.engine.skill import available_skills

from eval_routed import DEFAULT_ROUTING, load_net, stable_seed

_ARCH_OH_START = 7   # entities row layout — see obs._entity_row
_END_ARCH_START = 8  # end_features layout — see obs.END_FEATURES_DIM

DATASET_PATH = Path("models/distill_v1/teacher_pairs.npz")


def blind_self_identity_np(obs_np: dict) -> dict:
    """Zero the self-identity channels IN PLACE (numpy dataset arrays)."""
    obs_np["entities"][:, 0, _ARCH_OH_START:_ARCH_OH_START + N_ARCHETYPES] = 0.0
    obs_np["end_features"][:, _END_ARCH_START:_END_ARCH_START + N_ARCHETYPES] = 0.0
    return obs_np


def blind_self_identity_t(obs_t: dict) -> dict:
    """Same transform on a batched torch obs dict (inference path)."""
    obs_t = {k: v.clone() for k, v in obs_t.items()}
    obs_t["entities"][:, 0, _ARCH_OH_START:_ARCH_OH_START + N_ARCHETYPES] = 0.0
    obs_t["end_features"][:, _END_ARCH_START:_END_ARCH_START + N_ARCHETYPES] = 0.0
    return obs_t


def _net_turn(net, env, obs, actor, blind=False):
    """One greedy decision by `net` for `actor`. Returns (act, tt or None)."""
    ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
    if blind:
        ot = blind_self_identity_t(ot)
    with torch.no_grad():
        el, s, e, g = net(ot)
    s = apply_resource_mask(s, env.resources, env.ws, actor)
    e = apply_entity_mask(e, ot, env.ws, actor)
    act = tuple(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
    if act[0] == 0:
        return act, TT_END
    ag = env.ws.characters[actor]
    skills = available_skills(ag, env.ws)
    if act[0] < len(skills):
        return act, int(skills[act[0]].features.target_type)
    return act, None   # stale slot — caller drops the pair


def collect_1v1(teacher_nets: dict, eps_per_matchup: int) -> list:
    """Each teacher plays its arch vs every opponent arch; label own actions."""
    pairs = []
    for a_i, arch in enumerate(ARCHETYPE_LIST):
        net = teacher_nets[arch]
        for o_i, opp in enumerate(ARCHETYPE_LIST):
            for ep in range(eps_per_matchup):
                seed = stable_seed(f"distill_{arch}_{opp}_{ep}")
                env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
                obs, _ = env.reset(agent_archs=[arch], opp_archs=[opp])
                done = False
                while not done:
                    actor = env.current_agent_id
                    act, tt = _net_turn(teacher_nets[arch], env, obs, actor)
                    if tt is not None:
                        pairs.append((obs, act if tt != TT_END else (0, 0, 0), tt))
                    obs, _, term, trunc, info = env.step(list(act))
                    res = (info or {}).get("action_result") or {}
                    if pairs and res.get("type") == "ERROR":
                        pairs.pop()   # engine rejected → don't teach a no-op
                    done = term or trunc
        print(f"  1v1 teacher {arch}: cumulative {len(pairs)} pairs", flush=True)
    return pairs


def collect_team(teacher_nets: dict, n_eps: int, seed0: int = 77_000) -> list:
    """3v3 episodes, every agent seat driven by its routed teacher; ALL three
    seats' decisions become labels (one episode feeds three teachers' data)."""
    pairs = []
    rng = random.Random(seed0)
    for ep in range(n_eps):
        env = CombatEnvV2(seed=seed0 + ep, n_agents=3, n_opps=3)
        comp = sample_comp(rng)
        opp_comp = sample_comp(rng)
        obs, _ = env.reset(agent_archs=comp, opp_archs=opp_comp)
        arch_of = dict(zip(env.agent_ids, env.agent_archs))
        done = False
        while not done:
            actor = env.current_agent_id
            net = teacher_nets[arch_of[actor]]
            act, tt = _net_turn(net, env, obs, actor)
            if tt is not None:
                pairs.append((obs, act if tt != TT_END else (0, 0, 0), tt))
            obs, _, term, trunc, info = env.step(list(act))
            res = (info or {}).get("action_result") or {}
            if pairs and res.get("type") == "ERROR":
                pairs.pop()
            done = term or trunc
        if (ep + 1) % 50 == 0:
            print(f"  team {ep+1}/{n_eps} eps, cumulative {len(pairs)} pairs",
                  flush=True)
    return pairs


def save_dataset(pairs: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obs_keys = list(pairs[0][0].keys())
    arrs = {f"obs_{k}": np.stack([p[0][k] for p in pairs]) for k in obs_keys}
    arrs["actions"] = np.array([p[1] for p in pairs], dtype=np.int64)
    arrs["tts"] = np.array([p[2] for p in pairs], dtype=np.int64)
    np.savez_compressed(path, **arrs)
    print(f"saved {len(pairs)} pairs -> {path}")


def load_dataset(path: Path):
    z = np.load(path)
    obs_np = {k[4:]: z[k] for k in z.files if k.startswith("obs_")}
    return obs_np, z["actions"], z["tts"]


def train_student(obs_np, actions_np, tts_np, *, blind: bool, out_dir: Path,
                  epochs: int, lr: float, batch: int, skill_weight: bool,
                  hidden: int = 128):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(actions_np)
    if blind:
        obs_np = {k: v.copy() for k, v in obs_np.items()}
        blind_self_identity_np(obs_np)
        (out_dir / "BLIND").write_text(
            "self-identity channels zeroed at train AND inference\n")

    sample_w = np.ones(n, dtype=np.float32)
    if skill_weight:
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
        print(f"identity-weight: {len(cnt)} skills, range "
              f"[{min(id_w.values()):.2f}, {max(id_w.values()):.2f}]")

    net = CombatPolicyNet(hidden=hidden, n_head_groups=1)
    net.train()
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    n_end = int((actions_np[:, 0] == 0).sum())
    print(f"training student (blind={blind}) on {n} pairs "
          f"({n_end} end, {n - n_end} act)")

    idx = np.arange(n)
    for epoch in range(1, epochs + 1):
        np.random.shuffle(idx)
        losses, accs = [], []
        for s0 in range(0, n, batch):
            sel = idx[s0:s0 + batch]
            obs_b = {k: torch.from_numpy(v[sel]) for k, v in obs_np.items()}
            act_b = torch.from_numpy(actions_np[sel])
            tt_b = torch.from_numpy(tts_np[sel])
            sw_b = torch.from_numpy(sample_w[sel])
            loss, acc = bc_loss_step(net, obs_b, act_b, tt_b, optim,
                                     skill_sample_w=sw_b)
            losses.append(float(loss)); accs.append(acc)
        sk = np.nanmean([a["skill"] for a in accs])
        gr = np.nanmean([a["grid"] for a in accs])
        en = np.nanmean([a["end"] for a in accs])
        print(f"epoch {epoch}/{epochs} loss={np.mean(losses):.3f} "
              f"acc end={en:.2f} skill={sk:.2f} grid={gr:.2f}", flush=True)
        torch.save(net.state_dict(), out_dir / f"distill_e{epoch:02d}.pt")
    return net


def load_student(path: str, hidden: int = 128) -> CombatPolicyNet:
    """Distilled checkpoints are 1-head-group: no per-arch tiling, but the
    obs-era migration chain (v3 → v4) still applies."""
    net = CombatPolicyNet(hidden=hidden, n_head_groups=1)
    sd = torch.load(path, map_location="cpu")
    net.load_state_dict(CombatPolicyNet.adapt_state_dict_for_obs(sd))
    net.eval()
    return net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--collect", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--eps_1v1", type=int, default=3,
                   help="episodes per (teacher, opponent) matchup")
    p.add_argument("--team_eps", type=int, default=200)
    p.add_argument("--blind", action="store_true",
                   help="student B: zero self-identity inputs")
    p.add_argument("--skill_weight", action="store_true")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--dataset", type=str, default=str(DATASET_PATH))
    p.add_argument("--out_dir", type=str, default="models/distill_a")
    args = p.parse_args()

    ds_path = Path(args.dataset)
    if args.collect:
        print("loading teachers from routing table...")
        cache = {}
        teacher_nets = {}
        for arch, path in DEFAULT_ROUTING.items():
            if path not in cache:
                cache[path] = load_net(path)
            teacher_nets[arch] = cache[path]
        pairs = collect_1v1(teacher_nets, args.eps_1v1)
        pairs += collect_team(teacher_nets, args.team_eps)
        save_dataset(pairs, ds_path)

    if args.train:
        obs_np, actions_np, tts_np = load_dataset(ds_path)
        train_student(obs_np, actions_np, tts_np, blind=args.blind,
                      out_dir=Path(args.out_dir), epochs=args.epochs,
                      lr=args.lr, batch=args.batch,
                      skill_weight=args.skill_weight)


if __name__ == "__main__":
    main()
