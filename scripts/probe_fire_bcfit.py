"""Discriminator: is the fire-switch failure REPRESENTATIONAL (skill side lacks
damage_type) or an RL-EXPLORATION failure (PPO can't discover it)?

The corrected fire-lab (descriptor genuinely present, cache bug fixed) still
fails to teach evocation to drop fireball vs a fire-immune orc after 40 PPO
updates. Two hypotheses remain:
  (a) representational: with no damage_type on the skill, the AoE/save/target
      proxy is too indirect for the net to EVER condition skill choice on the
      enemy's typed_resist descriptor.
  (c) exploration: the net CAN represent it, but PPO starting from a ~80%-fire
      BC prior never explores its way to the immune-half switch.

Supervised BC removes exploration from the equation: we hand the net the oracle
labels (immune orc -> magic_missile, normal orc -> fireball) and ask whether it
can FIT them. The ONLY obs difference between the two halves is the descriptor
typed_resist[火] cell (verified by probe_desc_cache), so any successful fit MUST
route through the descriptor. The decisive readout is the desc-on vs desc-off
contrast after fitting:
  - IMMUNE desc-on fire% LOW and desc-off fire% HIGH  -> net learned to USE the
    descriptor; representation SUFFICES; damage_type NOT needed; PPO failure was
    exploration.
  - IMMUNE desc-on fire% STAYS HIGH (can't fit)        -> representational
    bottleneck; damage_type surgery justified.

States are collected on-policy from the warm net (sampled), labelled by the
oracle; skill choice is fit with masked cross-entropy. No engine play-to-win is
needed — only skill CHOICE is supervised.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse, random
import numpy as np
import torch
import torch.nn.functional as F

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.train_ppo import _sample_action
from trpg.engine.skill import available_skills
from trpg.scenarios.monsters import register_monsters, MONSTER_DEFS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single
from probe_descriptor_ab import blind_self_one_hot, zero_enemy_descriptor
from probe_fire_switch import action_damage_type
from eval_routed import stable_seed

ORACLE_IMMUNE = "magic_missile"   # 力場, auto-hit single target
ORACLE_NORMAL = "fireball_ev"     # 火, AoE point


def _oracle_slot(ws, aid, immune):
    """available_skills slot index of the oracle skill at this state, or None."""
    sks = available_skills(ws.characters[aid], ws)
    want = ORACLE_IMMUNE if immune else ORACLE_NORMAL
    for i, s in enumerate(sks):
        if s.skill_id == want:
            return i
    return None


def _inject(env, immune):
    if immune:
        for oid in env.opp_ids:
            env.ws.characters[oid].damage_multipliers["火"] = 0.0


def collect_states(net, n_states, seed, rng, level=6, base="orc"):
    """Roll the warm net (sampled) in both halves; label each agent state with
    the oracle skill slot. Returns (obs_list, label_list)."""
    net.eval()
    O, Y = [], []
    ep = 0
    while len(Y) < n_states:
        immune = rng.random() < 0.5
        env = CombatEnvV2(seed=seed * 1_000_003 + ep, n_agents=1, n_opps=1)
        env.reset(agent_archs=["evocation"], opp_archs=[base], level=level,
                  opp_level=MONSTER_DEFS[base].natural_level)
        _inject(env, immune)
        obs = blind_self_one_hot(_rebuild(env))
        done = False
        while not done:
            aid = env.current_agent_id
            slot = _oracle_slot(env.ws, aid, immune)
            if slot is not None:
                O.append(obs); Y.append(slot)
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                action, *_ = _sample_action(net, ot, env.resources, env.ws, aid)
            obs2, _, term, trunc, _ = env.step(action.numpy().tolist())
            done = term or trunc
            obs = blind_self_one_hot(obs2) if not done else obs2
        ep += 1
    return O, Y


def _rebuild(env):
    from trpg.rl.obs import build_obs
    return build_obs(env.ws, env.current_agent_id, env.resources)


def bc_fit(net, O, Y, steps, batch, lr):
    """Masked cross-entropy on skill choice toward the oracle slot."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    keys = list(O[0].keys())
    N = len(Y)
    Yt = torch.tensor(Y, dtype=torch.long)
    net.train()
    for st in range(steps):
        sel = np.random.randint(0, N, size=batch)
        ob = {k: torch.from_numpy(np.stack([O[i][k] for i in sel])) for k in keys}
        end_l, skill_l, ent_l, grid_l = net(ob)
        # mask invalid slots (padding) + the end slot (0) so CE is over real skills
        mask = ob["skill_mask"] < 0.5
        skill_l = skill_l.masked_fill(mask, -1e9)
        skill_l[:, 0] = -1e9
        loss = F.cross_entropy(skill_l, Yt[sel])
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 50 == 0 or st == steps - 1:
            with torch.no_grad():
                acc = (skill_l.argmax(-1) == Yt[sel]).float().mean().item()
            print(f"  step {st:4d}  loss={loss.item():.3f}  acc={acc:.2f}",
                  flush=True)


def eval_fire(net, immune, kill_desc, games, level=6, base="orc"):
    fire = ndmg = 0
    net.eval()
    for gi in range(games):
        env = CombatEnvV2(seed=stable_seed(f"bcfit_{base}_{immune}_{gi}"),
                          n_agents=1, n_opps=1)
        env.reset(agent_archs=["evocation"], opp_archs=[base], level=level,
                  opp_level=MONSTER_DEFS[base].natural_level)
        _inject(env, immune)
        obs = _rebuild(env)
        oid = env.opp_ids[0]
        done = False
        while not done:
            actor = env.current_agent_id
            ob = blind_self_one_hot(obs)
            if kill_desc:
                ob = zero_enemy_descriptor(ob)
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            ai = pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor)
            actor_c = env.ws.characters[actor]
            sks = available_skills(actor_c, env.ws)
            if 0 < ai[0] < len(sks):
                o = env.ws.characters[oid]
                built = sks[ai[0]].build_action(actor, oid,
                                                (o.position.x, o.position.y))
                dt = action_damage_type(built, actor_c)
                if dt:
                    ndmg += 1
                    if dt == "火":
                        fire += 1
            obs2, _, term, trunc, _ = env.step(list(ai))
            obs = obs2
            done = term or trunc
    return fire / max(1, ndmg)


def report(net, games, tag):
    print(f"[{tag}]")
    for immune in (True, False):
        f_on = eval_fire(net, immune, False, games)
        f_off = eval_fire(net, immune, True, games)
        name = "IMMUNE" if immune else "normal"
        print(f"  {name:6s} fire%: desc-on={f_on:.0%}  desc-off={f_off:.0%}",
              flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm", default="models/pop_mon/pop_u0005.pt")
    p.add_argument("--states", type=int, default=4000)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--games", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    register_monsters()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    net = load_student(args.warm)
    print(f"warm={args.warm}  states={args.states} steps={args.steps}\n")
    report(net, args.games, "baseline (pre-fit)")
    print("\ncollecting oracle-labelled states...", flush=True)
    O, Y = collect_states(net, args.states, args.seed * 7919 + 1, rng)
    immune_frac = np.mean([1 if y is not None else 0 for y in Y])
    print(f"collected {len(Y)} states\n\nBC-fitting skill choice to oracle:",
          flush=True)
    bc_fit(net, O, Y, args.steps, args.batch, args.lr)
    print()
    report(net, args.games, "after BC-fit")


if __name__ == "__main__":
    main()
