"""Clean learnability test: CAN the B-net learn to read enemy fire-immunity?

The earlier lab was confounded — fire_elemental/young_red_dragon are unwinnable
1v1 even with correct play (magic_missile WR 0% at L20), so the immune half lost
regardless of spell and gave no terminal gradient. This version removes that
confound: both halves use the SAME weak base enemy (orc, agent L6), differing
ONLY in an injected fire-immunity, which IS winnable by correct play:
  injected-fire-immune orc: always-magic_missile WR 100%, always-fireball WR 0%.

  - 50% immune half  : orc with 火→0 injected. fireball deals 0 → cannot win;
                       magic_missile (力場) wins. The descriptor typed_resist[火]
                       is the ONLY thing distinguishing this from the other half.
  - 50% normal half  : plain orc. fireball wins fastest.
The two halves are stat-identical except one descriptor bit, and BOTH are
winnable, so a CONDITIONAL (read-descriptor-and-switch) policy wins both while
any fixed policy loses one half — a strong terminal gradient toward using the
descriptor. Self one-hot blinded (B-net); orc reads all-zero one-hot so the
descriptor is the sole enemy signal. BC anchor preserves the standard kits.

Success: after training, vs immune orc the fire-share with descriptor ON is far
below descriptor OFF (and WR-on >> WR-off); vs normal orc fire-share stays high.

Usage: python scripts/train_fire_lab.py --updates 40 --out_dir models/fire_lab
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

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.train_ppo import _sample_action, _compute_gae, ppo_update
from trpg.rl.train_bc import bc_loss_step
from trpg.rl.obs import build_obs, migrate_entities_v3_to_v4
from trpg.engine.skill import available_skills
from trpg.scenarios.monsters import register_monsters, MONSTER_DEFS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import (load_student, load_dataset, DATASET_PATH,
                            blind_self_identity_np)
from train_population import blind_np_single, descriptor_weight_norm
from probe_descriptor_ab import blind_self_one_hot, zero_enemy_descriptor
from probe_fire_switch import action_damage_type
from eval_routed import stable_seed

CASTERS = ["evocation", "divination"]


def _inject_and_obs(env, immune):
    """Inject fire-immunity onto the opponent (if immune) and (re)build the obs
    so the capability descriptor reflects it from step 0."""
    if immune:
        for oid in env.opp_ids:
            env.ws.characters[oid].damage_multipliers["火"] = 0.0
    return build_obs(env.ws, env.current_agent_id, env.resources)


def collect(net, n_steps, seed, rng, base_enemy, level, gamma=0.99, lam=0.95):
    net.eval()
    obs_l, act_l, lp_l, rew_l, val_l, done_l = [], [], [], [], [], []
    skm_l, enm_l, grm_l = [], [], []
    ep = 0
    while len(rew_l) < n_steps:
        agent = rng.choice(CASTERS)
        immune = rng.random() < 0.5
        env = CombatEnvV2(seed=seed * 1_000_003 + ep, n_agents=1, n_opps=1)
        env.reset(agent_archs=[agent], opp_archs=[base_enemy],
                  level=level, opp_level=MONSTER_DEFS[base_enemy].natural_level)
        obs = blind_np_single(_inject_and_obs(env, immune))
        done = False
        while not done:
            aid = env.current_agent_id
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                action, lp, val, skm, enm, grm = _sample_action(
                    net, ot, env.resources, env.ws, aid)
            a = action.numpy().tolist()
            obs2, r, term, trunc, _ = env.step(a)
            obs_l.append(obs); act_l.append(a); lp_l.append(lp.numpy())
            rew_l.append(float(r)); val_l.append(val)
            skm_l.append(skm.numpy()); enm_l.append(enm.numpy())
            grm_l.append(grm.numpy())
            done = term or trunc
            done_l.append(bool(done))
            # keep injecting on the live obs each step (step rebuilds from ws,
            # and the injected mult persists on the character, so just blind it)
            obs = blind_np_single(obs2) if not done else obs2
        ep += 1
    rewards = np.array(rew_l, np.float32); values = np.array(val_l, np.float32)
    dones = np.array(done_l, np.float32)
    adv, ret = _compute_gae(rewards, values, dones, 0.0, gamma, lam)
    batch = {
        "obs": {k: np.stack([o[k] for o in obs_l]) for k in obs_l[0]},
        "actions": np.array(act_l, np.int64),
        "log_probs": np.array(lp_l, np.float32),
        "skill_masks": np.stack(skm_l).astype(np.bool_),
        "entity_masks": np.stack(enm_l).astype(np.bool_),
        "grid_masks": np.stack(grm_l).astype(np.bool_),
        "rewards": rewards, "values": values,
        "returns": ret, "advantages": adv, "dones": dones,
    }
    return batch, ep


def eval_variant(net, base_enemy, level, immune, kill_desc, games):
    """Greedy games; return (fire_share, win_rate)."""
    fire = ndmg = wins = 0
    for gi in range(games):
        env = CombatEnvV2(seed=stable_seed(f"flab_{base_enemy}_{immune}_{gi}"),
                          n_agents=1, n_opps=1)
        env.reset(agent_archs=["evocation"], opp_archs=[base_enemy], level=level,
                  opp_level=MONSTER_DEFS[base_enemy].natural_level)
        obs = _inject_and_obs(env, immune)
        aid = env.agent_ids[0]; oid = env.opp_ids[0]
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
        if env.ws.characters[oid].is_dead() and env.ws.characters[aid].is_alive():
            wins += 1
    return fire / max(1, ndmg), wins / games


def report(net, base_enemy, level, games, tag):
    d, _ = descriptor_weight_norm(net)
    out = [f"  [{tag}] desc-norm={d:.3f}"]
    for immune in (True, False):
        f_on, w_on = eval_variant(net, base_enemy, level, immune, False, games)
        f_off, w_off = eval_variant(net, base_enemy, level, immune, True, games)
        name = "IMMUNE" if immune else "normal"
        out.append(f"    {name:6s} 火%: on={f_on:.0%} off={f_off:.0%}  "
                   f"WR: on={w_on:.0%} off={w_off:.0%}")
    print("\n".join(out), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm", default="models/pop_mon/pop_u0005.pt")
    p.add_argument("--out_dir", default="models/fire_lab")
    p.add_argument("--base_enemy", default="orc")
    p.add_argument("--level", type=int, default=6)
    p.add_argument("--updates", type=int, default=40)
    p.add_argument("--steps", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--value_warmup", type=int, default=3)
    p.add_argument("--anchor_batches", type=int, default=4)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_games", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    register_monsters()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    net = load_student(args.warm)
    print(f"warm={args.warm} base={args.base_enemy} L{args.level} "
          f"steps={args.steps} updates={args.updates}", flush=True)

    anchor = None
    if args.anchor_batches > 0:
        obs_np, act_np, tt_np = load_dataset(Path(DATASET_PATH))
        obs_np = {k: v.copy() for k, v in obs_np.items()}
        obs_np["entities"] = migrate_entities_v3_to_v4(obs_np["entities"])
        blind_self_identity_np(obs_np)
        anchor = (obs_np, act_np, tt_np)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    print("baseline:", flush=True)
    report(net, args.base_enemy, args.level, args.eval_games, "u0")
    for update in range(1, args.updates + 1):
        batch, neps = collect(net, args.steps, args.seed * 7919 + update, rng,
                              args.base_enemy, args.level)
        info = ppo_update(net, batch, optim, n_epochs=args.epochs,
                          batch_size=args.batch, ent_coef=args.ent_coef,
                          device="cpu",
                          value_only=(update <= args.value_warmup))
        line = (f"U{update:3d}/{args.updates}"
                f"{' [warmup]' if update <= args.value_warmup else ''} "
                f"eps={neps} pol={info['policy_loss']:+.3f} "
                f"val={info['value_loss']:.2f} ent={info['entropy']:.3f}")
        if anchor is not None and update > args.value_warmup:
            a_obs, a_act, a_tt = anchor
            accs = []
            for _ in range(args.anchor_batches):
                sel = np.random.randint(0, len(a_act), size=args.batch)
                ob = {k: torch.from_numpy(v[sel]) for k, v in a_obs.items()}
                _, acc = bc_loss_step(net, ob, torch.from_numpy(a_act[sel]),
                                      torch.from_numpy(a_tt[sel]), optim)
                accs.append(acc.get("skill", float("nan")))
            line += f" anchor={np.nanmean(accs):.2f}"
        print(line, flush=True)
        if update % args.eval_every == 0:
            net.eval()
            torch.save(net.state_dict(), out_dir / f"fire_u{update:04d}.pt")
            report(net, args.base_enemy, args.level, args.eval_games, f"u{update}")
            net.train()
    print("done.", flush=True)


if __name__ == "__main__":
    main()
