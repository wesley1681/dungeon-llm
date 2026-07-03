"""BC-seed the EXPERT's high-value kit usage into the blinded student, then
verify. Companion to seed_switch_bc.py — same blinded pipeline, different
oracle: the label is simply the SCRIPTED EXPERT's action.

Why (probe_bcfit, 2026-06-14): the champion assigns ~0 probability to the
delayed-payoff control/heal/summon plays the expert leans on (menacing_attack
1.9%, cure_wounds 0.2%, spiritual_weapon 1.1%, sleep 0.2%), so it ends 1v2 at
half the expert's HP and loses. It never LEARNED these actions — a knowledge/BC
gap, not exploration/credit. So install them supervised from the experts (who
already use them), protected by a self-anchor on all 12 classes against
forgetting.

Pipeline:
  1. DEMOS — roll the SCRIPTED EXPERT in the seat over the FOCUS classes, in
     BOTH 1v1 and 1v2 (cure_wounds / control are 1v2-survival plays), recording
     (blinded obs, expert action, target_type) at every decision incl. END.
     Samples whose skill is a "high-value" dropped action are flagged for an
     adaptive CE boost (focus gradient on the tail the net currently misses).
  2. SELF-ANCHOR — the warm net's OWN greedy actions on all 12 classes (std +
     monster opponents), fresh current-obs → protects std12 behaviour.
  3. SEED — BC fine-tune alternating demo / self-anchor batches 1:1.
  4. VERIFY — std12 + vs-monster probes (warm vs seeded), and the dropped-action
     probability is checked externally by probe_bcfit / probe_usage.

No distill ("old") anchor: that dataset predates the los_grid / decision_context
obs channels and would need the full migration chain. The fresh self-anchor
carries all current keys and is the behaviour protector that actually matters.

Usage:
  python scripts/seed_kit_bc.py --warm models/pop_los_final/pop_u0044.pt \
         --out_dir models/seed_kit --states 12000 --self_states 12000 \
         --steps 800 --switch_boost 3.0
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import argparse, math, random
from collections import Counter
from pathlib import Path
import numpy as np
import torch

from trpg.scenarios.monsters import (register_monsters, MONSTER_DEFS,
                                      EQUIV_LEVEL_1V1, onev1_viable_monsters)
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.obs import build_obs
from trpg.rl.action import encode_action
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.train_bc import bc_loss_step
from trpg.engine.skill import available_skills
from trpg.engine.combat_policy import make_archetype_policy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single, standard_probe, monster_opp_probe
from synth_identity import STANDARD_IDS

# Classes whose expert leans on a high-value action the model drops (probe_usage).
FOCUS = ["battle_master", "war", "arcane_trickster", "assassin", "champion"]
# The dropped high-value skills — flagged for the adaptive CE boost.
HIGH_VALUE = {"menacing_attack", "spiritual_weapon", "spiritual_weapon_war",
              "spiritual_weapon_attack_war", "cure_wounds", "sleep",
              "spirit_guardians", "weapon:短劍", "distracting_strike"}


def _expert_act(script, env, actor):
    ch = env.ws.characters[actor]
    dec = script.decide(actor, ch, env.ws, env.resources,
                        env.ws.combat.round_number)
    if dec.action is None or getattr(dec, "fled", False):
        return [0, 0, 0], -1, None
    try:
        enc = list(encode_action(dec.action, env.ws, actor))
    except Exception:
        return [0, 0, 0], -1, None
    sks = available_skills(ch, env.ws)
    tt = int(sks[enc[0]].features.target_type) if 0 <= enc[0] < len(sks) else -1
    sid = sks[enc[0]].skill_id if 0 <= enc[0] < len(sks) else None
    return enc, tt, sid


def collect_demos(n_states, seed, rng):
    O, A, T, F = [], [], [], []
    ep = 0
    sid_hist = Counter()
    while len(A) < n_states:
        ident = rng.choice(FOCUS)
        n_opp = rng.choice([1, 2, 2])           # bias to 1v2 (where it matters)
        opps = [rng.choice(STANDARD_IDS) for _ in range(n_opp)]
        o_lvl = rng.randint(2, 5)
        env = CombatEnvV2(seed=seed * 1_000_003 + ep, n_agents=1, n_opps=n_opp)
        env.reset(agent_archs=[ident], opp_archs=opps, level=5, opp_level=o_lvl)
        aid = env.agent_ids[0]
        script = make_archetype_policy(ident)
        done = False
        while not done:
            actor = env.current_agent_id
            obs = blind_np_single(build_obs(env.ws, actor, env.resources))
            enc, tt, sid = _expert_act(script, env, actor)
            if actor == aid:
                O.append(obs); A.append(enc); T.append(tt)
                F.append(sid in HIGH_VALUE)
                if sid:
                    sid_hist[sid] += 1
            _, _, term, trunc, _ = env.step(enc)
            done = term or trunc
        ep += 1
        if ep % 300 == 0:
            print(f"  demos: {len(A)}/{n_states} ({ep} eps) "
                  f"HV={sum(F)}", flush=True)
    obs_np = {k: np.stack([o[k] for o in O]) for k in O[0]}
    return (obs_np, np.array(A, np.int64), np.array(T, np.int64),
            np.array(F, np.bool_), dict(sid_hist))


def collect_self_anchor(net, n_states, seed, rng):
    """Warm net's GREEDY actions on all 12 classes, std + monster opponents,
    fresh current-obs. Protects std12 behaviour (seed_switch_bc lesson)."""
    net.eval()
    mons = sorted(MONSTER_DEFS)
    O, A, T = [], [], []
    ep = 0
    while len(A) < n_states:
        agent = rng.choice(STANDARD_IDS)
        if rng.random() < 0.5:
            mon = rng.choice(mons); eq = EQUIV_LEVEL_1V1[mon]
            a_lvl = int(min(8, max(3, round(eq)))) if math.isfinite(eq) else 8
            opp, o_lvl = mon, MONSTER_DEFS[mon].natural_level
        else:
            a_lvl = rng.randint(5, 8)
            opp, o_lvl = rng.choice(STANDARD_IDS), None
            o_lvl = a_lvl
        env = CombatEnvV2(seed=seed * 1_000_003 + ep, n_agents=1, n_opps=1)
        env.reset(agent_archs=[agent], opp_archs=[opp],
                  level=a_lvl, opp_level=o_lvl)
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
            if act[0] == 0:
                tt = -1
            else:
                sks = available_skills(env.ws.characters[aid], env.ws)
                tt = int(sks[act[0]].features.target_type) if act[0] < len(sks) else -1
            O.append(obs); A.append(act); T.append(tt)
            obs2, _, term, trunc, _ = env.step(act)
            done = term or trunc
            obs = blind_np_single(obs2) if not done else obs2
        ep += 1
        if ep % 300 == 0:
            print(f"  self-anchor: {len(A)}/{n_states} ({ep} eps)", flush=True)
    obs_np = {k: np.stack([o[k] for o in O]) for k in O[0]}
    return obs_np, np.array(A, np.int64), np.array(T, np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm", default="models/pop_los_final/pop_u0044.pt")
    p.add_argument("--out_dir", default="models/seed_kit")
    p.add_argument("--states", type=int, default=12000)
    p.add_argument("--self_states", type=int, default=12000)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--switch_boost", type=float, default=3.0,
                   help="adaptive CE weight 1+boost on samples the net "
                        "currently gets wrong (focus gradient on the dropped "
                        "high-value tail)")
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--gate_std_games", type=int, default=1)
    p.add_argument("--gate_mon_games", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--demos_in", default="")
    p.add_argument("--self_anchor_in", default="")
    args = p.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    net = load_student(args.warm)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.demos_in:
        z = np.load(args.demos_in)
        act_np = z["actions"]; tt_np = z["target_types"]
        fl_np = z["hv_flags"]
        obs_np = {k[4:]: z[k] for k in z.files if k.startswith("obs_")}
        print(f"reused {len(act_np)} demos", flush=True)
    else:
        print("=== collecting expert demos (focus classes, 1v1+1v2) ===", flush=True)
        obs_np, act_np, tt_np, fl_np, hist = collect_demos(
            args.states, args.seed * 7919 + 1, rng)
        print(f"collected {len(act_np)} demo states; HV={int(fl_np.sum())}; "
              f"top skills={dict(sorted(hist.items(), key=lambda kv: -kv[1])[:10])}",
              flush=True)
        np.savez_compressed(out_dir / "kit_demos.npz", actions=act_np,
                            target_types=tt_np, hv_flags=fl_np,
                            **{f"obs_{k}": v for k, v in obs_np.items()})

    if args.self_anchor_in:
        z = np.load(args.self_anchor_in)
        s_act = z["actions"]; s_tt = z["target_types"]
        s_obs = {k[4:]: z[k] for k in z.files if k.startswith("obs_")}
        print(f"reused {len(s_act)} self-anchor", flush=True)
    else:
        print("=== collecting self-anchor (all 12 classes) ===", flush=True)
        s_obs, s_act, s_tt = collect_self_anchor(
            net, args.self_states, args.seed * 104729 + 7, rng)
        print(f"collected {len(s_act)} self-anchor states", flush=True)
        np.savez_compressed(out_dir / "self_anchor.npz", actions=s_act,
                            target_types=s_tt,
                            **{f"obs_{k}": v for k, v in s_obs.items()})

    mon_pool = onev1_viable_monsters(8.0)

    def gate(tag, n):
        n.eval()
        std, _ = standard_probe(n, games=args.gate_std_games)
        mon, _ = monster_opp_probe(n, mon_pool, games=args.gate_mon_games)
        print(f"  [{tag}] std12={std:.1%}  vs-mon={mon:.1%}", flush=True)

    print("=== baseline gate (warm) ===", flush=True)
    gate("warm", net)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(f"=== seeding {args.steps} steps (demo + self-anchor 1:1) ===", flush=True)
    net.train()
    for step in range(1, args.steps + 1):
        sel = np.random.randint(0, len(act_np), size=args.batch)
        ob = {k: torch.from_numpy(v[sel]) for k, v in obs_np.items()}
        acts = torch.from_numpy(act_np[sel])
        w = None
        if args.switch_boost > 0:
            with torch.no_grad():
                _, sl, _, _ = net(ob)
            w = 1.0 + args.switch_boost * (sl.argmax(-1) != acts[:, 0]).float()
        _, acc_d = bc_loss_step(net, ob, acts, torch.from_numpy(tt_np[sel]),
                                optim, skill_sample_w=w)
        sel = np.random.randint(0, len(s_act), size=args.batch)
        ob = {k: torch.from_numpy(v[sel]) for k, v in s_obs.items()}
        _, acc_sa = bc_loss_step(net, ob, torch.from_numpy(s_act[sel]),
                                 torch.from_numpy(s_tt[sel]), optim)
        if step % 50 == 0 or step == 1:
            print(f"  step {step:4d}  demo-acc={acc_d.get('skill', 0):.2f}  "
                  f"self-acc={acc_sa.get('skill', 0):.2f}", flush=True)
        if step % args.eval_every == 0:
            torch.save(net.state_dict(), out_dir / f"kit_s{step:04d}.pt")
            gate(f"s{step}", net)
            net.train()
    print("done.", flush=True)


if __name__ == "__main__":
    main()
