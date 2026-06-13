"""Why is mon-atk flat at 0.10-0.18 after 18 policy updates? (CLAUDE.md 解剖)

H1 advantage 不偏向攻擊：value 還爛（loss ~10）→ GAE 噪到攻擊步沒有穩定正
   advantage。量法：rollout 內怪席步按動作型分組的 advantage 均值±SE。
H2 anchor 拔河：switch/self anchor 的 BC 梯度經共享幹道把怪席攻擊 logit 壓
   回去。量法：固定怪席探針態 → 1 次 ppo_update 前後 vs 4 批 anchor 前後的
   攻擊-vs-move logit 差變化。
H3 純粹太慢：ppo_update 後攻擊 logit 漲但量級 ~0.0x → 36 更新不夠。

Usage: python scripts/diag_mon_actor.py models/mon_actor1/ma_u0020.pt
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse
import random
from collections import defaultdict

import numpy as np
import torch

from trpg.scenarios.monsters import MONSTER_DEFS, register_monsters
register_monsters()

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.train_ppo import ppo_update
from trpg.rl.train_bc import bc_loss_step
from trpg.engine.skill import available_skills

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single
from train_monster_actor import collect_rollout, train_pools


def classify(sks, a0):
    if a0 == 0:
        return "end"
    if a0 >= len(sks):
        return "pad?"
    s = sks[a0]
    if s.skill_id == "move":
        return "move"
    if s.features.expected_damage > 0:
        return "attack"
    return s.skill_id if s.skill_id in ("dodge", "hide", "disengage") \
        else "other"


def probe_states(net, n_probe=64, seed=99):
    """Collect monster-seat states where an ATTACK is currently usable
    (in range, affordable) — the states where passivity is a CHOICE."""
    rng = random.Random(seed)
    t1, tb = train_pools()
    out = []
    ep = 0
    while len(out) < n_probe:
        boss = rng.random() < 0.5
        if boss:
            mon = rng.choice(tb)
            opps = rng.sample(
                ["battle_master", "life", "evocation", "assassin",
                 "champion", "devotion"], 3)
            from trpg.scenarios.monsters import PARTY3_EQUIV_LEVEL
            lvl = max(1, round(PARTY3_EQUIV_LEVEL[mon]))
        else:
            mon = rng.choice(t1)
            opps = [rng.choice(["battle_master", "assassin", "life"])]
            from trpg.scenarios.monsters import EQUIV_LEVEL_1V1
            lvl = max(1, round(EQUIV_LEVEL_1V1[mon]))
        random.seed(7919 * ep + 13)
        env = CombatEnvV2(seed=ep, n_agents=1, n_opps=len(opps))
        obs, _ = env.reset(agent_archs=[mon], opp_archs=opps,
                           level=MONSTER_DEFS[mon].natural_level,
                           opp_level=lvl)
        done = False
        while not done and len(out) < n_probe:
            aid = env.current_agent_id
            ch = env.ws.characters[aid]
            sks = available_skills(ch, env.ws)
            # any damaging skill usable right now (range gate, melee ~1.5m)?
            tgt = min((c for i, c in env.ws.characters.items()
                       if env.ws.is_party_ally(i) != env.ws.is_party_ally(aid)
                       and c.is_alive()),
                      key=lambda c: c.position.distance_to(ch.position),
                      default=None)
            if tgt is not None:
                dist = ch.position.distance_to(tgt.position)
                atk_slots = [i for i, s in enumerate(sks)
                             if s.features.expected_damage > 0
                             and dist <= (s.features.range_m or 1.5) + 1e-6]
                if atk_slots:
                    out.append((blind_np_single(obs), atk_slots,
                                [i for i, s in enumerate(sks)
                                 if s.skill_id == "move"]))
            # step with the net's sampled action to diversify states
            ot = {k: torch.from_numpy(v).unsqueeze(0)
                  for k, v in blind_np_single(obs).items()}
            from trpg.rl.train_ppo import _sample_action
            with torch.no_grad():
                action, *_ = _sample_action(net, ot, env.resources,
                                            env.ws, aid)
            obs, _, term, trunc, _ = env.step(action.numpy().tolist())
            done = term or trunc
        ep += 1
    return out


def probe_gap(net, probes):
    """Mean (attack_logit_max − move_logit) over probe states."""
    gaps = []
    with torch.no_grad():
        for obs, atk_slots, mv_slots in probes:
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            _, s, _, _ = net(ot)
            s = s[0]
            atk = max(float(s[i]) for i in atk_slots)
            mv = float(s[mv_slots[0]]) if mv_slots else 0.0
            gaps.append(atk - mv)
    return float(np.mean(gaps))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", nargs="?", default="models/mon_actor1/ma_u0020.pt")
    p.add_argument("--steps", type=int, default=1024)
    args = p.parse_args()

    net = load_student(args.ckpt)
    t1, tb = train_pools()

    class A:   # mimic train args for collect_rollout
        p_boss, p_mon_agent, p_synth, p_mon_opp = 0.35, 0.35, 0.10, 0.30
    rng = random.Random(1234)

    print("collecting diagnostic rollout (monster-heavy mix)…", flush=True)
    batch, tags, st = collect_rollout(net, args.steps, seed=4242, rng=rng,
                                      t1=t1, tb=tb, args=A)
    print(f"  eps={st['eps']} boss={st['boss_eps']} "
          f"mon-atk={st['mon_atk_rate']:.2f}@{st['mon_turns']}")

    # ── H1: per-action-type advantage on monster-seat steps ─────────────
    # Re-walk the batch: tags mark identity; monster seats have tag in
    # MONSTER_DEFS. Action type needs available_skills at that state — not
    # stored, so approximate via skill slot semantics: slot from batch
    # actions + the obs skill features (expected_damage column is part of
    # the 66-d feature; damaging ⇔ feature > 0). We use obs_skills directly.
    from trpg.engine.skill import SKILL_FEATURE_DIM  # noqa
    obs_sk = batch["obs"]["skills"]          # [N, 20, 66]
    acts = batch["actions"]
    advs = batch["advantages"]
    # feature index of expected_damage in SkillFeatures.as_vector: probe via
    # a known attack: find max-correlating column is overkill — use the
    # convention that damaging slots have any positive value in the damage
    # block. Safer: recompute action type online is impossible post-hoc, so
    # bucket by (a0==0 end) / (a0==1 move-slot convention?) — slot 1 is move
    # for every kit (available_skills puts move first after end? verify).
    # Empirical check below prints the slot-id histogram per type instead.
    groups = defaultdict(list)
    for i, t in enumerate(tags):
        if t in MONSTER_DEFS:
            a0 = int(acts[i, 0])
            key = ("end" if a0 == 0 else f"slot{a0}")
            groups[key].append(float(advs[i]))
    print("\nH1 — monster-seat advantage by chosen SKILL SLOT "
          "(slot1=move by convention; check vs usage mix):")
    for k in sorted(groups, key=lambda x: -len(groups[x])):
        v = np.array(groups[k])
        if len(v) >= 20:
            print(f"  {k:<7} n={len(v):4d}  adv={v.mean():+.3f}±{v.std()/np.sqrt(len(v)):.3f}")

    # ── probe states for H2/H3 ───────────────────────────────────────────
    print("\ncollecting attack-available probe states…", flush=True)
    probes = probe_states(net, n_probe=64)
    g0 = probe_gap(net, probes)
    print(f"  probe gap BEFORE: attack−move logit = {g0:+.3f}")

    optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    ppo_update(net, batch, optim, n_epochs=4, batch_size=64,
               ent_coef=0.01, device="cpu", value_only=False)
    g1 = probe_gap(net, probes)
    print(f"H3 — after ONE ppo_update: gap = {g1:+.3f}  (Δ {g1-g0:+.3f})")

    sw = np.load("models/seed_dtype1/switch_demos.npz")
    sw_obs = {k[4:]: sw[k] for k in sw.files if k.startswith("obs_")}
    sa = np.load("models/mon_actor1/self_anchor.npz")
    sa_obs = {k[4:]: sa[k] for k in sa.files if k.startswith("obs_")}
    for b in range(4):
        src_obs, src_act, src_tt = ((sw_obs, sw["actions"], sw["target_types"])
                                    if b % 2 == 0 else
                                    (sa_obs, sa["actions"], sa["target_types"]))
        sel = np.random.randint(0, len(src_act), size=64)
        ob = {k: torch.from_numpy(v[sel]) for k, v in src_obs.items()}
        bc_loss_step(net, ob, torch.from_numpy(src_act[sel]),
                     torch.from_numpy(src_tt[sel]), optim)
    g2 = probe_gap(net, probes)
    print(f"H2 — after 4 anchor batches: gap = {g2:+.3f}  (Δ {g2-g1:+.3f})")
    print("\nverdict: |H3 Δ| vs |H2 Δ| tells whether anchors undo PPO's "
          "attack-logit gains; H1 tells whether the gradient even points at "
          "attacking.")


if __name__ == "__main__":
    main()
