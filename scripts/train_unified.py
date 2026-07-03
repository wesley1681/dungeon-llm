"""Unified self-play population PPO — the 'beat, not imitate' wave.

The /goal: ONE blind capability-conditioned net that PLAYS any identity
(standard class / monster / chimera-synth) on EITHER seat, positions (walls),
chooses the situationally-best skill, and BEATS the scripted expert — across
advantage / disadvantage / 1vN / Nv1. Crucially it must EXCEED the experts,
not merely imitate them (DAgger's ceiling), and must handle identities that
have NO scripted expert at all (synths / chimeras / odd monster matchups).

Why this is reward-driven, not imitation:
  - DAgger/BC pins the policy to the scripted expert's per-state choice -> its
    ceiling IS the expert. To exceed it the win signal (PPO + HP-PBRS) must be
    the gradient that moves the policy, and the only role of any anchor is to
    stop catastrophic collapse/forgetting — NOT to dictate actions.
  - So the anchor here is SELF-DISTILLATION (the warm net's OWN greedy actions
    over a broad identity panel), exactly the recipe that let mon_actor EXCEED
    the scripted monster policy. It preserves competence; reward finds better.
  - For identities with no script (synth / chimera), the only competent
    opponent is the net itself -> SELF-PLAY (a frozen blind snapshot drives the
    opponent seat). make_archetype_policy falls back to a weak generic
    heuristic for those, so self-play is the only way to train against them.

Recipe:
  - warm-start models/unified/uni_v2.pt (blind, 1 head group)
  - every episode samples identities for BOTH seats from {class, synth,
    chimera, monster}; arrangement from {1v1, ...teams...} with a disadvantage
    bias (the measured 1v2/1v3 = 0% gap).
  - opponent control: a synth/chimera opponent (no script) is ALWAYS driven by
    a frozen blind snapshot (self-play). A class/monster opponent is the
    scripted expert with prob (1-p_self), else the snapshot — so 'beat expert'
    is trained directly while self-play adds robustness + no-script coverage.
  - reward = env defaults (HP-PBRS + wasted-move cost [+ mitigation if set]) —
    identity-agnostic, geometric positioning, no skill names. No KL anchor.
  - self-identity channels blinded at the env boundary (rollout AND opponent).

Usage:
  python scripts/train_unified.py --updates 60 --out_dir models/unified_sp \
      --warm models/unified/uni_v2.pt --p_self 0.4 --p_chimera 0.1 \
      --p_team 0.45 --p_disadv 0.6 --anchor_states 12000 --anchor_batches 3
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch

from trpg.rl.env_v2 import CombatEnvV2, _TEAM_CONFIGS, LAYOUTS
from trpg.rl.model import (apply_resource_mask, apply_entity_mask, pick_action)
from trpg.rl.neural_policy import NeuralCombatPolicy
from trpg.rl.train_ppo import _sample_action, _compute_gae, ppo_update
from trpg.rl.train_bc import bc_loss_step
from trpg.engine.skill import available_skills

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from synth_identity import fresh_synth, STANDARD_IDS
from chimera_defs import register_chimeras, CHIMERA_IDS
from trpg.scenarios.monsters import (register_monsters, onev1_viable_monsters,
                                     EQUIV_LEVEL_1V1, MONSTER_DEFS)

# Reuse the proven measurement + blinding machinery verbatim (DRY).
from train_population import (blind_np_single, descriptor_weight_norm,
                              grad_cosines, standard_probe, monster_opp_probe,
                              chimera_probe, play_probe, fmt_chim)
from eval_routed import stable_seed


# ── Identity sampling ─────────────────────────────────────────────────────────

def sample_identity(rng, mon_pool, p_synth, p_chimera, p_mon, slot):
    """Return (ident_id, tag, is_mon, no_script). no_script=True for synth /
    chimera identities (no scripted expert -> a same-identity OPPONENT must be
    neural). Monsters and standard classes both have scripts."""
    r = rng.random()
    if mon_pool and r < p_mon:
        m = rng.choice(mon_pool)
        return m, "mon", True, False
    r -= p_mon
    if CHIMERA_IDS and r < p_chimera:
        return rng.choice(CHIMERA_IDS), "chimera", False, True
    r -= p_chimera
    if r < p_synth:
        return fresh_synth(rng, slot=slot), "synth", False, True
    c = rng.choice(STANDARD_IDS)
    return c, c, False, False


def _pair_levels(rng, a_is_mon, a_id, o_is_mon, o_id):
    """Fair 1v1 levels: a monster vs a non-monster sets the non-monster to the
    monster's MEASURED equiv level (coin-flip pairing); else symmetric."""
    if a_is_mon and not o_is_mon:
        return MONSTER_DEFS[a_id].natural_level, max(1, round(EQUIV_LEVEL_1V1[a_id]))
    if o_is_mon and not a_is_mon:
        return max(1, round(EQUIV_LEVEL_1V1[o_id])), MONSTER_DEFS[o_id].natural_level
    if a_is_mon and o_is_mon:
        return MONSTER_DEFS[a_id].natural_level, MONSTER_DEFS[o_id].natural_level
    lvl = rng.randint(3, 8)
    return lvl, lvl


# ── Rollout ──────────────────────────────────────────────────────────────────

def collect_unified_rollout(net, snapshot, n_steps, seed, rng, *,
                            mon_pool, p_synth, p_chimera, p_mon_agent,
                            p_team, p_disadv, p_self, p_inject=0.0,
                            gamma=0.99, lam=0.95):
    """Sequential rollout. Per episode: sample arrangement + both seats'
    identities, decide opponent control (scripted expert vs frozen self-play
    snapshot), run to completion. Returns (batch, tags, n_eps)."""
    net.eval()
    obs_l, act_l, lp_l, rew_l, val_l, done_l = [], [], [], [], [], []
    skm_l, enm_l, grm_l, tag_l = [], [], [], []
    # team configs: any side >1. Disadvantage = our side strictly smaller.
    team_cfgs = [c for c in _TEAM_CONFIGS if c[0] + c[1] > 2]
    disadv_cfgs = [c for c in team_cfgs if c[0] < c[1]]
    ep = 0
    while len(rew_l) < n_steps:
        is_team = team_cfgs and rng.random() < p_team
        if is_team:
            if disadv_cfgs and rng.random() < p_disadv:
                na, no_, _ = rng.choice(disadv_cfgs)
            else:
                na, no_, _ = rng.choices(team_cfgs,
                                         weights=[c[2] for c in team_cfgs], k=1)[0]
            # Teams use class/synth/chimera only (monster fair-levelling is a
            # per-creature 1v1 quantity that doesn't generalise to mixed teams).
            a_specs = [sample_identity(rng, (), p_synth, p_chimera, 0.0, ep * 16 + i)
                       for i in range(na)]
            o_specs = [sample_identity(rng, (), p_synth, p_chimera, 0.0,
                                       ep * 16 + 100 + i) for i in range(no_)]
            lvl = rng.randint(3, 8)
            a_lvl = o_lvl = lvl
            tag = "disadv" if na < no_ else ("team_adv" if na > no_ else "team")
        else:
            na = no_ = 1
            a_id, a_tag, a_mon, a_ns = sample_identity(
                rng, mon_pool, p_synth, p_chimera, p_mon_agent, ep)
            o_id, o_tag, o_mon, o_ns = sample_identity(
                rng, mon_pool, p_synth, p_chimera, 0.35, ep + 7)
            a_specs = [(a_id, a_tag, a_mon, a_ns)]
            o_specs = [(o_id, o_tag, o_mon, o_ns)]
            a_lvl, o_lvl = _pair_levels(rng, a_mon, a_id, o_mon, o_id)
            tag = a_tag

        a_archs = [s[0] for s in a_specs]
        o_archs = [s[0] for s in o_specs]
        opp_no_script = any(s[3] for s in o_specs)
        opp_has_mon = any(s[2] for s in o_specs)
        # A no-script opponent (synth/chimera) MUST be neural (self-play is the
        # only competent opponent for it). For SCRIPTED-capable opponents we
        # prefer their script: a MONSTER opponent always uses its script (the
        # measured u0->u15 run showed snapshot-driven monsters erode vs-mon by
        # ~12pp via a -0.97 grad conflict — the agent learns to beat a neural
        # monster, not the scripted one the eval measures, and 'beat the
        # scripted expert' IS the goal). A class opponent may be self-play with
        # prob p_self for coevolution robustness.
        use_self = snapshot is not None and (
            opp_no_script or (not opp_has_mon and rng.random() < p_self))

        layout = rng.choice(LAYOUTS)
        env = CombatEnvV2(seed=seed * 1_000_003 + ep, n_agents=na, n_opps=no_)
        obs, _ = env.reset(agent_archs=a_archs, opp_archs=o_archs,
                           level=a_lvl, opp_level=o_lvl, layout=layout)
        if use_self:
            for oid in env.opp_ids:
                env._opp_policies[oid] = NeuralCombatPolicy(
                    snapshot, device="cpu", blind=True)
            tag = tag + "_sp"

        # DISTRIBUTION EXTENSION (2026-07-01): inject immunity to the AGENT's
        # dominant damage type onto the opponent(s). Fills the measured hole —
        # training's immune enemies were only ever MATCHED-strength; the model
        # never saw "immune AND weak" (low-maxHP monster) so its secondary-vs-
        # dodge ranking there is untrained (maxHP-gate dodge collapse). By
        # injecting across ALL 1v1 opps — including the weak low-HP monsters
        # already in the pool — the caster-vs-weak-immune corner enters the
        # distribution. Reward (kill it) pushes attack over dodge; the self-
        # distill anchor (non-immune states only) preserves the rest = no erosion.
        if (not is_team) and p_inject > 0.0 and rng.random() < p_inject:
            from seed_switch_bc import damaging_options
            from trpg.rl.obs import build_obs
            aid0 = env.agent_ids[0]
            for oid in env.opp_ids:
                opts = damaging_options(env.ws, aid0, oid)
                if opts:
                    dom = max(opts, key=lambda t: t[4])[3]   # dominant dtype
                    env.ws.characters[oid].damage_multipliers[dom] = 0.0
            obs = build_obs(env.ws, env.current_agent_id, env.resources)
            tag = tag + "_imm"

        obs = blind_np_single(obs)
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
            done_l.append(bool(done)); tag_l.append(tag)
            obs = blind_np_single(obs2) if not done else obs2
        ep += 1

    rewards = np.array(rew_l, dtype=np.float32)
    values = np.array(val_l, dtype=np.float32)
    dones = np.array(done_l, dtype=np.float32)
    adv, ret = _compute_gae(rewards, values, dones, next_value=0.0,
                            gamma=gamma, gae_lambda=lam)
    batch = {
        "obs": {k: np.stack([o[k] for o in obs_l]) for k in obs_l[0]},
        "actions": np.array(act_l, dtype=np.int64),
        "log_probs": np.array(lp_l, dtype=np.float32),
        "skill_masks": np.stack(skm_l).astype(np.bool_),
        "entity_masks": np.stack(enm_l).astype(np.bool_),
        "grid_masks": np.stack(grm_l).astype(np.bool_),
        "rewards": rewards, "values": values,
        "returns": ret, "advantages": adv, "dones": dones,
    }
    return batch, tag_l, ep


# ── Self-distillation anchor (full identity panel) ───────────────────────────

def collect_self_distill_anchor(snapshot, n_states, seed, rng, mon_pool,
                                p_synth, p_chimera, p_mon):
    """The warm net's OWN greedy actions over the FULL identity panel (class /
    synth / chimera / monster), 1v1 + 1v2 — preserves competence WITHOUT
    pinning to any script. Records (blinded obs, action, target_type) for
    bc_loss_step, mirroring seed_switch_bc.collect_self_anchor but over the
    broader identity pool this wave trains on."""
    from trpg.rl.obs import build_obs
    snapshot.eval()
    O, A, T = [], [], []
    ep = 0
    while len(A) < n_states:
        a_id, _, a_mon, _ = sample_identity(rng, mon_pool, p_synth, p_chimera,
                                            p_mon, ep)
        # opponent: monster (with the agent's fair counterpart) or a class.
        if not a_mon and mon_pool and rng.random() < p_mon:
            o_id, o_mon = rng.choice(mon_pool), True
        else:
            o_id, o_mon = rng.choice(STANDARD_IDS), False
        a_lvl, o_lvl = _pair_levels(rng, a_mon, a_id, o_mon, o_id)
        # Include 1v2 class states (resource-timing is a 1v2 phenomenon).
        n_opp = 1
        if not a_mon and not o_mon:
            n_opp = rng.choice([1, 1, 2])
        o_archs = [o_id] + [rng.choice(STANDARD_IDS) for _ in range(n_opp - 1)]
        env = CombatEnvV2(seed=seed * 1_000_003 + ep, n_agents=1, n_opps=n_opp)
        env.reset(agent_archs=[a_id], opp_archs=o_archs,
                  level=a_lvl, opp_level=o_lvl)
        obs = blind_np_single(build_obs(env.ws, env.current_agent_id,
                                        env.resources))
        done = False
        while not done:
            aid = env.current_agent_id
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                el, s, e, g = snapshot(ot)
            s = apply_resource_mask(s, env.resources, env.ws, aid)
            e = apply_entity_mask(e, ot, env.ws, aid)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws,
                                   agent_id=aid))
            if act[0] == 0:
                tt = -1
            else:
                sks = available_skills(env.ws.characters[aid], env.ws)
                tt = (int(sks[act[0]].features.target_type)
                      if act[0] < len(sks) else -1)
            O.append(obs); A.append(act); T.append(tt)
            obs2, _, term, trunc, _ = env.step(act)
            done = term or trunc
            obs = blind_np_single(obs2) if not done else obs2
        ep += 1
        if ep % 300 == 0:
            print(f"  self-anchor: {len(A)}/{n_states} states ({ep} eps)",
                  flush=True)
    obs_np = {k: np.stack([o[k] for o in O]) for k in O[0]}
    return obs_np, np.array(A, np.int64), np.array(T, np.int64)


# ── Headline-gap probes (within-run comparable, fixed seeds) ─────────────────

FOCUS = ["battle_master", "war", "arcane_trickster", "assassin", "champion"]


def focus_probe(net, games=2):
    """Model WR driving each FOCUS class vs the 12 standard scripted experts."""
    per = {}
    for a in FOCUS:
        w = n = 0
        for o in STANDARD_IDS:
            base = stable_seed(f"uni_foc_{a}_{o}")
            for k in range(games):
                win, _ = play_probe(net, a, o, base + k, level=5, opp_level=5)
                w += int(win); n += 1
        per[a] = w / n
    return per


def disadv_probe(net, games=2):
    """Model in 1v2 (outnumbered) vs 2 standard scripted experts, per class."""
    per = {}
    wins = n = 0
    for a in FOCUS:
        w = na = 0
        for o in STANDARD_IDS:
            base = stable_seed(f"uni_dis_{a}_{o}")
            for k in range(games):
                env = CombatEnvV2(seed=base + k, n_agents=1, n_opps=2)
                o2 = STANDARD_IDS[(STANDARD_IDS.index(o) + 3) % len(STANDARD_IDS)]
                env.reset(agent_archs=[a], opp_archs=[o, o2],
                          level=5, opp_level=4)
                aid = env.agent_ids[0]
                obs = blind_np_single(_build(env))
                done = False
                while not done:
                    actor = env.current_agent_id
                    ot = {k_: torch.from_numpy(v).unsqueeze(0)
                          for k_, v in obs.items()}
                    with torch.no_grad():
                        el, s, e, g = net(ot)
                    s = apply_resource_mask(s, env.resources, env.ws, actor)
                    e = apply_entity_mask(e, ot, env.ws, actor)
                    act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws,
                                           agent_id=actor))
                    obs2, _, term, trunc, _ = env.step(act)
                    done = term or trunc
                    obs = blind_np_single(obs2) if not done else obs2
                win = env.ws.characters[aid].is_alive() and not any(
                    env.ws.characters[o_].is_alive() for o_ in env.opp_ids)
                w += int(win); na += 1
        per[a] = w / na
        wins += w; n += na
    return wins / n, per


def _build(env):
    from trpg.rl.obs import build_obs
    return build_obs(env.ws, env.current_agent_id, env.resources)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm", default="models/unified/uni_v2.pt")
    p.add_argument("--out_dir", default="models/unified_sp")
    p.add_argument("--updates", type=int, default=60)
    p.add_argument("--steps", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--value_warmup", type=int, default=3)
    # identity mix (agent seat)
    p.add_argument("--p_synth", type=float, default=0.25)
    p.add_argument("--p_chimera", type=float, default=0.10)
    p.add_argument("--p_mon_agent", type=float, default=0.15)
    p.add_argument("--mon_max_level", type=float, default=8.0)
    # arrangement
    p.add_argument("--p_team", type=float, default=0.40)
    p.add_argument("--p_disadv", type=float, default=0.55,
                   help="among team episodes, fraction forced to a "
                        "disadvantage (our side outnumbered) config")
    # opponent control
    p.add_argument("--p_self", type=float, default=0.35,
                   help="prob a SCRIPTED-capable (class/monster) opponent is "
                        "instead driven by the frozen self-play snapshot. "
                        "Synth/chimera opponents are ALWAYS self-play.")
    p.add_argument("--p_inject", type=float, default=0.0,
                   help="prob a 1v1 opponent is injected immune to the agent's "
                        "dominant damage type (fills the immune-x-weak-enemy "
                        "distribution hole; 0.0 = original behaviour).")
    p.add_argument("--self_refresh", type=int, default=0,
                   help="if >0, refresh the self-play snapshot to the current "
                        "net every N updates (coevolution). 0 = frozen warm net.")
    # anchor
    p.add_argument("--anchor_states", type=int, default=12000)
    p.add_argument("--anchor_batches", type=int, default=3,
                   help="self-distill BC minibatches per update (0 = none)")
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--std_games", type=int, default=1)
    p.add_argument("--mon_games", type=int, default=1)
    p.add_argument("--focus_games", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "BLIND").write_text("unified self-play PPO (beat-not-imitate)\n")
    register_chimeras(); register_monsters()
    mon_pool = onev1_viable_monsters(args.mon_max_level)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    net = load_student(args.warm)
    snapshot = copy.deepcopy(net); snapshot.eval()
    for p_ in snapshot.parameters():
        p_.requires_grad_(False)
    optim = torch.optim.Adam([p_ for p_ in net.parameters()
                              if p_.requires_grad], lr=args.lr)
    print(f"warm={args.warm}  steps={args.steps} updates={args.updates}\n"
          f"identity mix: synth={args.p_synth} chimera={args.p_chimera} "
          f"mon_agent={args.p_mon_agent} | team={args.p_team} "
          f"disadv={args.p_disadv} | p_self={args.p_self}\n"
          f"chimeras={CHIMERA_IDS} mon_pool({len(mon_pool)})", flush=True)

    anchor = None
    if args.anchor_batches > 0:
        print(f"building self-distill anchor ({args.anchor_states} states)...",
              flush=True)
        anchor = collect_self_distill_anchor(
            snapshot, args.anchor_states, seed=args.seed * 13 + 1, rng=rng,
            mon_pool=mon_pool, p_synth=args.p_synth, p_chimera=args.p_chimera,
            p_mon=0.4)
        print(f"  anchor: {len(anchor[1])} pairs (self-distill, full panel)",
              flush=True)

    def run_probes(tag):
        std, _ = standard_probe(net, games=args.std_games)
        mon, _ = monster_opp_probe(net, mon_pool, games=args.mon_games)
        foc = focus_probe(net, games=args.focus_games)
        dis, _ = disadv_probe(net, games=args.focus_games)
        d, _ = descriptor_weight_norm(net)
        fstr = " ".join(f"{k[:4]}={v:.0%}" for k, v in foc.items())
        print(f"  [{tag}] std12={std:.1%} vs-mon={mon:.1%} "
              f"disadv1v2={dis:.1%} desc={d:.3f}\n        focus: {fstr}",
              flush=True)
        return std, mon, foc, dis

    print("baseline probes (update 0):", flush=True)
    run_probes("u0")

    for update in range(1, args.updates + 1):
        if args.self_refresh and update % args.self_refresh == 0:
            snapshot = copy.deepcopy(net); snapshot.eval()
            for p_ in snapshot.parameters():
                p_.requires_grad_(False)
        batch, tags, n_eps = collect_unified_rollout(
            net, snapshot, args.steps, seed=args.seed * 7919 + update, rng=rng,
            mon_pool=mon_pool, p_synth=args.p_synth, p_chimera=args.p_chimera,
            p_mon_agent=args.p_mon_agent, p_team=args.p_team,
            p_disadv=args.p_disadv, p_self=args.p_self, p_inject=args.p_inject)
        info = ppo_update(net, batch, optim, n_epochs=args.epochs,
                          batch_size=args.batch, ent_coef=args.ent_coef,
                          device="cpu",
                          value_only=(update <= args.value_warmup))
        line = (f"Update {update:3d}/{args.updates}"
                f"{' [warmup]' if update <= args.value_warmup else ''}  "
                f"eps={n_eps}  policy={info['policy_loss']:+.4f}  "
                f"value={info['value_loss']:.4f}  ent={info['entropy']:.3f}")
        if anchor is not None and update > args.value_warmup:
            a_obs, a_act, a_tt = anchor
            accs = []
            for _ in range(args.anchor_batches):
                sel = np.random.randint(0, len(a_act), size=args.batch)
                ob = {k: torch.from_numpy(v[sel]) for k, v in a_obs.items()}
                _, acc = bc_loss_step(net, ob, torch.from_numpy(a_act[sel]),
                                      torch.from_numpy(a_tt[sel]), optim)
                accs.append(acc.get("skill", float("nan")))
            line += f"  anchor_acc={np.nanmean(accs):.2f}"
        print(line, flush=True)

        if update % args.eval_every == 0:
            net.eval()
            cos = grad_cosines(net, batch, tags)
            if cos:
                worst = min(cos, key=lambda x: x[2])
                print(f"  grad-cos mean={np.mean([c for *_, c in cos]):+.3f} "
                      f"worst={worst[2]:+.3f} ({worst[0]}~{worst[1]})",
                      flush=True)
            snap = out_dir / f"uni_u{update:04d}.pt"
            torch.save(net.state_dict(), snap)
            run_probes(f"u{update}")
            print(f"  snap={snap.name}", flush=True)
            net.train()

    print("done.", flush=True)


if __name__ == "__main__":
    main()
