"""BC-seed the typed-resist SWITCH behavior into the B-net, then verify.

Why this exists (MONSTER_CATALOG §6.6, 2026-06-12g): the net CAN represent
"read enemy typed_resist -> switch damage type" (supervised fit acc 0.98,
behavior keyed to the descriptor cell), but PPO cannot DISCOVER it from the
BC fireball prior even in a maximally dense lab. So we install the behavior
supervised (this script), and future PPO waves keep these demos in the anchor.

Pipeline:
  1. DEMOS — roll the warm net (sampled, blinded) over a seed distribution:
     agents = archetypes with >=2 damage types (discovered from engine data,
     probe_kit_dtypes); opponents = standard 12 + all 23 monsters (viable at
     fair pairing, the 1vN-excluded at L8 — labels don't need winnability);
     with prob p_inject the opponent gets a resist profile injected on one of
     the AGENT's damage types (mult in {0, 0.5, 2}). At every state where the
     net itself sampled a feasible DAMAGING skill, the label is the ORACLE
     choice — argmax of resist-adjusted EV (expected_damage x weapon
     attacks_per_action x target damage_multiplier) over the feasible damaging
     candidates — encoded via the standard build_action -> encode_action path.
     The oracle action is also EXECUTED (teacher forcing), so demo states
     follow corrected trajectories. Non-damage decisions (move/heal/end) are
     never relabeled — the distill anchor preserves them.
  2. SEED — BC fine-tune the warm net, alternating switch-demo batches and
     distill-anchor batches (1:1) to protect the standard kits.
  3. VERIFY — fixed acceptance battery (greedy, blinded):
       a. evocation vs injected-fire-immune orc: fire%-on low / off high,
          WR-on near always-switch ceiling (100%), WR-off low;
       b. self-calibrated second agent (dominant dtype measured desc-off,
          then immunity to THAT type injected) — generality;
       c. evocation vs REAL fire_elemental / young_red_dragon (natural
          immunity, unwinnable — behavior-only): fire%-on drops;
       d. normal-orc control: behavior unchanged;
       e. std12 standard_probe + monster_opp_probe vs the warm net's own
          numbers recomputed in-process (same protocol as the pop wave).

Oracle EV ignores hit/save probability (documented simplification — it cannot
flip a x0/x0.5/x2 ranking between options of similar magnitude).

Usage:
  python scripts/seed_switch_bc.py --oracle_check          # preflight only
  python scripts/seed_switch_bc.py --states 10000 --steps 800
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

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.train_ppo import _sample_action
from trpg.rl.train_bc import bc_loss_step
from trpg.rl.obs import build_obs, migrate_entities_v3_to_v4
from trpg.rl.action import encode_action
from trpg.engine.skill import available_skills
from trpg.scenarios.archetypes import STANDARD_ARCHETYPES
from trpg.scenarios.monsters import (register_monsters, MONSTER_DEFS,
                                     EQUIV_LEVEL_1V1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import (load_student, load_dataset, DATASET_PATH,
                            blind_self_identity_np)
from train_population import (blind_np_single, standard_probe,
                              monster_opp_probe, descriptor_weight_norm)
from probe_descriptor_ab import zero_enemy_descriptor
from probe_fire_switch import action_damage_type
from probe_kit_dtypes import agent_damage_types, multi_dtype_archetypes
from eval_routed import stable_seed

MULTS   = (0.0, 0.5, 2.0)
MULT_W  = (0.50, 0.25, 0.25)


# ── Oracle ───────────────────────────────────────────────────────────────────

def damaging_options(ws, aid, oid, skm=None):
    """Feasible damaging skills at this state: [(slot, skill, act, dtype, ev)].

    Feasibility comes from the SAMPLING mask (skm True = masked by
    apply_resource_mask — range/cost/slot gates), so the oracle can never
    label an action the engine would reject. EV = expected_damage x weapon
    attacks_per_action x the target's EFFECTIVE multiplier — the skill's
    damage-share-weighted soft one-hot (materialized features.damage_types,
    same data the obs dtype tail carries) dotted with the target's
    damage_multipliers. Since 12i this replaces single-packet action
    introspection: a divine_smite row now correctly keeps its 光耀 EV when
    the weapon type is immune (its expected_damage describes the rider).
    ``dtype`` is the row's primary (largest-share) type — used for flip
    targeting and reporting.
    """
    a = ws.characters[aid]
    o = ws.characters[oid]
    n_atk = float(getattr(a, "attacks_per_action", 1) or 1)
    opts = []
    for i, sk in enumerate(available_skills(a, ws)):
        if i == 0:
            continue
        if skm is not None and i < len(skm) and bool(skm[i]):
            continue
        if sk.features.expected_damage <= 0:
            continue
        pairs = list(sk.features.iter_damage_types())
        if not pairs:
            continue
        try:
            act = sk.build_action(aid, oid, (o.position.x, o.position.y))
        except Exception:
            continue
        if act is None:
            continue
        mult = sum(share * float((o.damage_multipliers or {}).get(tok, 1.0))
                   for tok, share in pairs)
        dt = max(pairs, key=lambda p: p[1])[0]
        ev = float(sk.features.expected_damage)
        if sk.skill_id.startswith("weapon:"):
            ev *= n_atk
        ev *= mult
        opts.append((i, sk, act, dt, ev))
    return opts


def oracle_pick(opts, net_slot):
    """Best option by EV; ties (within 1e-6) keep the net's own choice to
    minimize gratuitous drift. Returns the chosen tuple."""
    best = max(opts, key=lambda t: t[4])
    for t in opts:
        if t[0] == net_slot and t[4] >= best[4] - 1e-6:
            return t
    return best


# ── Episode distribution ─────────────────────────────────────────────────────

def sample_spec(rng, pools_by_level, p_mon=0.5):
    """-> (agent_arch, opp_id, agent_level, opp_level, is_monster)."""
    if rng.random() < p_mon:
        mon = rng.choice(sorted(MONSTER_DEFS))
        eq = EQUIV_LEVEL_1V1[mon]
        a_lvl = int(min(8, max(3, round(eq)))) if math.isfinite(eq) else 8
        agent = rng.choice(sorted(pools_by_level[a_lvl]))
        return agent, mon, a_lvl, MONSTER_DEFS[mon].natural_level, True
    lvl = rng.randint(5, 8)
    agent = rng.choice(sorted(pools_by_level[lvl]))
    opp = rng.choice(STANDARD_ARCHETYPES)
    return agent, opp, lvl, lvl, False


def maybe_inject(env, rng, agent_arch, p_inject, flip_p=0.5, allow=None):
    """With prob p_inject, inject (dtype from the AGENT's kit, mult) onto the
    opponent. Returns the injected (dtype, mult) or None. Descriptor picks it
    up on the next build_obs (cache keyed on damage_multipliers since 06-12g).

    flip_p: fraction of injections that TARGET the oracle-dominant dtype with
    mult 0.0 — a guaranteed ranking flip, so every agent gets dense switch
    episodes regardless of how many dtypes its kit spreads across (v1/v2
    failure: uniform dtype x mult left e.g. life with ~2% switch labels).

    allow: optional damage-type allowlist — the explicit HELD-OUT boundary
    for the 舉一反三 acceptance (types outside it are never injected, so the
    (skill-bit, resist-col) pair for them never co-occurs in training; only
    the shared join can serve them at eval). A dominant flip whose target
    type is disallowed is SKIPPED, not retargeted (no partial signal)."""
    if rng.random() >= p_inject:
        return None
    aid = env.agent_ids[0]
    oid = env.opp_ids[0]
    opts = damaging_options(env.ws, aid, oid)
    if not opts:
        return None
    if rng.random() < flip_p:
        dt = max(opts, key=lambda t: t[4])[3]   # dominant by oracle EV
        if allow and dt not in allow:
            return None
        mult = 0.0
    else:
        pool = sorted({t[3] for t in opts})
        if allow:
            pool = [d for d in pool if d in allow]
        if not pool:
            return None
        dt = rng.choice(pool)
        mult = rng.choices(MULTS, weights=MULT_W, k=1)[0]
    for o in env.opp_ids:
        env.ws.characters[o].damage_multipliers[dt] = mult
    return dt, mult


# ── Demo collection ──────────────────────────────────────────────────────────

def collect_demos(net, n_states, seed, rng, pools_by_level,
                  p_mon=0.5, p_inject_std=0.7, p_inject_mon=0.35,
                  ev_eps=1.0, flip_p=0.5, inject_allow=None):
    """ev_eps: when the best feasible damaging EV is below this, the state is
    NOT labeled and the net's own action is executed — labeling a zero-EV
    option (e.g. fireball once magic_missile's slots are dry vs a fire-immune
    enemy) teaches exactly the wrong thing (demo_fit_audit2: 42.8% of
    immune[火] labels were POINT skills = this poison)."""
    net.eval()
    O, A, T, F = [], [], [], []
    inj_hist = Counter()   # realized injected (dtype, mult-class) — the
                           # auditable train-side boundary for held-out claims
    n_switch = n_confirm = n_poison_skip = ep = 0
    while len(A) < n_states:
        agent, opp, a_lvl, o_lvl, is_mon = sample_spec(rng, pools_by_level, p_mon)
        env = CombatEnvV2(seed=seed * 1_000_003 + ep, n_agents=1, n_opps=1)
        env.reset(agent_archs=[agent], opp_archs=[opp],
                  level=a_lvl, opp_level=o_lvl)
        inj = maybe_inject(env, rng, agent,
                           p_inject_mon if is_mon else p_inject_std, flip_p,
                           allow=inject_allow)
        if inj:
            inj_hist[f"{inj[0]}x{inj[1]:g}"] += 1
        obs = blind_np_single(build_obs(env.ws, env.current_agent_id,
                                        env.resources))
        oid = env.opp_ids[0]
        done = False
        while not done:
            aid = env.current_agent_id
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                action, lp, val, skm, enm, grm = _sample_action(
                    net, ot, env.resources, env.ws, aid)
            act = action.numpy().tolist()
            exec_act = act
            opts = damaging_options(env.ws, aid, oid, skm.numpy())
            if any(t[0] == act[0] for t in opts):
                slot, sk, adict, dt, ev = oracle_pick(opts, act[0])
                if ev < ev_eps:
                    n_poison_skip += 1   # no worthwhile damage option: don't
                                         # label, don't teacher-force
                else:
                    enc = encode_action(adict, env.ws, aid)
                    if enc[0] >= 0:
                        O.append(obs)
                        A.append(enc)
                        T.append(int(sk.features.target_type))
                        is_sw = enc[0] != act[0]
                        F.append(is_sw)
                        if is_sw:
                            n_switch += 1
                        else:
                            n_confirm += 1
                        exec_act = list(enc)
            obs2, _, term, trunc, _ = env.step(exec_act)
            done = term or trunc
            obs = blind_np_single(obs2) if not done else obs2
        ep += 1
        if ep % 200 == 0:
            print(f"  demos: {len(A)}/{n_states} states  ({ep} eps, "
                  f"switch={n_switch} confirm={n_confirm} "
                  f"poison-skip={n_poison_skip})", flush=True)
    obs_np = {k: np.stack([o[k] for o in O]) for k in O[0]}
    return (obs_np, np.array(A, np.int64), np.array(T, np.int64),
            np.array(F, np.bool_),
            {"episodes": ep, "switch": n_switch, "confirm": n_confirm,
             "poison_skip": n_poison_skip,
             "inject_hist": dict(inj_hist)})


def collect_self_anchor(net, n_states, seed, rng, pools_by_level, p_mon=0.5,
                        greedy=True):
    """Behavior-preservation anchor: the warm net's OWN actions at REAL-
    descriptor states — all 12 agents, standard + monster opponents, NO
    injection. v1/v2 lesson: the distill anchor's obs carry all-ZERO
    descriptors (v3-era data, zero-padded by migration), so it cannot pin
    behavior in the real-descriptor obs space where std12 actually plays —
    seed v2 regressed std12 -5.6pp precisely there. END pairs included
    (end_head supervision), mirroring bc_collect's TT_END=-1 convention.

    greedy=True records the GREEDY action (argmax via pick_action, the policy
    std12 actually gates) — deterministic, consistent labels. v3 lesson:
    SAMPLED labels carry entropy noise (argmax-vs-sample ceiling ~30-50%),
    turning the anchor into distribution-matching mush that neither protects
    greedy behavior nor lets the switch carve cleanly."""
    net.eval()
    O, A, T = [], [], []
    ep = 0
    while len(A) < n_states:
        if rng.random() < p_mon:
            mon = rng.choice(sorted(MONSTER_DEFS))
            eq = EQUIV_LEVEL_1V1[mon]
            a_lvl = int(min(8, max(3, round(eq)))) if math.isfinite(eq) else 8
            agent = rng.choice(STANDARD_ARCHETYPES)
            opp, o_lvl = mon, MONSTER_DEFS[mon].natural_level
        else:
            a_lvl = rng.randint(5, 8)
            agent = rng.choice(STANDARD_ARCHETYPES)
            opp, o_lvl = rng.choice(STANDARD_ARCHETYPES), None
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
            if greedy:
                with torch.no_grad():
                    el, s, e, g = net(ot)
                s = apply_resource_mask(s, env.resources, env.ws, aid)
                e = apply_entity_mask(e, ot, env.ws, aid)
                act = list(pick_action(el[0], s[0], e[0], g[0],
                                       ws=env.ws, agent_id=aid))
            else:
                with torch.no_grad():
                    action, *_ = _sample_action(net, ot, env.resources,
                                                env.ws, aid)
                act = action.numpy().tolist()
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
        if ep % 200 == 0:
            print(f"  self-anchor: {len(A)}/{n_states} states ({ep} eps)",
                  flush=True)
    obs_np = {k: np.stack([o[k] for o in O]) for k in O[0]}
    return obs_np, np.array(A, np.int64), np.array(T, np.int64)


# ── Verification battery ─────────────────────────────────────────────────────

def eval_switch(net, agent_arch, enemy, a_lvl, o_lvl, inject, kill_desc,
                games, tag):
    """Greedy blinded games. Returns ({dtype: share}, win_rate)."""
    net.eval()
    types = Counter(); ndmg = 0; wins = 0
    for gi in range(games):
        env = CombatEnvV2(seed=stable_seed(f"swb_{tag}_{gi}"),
                          n_agents=1, n_opps=1)
        env.reset(agent_archs=[agent_arch], opp_archs=[enemy],
                  level=a_lvl, opp_level=o_lvl)
        if inject:
            dt_i, mult_i = inject
            for o in env.opp_ids:
                env.ws.characters[o].damage_multipliers[dt_i] = mult_i
        obs = build_obs(env.ws, env.current_agent_id, env.resources)
        aid = env.agent_ids[0]; oid = env.opp_ids[0]
        done = False
        while not done:
            actor = env.current_agent_id
            ob = blind_np_single(obs)
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
                try:
                    built = sks[ai[0]].build_action(
                        actor, oid, (o.position.x, o.position.y))
                except Exception:
                    built = None
                dt = action_damage_type(built, actor_c) if built else None
                if dt:
                    types[dt] += 1
                    ndmg += 1
            obs, _, term, trunc, _ = env.step(list(ai))
            done = term or trunc
        if (env.ws.characters[oid].is_dead()
                and env.ws.characters[aid].is_alive()):
            wins += 1
    shares = {dt: c / max(1, ndmg) for dt, c in types.items()}
    return shares, wins / games


def _fmt(shares, wr):
    s = " ".join(f"{dt}:{v:.0%}" for dt, v in
                 sorted(shares.items(), key=lambda kv: -kv[1]))
    return f"WR={wr:.0%}  [{s}]"


def battery(net, games, arms, tag):
    """arms: list of (name, agent, enemy, a_lvl, o_lvl, inject, watch_dtype).
    Prints on/off rows; returns {name: (share_on, share_off, wr_on, wr_off)}
    for each arm's watch_dtype."""
    print(f"[battery @{tag}] desc-norm={descriptor_weight_norm(net)[0]:.3f}",
          flush=True)
    out = {}
    for name, agent, enemy, a_lvl, o_lvl, inject, watch in arms:
        sh_on, wr_on = eval_switch(net, agent, enemy, a_lvl, o_lvl, inject,
                                   False, games, f"{name}_on")
        sh_off, wr_off = eval_switch(net, agent, enemy, a_lvl, o_lvl, inject,
                                     True, games, f"{name}_off")
        w_on = sh_on.get(watch, 0.0); w_off = sh_off.get(watch, 0.0)
        out[name] = (w_on, w_off, wr_on, wr_off)
        print(f"  {name:24s} on : {_fmt(sh_on, wr_on)}", flush=True)
        print(f"  {'':24s} off: {_fmt(sh_off, wr_off)}", flush=True)
    return out


def dominant_dtype(net, agent_arch, enemy, a_lvl, o_lvl, games, tag):
    """Self-calibration: the agent's dominant damage type measured DESC-OFF
    vs a neutral enemy — the type whose immunity forces a switch."""
    sh, _ = eval_switch(net, agent_arch, enemy, a_lvl, o_lvl, None, True,
                        games, f"dom_{tag}")
    return max(sh.items(), key=lambda kv: kv[1])[0] if sh else None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm", default="models/pop_mon/pop_u0005.pt")
    p.add_argument("--out_dir", default="models/seed_switch")
    p.add_argument("--states", type=int, default=10000)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--demos_in", default="",
                   help="reuse a saved switch_demos.npz instead of collecting")
    p.add_argument("--self_anchor_in", default="",
                   help="reuse a saved self_anchor.npz instead of collecting")
    p.add_argument("--self_states", type=int, default=10000)
    p.add_argument("--ev_eps", type=float, default=1.0,
                   help="skip labeling when best damaging EV is below this")
    p.add_argument("--flip_p", type=float, default=0.5,
                   help="fraction of injections that target the oracle-"
                        "dominant dtype with mult 0 (guaranteed flip)")
    p.add_argument("--inject_types", default="",
                   help="comma-separated damage-type allowlist for injection "
                        "(the explicit held-out boundary); empty = all kit "
                        "types")
    p.add_argument("--old_anchor_every", type=int, default=0,
                   help="run an old (zero-descriptor) distill-anchor batch "
                        "every N steps; 0 = never")
    p.add_argument("--adaptive_boost", action="store_true",
                   help="ignore stored switch flags; weight samples the net "
                        "currently gets wrong (self-balancing — fixed true-"
                        "flag weighting tilted the marginal to the fallback "
                        "skill in v3 and never carved the conditional)")
    p.add_argument("--switch_boost", type=float, default=0.0,
                   help="adaptive per-sample skill-CE weight: 1 + boost for "
                        "samples the net currently gets wrong (no-grad "
                        "pre-forward). Oracle labels are deterministic, so "
                        "this focuses gradient on the unfit (switch) tail "
                        "without noise-chasing; decays to uniform as fit.")
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--eval_games", type=int, default=20)
    p.add_argument("--gate_std_games", type=int, default=1)
    p.add_argument("--gate_mon_games", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--oracle_check", action="store_true",
                   help="preflight: print oracle choices on known cases, exit")
    p.add_argument("--skip_gates", action="store_true")
    args = p.parse_args()

    register_monsters()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    net = load_student(args.warm)

    pools_by_level = {lvl: multi_dtype_archetypes(lvl) for lvl in range(3, 9)}
    print("multi-dtype agent pools:",
          {lvl: sorted(d) for lvl, d in pools_by_level.items()}, flush=True)

    if args.oracle_check:
        print("\n=== oracle preflight ===")
        cases = [
            ("evocation", "orc", 6, None),
            ("evocation", "orc", 6, ("火", 0.0)),
            ("evocation", "orc", 6, ("火", 0.5)),
            ("evocation", "fire_elemental", 8, None),
            ("life", "orc", 6, None),
            ("life", "orc", 6, ("斬擊", 0.0)),
            ("arcane_trickster", "orc", 6, ("穿刺", 0.0)),
            ("devotion", "orc", 6, None),
            ("devotion", "orc", 6, ("斬擊", 0.0)),
        ]
        for agent, opp, lvl, inj in cases:
            o_lvl = (MONSTER_DEFS[opp].natural_level
                     if opp in MONSTER_DEFS else lvl)
            env = CombatEnvV2(seed=11, n_agents=1, n_opps=1)
            env.reset(agent_archs=[agent], opp_archs=[opp],
                      level=lvl, opp_level=o_lvl)
            if inj:
                for o in env.opp_ids:
                    env.ws.characters[o].damage_multipliers[inj[0]] = inj[1]
            aid = env.agent_ids[0]; oid = env.opp_ids[0]
            opts = damaging_options(env.ws, aid, oid)
            opts_s = ", ".join(f"{sk.skill_id}({dt} ev={ev:.1f})"
                               for _, sk, _, dt, ev in opts)
            pick = oracle_pick(opts, -1) if opts else None
            pick_s = pick[1].skill_id if pick else "—"
            print(f"  {agent:16s} vs {opp:15s} inj={inj}  -> {pick_s}\n"
                  f"      options: {opts_s}")
        return

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Battery arms — fixed BEFORE training. Second agent self-calibrated.
    dom2 = dominant_dtype(net, "life", "orc", 6,
                          MONSTER_DEFS["orc"].natural_level, 12, "life")
    print(f"life dominant dtype (desc-off vs orc) = {dom2}", flush=True)
    onat = MONSTER_DEFS["orc"].natural_level
    arms = [
        ("evo/orc+火immune",  "evocation", "orc", 6, onat, ("火", 0.0), "火"),
        ("evo/orc normal",    "evocation", "orc", 6, onat, None,        "火"),
        (f"life/orc+{dom2}immune", "life", "orc", 6, onat, (dom2, 0.0), dom2),
        ("evo/fire_elemental", "evocation", "fire_elemental", 8,
         MONSTER_DEFS["fire_elemental"].natural_level, None, "火"),
    ]

    print("\n=== baseline battery (warm net) ===", flush=True)
    battery(net, args.eval_games, arms, "u0")

    if args.demos_in:
        print(f"\n=== reusing demos from {args.demos_in} ===", flush=True)
        z = np.load(args.demos_in)
        act_np = z["actions"]; tt_np = z["target_types"]
        fl_np = z["switch_flags"] if "switch_flags" in z.files else None
        obs_np = {k[len("obs_"):]: z[k] for k in z.files
                  if k.startswith("obs_")}
        print(f"loaded {len(act_np)} states "
              f"(flags={'yes' if fl_np is not None else 'NO -> adaptive'})",
              flush=True)
    else:
        print("\n=== collecting switch demos ===", flush=True)
        inject_allow = ({t.strip() for t in args.inject_types.split(",")
                         if t.strip()} or None)
        if inject_allow:
            print(f"inject allowlist (held-out boundary): "
                  f"{sorted(inject_allow)}", flush=True)
        obs_np, act_np, tt_np, fl_np, stats = collect_demos(
            net, args.states, args.seed * 7919 + 1, rng, pools_by_level,
            ev_eps=args.ev_eps, flip_p=args.flip_p,
            inject_allow=inject_allow)
        print(f"collected {len(act_np)} states over {stats['episodes']} eps "
              f"(switch={stats['switch']} confirm={stats['confirm']} "
              f"poison-skip={stats['poison_skip']})", flush=True)
        print(f"realized injections: "
              f"{dict(sorted(stats['inject_hist'].items()))}", flush=True)
        np.savez_compressed(out_dir / "switch_demos.npz",
                            actions=act_np, target_types=tt_np,
                            switch_flags=fl_np,
                            **{f"obs_{k}": v for k, v in obs_np.items()})

    if args.self_anchor_in:
        print(f"\n=== reusing self-anchor from {args.self_anchor_in} ===",
              flush=True)
        z = np.load(args.self_anchor_in)
        s_act = z["actions"]; s_tt = z["target_types"]
        s_obs = {k[len("obs_"):]: z[k] for k in z.files
                 if k.startswith("obs_")}
        print(f"loaded {len(s_act)} states", flush=True)
    else:
        print("\n=== collecting self-anchor (real-descriptor states) ===",
              flush=True)
        s_obs, s_act, s_tt = collect_self_anchor(
            net, args.self_states, args.seed * 104729 + 7, rng,
            pools_by_level)
        print(f"collected {len(s_act)} self-anchor states", flush=True)
        np.savez_compressed(out_dir / "self_anchor.npz",
                            actions=s_act, target_types=s_tt,
                            **{f"obs_{k}": v for k, v in s_obs.items()})

    print("\n=== loading distill anchor ===", flush=True)
    a_obs, a_act, a_tt = load_dataset(Path(DATASET_PATH))
    a_obs = {k: v.copy() for k, v in a_obs.items()}
    a_obs["entities"] = migrate_entities_v3_to_v4(a_obs["entities"])
    blind_self_identity_np(a_obs)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(f"\n=== seeding: {args.steps} steps "
          f"(switch + self-anchor 1:1, old-anchor every "
          f"{args.old_anchor_every or 'never'}) ===", flush=True)
    for step in range(1, args.steps + 1):
        sel = np.random.randint(0, len(act_np), size=args.batch)
        ob = {k: torch.from_numpy(v[sel]) for k, v in obs_np.items()}
        acts = torch.from_numpy(act_np[sel])
        w = None
        if args.switch_boost > 0:
            if fl_np is not None and not args.adaptive_boost:
                w = 1.0 + args.switch_boost * torch.from_numpy(
                    fl_np[sel].astype(np.float32))
            else:
                with torch.no_grad():
                    _, sl, _, _ = net(ob)
                w = 1.0 + args.switch_boost * (
                    sl.argmax(-1) != acts[:, 0]).float()
        _, acc_s = bc_loss_step(net, ob, acts,
                                torch.from_numpy(tt_np[sel]), optim,
                                skill_sample_w=w)
        sel = np.random.randint(0, len(s_act), size=args.batch)
        ob = {k: torch.from_numpy(v[sel]) for k, v in s_obs.items()}
        _, acc_sa = bc_loss_step(net, ob, torch.from_numpy(s_act[sel]),
                                 torch.from_numpy(s_tt[sel]), optim)
        acc_a = {}
        if args.old_anchor_every and step % args.old_anchor_every == 0:
            sel = np.random.randint(0, len(a_act), size=args.batch)
            ob = {k: torch.from_numpy(v[sel]) for k, v in a_obs.items()}
            _, acc_a = bc_loss_step(net, ob, torch.from_numpy(a_act[sel]),
                                    torch.from_numpy(a_tt[sel]), optim)
        if step % 50 == 0 or step == 1:
            jw = net.skill_heads[0].weight[0, -1].item()
            print(f"  step {step:4d}  switch-acc={acc_s.get('skill', 0):.2f}  "
                  f"self-acc={acc_sa.get('skill', 0):.2f}  "
                  f"old-acc={acc_a.get('skill', float('nan')):.2f}  "
                  f"join-w={jw:+.3f}",
                  flush=True)
        if step % args.eval_every == 0:
            net.eval()
            torch.save(net.state_dict(), out_dir / f"seed_s{step:04d}.pt")
            battery(net, args.eval_games, arms, f"s{step}")
            net.train()

    if args.skip_gates:
        print("done (gates skipped).", flush=True)
        return

    print("\n=== full gates (same protocol as pop wave) ===", flush=True)
    from trpg.scenarios.monsters import onev1_viable_monsters
    mon_pool = onev1_viable_monsters(8.0)
    base = load_student(args.warm)
    for name, n in (("warm", base), ("seeded", net)):
        n.eval()
        std, per = standard_probe(n, games=args.gate_std_games)
        mon, _ = monster_opp_probe(n, mon_pool, games=args.gate_mon_games)
        worst = sorted(per.items(), key=lambda kv: kv[1])[:3]
        print(f"  {name:6s} std12={std:.1%}  vs-mon={mon:.1%}  "
              f"worst3={[(a, f'{v:.0%}') for a, v in worst]}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
