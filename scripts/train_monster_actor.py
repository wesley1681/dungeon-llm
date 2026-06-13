"""Monster-ACTOR population PPO (扮演怪物 wave).

Baseline (scripts/eval_monster_actor.py, seed_dtype1/seed_s1600):
  1v1 monster seat  mean Δ −10.2pp vs GenericMonsterPolicy in the same seat
                    (melee seats passive: move/dodge ~0.65/turn, weapon
                    0.03-0.13/turn; ranged/caster kits transfer fine)
  1v3 BOSS seat     mean Δ −27.1pp (same passivity)… EXCEPT archmage
                    +25pp — caster play transfers and already BEATS the
                    nearest-target script. Headroom is real.
Logit probe: adjacent-ogre weapon logit sits only ~2-3.4 below move/dodge —
a soft preference gap (≈3% sample rate), not hard suppression → pure
reward-driven PPO can close it. NO scripted monster demos anywhere in this
wave: the user's generalization bar forbids learn-by-imitating-my-script.
The script is the YARDSTICK (eval arm), never the teacher.

Recipe (extends train_population.py):
  - warm models/seed_dtype1/seed_s1600.pt (current champion)
  - per-episode seat sampling:
      p_boss      1v3 BOSS seat — train boss at natural_level vs 3 random
                  standard classes at the PARTY3 equiv-bracket level
                  (fraction-weighted floor/ceil)
      p_mon_agent 1v1 monster seat (fair pairing, natural vs round(equiv))
      p_synth     synthesized identity (chimera robustness mass)
      rest        standard class 1v1 (std12 stability mass)
      non-boss opponents are monsters with prob p_mon_opp.
  - HELD-OUT quarantine (the zero-shot instruments — this wave's rollouts
    and fresh anchor contain them on NEITHER side; historical opponent-side
    exposure in the warm net predates the wave and is disclosed):
      1v1: ghoul (rider), gargoyle (phys-resist), manticore (ranged volley)
      boss: hill_giant (chassis cousin ettin stays in-train)
  - anchors per update (1:1): 12j switch_demos (oracle resist-switching,
    never relabeled) + FRESH greedy self-anchor collected from the warm net
    on CLASS seats only (preserve class play; monster seats are exactly
    what PPO is supposed to change, so they are NOT anchored).
  - telemetry: monster-seat attack-rate per rollout (the passivity
    dissolving in numbers), seat mix, grad-cos, std12/vs-mon/seat probes.

Usage:
  python scripts/train_monster_actor.py --updates 40 --out_dir models/mon_actor1
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch

from trpg.scenarios.monsters import (MONSTER_DEFS, EQUIV_LEVEL_1V1,
                                     PARTY3_EQUIV_LEVEL, register_monsters,
                                     onev1_viable_monsters,
                                     party3_boss_monsters)

register_monsters()

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.train_ppo import _sample_action, _compute_gae, ppo_update
from trpg.rl.train_bc import bc_loss_step
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.obs import build_obs, migrate_entities_v3_to_v4
from trpg.engine.skill import available_skills
from trpg.engine.combat import _ally_adjacent_to
from trpg.scenarios.archetypes import STANDARD_ARCHETYPES

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import (blind_np_single, standard_probe,
                              monster_opp_probe, grad_cosines)
from synth_identity import fresh_synth, STANDARD_IDS
from eval_monster_actor import run_episode, PARTY

HELD_OUT_1V1 = ("ghoul", "gargoyle", "manticore")
HELD_OUT_BOSS = ("hill_giant",)

# Pack-tactics verification seats (obs v5 passive-trait channel). The model
# controls a WOLF PACK vs a party with an AoE caster (evocation fireball);
# pack_tactics is RANDOMIZED on/off per episode. The only way to know whether to
# flank (gang one target for advantage, eating the fireball) vs spread (dodge the
# fireball, no advantage to gain) is to READ the v5 trait channel — a
# channel-blind policy cannot do both, so the universal reward forces
# channel-conditioned flanking (and the trait-off ablation then reverts it =
# causal). Train on wolf packs ONLY; kobold / dire_wolf packs are held out as
# the zero-shot "trait→behaviour, not wolf-memorisation" generalization probe.
PACK_AGENT = "wolf"
PACK_SIZES = (3, 4)   # ≤4 so self + 3 ally slots stay visible (N_ALLY_SLOTS=3)
PACK_OPPS = ("evocation", "battle_master")   # AoE caster (burning_hands/fireball
                                             # punishes clumping) + a bruiser
HELD_OUT_PACK = ("kobold", "dire_wolf")

# During-training seat probe (fixed seeds, model arm; script reference run
# once at startup in the same process). Held-out seats included — their
# zero-shot trajectory is the generalization signal forming (or not).
SEAT_1V1 = ("ogre", "wolf", "basilisk", "mage_npc", "ghoul", "manticore")
SEAT_1V1_OPPS = ("battle_master", "assassin", "life", "evocation")
SEAT_BOSS = (("ettin", 3), ("troll", 4), ("archmage", 5), ("hill_giant", 3))


def train_pools():
    t1 = [m for m in onev1_viable_monsters(8.0) if m not in HELD_OUT_1V1]
    tb = [m for m in party3_boss_monsters(8.0) if m not in HELD_OUT_BOSS]
    return t1, tb


def sample_seat(rng, t1, tb, p_boss, p_mon_agent, p_synth, p_mon_opp, p_pack, ep):
    """One episode spec: (kind, agent_archs, a_lvl, opps, o_lvl, tag, pack_on).
    agent_archs is a LIST (len>1 only for the pack seat); pack_on is None except
    on pack seats."""
    r = rng.random()
    if r < p_pack:                              # wolf pack vs AoE party
        n_w = rng.choice(PACK_SIZES)
        pack_on = rng.random() < 0.5
        opps = list(PACK_OPPS)
        o_lvl = rng.choice((2, 3))
        return ("pack", [PACK_AGENT] * n_w,
                MONSTER_DEFS[PACK_AGENT].natural_level, opps, o_lvl,
                "pack", pack_on)
    r -= p_pack
    if r < p_boss:
        mon = rng.choice(tb)
        eq = PARTY3_EQUIV_LEVEL[mon]
        lo = max(1, math.floor(eq))
        lvl = min(8, lo + (1 if rng.random() < (eq - lo) else 0))
        opps = rng.sample(STANDARD_IDS, 3)
        return ("boss", [mon], MONSTER_DEFS[mon].natural_level, opps, lvl,
                mon, None)
    if r < p_boss + p_mon_agent:
        agent, tag = rng.choice(t1), None
        a_lvl = MONSTER_DEFS[agent].natural_level
        tag = agent
    elif r < p_boss + p_mon_agent + p_synth:
        agent, tag = fresh_synth(rng, slot=ep), "synth"
        a_lvl = None
    else:
        agent = rng.choice(STANDARD_IDS); tag = agent
        a_lvl = None
    if rng.random() < p_mon_opp:
        opp = rng.choice(t1)
        o_lvl = MONSTER_DEFS[opp].natural_level
        if a_lvl is None:                      # class/synth vs monster: fair
            a_lvl = max(1, round(EQUIV_LEVEL_1V1[opp]))
    else:
        opp = rng.choice(STANDARD_IDS)
        if a_lvl is None:                      # class vs class: symmetric
            a_lvl = rng.randint(3, 8)
            o_lvl = a_lvl
        else:                                  # monster vs class: fair
            o_lvl = max(1, round(EQUIV_LEVEL_1V1[agent]))
    return ("1v1", [agent], a_lvl, [opp], o_lvl, tag, None)


def _track_pack_flank(env, aid, pack_on, pack_steps, pack_flank):
    """On a pack-wolf decision where it can melee its nearest enemy, record
    whether a packmate already flanks that target (= attacking now lands
    pack-tactics advantage). Split by pack_on so the on/off flank-rate gap shows
    the channel-conditioned behaviour forming in-train."""
    ch = env.ws.characters[aid]
    live = [o for o in env.opp_ids if env.ws.characters[o].is_alive()]
    if not live:
        return
    tgt = min(live, key=lambda o: ch.position.distance_to(
        env.ws.characters[o].position))
    t = env.ws.characters[tgt]
    if ch.position.distance_to(t.position) <= 1.5 + 1e-6:
        pack_steps[bool(pack_on)] += 1
        if _ally_adjacent_to(env.ws, ch, t):
            pack_flank[bool(pack_on)] += 1


def collect_rollout(net, n_steps, seed, rng, t1, tb, args):
    """Sequential mixed-seat rollout (1v1 + 1v3 boss), blinded obs.
    Returns (batch, tags, stats)."""
    net.eval()
    obs_l, act_l, lp_l, rew_l, val_l, done_l = [], [], [], [], [], []
    skm_l, enm_l, grm_l, tag_l = [], [], [], []
    ep = 0
    kind_eps = {"boss": 0, "1v1": 0, "pack": 0}
    mon_turns = mon_atk = 0
    pack_steps = {True: 0, False: 0}      # flank-eligible wolf decisions
    pack_flank = {True: 0, False: 0}      # of those, attacked an ally-flanked tgt
    while len(rew_l) < n_steps:
        kind, agent_archs, a_lvl, opps, o_lvl, tag, pack_on = sample_seat(
            rng, t1, tb, args.p_boss, args.p_mon_agent, args.p_synth,
            args.p_mon_opp, args.p_pack, ep)
        kind_eps[kind] += 1
        is_mon_seat = agent_archs[0] in MONSTER_DEFS
        env = CombatEnvV2(seed=seed * 1_000_003 + ep,
                          n_agents=len(agent_archs), n_opps=len(opps))
        env.reset(agent_archs=list(agent_archs), opp_archs=list(opps),
                  level=a_lvl, opp_level=o_lvl)
        if kind == "pack":
            for aid in env.agent_ids:         # override default wolf pack_tactics
                env.ws.characters[aid].pack_tactics = bool(pack_on)
        obs = blind_np_single(build_obs(env.ws, env.current_agent_id,
                                        env.resources))
        done = False
        while not done:
            aid = env.current_agent_id
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                action, lp, val, skm, enm, grm = _sample_action(
                    net, ot, env.resources, env.ws, aid)
            a = action.numpy().tolist()
            if is_mon_seat:
                mon_turns += 1
                if a[0] > 0:
                    sks = available_skills(env.ws.characters[aid], env.ws)
                    if (a[0] < len(sks)
                            and sks[a[0]].features.expected_damage > 0):
                        mon_atk += 1
            if kind == "pack":
                _track_pack_flank(env, aid, pack_on, pack_steps, pack_flank)
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
                            gamma=0.99, gae_lambda=0.95)
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
    stats = {"eps": ep, "boss_eps": kind_eps["boss"],
             "pack_eps": kind_eps["pack"],
             "mon_atk_rate": mon_atk / max(1, mon_turns),
             "mon_turns": mon_turns,
             "flank_on": pack_flank[True] / max(1, pack_steps[True]),
             "flank_off": pack_flank[False] / max(1, pack_steps[False])}
    return batch, tag_l, stats


def collect_self_anchor_classes(net, n_states, seed, rng, opp_pool,
                                p_mon=0.5):
    """Greedy self-distill anchor on CLASS seats only (the behavior to
    PRESERVE). Monster opponents come from the TRAIN pool — held-outs stay
    quarantined out of this wave's data entirely. Mirrors
    seed_switch_bc.collect_self_anchor (greedy labels, END pairs, TT=-1)."""
    net.eval()
    O, A, T = [], [], []
    ep = 0
    while len(A) < n_states:
        if rng.random() < p_mon and opp_pool:
            opp = rng.choice(opp_pool)
            eq = EQUIV_LEVEL_1V1[opp]
            a_lvl = int(min(8, max(3, round(eq)))) if math.isfinite(eq) else 8
            o_lvl = MONSTER_DEFS[opp].natural_level
        else:
            a_lvl = rng.randint(5, 8)
            opp = rng.choice(list(STANDARD_ARCHETYPES))
            o_lvl = a_lvl
        agent = rng.choice(list(STANDARD_ARCHETYPES))
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
            act = list(pick_action(el[0], s[0], e[0], g[0],
                                   ws=env.ws, agent_id=aid))
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
            print(f"  self-anchor: {len(A)}/{n_states} ({ep} eps)", flush=True)
    obs_np = {k: np.stack([o[k] for o in O]) for k in O[0]}
    return obs_np, np.array(A, np.int64), np.array(T, np.int64)


def seat_probe(net, games_1v1=3, games_boss=8):
    """Model-arm WR on the fixed probe seats (within-run trend metric)."""
    out = {}
    for mon in SEAT_1V1:
        cls_lvl = max(1, round(EQUIV_LEVEL_1V1[mon]))
        m_lvl = MONSTER_DEFS[mon].natural_level
        w = n = 0
        for opp in SEAT_1V1_OPPS:
            for k in range(games_1v1):
                key = f"seatp|{mon}|{opp}|{m_lvl}v{cls_lvl}|{k}"
                won, *_ = run_episode(mon, [opp], m_lvl, cls_lvl, key, net)
                w += int(won); n += 1
        out[mon] = w / n
    for mon, p_lvl in SEAT_BOSS:
        m_lvl = MONSTER_DEFS[mon].natural_level
        w = n = 0
        for k in range(games_boss):
            key = f"seatpb|{mon}|{p_lvl}|{k}"
            won, *_ = run_episode(mon, list(PARTY), m_lvl, p_lvl, key, net)
            w += int(won); n += 1
        out[f"boss:{mon}@L{p_lvl}"] = w / n
    return out


def fmt_seats(d):
    return "  ".join(f"{k}={v:.0%}" for k, v in d.items())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm", default="models/seed_dtype1/seed_s1600.pt")
    p.add_argument("--out_dir", default="models/mon_actor1")
    p.add_argument("--switch_demos", default="models/seed_dtype1/switch_demos.npz")
    p.add_argument("--self_anchor_in", default="",
                   help="reuse a saved self-anchor npz (else collect fresh)")
    p.add_argument("--self_states", type=int, default=8000)
    p.add_argument("--updates", type=int, default=40)
    p.add_argument("--steps", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--value_warmup", type=int, default=3)
    p.add_argument("--anchor_batches", type=int, default=4,
                   help="anchor minibatches per update, split 1:1 "
                        "switch:self (0 = none)")
    p.add_argument("--p_pack", type=float, default=0.0,
                   help="wolf-pack seat fraction (obs v5 pack_tactics "
                        "verification; 0 = legacy monster-actor run)")
    p.add_argument("--p_boss", type=float, default=0.20)
    p.add_argument("--p_mon_agent", type=float, default=0.20)
    p.add_argument("--p_synth", type=float, default=0.20)
    p.add_argument("--p_mon_opp", type=float, default=0.30)
    p.add_argument("--eval_every", type=int, default=4)
    p.add_argument("--std_games", type=int, default=1)
    p.add_argument("--mon_games", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "BLIND").write_text("monster-actor population PPO (blind)\n")
    t1, tb = train_pools()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    net = load_student(args.warm)
    print(f"warm={args.warm}", flush=True)
    print(f"train pools: 1v1={t1}\n  boss={tb}\n"
          f"held-out: 1v1={HELD_OUT_1V1} boss={HELD_OUT_BOSS} "
          f"pack={HELD_OUT_PACK}\n"
          f"mix: p_pack={args.p_pack} p_boss={args.p_boss} "
          f"p_mon_agent={args.p_mon_agent} p_synth={args.p_synth} "
          f"p_mon_opp={args.p_mon_opp}", flush=True)
    if args.p_pack > 0:
        print(f"pack seat: {PACK_SIZES}×{PACK_AGENT} vs {PACK_OPPS} "
              f"@L2-3, pack_tactics randomised on/off", flush=True)

    # ── anchors ──────────────────────────────────────────────────────────
    # Anchors saved at an older obs schema (switch_demos is v4-width entities)
    # must be migrated to the live width before the net consumes them — else
    # bc_loss_step hits a (B×101)·(105×64) shape mismatch. migrate_entities_v3_to_v4
    # chains v4→v5 (and is a no-op on already-current arrays).
    def _migrate_anchor(o):
        if "entities" in o:
            o["entities"] = migrate_entities_v3_to_v4(o["entities"])
        return o

    sw = np.load(args.switch_demos)
    sw_obs = _migrate_anchor({k[4:]: sw[k] for k in sw.files
                              if k.startswith("obs_")})
    sw_act, sw_tt = sw["actions"], sw["target_types"]
    print(f"switch anchor: {len(sw_act)} pairs", flush=True)
    if args.self_anchor_in:
        z = np.load(args.self_anchor_in)
        sa_obs = _migrate_anchor({k[4:]: z[k] for k in z.files
                                  if k.startswith("obs_")})
        sa_act, sa_tt = z["actions"], z["target_types"]
        print(f"self anchor (reused): {len(sa_act)} pairs", flush=True)
    else:
        print("collecting fresh self-anchor from warm net "
              f"({args.self_states} states, class seats, train-pool opps)…",
              flush=True)
        sa_obs, sa_act, sa_tt = collect_self_anchor_classes(
            net, args.self_states, args.seed * 104729 + 11, rng, t1)
        np.savez_compressed(out_dir / "self_anchor.npz",
                            actions=sa_act, target_types=sa_tt,
                            **{f"obs_{k}": v for k, v in sa_obs.items()})
        print(f"self anchor: {len(sa_act)} pairs (saved)", flush=True)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    def probes():
        std = std_per = mon = None
        if args.std_games > 0:
            std, std_per = standard_probe(net, games=args.std_games)
        if args.mon_games > 0:
            mon, _ = monster_opp_probe(net, t1, games=args.mon_games)
        return std, mon, seat_probe(net)

    print("\nbaseline probes (update 0):", flush=True)
    std0, mon0, seats0 = probes()
    print(f"  std12={std0 if std0 is None else f'{std0:.1%}'}  "
          f"vs-mon(train)={mon0 if mon0 is None else f'{mon0:.1%}'}",
          flush=True)
    print(f"  seats: {fmt_seats(seats0)}\n", flush=True)

    for update in range(1, args.updates + 1):
        batch, tags, st = collect_rollout(
            net, args.steps, seed=args.seed * 7919 + update, rng=rng,
            t1=t1, tb=tb, args=args)
        info = ppo_update(net, batch, optim, n_epochs=args.epochs,
                          batch_size=args.batch, ent_coef=args.ent_coef,
                          device="cpu",
                          value_only=(update <= args.value_warmup))
        line = (f"U{update:3d}/{args.updates}"
                f"{'[wu]' if update <= args.value_warmup else '    '} "
                f"eps={st['eps']}(boss {st['boss_eps']} pack {st['pack_eps']}) "
                f"mon-atk={st['mon_atk_rate']:.2f}@{st['mon_turns']} ")
        if args.p_pack > 0:
            # the causal signal: flank rate should DIVERGE (on ≫ off) as the
            # policy learns to read the v5 pack channel.
            line += f"flank(on/off)={st['flank_on']:.2f}/{st['flank_off']:.2f} "
        line += (f"pol={info['policy_loss']:+.3f} val={info['value_loss']:.3f} "
                 f"ent={info['entropy']:.2f}")
        if args.anchor_batches > 0 and update > args.value_warmup:
            accs = []
            for b in range(args.anchor_batches):
                src = (sw_obs, sw_act, sw_tt) if b % 2 == 0 else \
                      (sa_obs, sa_act, sa_tt)
                sel = np.random.randint(0, len(src[1]), size=args.batch)
                ob = {k: torch.from_numpy(v[sel]) for k, v in src[0].items()}
                _, acc = bc_loss_step(net, ob, torch.from_numpy(src[1][sel]),
                                      torch.from_numpy(src[2][sel]), optim)
                accs.append(acc.get("skill", float("nan")))
            line += f" anc={np.nanmean(accs):.2f}"
        print(line, flush=True)

        if update % args.eval_every == 0:
            net.eval()
            cos = grad_cosines(net, batch, tags)
            if cos:
                worst = min(cos, key=lambda x: x[2])
                print(f"  grad-cos mean={np.mean([c for _, _, c in cos]):+.3f} "
                      f"worst={worst[2]:+.3f}({worst[0]}~{worst[1]})",
                      flush=True)
            std, mon, seats = probes()
            snap = out_dir / f"ma_u{update:04d}.pt"
            torch.save(net.state_dict(), snap)
            sstd = "-" if std is None else f"{std:.1%}(b {std0:.1%})"
            smon = "-" if mon is None else f"{mon:.1%}(b {mon0:.1%})"
            print(f"  => std12 {sstd}  vs-mon {smon}  snap={snap.name}",
                  flush=True)
            print(f"  => seats: {fmt_seats(seats)}", flush=True)
            net.train()

    print("done.", flush=True)


if __name__ == "__main__":
    main()
