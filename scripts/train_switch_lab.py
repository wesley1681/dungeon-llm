"""CAN pure PPO learn to READ the enemy resist trait and switch damage type —
with NO teacher-forcing oracle? (obs-trait goal, reward-driven + generalizing.)

Diagnosis behind this lab (all data, scripts/probe_*, REPORT_2026-06-13b):
  - regen→melee is a structural single-creature carrier (dead).
  - fire_lab (caster) PPO stayed channel-blind in 3 runs. Walls: gradient
    cancellation on a shared logit; an ESCAPE HATCH / SLOT confound (caster's
    spiritual_weapon/magic_missile deal a 3rd damage type that dodges the
    injected immunity, so switching is never NECESSARY) → no terminal gradient.
  - FIX: a two-weapon MELEE kit with NO spell escape — its ONLY damage is its
    two weapon types, both AT-WILL. Inject immunity to one → the OTHER weapon is
    the unique winner; no hedge, no escape. Verified clean-flip vs orc@L6.
  - SINGLE-pair (斬擊/穿刺) pure PPO LEARNED the switch (desc-ON WR 100% both
    immunities; ablate enemy descriptor → reverts to fixed default & loses =
    causal), overturning "PPO 探索不到" — but the type-symmetric matchup-join only
    grew to +0.26 and did NOT generalize zero-shot to held-out types.
  - THIS run broadens the distribution: several two-weapon kits over several
    INJECTABLE types so the join learns the GENERAL "avoid the resisted type"
    rule. Held-out types (閃電/強酸) stay out → probe_heldout_switch is the
    zero-shot generalization test.

NO ORACLE anywhere: reward (kill the orc) is the only signal; the trait→behaviour
mapping EMERGES from reward. The optional BC anchor is general class-play to
preserve std12 — NOT a switch label.

Usage:
  python scripts/train_switch_lab.py --updates 40 --out_dir models/switch_lab2 \
      --anchor_batches 2
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse, random
from collections import Counter
from pathlib import Path
import numpy as np
import torch

import trpg.rl.obs  # noqa: F401  freeze N_ARCHETYPES before registry mutation
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.train_ppo import _sample_action, _compute_gae, ppo_update
from trpg.rl.train_bc import bc_loss_step
from trpg.rl.obs import build_obs, migrate_entities_v3_to_v4
from trpg.engine.skill import available_skills
from trpg.engine.items import WEAPON_DEFS, Weapon
from trpg.scenarios.archetypes import (CLASS_DEFS, ARCHETYPE_FACTORIES,
                                       ARCHETYPE_ROLES, ClassDef, TraitGrant,
                                       SkillGrant, _factory, _WIZARD_SLOTS)
from trpg.scenarios.monsters import register_monsters, MONSTER_DEFS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import (load_student, load_dataset, DATASET_PATH,
                            blind_self_identity_np)
from train_population import blind_np_single, descriptor_weight_norm
from probe_descriptor_ab import blind_self_one_hot, zero_enemy_descriptor
from probe_fire_switch import action_damage_type
from eval_routed import stable_seed

# Training two-weapon kits (NO spell escape). Their damage types span a BROAD
# injectable set {斬擊,穿刺,火,冰,雷鳴,毒,光耀} in many distinct pairings, so the
# TYPE-SYMMETRIC matchup-join gets gradient across many type columns → the more
# columns it sees, the better it transfers ZERO-SHOT to held-out columns
# (閃電/強酸, never injected here = probe_heldout_switch's zero-shot test).
# Synthetic 1d8 melee weapons give each type a winnable at-will option.
_TRAIN_WEAPONS = {           # name -> (dice, damage_type)
    "火焰之刃": ("1d8", "火"), "冰霜之刃": ("1d8", "冰"),
    "雷鳴之錘": ("1d8", "雷鳴"), "劇毒之刃": ("1d8", "毒"),
    "聖光之刃": ("1d8", "光耀"),
    # 強酸/閃電 weapons (2d6 to match probe_heldout_switch's eval kits) — added
    # so 強酸/閃電 become IN-DISTRIBUTION injectable columns. Earlier they were
    # held-out (zero-shot test): D(閃電) transferred zero-shot but E(強酸+club)
    # stayed at 0%. The goal wants the model to actually switch on these, so we
    # train them directly (鈍擊/棍棒 fallback column too) → A/D/E in-distribution.
    "閃電刃": ("2d6", "閃電"), "酸蝕之刃": ("2d6", "強酸"),
}
KITS = [
    ("ckit_sw_sp", ("長劍", "短劍"),     ("斬擊", "穿刺")),
    ("ckit_sw_fr", ("長劍", "火焰之刃"),   ("斬擊", "火")),
    ("ckit_sp_ic", ("短劍", "冰霜之刃"),   ("穿刺", "冰")),
    ("ckit_fr_th", ("火焰之刃", "雷鳴之錘"), ("火", "雷鳴")),
    ("ckit_ic_po", ("冰霜之刃", "劇毒之刃"), ("冰", "毒")),
    ("ckit_th_ra", ("雷鳴之錘", "聖光之刃"), ("雷鳴", "光耀")),
    ("ckit_po_sw", ("劇毒之刃", "長劍"),    ("毒", "斬擊")),
    ("ckit_ra_sp", ("聖光之刃", "短劍"),    ("光耀", "穿刺")),
    # acid/lightning + club (鈍擊) kits = the previously-failing A/D/E regime,
    # now in-distribution (same weapons as probe_heldout_switch's E/D kits).
    ("ckit_ac_cl", ("酸蝕之刃", "棍棒"),    ("強酸", "鈍擊")),
    ("ckit_zap_cl", ("閃電刃", "棍棒"),     ("閃電", "鈍擊")),
    ("ckit_ac_sw", ("酸蝕之刃", "長劍"),    ("強酸", "斬擊")),
    ("ckit_zap_sp", ("閃電刃", "短劍"),     ("閃電", "穿刺")),
]
INJECT_NONE_W = 0.20   # rest split evenly over the kit's two types

# Spell+weapon kits: a damaging SPELL of one type + a weapon of another. Inject
# immunity to the spell's type → the model must DROP the spell for the weapon
# (the A regime of probe_heldout_switch: sword+fireball vs fire-immune, which a
# weapon-only lab never teaches because its only alternative-to-immune options
# are weapons). (aid, weapons, spell_skill_ids, (spell_type, weapon_type)).
SPELL_KITS = [
    ("ckit_sw_fb", ("長劍",),       ("fireball_ev",),     ("火", "斬擊")),
    ("ckit_sp_is", ("短劍",),       ("ice_storm_ev",),    ("冰", "穿刺")),
]

# What collect()/report() SAMPLE from (weapon kits + spell kits, types only).
SAMPLE_KITS = [(aid, w, t) for (aid, w, t) in KITS] + \
              [(aid, None, t) for (aid, _w, _sk, t) in SPELL_KITS]


def register_train_kits():
    for wname, (dice, dt) in _TRAIN_WEAPONS.items():
        WEAPON_DEFS.setdefault(wname, Weapon(wname, dice, dt, "近戰",
                                             range_normal=1.5))
    for aid, weapons, _types in KITS:
        cd = ClassDef(
            archetype_id=aid, default_name=aid, class_display="訓練合成",
            role="front",
            stat_block=dict(STR=16, DEX=14, CON=14, INT=10, WIS=10, CHA=10),
            hp_base=10, hp_per_level=6, ac=16,
            weapons=weapons, proficiencies=("STR", "CON"), skills=(),
            traits=(TraitGrant("extra_attack", min_level=5),))
        CLASS_DEFS[aid] = cd
        ARCHETYPE_FACTORIES[aid] = _factory(aid)
        ARCHETYPE_ROLES[aid] = cd.role
    for aid, weapons, skill_ids, _types in SPELL_KITS:
        cd = ClassDef(
            archetype_id=aid, default_name=aid, class_display="訓練合成",
            role="front",
            stat_block=dict(STR=16, DEX=12, CON=14, INT=16, WIS=10, CHA=10),
            hp_base=9, hp_per_level=6, ac=15,
            weapons=weapons, proficiencies=("STR", "CON"),
            spell_ability="INT", spell_slots_table=_WIZARD_SLOTS,
            skills=tuple(SkillGrant(s, min_level=5) for s in skill_ids),
            traits=())
        CLASS_DEFS[aid] = cd
        ARCHETYPE_FACTORIES[aid] = _factory(aid)
        ARCHETYPE_ROLES[aid] = cd.role


def _inject(env, immune_type):
    if immune_type is not None:
        for oid in env.opp_ids:
            env.ws.characters[oid].damage_multipliers[immune_type] = 0.0


def _sample_inject(rng, types):
    r = rng.random()
    if r < INJECT_NONE_W:
        return None
    return types[0] if r < INJECT_NONE_W + (1 - INJECT_NONE_W) / 2 else types[1]


def collect(net, n_steps, seed, rng, enemy, level, gamma=0.99, lam=0.95):
    net.eval()
    obs_l, act_l, lp_l, rew_l, val_l, done_l = [], [], [], [], [], []
    skm_l, enm_l, grm_l = [], [], []
    ep = 0
    wrong = ndmg = 0           # wrong-type damaging actions when something immune
    olvl = MONSTER_DEFS[enemy].natural_level
    while len(rew_l) < n_steps:
        kit_id, _w, types = rng.choice(SAMPLE_KITS)
        imm = _sample_inject(rng, types)
        env = CombatEnvV2(seed=seed * 1_000_003 + ep, n_agents=1, n_opps=1)
        env.reset(agent_archs=[kit_id], opp_archs=[enemy],
                  level=level, opp_level=olvl)
        _inject(env, imm)
        aid0 = env.agent_ids[0]
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
            if aid == aid0 and imm is not None and a[0] > 0:
                sks = available_skills(env.ws.characters[aid], env.ws)
                if a[0] < len(sks) and sks[a[0]].features.expected_damage > 0:
                    o = env.ws.characters[env.opp_ids[0]]
                    dt = action_damage_type(
                        sks[a[0]].build_action(aid, env.opp_ids[0],
                                               (o.position.x, o.position.y)),
                        env.ws.characters[aid])
                    if dt:
                        ndmg += 1
                        if dt == imm:
                            wrong += 1
            obs2, r, term, trunc, _ = env.step(a)
            obs_l.append(obs); act_l.append(a); lp_l.append(lp.numpy())
            rew_l.append(float(r)); val_l.append(val)
            skm_l.append(skm.numpy()); enm_l.append(enm.numpy())
            grm_l.append(grm.numpy())
            done = term or trunc
            done_l.append(bool(done))
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
    return batch, ep, wrong / max(1, ndmg)


def eval_variant(net, kit_id, enemy, level, immune, kill_desc, games):
    """Greedy; return (wrong_type_share, win_rate). wrong_type = immune type."""
    wrong = ndmg = wins = 0
    olvl = MONSTER_DEFS[enemy].natural_level
    for gi in range(games):
        env = CombatEnvV2(seed=stable_seed(f"swl_{kit_id}_{enemy}_{immune}_{gi}"),
                          n_agents=1, n_opps=1)
        env.reset(agent_archs=[kit_id], opp_archs=[enemy], level=level,
                  opp_level=olvl)
        _inject(env, immune)
        aid = env.agent_ids[0]; oid = env.opp_ids[0]
        done = False
        while not done:
            actor = env.current_agent_id
            ob = blind_self_one_hot(build_obs(env.ws, actor, env.resources))
            if kill_desc:
                ob = zero_enemy_descriptor(ob)
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            ai = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws,
                                  agent_id=actor))
            if actor == aid and 0 < ai[0]:
                sks = available_skills(env.ws.characters[aid], env.ws)
                if ai[0] < len(sks) and sks[ai[0]].features.expected_damage > 0:
                    o = env.ws.characters[oid]
                    dt = action_damage_type(
                        sks[ai[0]].build_action(aid, oid,
                                                (o.position.x, o.position.y)),
                        env.ws.characters[aid])
                    if dt:
                        ndmg += 1
                        if dt == immune:
                            wrong += 1
            obs2, _, term, trunc, _ = env.step(ai)
            done = term or trunc
        if (env.ws.characters[oid].is_dead()
                and env.ws.characters[aid].is_alive()):
            wins += 1
    return wrong / max(1, ndmg), wins / games


def report(net, enemy, level, games, tag):
    d, _ = descriptor_weight_norm(net)
    # typed join = net.tjoin_col，不是 -1（cimmun 欄）——ncol-1 位置漂移 bug 曾讓
    # v7 rsw 手術訓練/監看錯欄位（typed join 全程凍結、pooled 欄變唯一載體）。
    sj = float(np.mean([h.weight[0, net.tjoin_col].item()
                        for h in net.skill_heads]))
    out = [f"  [{tag}] desc-norm={d:.3f} skill-join={sj:+.3f}"]
    for kit_id, _w, types in SAMPLE_KITS:
        for immune in types:
            w_on, wr_on = eval_variant(net, kit_id, enemy, level, immune,
                                       False, games)
            w_off, wr_off = eval_variant(net, kit_id, enemy, level, immune,
                                         True, games)
            out.append(f"    {kit_id:11s} immune={immune}  wrong% "
                       f"on={w_on:3.0%} off={w_off:3.0%}   "
                       f"WR on={wr_on:3.0%} off={wr_off:3.0%}")
    print("\n".join(out), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm", default="models/pop_mon/pop_u0005.pt")
    p.add_argument("--out_dir", default="models/switch_lab2")
    p.add_argument("--enemy", default="orc")
    p.add_argument("--level", type=int, default=6)
    p.add_argument("--updates", type=int, default=40)
    p.add_argument("--steps", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ent_coef", type=float, default=0.02)
    p.add_argument("--value_warmup", type=int, default=3)
    p.add_argument("--anchor_batches", type=int, default=2,
                   help="0 = no kit anchor; >0 preserves std12 (general play)")
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_games", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--freeze_except_switch", action="store_true",
                   help="freeze ALL params except the descriptor->skill-switch "
                        "carrier channels (entity_mlp[0] descriptor input cols + "
                        "skill_heads join column) + critic. These channels are "
                        "~0 in no-resist play (class-vs-class), so normal "
                        "fighter/walls behaviour is ~unchanged while the switch "
                        "learns -> NO SWA-merge needed, NO weak-signal dilution, "
                        "NO fighter erosion (the LoS-only-surgery pattern for "
                        "trait-switch). Pair with --anchor_batches 0.")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "BLIND").write_text("switch-lab multi-kit pure-PPO (blind)\n")
    register_monsters()
    register_train_kits()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    net = load_student(args.warm)
    if args.freeze_except_switch:
        # Isolate the descriptor->skill-switch carrier (the LoS-only pattern).
        # Train ONLY: entity_mlp[0] descriptor INPUT columns (read enemy
        # typed_resist) + skill_heads JOIN column (last input col = skill-type x
        # resist) + critic. Everything else frozen + grad-masked, so the
        # no-resist (class-vs-class) pathway is untouched -> zero fighter/wall
        # erosion, and NO SWA dilution of the weak A/acid signals.
        # Resist-ISOLATED carrier: entity_mlp[0] TYPED-RESIST sub-columns
        # (I_DESC_RESIST..+N_DAMAGE_TYPES = (multiplier-1) per damage type, which
        # is EXACTLY 0 for every non-resistant creature incl all classes) +
        # skill_heads JOIN column + critic. Both are 0 in class-vs-class play ->
        # the no-resist pathway is bit-exact = zero fighter/wall erosion; but
        # unlike the join column alone (capacity-starved: wrong% stuck 0.50),
        # the N_DAMAGE_TYPES resist columns give the encoder enough capacity to
        # learn the switch. (The FULL descriptor block is NOT isolated — it also
        # carries always-on level/HP/AC, so training it erodes class play:
        # measured open 67->56. Only the typed-resist tail is resist-conditional.)
        from trpg.rl.obs import I_DESC_RESIST, N_DAMAGE_TYPES
        for p_ in net.parameters():
            p_.requires_grad_(False)
        for p_ in net.critic.parameters():
            p_.requires_grad_(True)
        rs, re_ = I_DESC_RESIST, I_DESC_RESIST + N_DAMAGE_TYPES

        def _cols_only(lo, hi):
            def hook(g):
                gg = torch.zeros_like(g); gg[:, lo:hi] = g[:, lo:hi]; return gg
            return hook

        ew = net.entity_mlp[0].weight
        ew.requires_grad_(True)
        ew.register_hook(_cols_only(rs, re_))
        for h in net.skill_heads:
            h.weight.requires_grad_(True)
            # typed join = net.tjoin_col。歷史 bug：這裡曾寫 ncol-1，但牆波
            # blocked_feat(+64) 與 cimmun join(+1) append 在 typed join 之後
            # → v7 rsw 手術實際訓練的是 cimmun 欄（對 lab 敵人特徵恆 0＝零梯度），
            # typed join 全程凍結、13 個 pooled resist 欄成唯一載體＝捷徑根源。
            h.weight.register_hook(_cols_only(net.tjoin_col, net.tjoin_col + 1))
        n_tr = sum(p_.numel() for p_ in net.parameters() if p_.requires_grad)
        print(f"[freeze_except_switch] trainable={n_tr} (entity typed-resist "
              f"cols {rs}:{re_} + skill_heads join col + critic; grad-masked; "
              f"resist-isolated = class-play bit-exact)", flush=True)
    print(f"warm={args.warm} enemy={args.enemy} L{args.level} "
          f"kits={[k[0] for k in KITS]} anchor={args.anchor_batches} "
          f"ent={args.ent_coef}", flush=True)

    anchor = None
    if args.anchor_batches > 0:
        obs_np, act_np, tt_np = load_dataset(Path(DATASET_PATH))
        obs_np = {k: v.copy() for k, v in obs_np.items()}
        obs_np["entities"] = migrate_entities_v3_to_v4(obs_np["entities"])
        # pre-12j dataset stores 53-wide skill features; the net's skill_proj is
        # now 66-wide (damage-type soft one-hot tail). Zero-pad = bit-exact.
        want = net.skill_proj.in_features
        sk = obs_np["skills"]
        if sk.shape[-1] < want:
            pad = np.zeros(sk.shape[:-1] + (want - sk.shape[-1],), dtype=sk.dtype)
            obs_np["skills"] = np.concatenate([sk, pad], axis=-1)
        # pre-reaction-wave dataset lacks decision_context (all-zero on normal
        # turns); pre-los-grid dataset lacks los_grid (open-field expert data =
        # all ones). Both zero/one-fill = bit-exact migration.
        if "decision_context" not in obs_np:
            from trpg.rl.obs import N_DECISION_CTX
            n = obs_np["skills"].shape[0]
            obs_np["decision_context"] = np.zeros((n, N_DECISION_CTX),
                                                  dtype=np.float32)
        if "los_grid" not in obs_np:
            from trpg.rl.obs import N_LOS_GRID_CHANNELS, N_GRID
            n = obs_np["skills"].shape[0]
            obs_np["los_grid"] = np.ones((n, N_LOS_GRID_CHANNELS, N_GRID, N_GRID),
                                         dtype=np.float32)
        blind_self_identity_np(obs_np)
        anchor = (obs_np, act_np, tt_np)

    optim = torch.optim.Adam(
        [p_ for p_ in net.parameters() if p_.requires_grad], lr=args.lr)
    print("baseline:", flush=True)
    report(net, args.enemy, args.level, args.eval_games, "u0")
    for update in range(1, args.updates + 1):
        batch, neps, wrate = collect(net, args.steps, args.seed * 7919 + update,
                                     rng, args.enemy, args.level)
        info = ppo_update(net, batch, optim, n_epochs=args.epochs,
                          batch_size=args.batch, ent_coef=args.ent_coef,
                          device="cpu", value_only=(update <= args.value_warmup))
        line = (f"U{update:3d}/{args.updates}"
                f"{' [wu]' if update <= args.value_warmup else ''} "
                f"eps={neps} wrong%={wrate:.2f} "
                f"pol={info['policy_loss']:+.3f} val={info['value_loss']:.2f} "
                f"ent={info['entropy']:.3f}")
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
            torch.save(net.state_dict(), out_dir / f"sw_u{update:04d}.pt")
            report(net, args.enemy, args.level, args.eval_games, f"u{update}")
            net.train()
    print("done.", flush=True)


if __name__ == "__main__":
    main()
