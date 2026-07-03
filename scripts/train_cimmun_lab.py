"""CAN pure PPO learn to READ the enemy CONDITION-IMMUNITY (obs v6) and stop
wasting a control ability on an immune target — with NO oracle? The
condition-immunity analog of train_switch_lab.py (which proved the SAME for
damage-type resist). This is the capacity experiment that answers "can the flat
model handle the immunity mechanic, or do we need a new architecture?".

Lab design (mirrors switch_lab's confound analysis, adapted):
  - Clean GISH kit: a melee weapon (長劍) + hold_person ONLY — no other spell,
    no escape hatch. Its one control option is hold_person (applies paralyzed
    via a WIS save).
  - Fixed melee-able enemy (orc). INJECT / don't-inject paralyzed-immunity onto
    it (char.condition_immunities) → the ONLY toggled variable, and the v6
    cimmun descriptor flips with it. Controlled A/B, same as switch_lab injects
    a damage immunity.
  - Why this is a STRICTER instrument than switch (a CONDITIONAL discrimination,
    not an unconditional one): hold_person is genuinely VALUABLE when it lands
    (paralyzed target → melee AUTO-CRIT + can't act, verified combat.py:1234) →
    the model WANTS to cast it; but on an immune target it is a wasted turn →
    slower kill → lose. So the model cannot pass by "never casting" — it must
    READ the immunity to condition its behaviour (cast when not-immune, skip
    when immune).
  - freeze-except-cimmun: train ONLY the entity_mlp cimmun INPUT columns
    (I_DESC_CIMMUN..) + the skill/entity head cimmun-JOIN column (last col) +
    critic. All are exactly 0 in no-immunity play → class-vs-class behaviour is
    bit-exact = zero erosion (the isolated-surgery pattern from switch_lab).

NO ORACLE: reward (kill the orc) is the only signal; the immunity→behaviour map
EMERGES from reward. Optional BC anchor is general class-play to preserve std12.

Read-out (report):
  ctrl%  = share of the agent's own action-turns that CAST hold_person.
           immune-enemy ctrl% should DROP (learned to skip); non-immune should
           stay HIGH (still valuable). The discrimination = non-immune − immune.
  WR     = win rate; immune-enemy WR should RISE as it stops wasting turns.
  desc-off = zero the enemy descriptor → discrimination must collapse (causal).

Usage:
  python scripts/train_cimmun_lab.py --updates 40 --out_dir models/cimmun_lab \
      --freeze_except_cimmun --anchor_batches 0
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

import trpg.rl.obs  # noqa: F401  freeze N_ARCHETYPES before registry mutation
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.train_ppo import _sample_action, _compute_gae, ppo_update
from trpg.rl.train_bc import bc_loss_step
from trpg.rl.obs import (build_obs, migrate_entities_v3_to_v4, I_DESC_CIMMUN,
                        N_V6_CIMMUN)
from trpg.engine.skill import available_skills, STATUS_SLOTS
from trpg.scenarios.archetypes import (CLASS_DEFS, ARCHETYPE_FACTORIES,
                                       ARCHETYPE_ROLES, ClassDef, TraitGrant,
                                       SkillGrant, _factory, _WIZARD_SLOTS)
from trpg.scenarios.monsters import register_monsters, MONSTER_DEFS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import (load_student, load_dataset, DATASET_PATH,
                            blind_self_identity_np)
from train_population import blind_np_single, descriptor_weight_norm
from probe_descriptor_ab import blind_self_one_hot, zero_enemy_descriptor
from eval_routed import stable_seed

CTRL_SKILL = "hold_person"          # the control option (applies paralyzed)
CTRL_COND = "paralyzed"
KIT_ID = "ckit_hp"                  # gish: 長劍 + hold_person only
INJECT_IMMUNE_W = 0.5               # half the episodes: enemy immune to paralyze


def register_cimmun_kit():
    cd = ClassDef(
        archetype_id=KIT_ID, default_name=KIT_ID, class_display="訓練合成",
        role="front",
        # High WIS → hold_person save DC bites a low-WIS orc often, so landing
        # it (non-immune) is genuinely decisive = strong value signal.
        stat_block=dict(STR=16, DEX=12, CON=14, INT=10, WIS=17, CHA=10),
        hp_base=10, hp_per_level=6, ac=16,
        weapons=("長劍",), proficiencies=("STR", "CON"),
        spell_ability="WIS", spell_slots_table=_WIZARD_SLOTS,
        skills=(SkillGrant(CTRL_SKILL, min_level=3),),
        traits=(TraitGrant("extra_attack", min_level=5),))
    CLASS_DEFS[KIT_ID] = cd
    ARCHETYPE_FACTORIES[KIT_ID] = _factory(KIT_ID)
    ARCHETYPE_ROLES[KIT_ID] = cd.role


def _inject_immune(env, immune: bool):
    if immune:
        for oid in env.opp_ids:
            ci = env.ws.characters[oid].condition_immunities
            if CTRL_COND not in ci:
                ci.append(CTRL_COND)


def _is_ctrl(skill) -> bool:
    """True if the skill applies the control condition (paralyzed)."""
    pi = STATUS_SLOTS.index(CTRL_COND)
    ap = getattr(skill.features, "applies_status", ())
    return len(ap) > pi and bool(ap[pi])


def collect(net, n_steps, seed, rng, enemy, level, gamma=0.99, lam=0.95):
    net.eval()
    obs_l, act_l, lp_l, rew_l, val_l, done_l = [], [], [], [], [], []
    skm_l, enm_l, grm_l = [], [], []
    ep = 0
    olvl = MONSTER_DEFS[enemy].natural_level
    while len(rew_l) < n_steps:
        immune = rng.random() < INJECT_IMMUNE_W
        env = CombatEnvV2(seed=seed * 1_000_003 + ep, n_agents=1, n_opps=1)
        env.reset(agent_archs=[KIT_ID], opp_archs=[enemy],
                  level=level, opp_level=olvl)
        _inject_immune(env, immune)
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
    return batch, ep


def eval_variant(net, enemy, level, immune, kill_desc, games):
    """Greedy. Returns (ctrl_share, win_rate). ctrl_share = fraction of the
    agent's own action-turns that cast the control skill (hold_person)."""
    ctrl = turns = wins = 0
    olvl = MONSTER_DEFS[enemy].natural_level
    for gi in range(games):
        env = CombatEnvV2(seed=stable_seed(f"cil_{enemy}_{immune}_{gi}"),
                          n_agents=1, n_opps=1)
        env.reset(agent_archs=[KIT_ID], opp_archs=[enemy], level=level,
                  opp_level=olvl)
        _inject_immune(env, immune)
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
                if ai[0] < len(sks):
                    turns += 1
                    if _is_ctrl(sks[ai[0]]):
                        ctrl += 1
            obs2, _, term, trunc, _ = env.step(ai)
            done = term or trunc
        if (env.ws.characters[oid].is_dead()
                and env.ws.characters[aid].is_alive()):
            wins += 1
    return ctrl / max(1, turns), wins / games


def report(net, enemy, level, games, tag):
    d, _ = descriptor_weight_norm(net)
    # 這裡的 -1 是「故意」的：cimmun join 就是最後一欄（本 lab 的訓練標的）。
    # typed join 在 net.tjoin_col——別把這兩欄搞混（ncol-1 漂移 bug 家族）。
    sj = float(np.mean([h.weight[0, -1].item() for h in net.skill_heads]))
    ej = float(np.mean([h.weight[0, -1].item() for h in net.entity_heads]))
    ci_on, wr_ci = eval_variant(net, enemy, level, True, False, games)   # immune
    ni_on, wr_ni = eval_variant(net, enemy, level, False, False, games)  # not
    ci_off, _ = eval_variant(net, enemy, level, True, True, games)       # immune desc-off
    print(f"  [{tag}] desc-norm={d:.3f} skill-join={sj:+.3f} ent-join={ej:+.3f}",
          flush=True)
    print(f"    IMMUNE     ctrl% on={ci_on:4.0%} desc-off={ci_off:4.0%}   WR={wr_ci:4.0%}",
          flush=True)
    print(f"    NOT-immune ctrl% on={ni_on:4.0%}                    WR={wr_ni:4.0%}",
          flush=True)
    print(f"    >> discrimination (not−immune ctrl%) = {(ni_on-ci_on)*100:+.0f}pp "
          f"| desc-off collapse: {(ci_off-ci_on)*100:+.0f}pp", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm", default="models/unified/uni_v9.pt")
    p.add_argument("--out_dir", default="models/cimmun_lab")
    p.add_argument("--enemy", default="orc")
    p.add_argument("--level", type=int, default=6)
    p.add_argument("--updates", type=int, default=40)
    p.add_argument("--steps", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ent_coef", type=float, default=0.02)
    p.add_argument("--value_warmup", type=int, default=3)
    p.add_argument("--anchor_batches", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_games", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--freeze_except_cimmun", action="store_true",
                   help="freeze ALL except the cimmun carrier: entity_mlp cimmun "
                        "INPUT cols + skill/entity head cimmun-join col + critic. "
                        "0 in no-immunity play → class-play bit-exact, zero erosion.")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "BLIND").write_text("cimmun-lab pure-PPO (blind)\n")
    register_monsters()
    register_cimmun_kit()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    net = load_student(args.warm)

    if args.freeze_except_cimmun:
        # Isolate the cimmun carrier (the LoS-only / switch-lab pattern):
        #   entity_mlp[0] cimmun INPUT columns (read enemy condition-immunity)
        #   + skill/entity head cimmun-JOIN column (LAST input col)
        #   + critic.
        # All are exactly 0 in no-immunity play → the class-vs-class pathway is
        # bit-exact = zero fighter/wall erosion.
        cs, ce = I_DESC_CIMMUN, I_DESC_CIMMUN + N_V6_CIMMUN
        for p_ in net.parameters():
            p_.requires_grad_(False)
        for p_ in net.critic.parameters():
            p_.requires_grad_(True)

        def _cols_only(lo, hi):
            def hook(g):
                gg = torch.zeros_like(g); gg[:, lo:hi] = g[:, lo:hi]; return gg
            return hook

        ew = net.entity_mlp[0].weight
        ew.requires_grad_(True)
        ew.register_hook(_cols_only(cs, ce))
        for h in list(net.skill_heads) + list(net.entity_heads):
            h.weight.requires_grad_(True)
            nc = h.weight.shape[1]
            h.weight.register_hook(_cols_only(nc - 1, nc))   # cimmun col = last
        n_tr = sum(p_.numel() for p_ in net.parameters() if p_.requires_grad)
        print(f"[freeze_except_cimmun] trainable={n_tr} (entity_mlp cimmun cols "
              f"{cs}:{ce} + skill/entity head join col + critic; grad-masked; "
              f"class-play bit-exact)", flush=True)

    optim = torch.optim.Adam(
        [p_ for p_ in net.parameters() if p_.requires_grad], lr=args.lr)
    print(f"warm={args.warm} enemy={args.enemy} L{args.level} kit={KIT_ID} "
          f"ctrl={CTRL_SKILL} inject_immune_w={INJECT_IMMUNE_W}", flush=True)
    print("baseline:", flush=True)
    report(net, args.enemy, args.level, args.eval_games, "u0")

    for update in range(1, args.updates + 1):
        batch, neps = collect(net, args.steps, args.seed * 7919 + update,
                              rng, args.enemy, args.level)
        info = ppo_update(net, batch, optim, n_epochs=args.epochs,
                          batch_size=args.batch, ent_coef=args.ent_coef,
                          device="cpu", value_only=(update <= args.value_warmup))
        print(f"U{update:3d}/{args.updates}"
              f"{' [wu]' if update <= args.value_warmup else ''} eps={neps} "
              f"pol={info['policy_loss']:+.3f} val={info['value_loss']:.2f} "
              f"ent={info['entropy']:.3f}", flush=True)
        if update % args.eval_every == 0:
            net.eval()
            torch.save(net.state_dict(), out_dir / f"ci_u{update:04d}.pt")
            report(net, args.enemy, args.level, args.eval_games, f"u{update}")
            net.train()
    print("done.", flush=True)


if __name__ == "__main__":
    main()
