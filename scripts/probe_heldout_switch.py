"""舉一反三 probe: does the seeded switch GENERALIZE to held-out combinations?

Two synthetic kits (runtime-registered ClassDefs, all-zero one-hot — the
B-net's native input regime), each holding TWO damage types, vs an orc with an
injected immunity to the kit's DOMINANT type:

  A. ckit_sword_fire  長劍(斬擊) + fireball(火)   — held-out KIT, TRAINED type.
     No training identity pairs a sword with fireball (evocation/divination
     have no weapon; life/war have no fire). The rule "enemy fire-cell -> drop
     the fireball-looking row" was trained on caster kits only. Does it
     transfer to a novel kit context?
  B. ckit_zap_mm      閃電刃(閃電, 2d6+extra_attack) + magic_missile(力場)
     — held-out TYPE. 閃電 was never an injectable immunity in training
     (injection samples the agent's own types; no pool agent has 閃電; the
     12i --inject_types allowlist makes the boundary explicit). The
     STRUCTURE "weapon-immune -> switch to mm" WAS trained (arcane_
     trickster, 穿刺 column). Only the resist COLUMN is new.
  C. ckit_acid_mm     酸蝕之刃(強酸, 2d6+extra_attack) + magic_missile(力場)
     — second held-out type (強酸 appears in NO pool kit and NO injection),
     same structure as B: two fully-untouched columns must both flip for the
     zero-shot claim to hold.

Pre-12i (53-d skills, no dtype channel) both held-out arms FAILED — the obs
could not link a weapon to its resist column, and one-hot orthogonality means
per-type weights get no gradient for never-injected types. The 12i surgery
adds the skill-side dtype soft one-hot + a shared (dtype ⋅ typed-resist)
matchup join in the heads; the join's single weight is type-symmetric, so
THIS probe flipping green is the acceptance for genuine 舉一反三.

Controls: same kits vs a plain orc (dominant skill should stay dominant),
desc-off arms (immunity invisible -> no switch expected anywhere), and
--ablate_dtype (zero the skill-side dtype tail at eval: join + one-hot dead
-> switch must revert, proving the new channel CAUSES the behavior).

Usage: python scripts/probe_heldout_switch.py [--games 40] [--ablate_dtype]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse
import torch

import trpg.rl.obs  # noqa: F401  freeze N_ARCHETYPES before registry mutation
from trpg.engine.skill import SKILL_DTYPE_START
from trpg.engine.items import WEAPON_DEFS, Weapon
from trpg.scenarios.archetypes import (
    CLASS_DEFS, ARCHETYPE_FACTORIES, ARCHETYPE_ROLES,
    ClassDef, SkillGrant, TraitGrant, _factory, _WIZARD_SLOTS,
)
from trpg.scenarios.monsters import register_monsters, MONSTER_DEFS
from trpg.rl.env_v2 import CombatEnvV2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from seed_switch_bc import eval_switch, damaging_options, oracle_pick


HELDOUT_DEFS = [
    ClassDef(
        archetype_id="ckit_sword_fire", default_name="Sword-Fire Kit",
        class_display="合成試驗", role="front",
        stat_block=dict(STR=16, DEX=12, CON=14, INT=16, WIS=10, CHA=10),
        hp_base=9, hp_per_level=6, ac=15,
        weapons=("長劍",),
        proficiencies=("STR", "CON"),
        spell_ability="INT", spell_slots_table=_WIZARD_SLOTS,
        skills=(SkillGrant("fireball_ev", min_level=5),),
        traits=(),
    ),
    ClassDef(
        archetype_id="ckit_zap_mm", default_name="Zap-MM Kit",
        class_display="合成試驗", role="front",
        stat_block=dict(STR=16, DEX=12, CON=14, INT=16, WIS=10, CHA=10),
        hp_base=9, hp_per_level=6, ac=15,
        weapons=("閃電刃",),
        proficiencies=("STR", "CON"),
        spell_ability="INT", spell_slots_table=_WIZARD_SLOTS,
        skills=(SkillGrant("magic_missile"),),
        traits=(TraitGrant("extra_attack", min_level=5),),
    ),
    ClassDef(
        archetype_id="ckit_acid_mm", default_name="Acid-MM Kit",
        class_display="合成試驗", role="front",
        stat_block=dict(STR=16, DEX=12, CON=14, INT=16, WIS=10, CHA=10),
        hp_base=9, hp_per_level=6, ac=15,
        weapons=("酸蝕之刃",),
        proficiencies=("STR", "CON"),
        spell_ability="INT", spell_slots_table=_WIZARD_SLOTS,
        skills=(SkillGrant("magic_missile"),),
        traits=(TraitGrant("extra_attack", min_level=5),),
    ),
    # Two-WEAPON variants: the spell-less fallback removes the mm attractor
    # (a net that just prefers casting on synthetic kits would mask the
    # switch). Default play must pick one weapon row; reading the held-out
    # immunity column means moving to the OTHER weapon row. 鈍擊 is itself a
    # never-injected column, so these arms are doubly held-out.
    ClassDef(
        archetype_id="ckit_zap_club", default_name="Zap-Club Kit",
        class_display="合成試驗", role="front",
        stat_block=dict(STR=16, DEX=12, CON=14, INT=10, WIS=10, CHA=10),
        hp_base=10, hp_per_level=6, ac=16,
        weapons=("閃電刃", "棍棒"),
        proficiencies=("STR", "CON"),
        skills=(),
        traits=(TraitGrant("extra_attack", min_level=5),),
    ),
    ClassDef(
        archetype_id="ckit_acid_club", default_name="Acid-Club Kit",
        class_display="合成試驗", role="front",
        stat_block=dict(STR=16, DEX=12, CON=14, INT=10, WIS=10, CHA=10),
        hp_base=10, hp_per_level=6, ac=16,
        weapons=("酸蝕之刃", "棍棒"),
        proficiencies=("STR", "CON"),
        skills=(),
        traits=(TraitGrant("extra_attack", min_level=5),),
    ),
]


def register_heldout_kits():
    WEAPON_DEFS.setdefault(
        "閃電刃", Weapon("閃電刃", "2d6", "閃電", "近戰", range_normal=1.5))
    WEAPON_DEFS.setdefault(
        "酸蝕之刃", Weapon("酸蝕之刃", "2d6", "強酸", "近戰", range_normal=1.5))
    for cd in HELDOUT_DEFS:
        CLASS_DEFS[cd.archetype_id] = cd
        ARCHETYPE_FACTORIES[cd.archetype_id] = _factory(cd.archetype_id)
        ARCHETYPE_ROLES[cd.archetype_id] = cd.role


def preflight(level=6):
    """Verify the ORACLE direction on each kit before judging the net."""
    print("=== oracle preflight (engine ground truth) ===")
    cases = [
        ("ckit_sword_fire", None),
        ("ckit_sword_fire", ("火", 0.0)),
        ("ckit_zap_mm", None),
        ("ckit_zap_mm", ("閃電", 0.0)),
        ("ckit_acid_mm", None),
        ("ckit_acid_mm", ("強酸", 0.0)),
        ("ckit_zap_club", None),
        ("ckit_zap_club", ("閃電", 0.0)),
        ("ckit_acid_club", None),
        ("ckit_acid_club", ("強酸", 0.0)),
    ]
    for arch, inj in cases:
        env = CombatEnvV2(seed=23, n_agents=1, n_opps=1)
        env.reset(agent_archs=[arch], opp_archs=["orc"], level=level,
                  opp_level=MONSTER_DEFS["orc"].natural_level)
        if inj:
            for o in env.opp_ids:
                env.ws.characters[o].damage_multipliers[inj[0]] = inj[1]
        aid = env.agent_ids[0]; oid = env.opp_ids[0]
        opts = damaging_options(env.ws, aid, oid)
        pick = oracle_pick(opts, -1) if opts else None
        opts_s = ", ".join(f"{sk.skill_id}({dt} ev={ev:.1f})"
                           for _, sk, _, dt, ev in opts)
        print(f"  {arch:16s} inj={str(inj):14s} -> "
              f"{pick[1].skill_id if pick else '—'}\n      [{opts_s}]")
    print()


class ZeroDtypeNet(torch.nn.Module):
    """Eval wrapper: zero the skill-side dtype tail (one-hot AND the join
    both die — the join is computed from these columns in forward). If the
    seeded switch is driven by the new channel, it must revert under this."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, obs):
        o = dict(obs)
        sk = o["skills"].clone()
        sk[..., SKILL_DTYPE_START:] = 0.0
        o["skills"] = sk
        return self.net(o)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeded", default="models/seed_switch4/seed_s1600.pt")
    p.add_argument("--warm", default="models/pop_mon/pop_u0005.pt")
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--level", type=int, default=6)
    p.add_argument("--ablate_dtype", action="store_true",
                   help="also run the seeded net with the skill dtype tail "
                        "zeroed (channel-causality control)")
    p.add_argument("--skip_warm", action="store_true")
    args = p.parse_args()
    register_monsters()
    register_heldout_kits()
    preflight(args.level)

    onat = MONSTER_DEFS["orc"].natural_level
    arms = [
        ("A  sword_fire vs 火immune", "ckit_sword_fire", ("火", 0.0), "火"),
        ("A' sword_fire vs normal",   "ckit_sword_fire", None,        "火"),
        ("B  zap_mm    vs 閃電immune", "ckit_zap_mm",    ("閃電", 0.0), "閃電"),
        ("B' zap_mm    vs normal",    "ckit_zap_mm",     None,        "閃電"),
        ("C  acid_mm   vs 強酸immune", "ckit_acid_mm",   ("強酸", 0.0), "強酸"),
        ("C' acid_mm   vs normal",    "ckit_acid_mm",    None,        "強酸"),
        ("D  zap_club  vs 閃電immune", "ckit_zap_club",  ("閃電", 0.0), "閃電"),
        ("D' zap_club  vs normal",    "ckit_zap_club",   None,        "閃電"),
        ("E  acid_club vs 強酸immune", "ckit_acid_club", ("強酸", 0.0), "強酸"),
        ("E' acid_club vs normal",    "ckit_acid_club",  None,        "強酸"),
    ]
    runs = [] if args.skip_warm else [("WARM", args.warm, False)]
    runs.append(("SEEDED", args.seeded, False))
    if args.ablate_dtype:
        runs.append(("SEEDED dtype-ablated", args.seeded, True))
    for name, path, ablate in runs:
        net = load_student(path); net.eval()
        if ablate:
            net = ZeroDtypeNet(net); net.eval()
        print(f"=== {name} ({args.games} games/arm) ===")
        for label, arch, inj, watch in arms:
            sh_on, wr_on = eval_switch(net, arch, "orc", args.level, onat,
                                       inj, False, args.games,
                                       f"ho_{name}_{label}_on")
            sh_off, wr_off = eval_switch(net, arch, "orc", args.level, onat,
                                         inj, True, args.games,
                                         f"ho_{name}_{label}_off")
            f_on = " ".join(f"{d}:{v:.0%}" for d, v in
                            sorted(sh_on.items(), key=lambda kv: -kv[1]))
            f_off = " ".join(f"{d}:{v:.0%}" for d, v in
                             sorted(sh_off.items(), key=lambda kv: -kv[1]))
            print(f"  {label:26s} on : WR={wr_on:3.0%}  [{f_on}]")
            print(f"  {'':26s} off: WR={wr_off:3.0%}  [{f_off}]", flush=True)
        print()


if __name__ == "__main__":
    main()
