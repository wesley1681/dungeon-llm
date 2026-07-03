"""Reproduce + diagnose the GUI 'disengage-loop' bug: model drives a ranged
CASTER, enemy is PASSIVE and STATIONARY at range. Does the caster engage
(cast a ranged spell / close distance) or spin on a useless defensive action
(disengage) doing 0 damage?

The bug regime my audits missed: every scripted opponent in diag_degen_audit /
diag_info_channels is AGGRESSIVE (closes + attacks), so the model always had a
near target. A passive enemy that stays far (a human ending turn every round)
is a regime never tested.

Per model turn we print: distance, each available skill (masked? = out of
range/illegal after apply_resource_mask), the top action logits, and the chosen
action — so we see WHY it picks what it picks.
"""
from __future__ import annotations
import sys, os, argparse
from types import SimpleNamespace
from zlib import crc32
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import random
import torch

from trpg.engine.vec2 import Vec2
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.engine.skill import available_skills
from trpg.scenarios.monsters import register_monsters
register_monsters()
from distill_routed import load_student
from train_population import blind_np_single
from trpg.scenarios.archetypes import (CLASS_DEFS, ARCHETYPE_FACTORIES,
                                       ARCHETYPE_ROLES, ClassDef, SkillGrant,
                                       TraitGrant, _factory, _WIZARD_SLOTS)


def _register_caster(aid, skills):
    if aid in CLASS_DEFS:
        return aid
    cd = ClassDef(
        archetype_id=aid, default_name=aid, class_display="法師",
        role="ranged",
        stat_block=dict(STR=8, DEX=14, CON=14, INT=16, WIS=12, CHA=10),
        hp_base=6, hp_per_level=4, ac=12, weapons=(),
        proficiencies=("INT", "WIS"),
        spell_ability="INT", spell_slots_table=_WIZARD_SLOTS,
        skills=tuple(SkillGrant(s, min_level=1) for s in skills),
        traits=())
    CLASS_DEFS[aid] = cd
    ARCHETYPE_FACTORIES[aid] = _factory(aid)
    ARCHETYPE_ROLES[aid] = cd.role
    return aid


_FULL = ("magic_missile", "shield_spell", "burning_hands_ev", "web_ev",
         "misty_step", "hold_person", "fireball_ev", "ice_storm_ev",
         "cold_breath")


def register_gui_evoker():
    """EXACT combat_log.txt Evoker kit + single-variable isolation variants."""
    _register_caster("gui_evoker", _FULL)
    # baseline caster (no shield/cold_breath): fireball+mm+ice+utility
    _register_caster("evo_base", ("magic_missile", "burning_hands_ev", "web_ev",
                                  "misty_step", "hold_person", "fireball_ev",
                                  "ice_storm_ev"))
    # +cold_breath only
    _register_caster("evo_cold", ("magic_missile", "burning_hands_ev", "web_ev",
                                  "misty_step", "hold_person", "fireball_ev",
                                  "ice_storm_ev", "cold_breath"))
    # +shield_spell only
    _register_caster("evo_shield", ("magic_missile", "shield_spell",
                                    "burning_hands_ev", "web_ev", "misty_step",
                                    "hold_person", "fireball_ev", "ice_storm_ev"))
    # minimal fire+mm (the probe_heldout_switch shape that DID switch)
    _register_caster("evo_min", ("fireball_ev", "magic_missile"))
    return "gui_evoker"


MASK_NULL_DISENGAGE = [False]
CLAMP_DESC = [False]    # clamp self capability-descriptor scalars to [0,1]
MASK_SKILLS = [set()]   # skill_ids to force-mask (test: unpickable but visible)
ACTIVE_ENEMY = [False]  # False = stationary passive; True = env's scripted opp


class StationaryPolicy:
    def decide(self, opp_id, opp, ws, resources, round_number):
        return SimpleNamespace(action=None, fled=False, ended=True)


def run_quiet(net, ident, dist, level, turns_max, seed, immune, enemy):
    """Same episode, no per-decision printing; returns (dmg, hp0, picks)."""
    import io, contextlib
    from collections import Counter
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run(net, ident, dist, level, turns_max, seed, immune=immune, enemy=enemy)
    # parse the summary line we printed
    line = [l for l in buf.getvalue().splitlines() if "dmg dealt=" in l]
    if not line:
        return 0.0, 1.0
    import re
    m = re.search(r"dmg dealt=(\d+)/(\d+)", line[-1])
    return (float(m.group(1)), float(m.group(2))) if m else (0.0, 1.0)


def sweep(net, dist, level, turns):
    """Passive-standoff audit matrix: a caster whose PRIMARY damage the enemy is
    immune to must still ENGAGE (fall back to a non-immune attack), not collapse
    into disengage/kite. RED = ~0 dmg vs a passive enemy it could damage."""
    kits = ["evocation", "divination", "gui_evoker"]
    enemies = ["battle_master", "champion", "vengeance"]
    print(f"\n===== PASSIVE-STANDOFF SWEEP  (enemy stationary @ {dist}m, L{level}) =====")
    print(f"{'kit':14s} {'enemy':14s} {'immune':6s} {'dmg':>10s}  verdict")
    worst = 1.0
    for kit in kits:
        primary = "火"   # all three fire-primary casters; immune => must switch
        for enemy in enemies:
            for immune in (primary, None):
                dmg, hp0 = run_quiet(net, kit, dist, level, turns, 0, immune, enemy)
                frac = dmg / max(1.0, hp0)
                worst = min(worst, frac) if immune else worst
                v = ("紅 0傷退化" if immune and frac < 0.02
                     else "綠" if frac >= 0.02 else "灰")
                print(f"{kit:14s} {enemy:14s} {str(immune or '-'):6s} "
                      f"{dmg:5.0f}/{hp0:<4.0f} {v}")
    print(f"worst immune-arm dmg fraction = {worst:.1%}")


def run(net, ident, dist, level, turns_max, seed, immune=None, enemy="champion"):
    k = crc32(f"standoff_{ident}_{dist}_{seed}".encode())
    random.seed(k)
    env = CombatEnvV2(seed=k ^ 0x33, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[ident], opp_archs=[enemy],
                       level=level, opp_level=level, layout="open")
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    if not ACTIVE_ENEMY[0]:
        env._opp_policies[oid] = StationaryPolicy()
    agent = env.ws.characters[aid]; enemy = env.ws.characters[oid]
    if immune:
        enemy.damage_multipliers[immune] = 0.0
        obs = None  # force rebuild below
        from trpg.rl.obs import build_obs
        obs = build_obs(env.ws, env.current_agent_id, env.resources)
    bf = env.ws.combat.battlefield
    W, H = bf.width, bf.height
    agent.position = Vec2(W * 0.5, H * 0.5)
    enemy.position = Vec2(min(W - 0.5, W * 0.5 + dist), H * 0.5)
    print(f"\n=== {ident}  L{level}  start dist={agent.position.distance_to(enemy.position):.1f}m ===")

    decisions = 0
    chosen = []
    done = False
    term = trunc = False
    while not done and decisions < turns_max:
        actor = env.current_agent_id
        if actor == aid:
            decisions += 1
            ch = env.ws.characters[actor]
            d = ch.position.distance_to(enemy.position)
            ob = blind_np_single(obs)
            if CLAMP_DESC[0]:
                from trpg.rl.obs import ENT_DESC_START
                import numpy as _np
                ent = ob["entities"]
                # clamp the 6 magnitude scalars (dmg/heal/rng/aoe/atk/dc) on ALL
                # rows to [0,1] — the normalization ceiling the model trained under.
                ent[:, ENT_DESC_START:ENT_DESC_START + 6] = _np.minimum(
                    ent[:, ENT_DESC_START:ENT_DESC_START + 6], 1.0)
            ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            sks = available_skills(ch, env.ws)
            if MASK_SKILLS[0]:
                for i in range(min(len(sks), s.shape[-1])):
                    if sks[i].skill_id in MASK_SKILLS[0]:
                        s[0, i] = -1e9
            if MASK_NULL_DISENGAGE[0]:
                # null-effect disengage: no adjacent enemy => nothing to avoid.
                near = min((ch.position.distance_to(env.ws.characters[o].position)
                            for o in env.opp_ids
                            if env.ws.characters[o].is_alive()), default=99.0)
                if near > 2.0:
                    for i in range(min(len(sks), s.shape[-1])):
                        if sks[i].skill_id == "disengage":
                            s[0, i] = -1e9
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
            sv = s[0].tolist()
            legal = [(i, sv[i]) for i in range(min(len(sks), len(sv)))
                     if sv[i] > -1e8]
            legal.sort(key=lambda t: -t[1])
            top = ", ".join(f"{sks[i].skill_id}:{v:.2f}" for i, v in legal[:6])
            masked = [sks[i].skill_id for i in range(min(len(sks), len(sv)))
                      if sv[i] <= -1e8 and sks[i].skill_id != "move"]
            chosen_id = sks[act[0]].skill_id if 0 <= act[0] < len(sks) else f"slot{act[0]}"
            chosen.append(chosen_id)
            rnd = env.ws.combat.round_number
            print(f" r{rnd} d={d:4.1f}m  pick={chosen_id:16s}"
                  f"  legal_top=[{top}]")
            if masked:
                print(f"        out-of-range/masked: {masked}")
            obs, _, term, trunc, _ = env.step(act)
        else:
            obs, _, term, trunc, _ = env.step([0, 0, 0])
        done = term or trunc
    hp0 = enemy.max_hp
    dmg = hp0 - max(0, enemy.hp)
    from collections import Counter
    print(f"  -> {decisions} decisions, dmg dealt={dmg:.0f}/{hp0:.0f}, "
          f"term={term} trunc={trunc}, picks={dict(Counter(chosen))}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/unified/uni_v9.pt")
    p.add_argument("--idents", nargs="*",
                   default=["evocation", "divination", "assassin",
                            "battle_master", "war"])
    p.add_argument("--dist", type=float, default=11.0)
    p.add_argument("--level", type=int, default=8)
    p.add_argument("--turns", type=int, default=8)
    p.add_argument("--immune", default=None, help="damage type the enemy is immune to (e.g. 火)")
    p.add_argument("--enemy", default="champion", help="enemy archetype (target)")
    p.add_argument("--mask_null_disengage", action="store_true",
                   help="mask disengage when no enemy is adjacent (null-effect)")
    p.add_argument("--sweep", action="store_true",
                   help="run the passive-standoff audit matrix (kit x enemy x immune)")
    p.add_argument("--clamp_desc", action="store_true",
                   help="clamp self capability-descriptor magnitude scalars to [0,1]")
    p.add_argument("--mask_skills", nargs="*", default=[],
                   help="skill_ids to force-mask (unpickable but still visible)")
    p.add_argument("--active", action="store_true",
                   help="enemy uses the env's scripted policy (chases/attacks) "
                        "instead of standing still")
    args = p.parse_args()
    MASK_NULL_DISENGAGE[0] = args.mask_null_disengage
    CLAMP_DESC[0] = args.clamp_desc
    MASK_SKILLS[0] = set(args.mask_skills)
    ACTIVE_ENEMY[0] = args.active
    register_gui_evoker()
    net = load_student(args.ckpt); net.eval()
    if args.sweep:
        sweep(net, args.dist, args.level, args.turns)
        return
    print(f"ckpt={args.ckpt}  enemy=STATIONARY passive @ {args.dist}m open"
          f"{'  immune='+args.immune if args.immune else ''}")
    for ident in args.idents:
        run(net, ident, args.dist, args.level, args.turns, 0, immune=args.immune,
            enemy=args.enemy)


if __name__ == "__main__":
    main()
