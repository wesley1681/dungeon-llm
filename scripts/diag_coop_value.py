"""Is cooperation reward-aligned? Controlled experiments on the expert team.

The user's goal asks for (a) cleric healing wounded allies and (b) wizards
avoiding friendly fire — IF those behaviours are consistent with reward
maximisation (terminal win reward + team PBRS). This script measures that
directly, with same-seed controlled comparisons:

  EXP A (heal targeting value):
      full expert team (cleric does lowest-HP%-ally triage)   [base]
   vs same but cleric heals ONLY ITSELF (selfish variant)     [selfheal]
      -> WR difference = win-value of healing allies.

  EXP B (friendly-fire avoidance value), divination-striker comps
      (evocation has sculpt_spells: engine already excludes allies):
      full expert team (AoE centred blindly on nearest enemy) [base]
   vs AoE retargeted to an ally-free enemy centre / skipped   [careful]
      -> WR difference = win-value of avoiding friendly fire.
      Also plays the routed MODEL team for properly-attributed FF numbers.

  EXP C (attribution sanity): model team on evocation comps — sculpted
      fireballs should show ZERO ally AoE damage; the 64.7/game number in
      the earlier trace_coop was hostile-aura/terrain ticks misattributed.

Friendly fire here is attributed from the engine's own cast result
(target_results per target), not from HP diffs, so turn-start ticks of the
NEXT character can no longer pollute the number. Character names are made
unique after reset so target_name -> char id is unambiguous.

Usage: python scripts/diag_coop_value.py [games] [n_opp_comps]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import itertools
import random
from collections import defaultdict
import torch

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.action import encode_action
from trpg.engine.combat_policy import (make_archetype_policy, _ClericBase,
                                       _WizardBase)
from trpg.engine.abilities import ABILITY_REGISTRY
from trpg.engine.skill import available_skills
from trpg.scenarios.archetypes import ARCHETYPE_ROLES

from eval_routed import DEFAULT_ROUTING, load_net, stable_seed

games = int(sys.argv[1]) if len(sys.argv) > 1 else 12
n_opp_comps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
which = (sys.argv[3] if len(sys.argv) > 3 else "all").lower()

HEAL_SKILLS = ("cure_wounds", "lay_on_hands_ability")
DIAG = defaultdict(float)          # variant-internal counters (retarget/skip)


# ── policy variants (diagnostic-only, never used for training) ───────────────

class _SelfishHealMixin:
    """Cleric that triages ONLY itself — removes ally-healing, keeps the
    same threshold trigger. Isolates the value of the heal TARGET choice."""
    def _find_heal_target(self, actor_id, ws, threshold: float = 0.45):
        a = ws.characters[actor_id]
        if a.max_hp and a.is_alive() and a.hp / a.max_hp < threshold:
            return actor_id
        return None


class _EagerBonusHealMixin:
    """Cleric that spends its BONUS ACTION on a ranged ally-heal whenever any
    party member is below 80% HP, before anything else. Tests whether heal
    VOLUME (not just target choice) has win value: the bonus-action heal has
    near-zero opportunity cost (competes only with spiritual weapon and the
    one-leveled-spell-per-turn rule, both correctly charged by the engine).
    The heal skill is found generically: bonus-cost ally-target ability with
    expected_healing > 0, target in range."""
    def decide(self, actor_id, actor, ws, resources, round_num):
        if resources.get("bonus_action", 0) > 0:
            heal_id = self._find_heal_target(actor_id, ws, 0.8)
            if heal_id is not None:
                from trpg.engine.skill import available_skills, TargetType
                tgt = ws.characters[heal_id]
                for sk in available_skills(actor, ws):
                    f = sk.features
                    if (getattr(f, "cost_bonus", 0) > 0
                            and getattr(f, "expected_healing", 0) > 0
                            and f.target_type == TargetType.SINGLE_ALLY
                            and actor.position.distance_to(tgt.position)
                                < float(f.range_m) - 0.01):
                        a = self._use_skill(actor_id, actor, ws,
                                            sk.skill_id, heal_id)
                        if a:
                            from trpg.engine.combat_policy import CombatDecision
                            return CombatDecision(action=a)
        return super().decide(actor_id, actor, ws, resources, round_num)


class _CarefulAoEMixin:
    """Wizard that refuses to centre a damaging AoE on a point that splashes
    its own side (self included). Tries other enemy centres in range first;
    if none is clean, skips the spell (falls through to the next priority).
    Casters with sculpt_spells pass through untouched (engine already safe).
    """
    def _use_skill(self, actor_id, actor, ws, skill_id, target_id=None,
                   coord=None):
        ab = ABILITY_REGISTRY.get(skill_id)
        f = ab.features if ab is not None else None
        if (f is None or coord is None
                or getattr(f, "aoe_radius_m", 0.0) <= 0.0
                or getattr(f, "expected_damage", 0.0) <= 0.0
                or getattr(actor, "sculpt_spells", False)):
            return super()._use_skill(actor_id, actor, ws, skill_id,
                                      target_id, coord)
        my_side = ws.is_party_ally(actor_id)
        radius = float(f.aoe_radius_m)
        alive = [(cid, c) for cid, c in ws.characters.items() if c.is_alive()]

        def splash(center):
            a = sum(1 for cid, c in alive
                    if ws.is_party_ally(cid) == my_side
                    and c.position.distance_to(center) <= radius + 1e-6)
            e = sum(1 for cid, c in alive
                    if ws.is_party_ally(cid) != my_side
                    and c.position.distance_to(center) <= radius + 1e-6)
            return a, e

        a0, _ = splash(coord)
        if a0 == 0:
            return super()._use_skill(actor_id, actor, ws, skill_id,
                                      target_id, coord)
        best = None   # (enemies_hit, cid, pos)
        for cid, c in alive:
            if ws.is_party_ally(cid) == my_side:
                continue
            if actor.position.distance_to(c.position) > float(f.range_m) + 1e-6:
                continue
            a1, e1 = splash(c.position)
            if a1 == 0 and e1 >= 1 and (best is None or e1 > best[0]):
                best = (e1, cid, c.position)
        if best is not None:
            DIAG["aoe_retarget"] += 1
            return super()._use_skill(actor_id, actor, ws, skill_id,
                                      best[1], best[2])
        DIAG["aoe_skip"] += 1
        return None


def variant_policy(arch: str, mode: str):
    base_cls = type(make_archetype_policy(arch))
    if mode == "selfheal" and issubclass(base_cls, _ClericBase):
        return type("Selfish" + base_cls.__name__,
                    (_SelfishHealMixin, base_cls), {})()
    if mode == "healword" and issubclass(base_cls, _ClericBase):
        return type("Eager" + base_cls.__name__,
                    (_EagerBonusHealMixin, base_cls), {})()
    if mode == "careful" and issubclass(base_cls, _WizardBase):
        return type("Careful" + base_cls.__name__,
                    (_CarefulAoEMixin, base_cls), {})()
    return base_cls()


# ── routed model (for the model arms) ────────────────────────────────────────

_cache: dict = {}
ARCH_TO_NET: dict = {}
def _nets():
    if not ARCH_TO_NET:
        for arch, path in DEFAULT_ROUTING.items():
            if path not in _cache:
                _cache[path] = load_net(path)
            ARCH_TO_NET[arch] = _cache[path]
    return ARCH_TO_NET


# ── episode driver with engine-attributed accounting ─────────────────────────

def play(mode: str, agent_comp, opp_comp, seed, st) -> None:
    """mode: 'base' | 'selfheal' | 'careful' | 'model'."""
    env = CombatEnvV2(seed=seed, n_agents=3, n_opps=3)
    obs, _ = env.reset(agent_archs=agent_comp, opp_archs=opp_comp, level=5)
    # Unique display names so SPELL target_results map back to char ids.
    for cid, c in env.ws.characters.items():
        c.name = f"{c.name}#{cid}"
    name2cid = {c.name: cid for cid, c in env.ws.characters.items()}
    side = {cid: env.ws.is_party_ally(cid) for cid in env.ws.characters}
    arch_of = dict(zip(env.agent_ids, env.agent_archs))
    experts = ({aid: variant_policy(a, mode) for aid, a in arch_of.items()}
               if mode != "model" else None)

    done = False
    while not done:
        actor = env.current_agent_id
        ag = env.ws.characters[actor]
        rnd = env.ws.combat.round_number
        if experts is not None:
            dec = experts[actor].decide(actor, ag, env.ws, env.resources, rnd)
            act = ([0, 0, 0] if (dec.action is None or dec.fled)
                   else list(encode_action(dec.action, env.ws, actor)))
            if act[0] < 0:
                act = [0, 0, 0]
        else:
            net = _nets()[arch_of[actor]]
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            act = list(pick_action(el[0], s[0], e[0], g[0],
                                   ws=env.ws, agent_id=actor))

        skills = available_skills(ag, env.ws)
        sid = skills[act[0]].skill_id if 0 <= act[0] < len(skills) else None

        hp_before = {cid: c.hp for cid, c in env.ws.characters.items()}
        obs, _, term, trunc, info = env.step(act)
        done = term or trunc
        res = (info or {}).get("action_result") or {}

        if res.get("type") == "SPELL":
            for tr in res.get("target_results", ()):
                cid = name2cid.get(tr["target_name"])
                dmg = float(tr.get("damage", 0) or 0)
                if cid is None or dmg <= 0:
                    continue
                if side[cid] == side[actor]:
                    st["aoe_ally_dmg"] += dmg
                    if cid == actor:
                        st["aoe_self_dmg"] += dmg
                    st["aoe_ally_hits"] += 1
                else:
                    st["aoe_enemy_dmg"] += dmg
        elif res.get("type") == "HEAL":
            cid = name2cid.get(res.get("target_name"))
            if cid is not None:
                st["heal_casts"] += 1
                st["heal_amt"] += float(res.get("amount", 0) or 0)
                st["heal_on_ally" if cid != actor else "heal_on_self"] += 1
                st["heal_hpfrac"] += (hp_before[cid]
                                      / max(1, env.ws.characters[cid].max_hp))
        elif (res.get("type") == "ERROR" and sid in HEAL_SKILLS):
            st["heal_err"] += 1   # heal attempted, engine rejected (range/slot)

        # residual same-side HP loss NOT explained by the cast result =
        # ticks (hostile aura / terrain / opportunity attacks etc.)
        same_drop = sum(hp_before[cid] - env.ws.characters[cid].hp
                        for cid in env.agent_ids
                        if env.ws.characters[cid].hp < hp_before[cid])
        aoe_part = sum(float(tr.get("damage", 0) or 0)
                       for tr in res.get("target_results", ())
                       if res.get("type") == "SPELL"
                       and name2cid.get(tr["target_name"]) in env.agent_ids)
        st["team_hp_lost_other"] += max(0.0, same_drop - aoe_part)

    st["games"] += 1
    front = env.agent_ids[0]
    st["front_dead"] += int(not env.ws.characters[front].is_alive())
    opps_dead = all(not env.ws.characters[o].is_alive() for o in env.opp_ids)
    alive = any(env.ws.characters[a].is_alive() for a in env.agent_ids)
    st["wins"] += int(opps_dead and alive)


# ── experiment runner ─────────────────────────────────────────────────────────

FRONTS   = sorted(a for a, r in ARCHETYPE_ROLES.items() if r == "front")
SUPPORTS = sorted(a for a, r in ARCHETYPE_ROLES.items() if r == "support")
STRIKERS = sorted(a for a, r in ARCHETYPE_ROLES.items() if r == "striker")
ALL_COMPS = [list(c) for c in itertools.product(FRONTS, STRIKERS, SUPPORTS)]
_rng = random.Random(20260610)
OPP_SET = [ALL_COMPS[i] for i in _rng.sample(range(len(ALL_COMPS)), n_opp_comps)]

SPREAD_COMPS = [c.split("+") for c in (
    "battle_master+evocation+life",  "battle_master+assassin+war",
    "champion+divination+life",      "champion+arcane_trickster+war",
    "totem_bear+evocation+war",      "totem_bear+assassin+life",
    "berserker+divination+war",      "berserker+arcane_trickster+life",
    "devotion+evocation+life",       "devotion+assassin+war",
    "vengeance+divination+life",     "vengeance+arcane_trickster+war",
)]
DIV_COMPS = [[f, "divination", s] for f in FRONTS for s in SUPPORTS]
EVO_COMPS = [[f, "evocation", s] for f in FRONTS for s in SUPPORTS]


def run_arm(tag: str, mode: str, comps) -> dict:
    st = defaultdict(float)
    DIAG.clear()
    for comp in comps:
        for opp in OPP_SET:
            base = stable_seed(f"diagcoop_{'+'.join(comp)}_{'+'.join(opp)}")
            for g in range(games):
                play(mode, comp, opp, base + g, st)
    n = st["games"]; hc = max(1.0, st["heal_casts"])
    wr = st["wins"] / n
    sig = (wr * (1 - wr) / n) ** 0.5
    print(f"  {tag:18s} WR={wr:5.1%} (±{2*sig:.1%} 2σ, n={n:.0f})  "
          f"heals/g={st['heal_casts']/n:4.2f} "
          f"(ally {st['heal_on_ally']/hc:3.0%}, hp%@cast {st['heal_hpfrac']/hc:3.0%}, "
          f"err/g {st['heal_err']/n:4.2f})  "
          f"AoE ally-dmg/g={st['aoe_ally_dmg']/n:4.1f} "
          f"(self {st['aoe_self_dmg']/n:4.1f})  enemy-dmg/g={st['aoe_enemy_dmg']/n:5.1f}  "
          f"tick-dmg/g={st['team_hp_lost_other']/n:5.1f}  "
          f"front-death={st['front_dead']/n:3.0%}"
          + (f"  retarget/g={DIAG['aoe_retarget']/n:4.2f} skip/g={DIAG['aoe_skip']/n:4.2f}"
             if mode == "careful" else ""))
    return {"wr": wr, "sig": sig, "n": n}


def diff(a, b, la, lb):
    d = a["wr"] - b["wr"]
    s2 = 2 * (a["sig"] ** 2 + b["sig"] ** 2) ** 0.5
    verdict = ("ALIGNED (helps win)" if d > s2
               else ("HURTS" if d < -s2 else "NEUTRAL (within noise)"))
    print(f"  >> {la} − {lb} = {d:+.1%}  (2σ ±{s2:.1%})  → {verdict}")


LIFE_COMPS = [[f, s, "life"] for f in FRONTS for s in ("evocation", "divination")]

if which in ("a", "all"):
    print(f"=== EXP A: heal-target value — 12 spread comps × {n_opp_comps} opp × {games}g ===")
    a_base = run_arm("expert (triage)", "base", SPREAD_COMPS)
    a_self = run_arm("expert (selfish)", "selfheal", SPREAD_COMPS)
    diff(a_base, a_self, "triage", "selfish")

if which in ("a2", "all"):
    print(f"\n=== EXP A2: heal-VOLUME value — 12 life comps × {n_opp_comps} opp × {games}g ===")
    a2_base = run_arm("expert (triage)", "base", LIFE_COMPS)
    a2_hw = run_arm("expert (bonus-heal)", "healword", LIFE_COMPS)
    diff(a2_hw, a2_base, "bonus-heal", "triage")

if which in ("b", "all"):
    print(f"\n=== EXP B: FF-avoidance value — 12 divination comps × {n_opp_comps} opp × {games}g ===")
    b_base = run_arm("expert (blind AoE)", "base", DIV_COMPS)
    b_safe = run_arm("expert (careful)", "careful", DIV_COMPS)
    diff(b_safe, b_base, "careful", "blind")
    if which == "all":
        b_model = run_arm("model (routed)", "model", DIV_COMPS)

if which in ("c", "all"):
    print(f"\n=== EXP C: attribution sanity — model on 12 evocation comps (sculpted) ===")
    c_model = run_arm("model (routed)", "model", EVO_COMPS)
    print("    (sculpt_spells: expected AoE ally-dmg ≈ 0 — the old 64.7 'friendly "
          "fire' number was hostile-aura/terrain ticks, see tick-dmg column)")
