"""WHEN does ally-healing pay off? Three probes behind the neutral verdict.

diag_coop_value measured ally-heal as WR-neutral (+0.1%/+0.5% within noise).
This script explains WHY and tests the conditions under which it stops being
neutral:

  PART 0 (tempo): how fast is combat relative to heal throughput? Measures
      rounds/game, front max HP, front damage taken per round, and computes
      the coverage ratio (max heal/round vs front damage/round). If a turn
      of healing undoes only a small fraction of a turn of damage, healing
      can shift time-to-death by less than one round and rarely crosses the
      live/die boundary.

  PART 1 (focus fire): the current expert opponents spread damage (attack
      nearest). Against FOCUS-FIRE opponents (always attack the lowest-HP%
      target), damage concentrates on one member, so triage heals land
      exactly on the death boundary. Re-runs triage-vs-selfish under
      focus-fire opponents.

  PART 2 (heal magnitude x3): patches cure_wounds 1d8+3 -> 3d8+9 and
      lay_on_hands 5 -> 15 per cast (diagnostic-only), so one heal action
      undoes roughly one enemy turn. Re-runs triage-vs-selfish.

NOT testable here without an engine change: REVIVE. The engine has PC death
saves (is_dying) and apply_heal can pick a dying PC back up, but the HEAL
action rejects non-alive targets (combat.py "已倒下，無法治療"), the expert's
_find_heal_target skips non-alive allies, and the obs/entity-mask treat dying
as dead. In real 5e the cleric's main win-value is exactly the pick-up
(1 HP = a whole action stream back); that lever is gated off here.

Usage: python scripts/diag_heal_when.py [games] [n_opp_comps]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import itertools
import random
from collections import defaultdict

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.action import encode_action
from trpg.engine.combat_policy import make_archetype_policy, _ClericBase
from trpg.engine.abilities import ABILITY_REGISTRY
from trpg.scenarios.archetypes import ARCHETYPE_ROLES

from eval_routed import stable_seed

games = int(sys.argv[1]) if len(sys.argv) > 1 else 16
n_opp_comps = int(sys.argv[2]) if len(sys.argv) > 2 else 6

FRONTS   = sorted(a for a, r in ARCHETYPE_ROLES.items() if r == "front")
STRIKERS = sorted(a for a, r in ARCHETYPE_ROLES.items() if r == "striker")
SUPPORTS = sorted(a for a, r in ARCHETYPE_ROLES.items() if r == "support")
ALL_COMPS = [list(c) for c in itertools.product(FRONTS, STRIKERS, SUPPORTS)]
_rng = random.Random(20260610)
OPP_SET = [ALL_COMPS[i] for i in _rng.sample(range(len(ALL_COMPS)), n_opp_comps)]
LIFE_COMPS = [[f, s, "life"] for f in FRONTS for s in ("evocation", "divination")]


class _SelfishHealMixin:
    def _find_heal_target(self, actor_id, ws, threshold: float = 0.45):
        a = ws.characters[actor_id]
        if a.max_hp and a.is_alive() and a.hp / a.max_hp < threshold:
            return actor_id
        return None


class _FocusFireMixin:
    """Opponent variant: every targeting decision aims at the lowest-HP%
    living enemy instead of the nearest — concentrated damage."""
    def _nearest_enemy(self, actor, ws, actor_id):
        enemies = self._enemy_candidates(actor_id, ws)
        if not enemies:
            return None, None
        return min(enemies, key=lambda kv: kv[1].hp / max(1, kv[1].max_hp))


def agent_policy(arch: str, selfish: bool):
    base_cls = type(make_archetype_policy(arch))
    if selfish and issubclass(base_cls, _ClericBase):
        return type("S" + base_cls.__name__, (_SelfishHealMixin, base_cls), {})()
    return base_cls()


def focus_policy(arch: str):
    base_cls = type(make_archetype_policy(arch))
    return type("F" + base_cls.__name__, (_FocusFireMixin, base_cls), {})()


def play(agent_comp, opp_comp, seed, st, selfish: bool, focus_opps: bool):
    env = CombatEnvV2(seed=seed, n_agents=3, n_opps=3)
    obs, _ = env.reset(agent_archs=agent_comp, opp_archs=opp_comp, level=5)
    if focus_opps:
        env._opp_policies = {oid: focus_policy(a)
                             for oid, a in zip(env.opp_ids, env.opp_archs)}
    experts = {aid: agent_policy(a, selfish)
               for aid, a in zip(env.agent_ids, env.agent_archs)}
    front = env.agent_ids[0]
    st["front_maxhp"] += env.ws.characters[front].max_hp
    done = False
    while not done:
        actor = env.current_agent_id
        ag = env.ws.characters[actor]
        rnd = env.ws.combat.round_number
        dec = experts[actor].decide(actor, ag, env.ws, env.resources, rnd)
        act = ([0, 0, 0] if (dec.action is None or dec.fled)
               else list(encode_action(dec.action, env.ws, actor)))
        if act[0] < 0:
            act = [0, 0, 0]
        hp_before = {cid: c.hp for cid, c in env.ws.characters.items()}
        obs, _, term, trunc, info = env.step(act)
        done = term or trunc
        res = (info or {}).get("action_result") or {}
        if res.get("type") in ("HEAL", "LAY_ON_HANDS"):
            st["heal_casts"] += 1
            st["heal_amt"] += sum(env.ws.characters[c].hp - hp_before[c]
                                  for c in env.agent_ids
                                  if env.ws.characters[c].hp > hp_before[c])
        if env.ws.characters[front].hp < hp_before[front]:
            st["front_dmg"] += hp_before[front] - env.ws.characters[front].hp
    st["rounds"] += env.ws.combat.round_number
    st["games"] += 1
    st["front_dead"] += int(not env.ws.characters[front].is_alive())
    opps_dead = all(not env.ws.characters[o].is_alive() for o in env.opp_ids)
    alive = any(env.ws.characters[a].is_alive() for a in env.agent_ids)
    st["wins"] += int(opps_dead and alive)


def run_arm(tag, comps, selfish, focus_opps):
    st = defaultdict(float)
    for comp in comps:
        for opp in OPP_SET:
            base = stable_seed(f"healwhen_{'+'.join(comp)}_{'+'.join(opp)}")
            for g in range(games):
                play(comp, opp, base + g, st, selfish, focus_opps)
    n = st["games"]
    wr = st["wins"] / n
    sig = (wr * (1 - wr) / n) ** 0.5
    print(f"  {tag:22s} WR={wr:5.1%} (±{2*sig:.1%})  heals/g={st['heal_casts']/n:4.2f} "
          f"healed-HP/g={st['heal_amt']/n:4.1f}  front-death={st['front_dead']/n:3.0%}  "
          f"rounds/g={st['rounds']/n:3.1f}  front-dmg/g={st['front_dmg']/n:4.1f} "
          f"(maxHP {st['front_maxhp']/n:3.0f})", flush=True)
    return {"wr": wr, "sig": sig}


def diff(a, b, la, lb):
    d = a["wr"] - b["wr"]
    s2 = 2 * (a["sig"] ** 2 + b["sig"] ** 2) ** 0.5
    v = "POSITIVE" if d > s2 else ("NEGATIVE" if d < -s2 else "NEUTRAL")
    print(f"  >> {la} − {lb} = {d:+.1%} (2σ ±{s2:.1%}) → {v}")


print(f"=== PART 0+1: focus-fire opponents — 12 life comps × {n_opp_comps} × {games}g ===")
print("  [normal opponents]")
n_tri = run_arm("triage / normal", LIFE_COMPS, selfish=False, focus_opps=False)
n_sel = run_arm("selfish / normal", LIFE_COMPS, selfish=True, focus_opps=False)
diff(n_tri, n_sel, "triage", "selfish")
print("  [FOCUS-FIRE opponents]")
f_tri = run_arm("triage / focus", LIFE_COMPS, selfish=False, focus_opps=True)
f_sel = run_arm("selfish / focus", LIFE_COMPS, selfish=True, focus_opps=True)
diff(f_tri, f_sel, "triage", "selfish")

print(f"\n=== PART 2: heal magnitude ×3 (cure 3d8+9, lay-on-hands 15) — normal opponents ===")
_cw = ABILITY_REGISTRY["cure_wounds"]
_loh = ABILITY_REGISTRY["lay_on_hands_ability"]
_cw_orig, _loh_orig = _cw.builder, _loh.builder
_cw.builder = lambda *a, **k: {**_cw_orig(*a, **k), "dice": "3d8+9"}
_loh.builder = lambda *a, **k: {**_loh_orig(*a, **k), "amount": 15}
b_tri = run_arm("triage / bigheal", LIFE_COMPS, selfish=False, focus_opps=False)
b_sel = run_arm("selfish / bigheal", LIFE_COMPS, selfish=True, focus_opps=False)
diff(b_tri, b_sel, "triage", "selfish")
_cw.builder, _loh.builder = _cw_orig, _loh_orig

print(f"\n=== PART 3: both (focus-fire opponents + ×3 heals) ===")
_cw.builder = lambda *a, **k: {**_cw_orig(*a, **k), "dice": "3d8+9"}
_loh.builder = lambda *a, **k: {**_loh_orig(*a, **k), "amount": 15}
fb_tri = run_arm("triage / focus+big", LIFE_COMPS, selfish=False, focus_opps=True)
fb_sel = run_arm("selfish / focus+big", LIFE_COMPS, selfish=True, focus_opps=True)
diff(fb_tri, fb_sel, "triage", "selfish")
_cw.builder, _loh.builder = _cw_orig, _loh_orig
