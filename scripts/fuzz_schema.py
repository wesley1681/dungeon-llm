"""Schema fuzz — verify builder output and resource accounting.

Two passes:
  1. Static: for every Spell + Ability with engine_ready=True, build an
     action_dict with a plausible target and check structural invariants
     (target vs target_position, slot_level, consumes list).
  2. Dynamic: run random rollouts and verify that any action consuming a
     resource (spell slot, ability use, concentration) produced a non-empty
     effect (target_results / status applied / damage dealt).

Both passes raise on the first violation. Run before training.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import torch
import numpy as np

from trpg.engine.skill import available_skills, from_spell, TargetType
from trpg.engine.spells import SPELLS
from trpg.engine.abilities import ABILITY_REGISTRY
from trpg.engine.vec2 import Vec2
from trpg.rl.env_v2 import CombatEnvV2, _AGENT_ID, ARCHETYPE_LIST
from trpg.rl.model import CombatPolicyNet, apply_resource_mask, apply_entity_mask


# ─── Pass 1: static structural check ───────────────────────────────────────

def check_spell_schema():
    """Every Spell's from_spell builder must match its aoe semantics."""
    violations = []
    for name, spell in SPELLS.items():
        sk = from_spell(spell, _DummyCaster())
        # AOE → POINT, single → SINGLE_ENEMY
        expected_tt = TargetType.POINT if spell.aoe_radius_m > 0 else TargetType.SINGLE_ENEMY
        if sk.features.target_type != expected_tt:
            violations.append(
                f"  Spell {name!r}: features.target_type={sk.features.target_type} "
                f"but aoe_radius_m={spell.aoe_radius_m} → expected {expected_tt}"
            )
            continue
        # Build with both target_id and coord — check builder picks correctly
        action = sk.builder("caster_x", "victim_y", Vec2(5.0, 5.0))
        if spell.aoe_radius_m > 0:
            if "target_position" not in action:
                violations.append(
                    f"  Spell {name!r} (AOE): builder did not set target_position "
                    f"→ engine would treat as single-target. action={action}"
                )
            if "target" in action:
                violations.append(
                    f"  Spell {name!r} (AOE): builder set both target & target_position "
                    f"→ ambiguous. action={action}"
                )
        else:
            if "target" not in action:
                violations.append(
                    f"  Spell {name!r} (single): builder did not set target — "
                    f"engine would consume slot for nobody. action={action}"
                )
            if "target_position" in action:
                violations.append(
                    f"  Spell {name!r} (single): builder set target_position — "
                    f"engine would treat as coord cast and miss the creature. "
                    f"action={action}"
                )
    return violations


def check_ability_schema():
    """For each engine_ready Ability, build and check structural invariants."""
    violations = []
    for skill_id, ab in ABILITY_REGISTRY.items():
        if not ab.engine_ready or ab.is_reaction or ab.builder is None:
            continue
        try:
            action = ab.builder("caster_x", "victim_y", Vec2(5.0, 5.0),
                                char=_DummyCaster())
        except Exception as exc:
            violations.append(f"  Ability {skill_id!r}: builder raised {exc!r}")
            continue
        if action is None:
            continue   # builder said "no", fine
        # Slot consumption consistency
        if ab.features.cost_slot_level > 0:
            slot_level = action.get("slot_level", 0)
            min_lvl = int(ab.features.cost_slot_level)
            if slot_level not in (min_lvl, 0):
                # 0 means "engine auto-picks"; otherwise must match the declared min
                violations.append(
                    f"  Ability {skill_id!r}: features.cost_slot_level={min_lvl} "
                    f"but builder produced slot_level={slot_level}"
                )
        # Concentration consistency
        if ab.features.requires_concentration:
            if not action.get("requires_concentration"):
                # APPLY_MOD reads this flag; missing means concentration won't be set
                if action.get("type") == "APPLY_MOD":
                    violations.append(
                        f"  Ability {skill_id!r}: features.requires_concentration=True "
                        f"but builder did not set requires_concentration on the action"
                    )
    return violations


class _DummyCaster:
    """Stand-in character for static schema checks. Provides the attributes
    builders / from_spell read without needing a full WorldState."""
    name = "Dummy"
    level = 5
    proficiency_bonus = 3
    spellcasting_ability = "WIS"
    spell_slots = {1: 4, 2: 3, 3: 2, 4: 1, 5: 1}
    lay_on_hands_pool = 25
    ability_uses = {}
    weapons = []
    position = Vec2(0.0, 0.0)
    sculpt_spells = False
    spells = []
    known_abilities = []

    class _Stats:
        def modifier(self, _):
            return 3
    stats = _Stats()


# ─── Pass 2: dynamic resource-vs-effect audit ──────────────────────────────

def check_resource_audit(n_trials: int = 1000, max_steps: int = 20):
    """Flag actions whose result violates the engine contract.

    We look at the action_result itself, not before/after resource diffs —
    deltas pick up reaction triggers and opponent turn effects, which are
    legitimate concurrent state changes, not silent failures of this action.

    Two real violations:
      (a) Non-AOE SPELL succeeded with empty target_results → contract bug
          (engine should have raised, or builder produced wrong shape).
      (b) APPLY_MOD with slot_level > 0 succeeded with no targets_affected
          → slot + concentration consumed for nobody.
    """
    net = CombatPolicyNet().eval()
    violations = []

    for trial in range(n_trials):
        env = CombatEnvV2(seed=trial)
        obs, _ = env.reset()
        for step in range(max_steps):
            obs_t = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                _, sl, e, g = net(obs_t)
            sl = apply_resource_mask(sl, env.resources, env.ws, _AGENT_ID)
            e = apply_entity_mask(e, obs_t)
            # Hierarchical sample: skill, then conditional entity/grid
            skill_idx = int(torch.distributions.Categorical(logits=sl.squeeze(0)).sample())
            ent_idx = int(torch.distributions.Categorical(logits=e[0, skill_idx, :]).sample())
            grid_idx = int(torch.distributions.Categorical(logits=g[0, skill_idx, :]).sample())
            action = [skill_idx, ent_idx, grid_idx]
            try:
                obs, r, term, trunc, info = env.step(action)
            except RuntimeError as exc:
                violations.append(f"trial={trial} step={step} RAISE: {exc!s}"[:140])
                break

            result = info.get("action_result") or {}
            t = result.get("type")
            spell_name = result.get("spell_name")
            if t == "SPELL" and spell_name:
                spell = SPELLS.get(spell_name)
                # Non-AOE spells must always affect at least one creature; an
                # empty target_results means the engine consumed the slot
                # without resolving the spell against any target.
                if spell and spell.aoe_radius_m == 0 and not result.get("target_results"):
                    violations.append(
                        f"  trial={trial} step={step}: non-AOE SPELL "
                        f"{spell_name!r} returned empty target_results "
                        f"(slot {result.get('slot_level')} consumed)"
                    )
            if t == "APPLY_MOD" and int(result.get("slot_level", 0)) > 0:
                if not result.get("targets_affected"):
                    violations.append(
                        f"  trial={trial} step={step}: APPLY_MOD "
                        f"{result.get('modifier')!r} consumed slot "
                        f"{result.get('slot_level')} but no targets affected"
                    )
            if term or trunc:
                break
        if len(violations) > 30:
            break
    return violations


def _snapshot_resources(char) -> dict:
    return {
        "slots": dict(char.spell_slots),
        "uses":  dict(char.ability_uses),
        "pool":  getattr(char, "lay_on_hands_pool", 0),
        "conc":  char.concentrating_on,
        "ammo":  {c.name: c.quantity for c in char.consumables},
    }


def _diff(before: dict, after: dict) -> dict:
    out = {}
    for lvl, v in before["slots"].items():
        if after["slots"].get(lvl, 0) < v:
            out[f"slot_{lvl}"] = v - after["slots"].get(lvl, 0)
    for sid, v in before["uses"].items():
        if after["uses"].get(sid, 0) < v:
            out[f"use_{sid}"] = v - after["uses"].get(sid, 0)
    if after["pool"] < before["pool"]:
        out["loh_pool"] = before["pool"] - after["pool"]
    if before["conc"] != after["conc"] and after["conc"]:
        out["concentration"] = after["conc"]
    for name, q in before["ammo"].items():
        if after["ammo"].get(name, 0) < q:
            out[f"ammo_{name}"] = q - after["ammo"].get(name, 0)
    return out


def _has_effect(result: dict) -> bool:
    """True iff the action_result shows the action actually did something."""
    if not result:
        return False
    t = result.get("type")
    if t in ("ATTACK", "MULTI_ATTACK"):
        # Even a miss is a valid effect — the action was resolved
        return True
    if t in ("MOVE", "HIDE", "DODGE", "DISENGAGE"):
        return True
    if t == "LAY_ON_HANDS":
        return result.get("healed", 0) > 0
    if t == "HEAL":
        return result.get("amount", 0) > 0
    if t == "AUTO_DAMAGE":
        return bool(result.get("target_results"))
    if t == "APPLY_MOD":
        return bool(result.get("targets_affected"))
    if t == "SPELL":
        return bool(result.get("target_results"))
    if t in ("COUNTERSPELLED", "ERROR"):
        return True   # not a silent failure
    return True   # unknown type: don't flag


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=== Pass 1a: spell schema ===")
    v1a = check_spell_schema()
    if v1a:
        print(f"❌ {len(v1a)} violations:")
        for x in v1a:
            print(x)
    else:
        print(f"✓ {len(SPELLS)} spells: all builder shapes consistent")

    print("\n=== Pass 1b: class-ability schema ===")
    v1b = check_ability_schema()
    if v1b:
        print(f"❌ {len(v1b)} violations:")
        for x in v1b:
            print(x)
    else:
        n = sum(1 for ab in ABILITY_REGISTRY.values()
                if ab.engine_ready and not ab.is_reaction and ab.builder)
        print(f"✓ {n} engine-ready abilities: all builder shapes consistent")

    print("\n=== Pass 2: dynamic resource-vs-effect audit ===")
    v2 = check_resource_audit(n_trials=1000)
    if v2:
        print(f"❌ {len(v2)} violations:")
        for x in v2[:20]:
            print(x)
    else:
        print("✓ No silent consumption detected in 1000 trials")

    total = len(v1a) + len(v1b) + len(v2)
    print(f"\nTotal violations: {total}")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
