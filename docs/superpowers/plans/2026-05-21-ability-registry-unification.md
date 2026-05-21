# Ability Registry Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the dual `char.spells` + `char.known_abilities` system so all executable abilities flow through a single `ABILITY_REGISTRY`, preventing future bugs like 治療術 silently dealing 0 damage.

**Architecture:** Remove `char.spells` field, `from_spell()` function, and the spells-loop in `available_skills()`. Rename `ClassAbility` → `Ability` and `CLASS_ABILITIES` → `ABILITY_REGISTRY`. Remove `class_id` / `archetype_id` fields from `Ability` (the character's `known_abilities` list is now the sole gate for "what this character can use"). All 10 spells already have corresponding `ClassAbility` entries, so no new ability definitions are needed — only the redundant `c.spells = [...]` lines in archetype builders get removed.

**Tech Stack:** Python 3, dataclasses, pytest.

---

## File Inventory

**Modified:**
- `trpg/engine/abilities.py` — rename class, registry dict; remove `class_id`/`archetype_id` fields and all `class_id=...`/`archetype_id=...` kwargs in `_register` calls
- `trpg/engine/skill.py` — delete `from_spell()`, delete spells loop in `available_skills()`, remove archetype filter, rename `_from_class_ability` → `_from_ability`
- `trpg/engine/character.py` — delete `spells` field; update `rest_character` import
- `trpg/engine/combat.py` — replace `char.spells` usages (caster detection, concentration check, SPELL handler validation)
- `trpg/engine/combat_policy.py` — rename import
- `trpg/scenarios/archetypes.py` — delete all `c.spells = [...]` lines from 12 archetype factories
- `trpg/rl/env_v2.py` — rename import
- `trpg/rl/skill_probe.py` — rename import
- `trpg/sandbox/driver.py` — rename import
- `trpg/sandbox/setup.py` — rename imports

**Created:**
- `tests/engine/test_ability_unification.py` — regression suite for healing-correctness + no-duplicate-skills

---

## Task 1: Baseline regression tests

Establish tests that capture the behavior we want preserved across the refactor. These must PASS at HEAD (before any refactor) and continue to PASS after every later task. If they ever fail, the refactor broke something.

**Files:**
- Create: `tests/engine/test_ability_unification.py`

- [ ] **Step 1: Write the regression test file**

This test file deliberately avoids constructing a full `WorldState` so the tests are robust to peripheral API changes. `available_skills` accepts `world_state=None` and returns the skill list; builders are pure functions that return action dicts.

```python
"""Regression tests for the ability registry unification refactor.

These tests capture behavior that MUST hold both before and after the refactor:
  1. Healing skills produce HEAL actions (not SPELL actions).
  2. No archetype's available_skills returns duplicate display_names
     (catches the dual-path 治療術 + cure_wounds bug class).
  3. Every skill returned by available_skills has a builder that returns
     either None or a valid action dict (never raises).
"""
from __future__ import annotations
import pytest

from trpg.engine.skill import available_skills
from trpg.scenarios.archetypes import (
    make_life_cleric, make_war_cleric, make_devotion_paladin,
    make_vengeance_paladin, make_evocation_wizard, make_divination_wizard,
    make_berserker, make_totem_bear, make_battle_master, make_champion,
    make_assassin, make_arcane_trickster,
)


ARCHETYPE_BUILDERS = [
    make_life_cleric, make_war_cleric, make_devotion_paladin,
    make_vengeance_paladin, make_evocation_wizard, make_divination_wizard,
    make_berserker, make_totem_bear, make_battle_master, make_champion,
    make_assassin, make_arcane_trickster,
]


def test_cure_wounds_builder_produces_HEAL_action():
    caster = make_life_cleric("Caster", level=3)
    skills = available_skills(caster, world_state=None)
    healers = [s for s in skills if s.display_name == "治療術"]
    assert len(healers) >= 1, "Life cleric must offer 治療術"
    action = healers[0].builder("caster", "caster", None)
    assert action is not None
    assert action["type"] == "HEAL", \
        f"治療術 must build HEAL, got {action['type']}"


def test_healing_word_builder_produces_HEAL_action():
    caster = make_life_cleric("Caster", level=3)
    skills = available_skills(caster, world_state=None)
    healers = [s for s in skills if s.display_name == "治療語"]
    assert len(healers) >= 1
    action = healers[0].builder("caster", "caster", None)
    assert action is not None
    assert action["type"] == "HEAL"


@pytest.mark.parametrize("builder", ARCHETYPE_BUILDERS)
def test_no_duplicate_display_names_in_available_skills(builder):
    """After unification there must be exactly one entry per spell.
    Currently this test FAILS for wizards/clerics/paladins because the
    spells list and abilities list both register the same spell. The
    refactor makes this test pass for everyone."""
    caster = builder("Caster", level=5)
    skills = available_skills(caster, world_state=None)
    names = [s.display_name for s in skills]
    duplicates = {n for n in names if names.count(n) > 1}
    assert not duplicates, (
        f"{builder.__name__} has duplicate skill names: {duplicates}"
    )


@pytest.mark.parametrize("builder", ARCHETYPE_BUILDERS)
def test_every_skill_builder_returns_action_dict_or_none(builder):
    """Every skill returned by available_skills must have a builder that
    returns either None or a dict — never raises and never returns garbage."""
    caster = builder("Caster", level=5)
    skills = available_skills(caster, world_state=None)
    for sk in skills:
        action = sk.builder("caster", "enemy", (1.0, 0.0))
        assert action is None or isinstance(action, dict), (
            f"{builder.__name__}/{sk.skill_id} builder returned {type(action)}"
        )
```

- [ ] **Step 2: Run the duplicates test to confirm it currently FAILS for spellcasters**

Run: `python -m pytest tests/engine/test_ability_unification.py::test_no_duplicate_display_names_in_available_skills -v`

Expected: FAIL on wizards/clerics/paladins (they have duplicate entries: e.g., 燃燒之手 appears twice, once from spells list and once from abilities). This failure is EXPECTED and proves the test correctly captures the bug we're fixing.

- [ ] **Step 3: Run the healing tests to confirm they currently PASS**

Run: `python -m pytest tests/engine/test_ability_unification.py::test_cure_wounds_builder_produces_HEAL_action tests/engine/test_ability_unification.py::test_healing_word_builder_produces_HEAL_action -v`

Expected: PASS (cure_wounds and healing_word_life are already wired correctly via ClassAbility; we're verifying we don't break them).

- [ ] **Step 4: Commit baseline tests**

```bash
git add tests/engine/test_ability_unification.py
git commit -m "test: regression suite for ability registry unification"
```

---

## Task 2: Remove redundant `c.spells = [...]` lines from archetypes

After this task, no archetype assigns `c.spells`, so the spells-loop in `available_skills()` becomes dead code (still runs but `char.spells` is always the default empty list). The corresponding ClassAbility entries (burning_hands_ev, fireball_ev, cure_wounds, etc.) already provide the same skills, so available_skills output stays correct but no longer has duplicates.

**Files:**
- Modify: `trpg/scenarios/archetypes.py:163-491` (12 factory functions)

Each archetype factory currently has lines like:

```python
spells = ["燃燒之手"]
abilities = ["magic_missile", "burning_hands_ev"]
```

with a later `c.spells = spells`. We delete the `spells` local variable and the `c.spells = spells` assignment.

- [ ] **Step 1: Edit `make_evocation_wizard`**

Open `trpg/scenarios/archetypes.py` lines 163-201. Replace the function body so that:
- The local `spells = [...]` list (and any `spells.append(...)` / `spells.extend(...)` calls in level branches) is removed
- The `c.spells = spells` assignment line is removed
- `spell_slots`, `spellcasting_ability`, and `known_abilities` assignments stay
- All `abilities` modifications stay untouched

Concretely: open the file and delete every line that mentions a local `spells` variable in this function. Leave `spell_slots`, `spellcasting_ability`, and `abilities` as-is.

- [ ] **Step 2: Repeat for the other 11 archetype factories**

Apply the same deletion to each of:
- `make_divination_wizard` (lines 204-237)
- `make_life_cleric` (lines 247-280)
- `make_war_cleric` (lines 283-316)
- `make_arcane_trickster` (lines 358-392)
- `make_devotion_paladin` (lines 412-451)
- `make_vengeance_paladin` (lines 454-491)

The four melee archetypes (`make_battle_master`, `make_champion`, `make_totem_bear`, `make_berserker`, `make_assassin`) have no `spells` variable; verify they don't and leave them alone.

- [ ] **Step 3: Run the regression tests**

Run: `python -m pytest tests/engine/test_ability_unification.py -v`

Expected:
- `test_no_duplicate_display_names_in_available_skills` now PASSES for all archetypes (duplicates gone)
- `test_cure_wounds_builder_produces_HEAL_action` still PASSES
- `test_healing_word_builder_produces_HEAL_action` still PASSES
- `test_every_skill_builder_returns_action_dict_or_none` still PASSES

- [ ] **Step 4: Run the archetype catalog tests**

Run: `python -m pytest tests/engine/test_phase3_catalog.py -v`

Expected: PASS. These tests check that each archetype includes the expected abilities; they don't inspect `c.spells`, so removing it should not affect them.

- [ ] **Step 5: Commit**

```bash
git add trpg/scenarios/archetypes.py
git commit -m "refactor(archetypes): drop redundant spells list (covered by abilities)"
```

---

## Task 3: Delete `from_spell()` and the spells loop in `available_skills()`

Now `char.spells` is always empty, so the `for name in char.spells:` loop in `available_skills()` runs zero times. Delete it and `from_spell()`.

**Files:**
- Modify: `trpg/engine/skill.py:324-376` (delete `from_spell` function)
- Modify: `trpg/engine/skill.py:495-514` (delete the spells loop inside `available_skills`)

- [ ] **Step 1: Delete `from_spell()` function**

Open `trpg/engine/skill.py`. Locate `def from_spell(spell, char) -> Skill:` (around line 324). Delete the entire function body, ending at the closing `return Skill(...)` block (around line 376).

- [ ] **Step 2: Delete the spells loop inside `available_skills()`**

In the same file, locate `available_skills()` (around line 463). Inside it, find:

```python
    from .spells import SPELLS
    from .abilities import CLASS_ABILITIES
```

Remove `from .spells import SPELLS` (keep the abilities import for now).

Then locate the block:

```python
    if char.spells and char.spellcasting_ability:
        for name in char.spells:
            spell = SPELLS.get(name)
            if spell is None:
                continue
            # ... slot checks ...
            if spell.requires_concentration and char.concentrating_on:
                continue
            out.append(from_spell(spell, char))
```

Delete the entire `if char.spells and char.spellcasting_ability:` block.

- [ ] **Step 3: Run the regression tests**

Run: `python -m pytest tests/engine/test_ability_unification.py -v`

Expected: All 4 test functions PASS (output unchanged — removing dead code).

- [ ] **Step 4: Run all engine tests**

Run: `python -m pytest tests/engine/ -v`

Expected: PASS. Catches any test that imported `from_spell` directly (none expected, but verify).

- [ ] **Step 5: Commit**

```bash
git add trpg/engine/skill.py
git commit -m "refactor(skill): remove from_spell() and dead spells loop in available_skills"
```

---

## Task 4: Remove `char.spells` field from Character

Now the field is unused by `available_skills`. But `combat.py` still reads it in three places (caster detection, concentration check, SPELL handler validation). Fix those, then delete the field.

**Files:**
- Modify: `trpg/engine/combat.py:81` (`is_caster` detection)
- Modify: `trpg/engine/combat.py:1497` (SPELL handler validation)
- Modify: `trpg/engine/combat.py:2124-2129` (concentration check function)
- Modify: `trpg/engine/character.py:36` (delete `spells` field)
- Modify: `trpg/engine/character.py:188-224` (`rest_character` — only references `spell_slots`, not `spells`; verify)

- [ ] **Step 1: Replace `combat.py:81` caster detection**

Open `trpg/engine/combat.py`. Locate line 81:

```python
is_caster = bool(char.spells and char.spellcasting_ability)
```

Replace with:

```python
is_caster = bool(char.spellcasting_ability)
```

(After the refactor, having `spellcasting_ability` set is the canonical "this character is a caster" signal. `known_abilities` doesn't have to contain spells — but if `spellcasting_ability` is non-empty, the character was designed as a caster.)

- [ ] **Step 2: Remove the SPELL handler validation**

Locate `combat.py:1497`, inside the `SPELL` handler:

```python
if spell_name not in caster.spells:
    return {"type": "ERROR", "message": f"{caster.name} 不知道 {spell_name}"}
```

Delete this check entirely. The Ability builder is the only thing that produces SPELL actions, and it always specifies a spell the character knows (because the character has the corresponding ability in `known_abilities`).

- [ ] **Step 3: Find and fix the concentration check at combat.py:2124-2129**

Open the file around line 2124. The Explore agent reported this loops over `char.spells` to "check concentration requirements." Read the function. If it iterates `char.spells` to determine which concentration spells the character can break, rewrite it to iterate `char.known_abilities` and look up each ability's spell properties via its builder, OR delete the check if it's a stale defensive guard. Specifically:

- If the function returns "is this an active concentration spell" based on iterating spells, replace with: directly check `char.concentrating_on` (already tracked on the Character).
- If the function provides UI-facing data (e.g., listing concentration spells), rewrite to derive from `ABILITY_REGISTRY` entries whose `requires_concentration` would be true — but if you can't trace the consumer easily, prefer deleting the function and seeing what breaks.

Concretely: read combat.py lines 2120-2140 carefully. Make the smallest change that preserves observable behavior (e.g., the concentration tracking system, if any tests cover it).

- [ ] **Step 4: Delete `char.spells` field**

Open `trpg/engine/character.py`. At line 36, delete:

```python
spells: list = field(default_factory=list)        # list[str] — spell names; look up via engine.spells.SPELLS
```

- [ ] **Step 5: Run all engine tests**

Run: `python -m pytest tests/engine/ -v`

Expected: PASS. Any failure means a missed reference to `char.spells`.

- [ ] **Step 6: Run grep for any remaining references**

Run: `python -c "import subprocess; r = subprocess.run(['grep', '-rn', 'char.spells\\|\\.spells\\b', 'trpg/', '--include=*.py'], capture_output=True, text=True); print(r.stdout)"`

Or via PowerShell:

Run: `Select-String -Path trpg\**\*.py -Pattern '\.spells\b' -SimpleMatch:$false`

Expected: Only matches for `spell_slots` (which we keep) or `SPELLS` (the dict). No matches for `char.spells` or `.spells` attribute access on a Character.

If any match remains, fix it before continuing.

- [ ] **Step 7: Run full test suite**

Run: `python -m pytest tests/ -v --timeout=60`

Expected: PASS (excluding pre-existing failures unrelated to this refactor).

- [ ] **Step 8: Commit**

```bash
git add trpg/engine/character.py trpg/engine/combat.py
git commit -m "refactor(combat,character): remove char.spells field and references"
```

---

## Task 5: Rename `ClassAbility` → `Ability`, `CLASS_ABILITIES` → `ABILITY_REGISTRY`

Mechanical rename across the codebase. Affects all importers.

**Files:**
- Modify: `trpg/engine/abilities.py` — class definition, dict definition, `_register` body
- Modify: `trpg/engine/skill.py` — import + reference in `available_skills` + `_from_class_ability` → `_from_ability`
- Modify: `trpg/engine/character.py` — import in `rest_character`
- Modify: `trpg/engine/combat_policy.py` — import
- Modify: `trpg/rl/env_v2.py` — import
- Modify: `trpg/rl/skill_probe.py` — import
- Modify: `trpg/sandbox/driver.py` — import
- Modify: `trpg/sandbox/setup.py` — import

- [ ] **Step 1: Rename inside `abilities.py`**

Open `trpg/engine/abilities.py`. Replace:
- `class ClassAbility:` → `class Ability:`
- `CLASS_ABILITIES: dict[str, ClassAbility] = {}` → `ABILITY_REGISTRY: dict[str, Ability] = {}`
- Inside `_register(ab: ClassAbility)`: change type hint to `Ability`, change `CLASS_ABILITIES[ab.skill_id]` → `ABILITY_REGISTRY[ab.skill_id]`
- All `_register(ClassAbility(...))` calls → `_register(Ability(...))`. (60 entries; use find-and-replace.)

- [ ] **Step 2: Update all importers**

For each of the following files, update the import line:

- `trpg/engine/skill.py`: `from .abilities import CLASS_ABILITIES` → `from .abilities import ABILITY_REGISTRY` (two occurrences: in `materialize` and in `available_skills`). Update every `CLASS_ABILITIES` reference in the body to `ABILITY_REGISTRY`.
- `trpg/engine/character.py`: same rename in `rest_character`.
- `trpg/engine/combat_policy.py`: same rename.
- `trpg/rl/env_v2.py`: same rename.
- `trpg/rl/skill_probe.py`: same rename.
- `trpg/sandbox/driver.py`: same rename.
- `trpg/sandbox/setup.py`: same rename.

- [ ] **Step 3: Rename `_from_class_ability` → `_from_ability`**

In `trpg/engine/skill.py`, rename the function `_from_class_ability` to `_from_ability`. Update its single call site inside `available_skills`.

- [ ] **Step 4: Search for any missed references**

Run: `Select-String -Path trpg\**\*.py,tests\**\*.py -Pattern 'ClassAbility|CLASS_ABILITIES|_from_class_ability'`

Expected: zero matches. Fix any that remain.

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v --timeout=60`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add trpg/
git commit -m "refactor: rename ClassAbility -> Ability, CLASS_ABILITIES -> ABILITY_REGISTRY"
```

---

## Task 6: Remove `class_id` and `archetype_id` fields from `Ability`

The character's `known_abilities` list is now the sole source of truth for "which abilities this character has," so per-ability archetype filtering in `available_skills()` is redundant. Remove the fields entirely.

**Files:**
- Modify: `trpg/engine/abilities.py` — remove field definitions, remove all `class_id=...` and `archetype_id=...` kwargs from `_register(Ability(...))` calls
- Modify: `trpg/engine/skill.py` — remove the archetype filter in `available_skills`

- [ ] **Step 1: Remove the archetype filter from `available_skills`**

Open `trpg/engine/skill.py`, inside `available_skills()`. Locate the block:

```python
        if ab.archetype_id and ab.archetype_id != char.archetype_id:
            continue
```

Delete this block. The character only has abilities in `known_abilities` that the archetype factory put there, so this filter is redundant.

- [ ] **Step 2: Remove fields from the `Ability` dataclass**

Open `trpg/engine/abilities.py`. In the `Ability` dataclass, delete:

```python
    class_id: str                    # "fighter", "wizard", "cleric", etc.
    archetype_id: str = ""
```

- [ ] **Step 3: Remove `class_id=...` and `archetype_id=...` from all `_register(Ability(...))` calls**

There are 60 `_register(Ability(...))` calls. Each contains `class_id="..."` and may contain `archetype_id="..."`. Remove these kwargs.

A regex-based search-and-replace is the cleanest approach. Use your editor's find-and-replace with these patterns:

- Pattern: `^\s*class_id="[^"]*",\s*$` → empty (delete line)
- Pattern: `class_id="[^"]*",\s*` → empty (delete inline)
- Pattern: `^\s*archetype_id="[^"]*",\s*$` → empty (delete line)
- Pattern: `archetype_id="[^"]*",\s*` → empty (delete inline)

Verify no `Ability(` constructor call still has `class_id=` or `archetype_id=`.

- [ ] **Step 4: Run the regression suite**

Run: `python -m pytest tests/engine/test_ability_unification.py tests/engine/test_phase3_catalog.py -v`

Expected: PASS.

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v --timeout=60`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add trpg/engine/abilities.py trpg/engine/skill.py
git commit -m "refactor(abilities): drop class_id and archetype_id from Ability"
```

---

## Task 7: End-to-end sandbox smoke check

Validate that real combat still works end-to-end with all 12 archetypes (not just unit tests).

**Files:** (no code changes — verification only)

- [ ] **Step 1: Run a one-round sandbox combat for each spellcaster**

For each of: `life_cleric`, `war_cleric`, `devotion_paladin`, `vengeance_paladin`, `evocation_wizard`, `divination_wizard` — run a non-interactive sparring session and confirm no crashes and no `"type": "ERROR"` results except range-related ones.

Run (one example; repeat per archetype):

```bash
python scripts/sparring.py --model models/ppo_v9/ppo_best.pt --archetype life_cleric --auto --rounds 3 --log fight_smoke_life_cleric.jsonl
```

Expected: completes without crashing; the JSONL contains valid action/result entries for the caster.

Note: if `sparring.py` does not support `--auto` or `--archetype` flags, run the human-interactive version with simple inputs, or write a one-off Python snippet that loops `available_skills(caster, ws)` for each archetype and calls each ability's builder + `execute_action`.

- [ ] **Step 2: Confirm healing now works in sandbox**

Inspect `fight_smoke_life_cleric.jsonl` for a 治療術 or 治療語 action. Confirm:
- `"action": {"type": "HEAL", ...}` (NOT `{"type": "SPELL", ...}`)
- `"hp_after"` shows the target's HP increasing after the action

- [ ] **Step 3: Clean up smoke-test log files**

Run: `Remove-Item fight_smoke_*.jsonl`

- [ ] **Step 4: No commit needed (verification only)**

---

## Done Criteria

- All tests in `tests/engine/test_ability_unification.py` PASS
- All tests in `tests/engine/test_phase3_catalog.py` PASS
- Full `pytest tests/` runs green (excluding pre-existing failures)
- `grep` for `char.spells`, `from_spell`, `ClassAbility`, `CLASS_ABILITIES`, `_from_class_ability` returns ZERO matches in `trpg/`
- Sandbox smoke tests show healing spells producing HEAL actions and restoring HP
- 6 commits land on the branch, one per task (Task 7 has no commit)
