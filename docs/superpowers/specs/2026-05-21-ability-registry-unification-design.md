# Ability Registry Unification Design

## Goal

Eliminate the dual-path skill system (`char.spells` + `char.known_abilities`) by unifying all executable abilities into a single `ABILITY_REGISTRY`. Prevents future bugs where a skill is defined in one system but executed incorrectly by another.

## Background

The current system has two parallel registries:

- `SPELLS` dict (`spells.py`) + `char.spells` list → all go through `from_spell()` → always produce `{"type": "SPELL"}` action
- `CLASS_ABILITIES` dict (`abilities.py`) + `char.known_abilities` list → each has a custom builder → can produce any action type

This caused 治療術 and 治療語 to silently fail: they were added to `char.spells`, routed through the SPELL handler, found no `damage_dice`, and produced 0 damage instead of healing.

## Architecture

### Layer 1 — Ability Definition (`abilities.py`)

`ClassAbility` is renamed to `Ability`. The fields `class_id` and `archetype_id` are removed; they described "who can use it" rather than "what it does", and that responsibility now belongs entirely to the character's `known_abilities` list.

Remaining fields:

```
skill_id        — unique identifier, e.g. "spell:火球術", "cure_wounds"
display_name    — shown in UI and logs
features        — SkillFeatures numeric vector for RL obs
builder         — callable(actor_id, target_id, coord, char=None) → action dict
engine_ready    — False = not yet implemented, excluded from available_skills()
is_reaction     — True = not offered as active turn action
min_level       — character level required (safety check only)
max_uses        — uses per rest (0 = unlimited)
refresh_on      — "short_rest" | "long_rest" | "never"
is_usable       — optional callable for non-standard resource gates
```

`CLASS_ABILITIES` dict is renamed to `ABILITY_REGISTRY`.

### Layer 2 — Spell Data (`spells.py`)

`SPELLS` dict is retained as **pure reference data** only. It is used by:
- The obs encoder to read spell properties (damage_dice, aoe_radius_m, etc.)
- Ability builders in `abilities.py` to read spell attributes when constructing actions

`SPELLS` no longer participates in `available_skills()`. `from_spell()` is deleted.

### Layer 3 — Spell Registrations (`abilities.py`)

All 10 spells get `Ability` entries in `ABILITY_REGISTRY` with correct builders:

| Spell | skill_id | builder produces |
|-------|----------|-----------------|
| 火球術 | `spell:火球術` | `SPELL` (AOE save) |
| 神聖光輝 | `spell:神聖光輝` | `SPELL` (single target save) |
| 燃燒之手 | `spell:燃燒之手` | `SPELL` (AOE save) |
| 蜘蛛網 | `spell:蜘蛛網` | `SPELL` (AOE status) |
| 冰風暴 | `spell:冰風暴` | `SPELL` (AOE save) |
| 定身術 | `spell:定身術` | `SPELL` (single target status) |
| 定怪術 | `spell:定怪術` | `SPELL` (single target status) |
| 霧步 | `spell:霧步` | already exists as ClassAbility, keep as-is |
| 治療術 | `spell:治療術` | `HEAL` |
| 治療語 | `spell:治療語` | `HEAL` |

Save DC is computed dynamically in each builder using `char.spellcasting_ability` and `char.proficiency_bonus`, matching the existing `from_spell()` behaviour.

### Layer 4 — Character (`character.py`)

`char.spells` field is removed.

`char.spellcasting_ability` is retained — used by spell builders to compute save DC dynamically.

`char.known_abilities` is the single source of all executable skills for a character.

### Layer 5 — Archetypes (`archetypes.py`)

Each archetype builder replaces the `spells=` list with entries in `known_abilities`. Order convention (maintained by each archetype):

```
[MOVE implicit, weapons implicit, spells..., class abilities..., DODGE/HIDE implicit]
```

Example — Life Cleric before:
```python
spells = ["祝福術", "治療語", "神聖光輝"]
abilities = ["cure_wounds", "bless", "sacred_flame", "healing_word_life"]
```

After:
```python
known_abilities = ["spell:祝福術", "spell:治療語", "spell:神聖光輝",
                   "cure_wounds", "bless", "sacred_flame", "healing_word_life"]
```

### Layer 6 — `available_skills()` (`skill.py`)

The `char.spells` loop and `from_spell()` call are removed. The `class_id` / `archetype_id` filtering is removed. A single loop over `known_abilities`:

```python
for skill_id in (char.known_abilities or []):
    ab = ABILITY_REGISTRY.get(skill_id)
    if ab is None: continue
    if not ab.engine_ready or ab.is_reaction: continue
    if char.level < ab.min_level: continue
    # max_uses / spell slot checks (unchanged)
    out.append(_from_ability(ab, char))
```

`END`, `MOVE`, weapons, `DODGE`, `HIDE` continue to be added by their existing dedicated code paths (unchanged).

## Invariants

- Every skill that appears in `available_skills()` output must have an entry in `ABILITY_REGISTRY`.
- `SPELLS` dict entries must never be added to `char.known_abilities` directly; only `ABILITY_REGISTRY` entries are valid.
- A spell builder that reads `heal_dice` must produce `{"type": "HEAL"}`, not `{"type": "SPELL"}`.

## What Does NOT Change

- `SPELLS` dict structure and content
- `Spell` dataclass
- All engine action handlers (`SPELL`, `HEAL`, `APPLY_MOD`, etc.)
- RL obs encoding (reads `SkillFeatures` from `Skill` objects, unchanged)
- `pick_action`, `apply_resource_mask`, `apply_entity_mask`
- BC training data collection (uses `available_skills()` output, which has same Skill structure)

## Impact on Existing Models

Existing BC / PPO checkpoints become invalid because skill slot ordering changes. Retraining is required after this refactor.
