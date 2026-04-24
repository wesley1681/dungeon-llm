# LLM-based TRPG Game — Design Spec
Date: 2026-04-24

## Overview

A text-based tabletop RPG game running in a single terminal. The game has one AI GM, one human player, and one AI player. Rules-based mechanics (dice, combat, HP) are handled entirely in Python; LLMs handle only narrative and decision-making. D&D 5e rules are used as the foundation.

## Players & Roles

- **GM (AI)**: Narrates scenes, controls NPCs, embeds structured tags to trigger game engine calls
- **Human player**: Types actions freely in the terminal each turn
- **AI player (x1 for MVP)**: An LLM with its own character perspective; only sees what its character would know

## Architecture

Five independent modules:

```
Session Loop (main.py)
    │
    ├── LLM Layer
    │     ├── gm_agent.py       — GM LLM with full world context
    │     ├── player_agent.py   — AI player LLM with character-scoped context
    │     └── tag_parser.py     — extracts tags from GM output, dispatches to engine
    │
    ├── Game Engine (pure Python, no LLM)
    │     ├── dice.py           — dice notation parser and roller
    │     ├── character.py      — Character / Stats dataclasses
    │     ├── combat.py         — initiative, attack resolution, damage, saving throws
    │     └── world_state.py    — WorldState (single source of truth)
    │
    └── scenarios/
          └── dungeon.py        — pre-built scenario: characters, opening scene
```

**Core principle**: The Game Engine never calls an LLM. The GM LLM never directly mutates WorldState — it requests changes via tags that the tag parser and game engine resolve.

## Data Model

```python
@dataclass
class Stats:
    STR: int; DEX: int; CON: int
    INT: int; WIS: int; CHA: int

    def modifier(self, stat: str) -> int:
        return (getattr(self, stat) - 10) // 2

@dataclass
class Character:
    name: str
    race: str
    class_: str
    level: int
    stats: Stats
    hp: int
    max_hp: int
    ac: int
    spell_slots: dict[int, int]    # {spell_level: remaining}
    inventory: list[str]
    status_effects: list[str]      # e.g. ["poisoned", "stunned"]
    proficiencies: list[str]
    is_npc: bool

@dataclass
class CombatState:
    initiative_order: list[str]    # character names sorted by initiative roll
    current_turn_index: int
    round_number: int
    active: bool

@dataclass
class WorldState:
    characters: dict[str, Character]
    scene: str
    combat: Optional[CombatState]
    event_log: list[str]           # chronological log of all events (for save/load later)
    scenario_name: str
```

## Tag Protocol

The GM embeds structured tags in its narrative output. The tag parser extracts and resolves them before the text is shown to players.

| Tag | Meaning |
|-----|---------|
| `[ROLL: <character> <skill/save> DC<n>]` | Skill check or saving throw |
| `[ATTACK: <attacker> -> <target>]` | Attack roll vs target AC |
| `[DAMAGE: <dice> to <target>]` | Apply damage to target |
| `[HEAL: <character> <dice>]` | Restore HP |
| `[INITIATIVE: start]` | Begin combat, roll initiative for all combatants |
| `[STATUS: <character> +/-<effect>]` | Add or remove a status effect |

After resolving a tag, the result (e.g. "擲出 17，成功！") is injected back into the GM's context so it can continue the narrative with accurate information.

**Example GM output:**
```
地精揮刀衝向你！[ATTACK: Goblin -> Aria]
你勉強格開了攻擊，但對方緊接著又劈來一刀[DAMAGE: 1d6+2 to Aria]
```

## LLM Context Design

**GM Agent**
- System prompt: world background, D&D 5e narration style, tag usage instructions, current WorldState summary
- Message history: all player actions + resolved tag results
- Receives: all players' actions each turn
- Must output: narrative text with tags where mechanics apply

**AI Player Agent**
- System prompt: character sheet, personality description, "you only know what your character has witnessed"
- Message history: only the GM narration directed at or visible to this character
- Receives: GM narration (no info hidden from this character's perspective)
- Must output: a single in-character action or dialogue

## Turn Loop

```
1. GM narrates current scene
2. Tag parser resolves any tags in GM output → results injected to GM context
3. GM output (with tag results rendered as plain text) displayed to all
4. AI player generates action (based on its filtered context)
5. Human player types action
6. Both actions collected and sent to GM
7. Go to step 1
```

In combat, step 4-5 are gated by initiative order; each character acts on their turn.

## Pre-built Scenario: Dungeon

**Opening scene**: A party of two enters a torchlit dungeon to retrieve a stolen amulet.

**Pre-built characters (MVP):**

| Name | Class | Role |
|------|-------|------|
| 艾里亞 (Aria) | 盜賊 Lv3 | Human player |
| 索爾 (Thor) | 戰士 Lv3 | AI player |
| 地精巡邏隊 (Goblin x3) | — | NPC enemies |

**Story beats**: patrol encounter → locked door puzzle → boss room (hobgoblin captain)

## MVP Scope

### Included
- Dice engine (e.g. `2d6+3` notation)
- HP tracking, death at 0 HP
- Basic combat: initiative, attack roll vs AC, damage
- Ability checks and saving throws
- Status effects (applied/removed via tags)
- GM LLM agent with tag protocol
- AI player LLM agent with character-scoped context
- Single terminal turn loop
- Pre-built dungeon scenario with pre-built characters

### Excluded (post-MVP)
- Spell system
- Save / load
- Multiple AI players (>1)
- Character creation
- Multiple scenarios
- Experience points / leveling

## File Structure

```
trpg/
├── engine/
│   ├── dice.py
│   ├── character.py
│   ├── combat.py
│   └── world_state.py
├── llm/
│   ├── gm_agent.py
│   ├── player_agent.py
│   └── tag_parser.py
├── scenarios/
│   └── dungeon.py
└── main.py
```

## Future Considerations (not in scope)
- Save/load: WorldState is already serialisation-friendly (dataclasses → JSON)
- Spell system: add `spell_slots` consumption logic to game engine + new tags
- More AI players: player_agent.py is already designed to be instantiated multiple times
- Web UI: session loop in main.py can be extracted into an API layer
