# Phase 2 RL Combat Environment Design

**Date:** 2026-05-19  
**Branch:** feat/2d-combat  
**Status:** Approved

---

## Overview

Upgrade the RL training environment from Phase 1 (10-d flat obs, 3 action types) to Phase 2 (structured Dict obs, full skill-based action space). The goal is to train a generalist combat model via Behavior Cloning (BC) followed by PPO free exploration, then fine-tune specialized variants per archetype or play style.

Training pipeline:
```
Expert policy rollouts → BC (supervised) → PPO (self-improvement)
                                              ↓
                              save checkpoints by win-rate
                              low / mid / boss-level model
                                              ↓
                              fine-tune specialized models
                              (aggressive, conservative, archetype-specific)
```

---

## Files

- **New:** `trpg/rl/env_v2.py` — `CombatEnvV2` class (this spec)
- **Keep:** `trpg/rl/env.py` — Phase 1 env, untouched (baseline reference)
- **New:** `trpg/rl/bc_collect.py` — BC rollout collection script
- **New:** `trpg/rl/train_bc.py` — BC training loop (PyTorch)
- **New:** `trpg/rl/train_ppo.py` — PPO training loop (PyTorch)

---

## Observation Space

Type: `Dict` (5 keys). Total flat size ≈ 1,473 floats.

```python
obs = {
    "skills":      np.float32[20, 53],   # SkillFeatures matrix
    "skill_mask":  np.float32[20],       # 1=valid skill, 0=padding
    "entities":    np.float32[6, 8],     # self + allies + enemies
    "resources":   np.float32[4],        # action economy
    "terrain":     np.float32[20, 20],   # battlefield grid (1.5m/cell)
}
```

### `skills` / `skill_mask`

Populated by calling `available_skills(actor, world_state)`, which returns the current actor's usable skills (filtered by level, archetype, ability uses). Each skill is encoded as its 53-d `SkillFeatures` vector via `features.materialize(actor, skill_id)`. The list is padded with zeros to exactly 20 slots; `skill_mask[i] = 0` for padding slots.

Slot 0 is always the `end` skill (END turn).

### `entities` — 6 rows × 8 columns

Row order (fixed every episode):
```
[0] self
[1] ally_1
[2] ally_2
[3] enemy_1   ← sorted by distance ascending
[4] enemy_2
[5] enemy_3
```
Missing slots (e.g. no allies) are zero-padded.

Per-entity features (8 values):
```
[0] hp_ratio          — hp / max_hp
[1] pos_x_norm        — position.x / battlefield.width
[2] pos_y_norm        — position.y / battlefield.height
[3] dist_norm         — distance to self / battlefield.width
[4] is_enemy          — 1.0 if hostile, 0.0 if ally/self
[5] is_alive          — 1.0 if alive
[6] is_self           — 1.0 for row 0 only
[7] has_debuff        — 1.0 if any of: paralyzed, restrained, stunned, poisoned
```

### `resources` — 4 values

```
[0] action_available      — 1.0 or 0.0
[1] bonus_action_available — 1.0 or 0.0
[2] movement_ratio        — resources["movement"] / MOVE_BUDGET_M
[3] round_norm            — min(round_number / 10.0, 1.0)
```

### `terrain` — 20×20 grid

Battlefield is 30m × 30m. Each cell = 1.5m × 1.5m.

```
 0.0 = open ground
 0.5 = difficult terrain (half movement)
 1.0 = obstacle / wall (impassable)
-1.0 = dangerous terrain (damage on entry)
```

Built once at episode start from `combat.battlefield`.

---

## Action Space

`MultiDiscrete([20, 6, 400])`

```python
action = [skill_idx, entity_idx, grid_cell]
```

| Head | Range | Meaning |
|------|-------|---------|
| `skill_idx` | 0–19 | Index into `available_skills()` list (0 = END) |
| `entity_idx` | 0–5 | Row index in `entities` obs (same ordering) |
| `grid_cell` | 0–399 | Flattened 20×20 grid index for position target |

**Interpretation depends on skill's `target_type`:**

| `TargetType` | Uses |
|-------------|------|
| `SELF` | ignore entity_idx and grid_cell |
| `SINGLE_ENEMY` / `SINGLE_ALLY` | use entity_idx → look up char_id |
| `POINT` | use grid_cell → decode to (x, y) world coords |
| `MULTI_*` | use entity_idx (simplified to single target) |

Invalid actions (e.g. entity_idx pointing to a dead slot, skill not available) are caught by `execute_action` and result in a small negative reward with no state change.

### `encode_action()` — for BC data collection

Reverse-maps an expert policy's action dict back to `[skill_idx, entity_idx, grid_cell]`:

1. Match action dict `type` + key fields to a skill in `available_skills()`
2. Look up `entity_idx` from the fixed entity ordering
3. Convert world coords to grid_cell index

---

## Episode Setup

```python
def reset(agent_arch=None, opponent_arch=None, level=None, layout=None):
    agent_arch    = agent_arch    or random.choice(ARCHETYPE_LIST)
    opponent_arch = opponent_arch or random.choice(ARCHETYPE_LIST)
    level         = level         or random.randint(3, 8)
    layout        = layout        or random.choice(["open","open","walls","difficult","lava"])
    # "open" appears twice → 50% chance open map, 50% varied terrain
```

- Agent character: `ARCHETYPE_FACTORIES[agent_arch](level=level)`
- Opponent character: `ARCHETYPE_FACTORIES[opponent_arch](level=level)`
- Opponent policy: `make_archetype_policy(opponent_arch)`
- Positions: `setup_combat_positions(ws, cs)`

---

## Reward Function

Applied every `step()`:

```python
reward  = dmg_dealt  / enemy_max_hp     # +1.0 max per step
reward -= dmg_taken  / self_max_hp      # -1.0 max per step
reward -= 0.01                           # step penalty (discourages stalling)

# Skill-use bonus — encourage active ability usage
if action used a non-trivial skill (not END / MOVE):
    reward += 0.05 * skill_features.expected_damage / enemy_max_hp
```

Terminal bonus (added once at episode end):
```python
if all enemies dead:  reward += 5.0
if agent dead:        reward -= 5.0
```

The skill-use bonus coefficient (0.05) can be annealed down during PPO to avoid over-incentivizing suboptimal skill use.

---

## BC Data Collection

`trpg/rl/bc_collect.py` runs expert policies to collect `(obs, action)` pairs:

```python
for archetype in ARCHETYPE_LIST:
    for episode in range(N_EPISODES_PER_ARCH):
        env.reset(agent_arch=archetype)
        expert = make_archetype_policy(archetype)
        while not done:
            expert_action_dict = expert.decide(actor_id, actor, ws, resources, round_num)
            encoded = env.encode_action(expert_action_dict)
            obs_t = env.current_obs()
            data.append((obs_t, encoded))
            env.step(encoded)
```

Output: `data/bc_rollouts.npz` — arrays of obs dicts + action arrays.

---

## Model Architecture (sketch — implemented in training scripts)

```
skills [20×53]  → TransformerEncoder (4 heads, 2 layers) → skill_emb [20×64]
                   ↓ mean-pool over valid skills
                   skill_ctx [64]

entities [6×8]  → MLP(8→32) per entity → entity_emb [6×32]
                   ↓ mean-pool
                   entity_ctx [32]

terrain [20×20] → Conv2d(1→8, k=3) → flatten → MLP → terrain_ctx [32]

resources [4]   → MLP(4→16) → resource_ctx [16]

concat([skill_ctx, entity_ctx, terrain_ctx, resource_ctx]) [144]
  → MLP(144→256→256)
  → 3 output heads:
      skill_logits  [20]   (softmax + skill_mask)
      entity_logits [6]    (softmax)
      grid_logits   [400]  (softmax)
```

---

## Model Checkpoint Strategy

Save checkpoints by **win-rate bracket** against a fixed set of opponent archetypes:

| Label | Win-rate vs. mixed opponents | Use in game |
|-------|------------------------------|-------------|
| `model_low` | 30–50% | Weak enemies |
| `model_mid` | 51–70% | Mid-game enemies |
| `model_boss` | 70%+ | Boss encounters |

Fine-tuned variants (from `model_boss` base):
- `model_<archetype>` — archetype-specific (e.g. `model_vengeance`)
- `model_aggressive` — reward shaped for max damage
- `model_conservative` — reward shaped for min damage taken

---

## Out of Scope (Phase 2)

- Monster RL model (monsters use expert policy as opponents for now)
- Multi-agent simultaneous decisions
- Gymnasium wrapping (can add later, one subclass)
- Encounter with more than 1 enemy vs 1 agent
