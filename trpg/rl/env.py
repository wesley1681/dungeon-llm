"""Gymnasium-shape combat environment.

The class deliberately does NOT inherit from gymnasium.Env so the package has
no third-party RL dependency at import time. Method signatures and return
shapes match the gymnasium API exactly — wrapping it in `gymnasium.Env` later
takes one subclass declaration and two `Space` attributes.

Phase 1 scope (deliberately narrow; expand later):
  - Single agent (default: aria, a level-3 warrior)
  - Single opponent (goblin, melee)
  - No spells, no allies, no obstacles
  - Flat float32 observation, MultiDiscrete factored action

Episode = one combat encounter. Step = one sub-action by the agent. Between
agent decisions the environment auto-plays every other combatant via its
assigned CombatPolicy (HeuristicCombatPolicy by default — also the scripted
baseline for evaluating any trained policy).

Schema headroom planned for Phase 2 (not yet wired):
  - spell_id head (0..31) and target_entity head (0..9)
  - Per-entity slot vector (allies + enemies) instead of single enemy
  - Multi-channel obstacle image
The current schema is a strict subset — adding the wider heads later costs a
retrain but no engine changes.
"""
from __future__ import annotations
import random
from typing import Any

import numpy as np

from ..engine.character import Character, Stats, CombatState
from ..engine.world_state import WorldState
from ..engine.combat import (
    execute_action, consume_resources, setup_combat_positions, MOVE_BUDGET_M,
    tick_terrain_damage,
)
from ..engine.combat_policy import CombatPolicy, HeuristicCombatPolicy
from ..engine.items import WEAPON_DEFS
from ..engine.vec2 import TerrainType


# ── Schema constants (everything trainable depends on these being stable) ────

# MOVE target is quantised to an N×N grid covering the whole battlefield.
N_MOVE_CELLS = 10   # 30m / 10 = 3m per cell
N_ACTION_TYPES = 3  # 0=END, 1=ATTACK_nearest, 2=MOVE_to_cell
OBS_DIM = 10

# Safety caps mirror game.py.
_MAX_SUB_ACTIONS_PER_TURN = 5
_MAX_AGENT_STEPS_PER_EPISODE = 50


def _build_1v1_world(layout: str = "open") -> WorldState:
    """Build a fresh 1v1 encounter: aria (warrior, lvl 3) vs goblin.

    `layout` controls the battlefield terrain:
      - "open"          empty 30x30 field (default)
      - "walls"         two pillars between spawn points (LoS / pathing test)
      - "difficult"     wide band of difficult terrain in the middle
      - "lava"          two dangerous-terrain pools off-axis
    """
    agent = Character(
        name="aria", race="人類", class_="戰士", level=3,
        stats=Stats(STR=14, DEX=12, CON=12), hp=24, max_hp=24, ac=14,
        weapons=[WEAPON_DEFS["長劍"]],
    )
    enemy = Character(
        name="goblin", race="哥布林", class_="戰士", level=1,
        stats=Stats(STR=10, DEX=12), hp=10, max_hp=10, ac=12,
        weapons=[WEAPON_DEFS["短劍"]], is_npc=True, attitude=0,
    )
    ws = WorldState(
        characters={"aria": agent, "goblin": enemy},
        scene="rl_training",
        party_ids=["aria"], pc_ids=["aria"],
    )
    cs = CombatState(initiative_order=["aria", "goblin"])
    ws.combat = cs
    setup_combat_positions(ws, cs)

    bf = cs.battlefield
    if layout == "walls":
        bf.add_rect_obstacle(7.5, 11.5, 8.5, 13.5)
        bf.add_rect_obstacle(7.5, 16.5, 8.5, 18.5)
    elif layout == "difficult":
        bf.add_rect_terrain(8.0, 0.0, 10.0, 30.0, TerrainType.DIFFICULT)
    elif layout == "lava":
        bf.add_rect_terrain(9.0, 13.5, 12.0, 16.5, TerrainType.DANGEROUS)

    return ws


class CombatEnv:
    """Single-agent combat env. See module docstring."""

    # Static schema so downstream code (training scripts, network builders) can
    # introspect dimensions without instantiating.
    observation_dim: int = OBS_DIM
    action_dims: tuple[int, int, int] = (N_ACTION_TYPES, N_MOVE_CELLS, N_MOVE_CELLS)

    def __init__(self, agent_id: str = "aria",
                 opponent_policy: CombatPolicy | None = None,
                 layout: str = "open",
                 seed: int | None = None):
        self.agent_id = agent_id
        self.opponent_policy = opponent_policy or HeuristicCombatPolicy()
        self.layout = layout
        self._rng = random.Random(seed)
        self.ws: WorldState | None = None
        self.resources: dict = {}
        self._turn_idx: int = 0
        self._step_count: int = 0

    # ── Gym API ─────────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = random.Random(seed)
        self.ws = _build_1v1_world(self.layout)
        self.resources = {"action": 1, "movement": MOVE_BUDGET_M}
        self._turn_idx = 0
        self._step_count = 0
        # Tick start-of-turn terrain damage for whoever's turn comes first
        self._tick_terrain(self.ws.characters[self.ws.combat.initiative_order[0]])
        self._advance_to_agent_turn()
        return self._extract_obs(), {}

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.ws is None:
            raise RuntimeError("call reset() before step()")
        self._step_count += 1

        prev_agent_hp = self._hp(self.agent_id)
        prev_enemy_hp = self._enemy_hp()

        action_dict = self._decode_action(action)
        result: dict[str, Any] | None = None
        if action_dict is not None:
            result = execute_action(action_dict, self.ws)
            if result.get("type") != "ERROR":
                consume_resources(self.resources, action_dict, result)

        turn_done = (
            action_dict is None
            or (self.resources["action"] <= 0
                and self.resources["movement"] <= 1e-6)
        )
        if turn_done:
            self.resources = {"action": 1, "movement": MOVE_BUDGET_M}
            self._turn_idx = (self._turn_idx + 1) % len(self.ws.combat.initiative_order)
            if self._turn_idx == 0:
                self.ws.combat.round_number += 1
            self._advance_to_agent_turn()

        reward = self._compute_reward(prev_agent_hp, prev_enemy_hp)
        terminated = self._is_terminal()
        if terminated:
            reward += 5.0 if self._enemy_count() == 0 else -5.0
        truncated = self._step_count >= _MAX_AGENT_STEPS_PER_EPISODE

        return self._extract_obs(), reward, terminated, truncated, {"action_result": result}

    # ── Internal: turn cycling ──────────────────────────────────────────────

    def _tick_terrain(self, char: Character) -> int:
        """Apply start-of-turn ticks: refresh reaction budget then dangerous
        terrain damage. Returns the damage dealt (0 if none)."""
        char.reaction_used = False
        char.leveled_spell_cast_this_turn = False
        return tick_terrain_damage(char, self.ws.combat.battlefield)

    def _advance_to_agent_turn(self) -> None:
        """Auto-play every actor before the agent's turn comes round again."""
        cs = self.ws.combat
        order = cs.initiative_order
        agent_idx = order.index(self.agent_id)

        # Safety bound: each non-agent turn either makes progress or ends,
        # so |order| * MAX_SUB_ACTIONS is a hard ceiling per cycle.
        for _ in range(len(order) * _MAX_SUB_ACTIONS_PER_TURN + 10):
            if self._is_terminal():
                return
            if self._turn_idx == agent_idx:
                # Agent's turn starts — apply terrain tick before handing over.
                agent = self.ws.characters[self.agent_id]
                if agent.is_alive():
                    self._tick_terrain(agent)
                return

            cid = order[self._turn_idx]
            char = self.ws.characters.get(cid)
            if char and char.is_alive():
                self._run_npc_turn(cid, char)

            self._turn_idx = (self._turn_idx + 1) % len(order)
            if self._turn_idx == 0:
                cs.round_number += 1

    def _run_npc_turn(self, cid: str, char: Character) -> None:
        """Drain one full turn for an NPC via its policy. Includes a
        start-of-turn terrain damage tick."""
        self._tick_terrain(char)
        if not char.is_alive():
            return

        resources = {"action": 1, "movement": MOVE_BUDGET_M}
        for _ in range(_MAX_SUB_ACTIONS_PER_TURN):
            decision = self.opponent_policy.decide(
                cid, char, self.ws, resources, self.ws.combat.round_number,
            )
            if decision.fled:
                return
            if decision.action is not None:
                result = execute_action(decision.action, self.ws)
                if result.get("type") != "ERROR":
                    consume_resources(resources, decision.action, result)
            if decision.ended:
                return
            if resources["action"] <= 0 and resources["movement"] <= 1e-6:
                return

    # ── Internal: observation / action / reward ─────────────────────────────

    def _extract_obs(self) -> np.ndarray:
        agent = self.ws.characters[self.agent_id]
        bf = self.ws.combat.battlefield
        feats = [
            agent.position.x / bf.width,
            agent.position.y / bf.height,
            agent.hp / max(1, agent.max_hp),
            1.0 if self.resources["action"] > 0 else 0.0,
            self.resources["movement"] / MOVE_BUDGET_M,
        ]
        enemy = self._find_enemy()
        if enemy is not None:
            feats.extend([
                1.0,
                enemy.position.x / bf.width,
                enemy.position.y / bf.height,
                enemy.hp / max(1, enemy.max_hp),
                min(1.0, agent.position.distance_to(enemy.position) / bf.width),
            ])
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        return np.array(feats, dtype=np.float32)

    def _decode_action(self, action) -> dict | None:
        """[a_type, dx, dy] → engine action dict (or None for END)."""
        a_type, dx, dy = int(action[0]), int(action[1]), int(action[2])
        agent = self.ws.characters[self.agent_id]
        bf = self.ws.combat.battlefield

        if a_type == 0:
            return None
        if a_type == 1:
            enemy = self._find_enemy()
            if enemy is None or not agent.weapons:
                return None
            target_id = self._id_of(enemy)
            return {
                "type":     "ATTACK",
                "attacker": self.agent_id,
                "target":   target_id,
                "weapon":   agent.get_weapon().name,
                "consumes": ["action"],
            }
        if a_type == 2:
            cell_w = bf.width / N_MOVE_CELLS
            cell_h = bf.height / N_MOVE_CELLS
            tx = (dx + 0.5) * cell_w
            ty = (dy + 0.5) * cell_h
            return {
                "type":            "MOVE",
                "character":       self.agent_id,
                "target_position": [tx, ty],
                "consumes":        ["movement"],
            }
        return None

    def _compute_reward(self, prev_agent_hp: int, prev_enemy_hp: int) -> float:
        agent = self.ws.characters[self.agent_id]
        enemy = self._find_enemy()
        new_agent_hp = agent.hp
        new_enemy_hp = enemy.hp if enemy is not None else 0

        dmg_dealt  = max(0, prev_enemy_hp - new_enemy_hp)
        dmg_taken  = max(0, prev_agent_hp - new_agent_hp)

        # Normalised by max HP so reward magnitude doesn't scale with character size
        enemy_max  = enemy.max_hp if enemy is not None else 1
        reward = dmg_dealt / max(1, enemy_max) - dmg_taken / max(1, agent.max_hp)
        reward -= 0.01   # small step penalty discourages stalling
        return reward

    # ── Internal: world queries ─────────────────────────────────────────────

    def _hp(self, cid: str) -> int:
        return self.ws.characters[cid].hp

    def _enemy_hp(self) -> int:
        enemy = self._find_enemy()
        return enemy.hp if enemy is not None else 0

    def _enemy_count(self) -> int:
        return sum(
            1 for cid, c in self.ws.characters.items()
            if cid != self.agent_id and c.is_alive() and c.is_npc and c.attitude == 0
        )

    def _find_enemy(self) -> Character | None:
        for cid, c in self.ws.characters.items():
            if cid != self.agent_id and c.is_alive() and c.is_npc and c.attitude == 0:
                return c
        return None

    def _id_of(self, char: Character) -> str:
        for cid, c in self.ws.characters.items():
            if c is char:
                return cid
        raise ValueError("char not in world")

    def _is_terminal(self) -> bool:
        return (
            not self.ws.characters[self.agent_id].is_alive()
            or self._enemy_count() == 0
        )
