
"""Phase 2 RL combat environment.

Dict observation, MultiDiscrete action. See
docs/superpowers/specs/2026-05-19-rl-env-v2-design.md for the full spec.

Phase 1 (env.py) stays untouched so we can A/B compare baselines.
"""
from __future__ import annotations
import os
import random
from typing import Any

from ..engine.world_state import WorldState, CombatState
from ..engine.combat import (
    execute_action, consume_resources, setup_combat_positions,
    MOVE_BUDGET_M, tick_terrain_damage, tick_aura_damage, roll_death_save,
)
from ..engine.combat_policy import make_archetype_policy, run_legendary_actions
from ..engine.status import tick_status_effects
from ..scenarios.archetypes import ARCHETYPE_FACTORIES, STANDARD_ARCHETYPES
from .obs import build_obs
from .action import decode_action, ACTION_DIMS


# The standard-12 panel (frozen at archetypes.py module body) — NOT the live
# factory registry, so registering monsters/synths before importing this
# module can't leak them into random archetype sampling or eval panels.
ARCHETYPE_LIST: tuple[str, ...] = STANDARD_ARCHETYPES
LAYOUTS = ("open", "open", "walls", "difficult", "lava")

_MAX_SUB_ACTIONS_PER_TURN = 5

# Potential-based reward shaping coefficient. Dense per-step reward =
# COEF * (Φ(s') - Φ(s)) where Φ = Σ team hp-frac - Σ opp hp-frac. PBRS is
# policy-invariant (Ng et al. 1999 — the optimal policy is unchanged) but
# densifies the otherwise terminal-only signal (+5/-5/-10) so PPO gets a
# gradient EVERY step instead of once per episode. The ~41% win-rate plateau
# that survived every architecture / BC-seed / hparam change traces to this
# sparse signal — it is the one thing all those configs shared and never
# varied. The fixed-turn env makes episodes longer, diluting the terminal
# signal further, so dense shaping matters more now. 0.0 disables it (eval
# ignores reward anyway, so this only affects PPO training).
REWARD_SHAPING_COEF = 3.0

# Per-sub-action cost (the "cost of acting"). Every non-end sub-action the agent
# takes subtracts this from the step reward. This is a SINGLE archetype-agnostic
# scalar — it encodes nothing about any skill, class, or tactic, only the
# universal opportunity cost of spending a turn-action. Unlike the PBRS term
# (policy-invariant), this DELIBERATELY shifts the optimum toward efficiency:
# an action is only worth taking if its expected value (PBRS damage/heal, or the
# downstream value of an action it enables) exceeds the cost. The policy must
# DISCOVER for itself which actions clear that bar — e.g. that firing a free
# action-grant (action_surge) while out of reach buys an action it cannot
# convert, so it nets negative; while firing it after closing to melee converts
# to a real attack and nets positive. It is calibrated well below the value of a
# single landed attack (one weapon hit moves Φ by ~0.1-0.2 → ~0.3-0.6 reward at
# COEF=3) so productive actions stay clearly positive and only genuinely empty
# actions are discouraged. 0.0 disables it. Overridable via env var
# TRPG_ACTION_STEP_COST for tuning the single scalar without code edits.
#
# NOTE: a FLAT per-action cost was measured to fail — it kills the lowest
# IMMEDIATE-reward actions first (action_surge, second_wind heal — whose payoff
# is delayed/instrumental, so the critic under-credits them) while leaving the
# real waste untouched. Default 0 (disabled); kept only as a tunable knob.
ACTION_STEP_COST = float(os.environ.get("TRPG_ACTION_STEP_COST", "0.0"))

# Wasted-movement cost. A MOVE that fails to change the distance to the nearest
# living enemy by at least WASTED_MOVE_THRESH metres accomplished nothing — it
# neither closed for an attack nor opened to kite. Such "lateral / dithering"
# moves are advantage-NEUTRAL under the HP-based PBRS (no HP changes, next-state
# value ≈ current), so PPO neither rewards nor removes them and the policy keeps
# whatever mass the (multi-class-contaminated) BC prior put on them — measured
# on champion: 4.3 such moves/game vs the expert's 0.2, each one a wasted
# sub-action that crowds out an attack. This penalty makes ONLY those no-op
# moves negative-advantage, so PPO learns to drop them; a move that genuinely
# changes engagement range (closing OR kiting) pays nothing, and non-move
# actions (attacks, surge, heal) are never touched. It is a single global,
# archetype-agnostic rule keyed on the move's GEOMETRIC EFFECT — not on any
# class, skill, or hand-written tactic. 0.0 disables it.
WASTED_MOVE_COST = float(os.environ.get("TRPG_WASTED_MOVE_COST", "0.4"))
WASTED_MOVE_THRESH = 0.5

# Weighted team configs: (n_agents, n_opps, weight)
# Balanced(1v1,2v2,3v3) ≈55%, ±1(2v1,3v2,1v2,2v3) ≈36%, ±2(3v1,1v3) ≈9%
_TEAM_CONFIGS = [
    (1, 1, 4), (2, 2, 4), (3, 3, 4),   # balanced
    (2, 1, 2), (3, 2, 2),               # advantage
    (1, 2, 2), (2, 3, 2),               # disadvantage
    (3, 1, 1), (1, 3, 1),               # extreme
]
_TEAM_CONFIG_WEIGHTS = [c[2] for c in _TEAM_CONFIGS]


class CombatEnvV2:
    """Variable-size team combat env (up to 3v3) with Dict obs and skill-based
    action space.

    n_agents / n_opps: fixed team sizes (1-3 each). Pass None (default) to
    sample from _TEAM_CONFIGS each episode — balanced matchups most common,
    extreme rare.

    Model-controlled agents alternate turns against opponents whose turns are
    auto-resolved by their policy. The trainer sees a flat stream of
    (obs, action, reward, done) transitions keyed by env.current_agent_id.
    Step budget scales with team size: 25 steps per character.
    """

    action_dims = ACTION_DIMS

    def __init__(self, seed: int | None = None,
                 n_agents: int | None = None,
                 n_opps: int | None = None):
        self._rng = random.Random(seed)
        self._n_agents_fixed = n_agents
        self._n_opps_fixed = n_opps
        self.ws: WorldState | None = None
        self._agent_resources: dict[str, dict] = {}
        self._step_count = 0
        self._max_steps = 100
        self._initiative_order: list[str] = []
        self._initiative_pos: int = 0
        self._current_agent_id: str = ""
        self._agent_ids: tuple[str, ...] = ()
        self._opp_ids: tuple[str, ...] = ()
        self.agent_archs: list[str] = []
        self.opp_archs: list[str] = []
        self._opp_policies: dict[str, Any] = {}
        self._opponent_override = None
        self._reward_shaping = REWARD_SHAPING_COEF
        self._action_step_cost = ACTION_STEP_COST
        self._wasted_move_cost = WASTED_MOVE_COST
        self._prev_phi = 0.0

    @property
    def current_agent_id(self) -> str:
        return self._current_agent_id

    @property
    def agent_ids(self) -> tuple[str, ...]:
        return self._agent_ids

    @property
    def opp_ids(self) -> tuple[str, ...]:
        return self._opp_ids

    @property
    def resources(self) -> dict:
        return self._agent_resources.get(
            self._current_agent_id,
            {"action": 0, "bonus_action": 0, "movement": 0.0},
        )

    def use_heuristic_opponent(self) -> None:
        self._opponent_override = "heuristic"

    def use_self_play_opponent(self, net) -> None:
        self._opponent_override = net

    def reset(self, *, level: int | None = None, opp_level: int | None = None,
              layout: str | None = None,
              seed: int | None = None,
              agent_archs: list[str] | None = None,
              opp_archs: list[str] | None = None) -> tuple[dict, dict]:
        """``level`` sets the agent side; ``opp_level`` the opponent side
        (default: same as ``level`` — the historical symmetric behaviour).
        A cross-side gap creates asymmetric encounters (stomp / be stomped)."""
        if seed is not None:
            self._rng = random.Random(seed)
        lvl = level if level is not None else self._rng.randint(3, 8)
        opp_lvl = opp_level if opp_level is not None else lvl
        layout = layout or self._rng.choice(LAYOUTS)

        if self._n_agents_fixed is not None:
            n_agents = self._n_agents_fixed
            n_opps   = self._n_opps_fixed if self._n_opps_fixed is not None else n_agents
        else:
            cfg = self._rng.choices(_TEAM_CONFIGS, weights=_TEAM_CONFIG_WEIGHTS, k=1)[0]
            n_agents, n_opps = cfg[0], cfg[1]

        self._agent_ids = tuple(f"agent_{i}" for i in range(n_agents))
        self._opp_ids   = tuple(f"opp_{i}"   for i in range(n_opps))
        self._max_steps = 25 * (n_agents + n_opps)

        self.agent_archs = [
            (agent_archs[i] if agent_archs and i < len(agent_archs)
             else self._rng.choice(ARCHETYPE_LIST))
            for i in range(n_agents)
        ]
        self.opp_archs = [
            (opp_archs[i] if opp_archs and i < len(opp_archs)
             else self._rng.choice(ARCHETYPE_LIST))
            for i in range(n_opps)
        ]

        chars: dict = {}
        for aid, arch in zip(self._agent_ids, self.agent_archs):
            c = ARCHETYPE_FACTORIES[arch](level=lvl)
            c.char_id = aid
            c.is_npc = False
            chars[aid] = c
        for oid, arch in zip(self._opp_ids, self.opp_archs):
            c = ARCHETYPE_FACTORIES[arch](level=opp_lvl)
            c.char_id = oid
            c.is_npc = True
            c.attitude = 0
            chars[oid] = c

        all_ids = list(self._agent_ids) + list(self._opp_ids)
        self._rng.shuffle(all_ids)

        self.ws = WorldState(
            characters=chars, scene="rl_team",
            pc_ids=list(self._agent_ids), party_ids=list(self._agent_ids),
        )
        self.ws.combat = CombatState(
            active=True, initiative_order=all_ids, round_number=1,
        )
        setup_combat_positions(self.ws, self.ws.combat, rng=self._rng)
        self._apply_layout(layout)

        self._opp_policies = {}
        for oid, arch in zip(self._opp_ids, self.opp_archs):
            if self._opponent_override is None:
                self._opp_policies[oid] = make_archetype_policy(arch)
            elif self._opponent_override == "heuristic":
                from ..engine.combat_policy import HeuristicCombatPolicy
                self._opp_policies[oid] = HeuristicCombatPolicy()
            else:
                from .neural_policy import NeuralCombatPolicy
                self._opp_policies[oid] = NeuralCombatPolicy(
                    self._opponent_override, device="cpu")

        self._step_count = 0
        self._initiative_order = all_ids
        self._initiative_pos = 0
        self._agent_resources = {}

        if not self._find_first_agent():
            # Re-roll with the SAME episode spec (archs/levels included —
            # dropping them here used to silently swap the comp to random).
            return self.reset(seed=(seed or 0) + 99999,
                              level=level, opp_level=opp_level, layout=layout,
                              agent_archs=agent_archs, opp_archs=opp_archs)
        self._prev_phi = self._potential()
        return build_obs(self.ws, self._current_agent_id, self.resources), {}

    def _potential(self) -> float:
        """Φ(s) for potential-based shaping: Σ team hp-frac − Σ opp hp-frac.

        Rises when the agent damages opponents, falls when the team takes
        damage — so COEF·(Φ' − Φ) densely rewards favourable HP trades.
        """
        team = sum(max(0.0, self.ws.characters[a].hp)
                   / max(1, self.ws.characters[a].max_hp)
                   for a in self._agent_ids)
        opp = sum(max(0.0, self.ws.characters[o].hp)
                  / max(1, self.ws.characters[o].max_hp)
                  for o in self._opp_ids)
        return team - opp

    def _apply_layout(self, layout: str) -> None:
        if layout == "open":
            return
        from ..engine.vec2 import TerrainType
        bf = self.ws.combat.battlefield
        if layout == "walls":
            bf.add_rect_obstacle(7.5, 11.5, 8.5, 13.5)
            bf.add_rect_obstacle(7.5, 16.5, 8.5, 18.5)
        elif layout == "difficult":
            bf.add_rect_terrain(8.0, 0.0, 10.0, 30.0, TerrainType.DIFFICULT)
        elif layout == "lava":
            bf.add_rect_terrain(13.5, 13.5, 16.5, 16.5, TerrainType.DANGEROUS)

    def _find_first_agent(self) -> bool:
        """Scan from pos 0 to find first alive agent, auto-resolving preceding
        opponents. Returns False if all agents die before getting a turn."""
        n = len(self._initiative_order)
        for _ in range(n):
            cid = self._initiative_order[self._initiative_pos]
            c = self.ws.characters[cid]
            if cid in self._agent_ids and c.is_dying():
                roll_death_save(c)   # nat 20 → is_alive() 變 True，下行接手
                if not c.is_alive():
                    run_legendary_actions(self.ws, cid, self.ws.combat.round_number)
            if cid in self._agent_ids and c.is_alive():
                self._current_agent_id = cid
                self._begin_turn(cid)
                return True
            if cid in self._opp_ids and c.is_alive():
                self._run_opponent_turn(cid)
                if all(not self.ws.characters[aid].is_alive()
                       for aid in self._agent_ids):
                    return False
            self._initiative_pos = (self._initiative_pos + 1) % n
        return False

    def _nearest_enemy_dist(self, agent) -> float | None:
        """Distance from `agent` to the nearest living opponent, or None."""
        best = None
        for oid in self._opp_ids:
            o = self.ws.characters.get(oid)
            if o is not None and o.is_alive():
                d = agent.position.distance_to(o.position)
                if best is None or d < best:
                    best = d
        return best

    def _begin_turn(self, agent_id: str) -> None:
        agent = self.ws.characters[agent_id]
        agent.reaction_used = False
        agent.leveled_spell_cast_this_turn = False
        tick_status_effects(agent, "self_turn_start", self.ws.combat.round_number)
        tick_terrain_damage(agent, self.ws.combat.battlefield)
        tick_aura_damage(agent, self.ws, self.ws.combat.round_number)
        self._agent_resources[agent_id] = {
            "action": 1, "bonus_action": 1, "movement": MOVE_BUDGET_M,
        }
        # Sub-action counter for this turn. The agent — like the scripted
        # opponent in _run_opponent_turn — may take several sub-actions per
        # turn (move + attack + bonus). Capped at _MAX_SUB_ACTIONS_PER_TURN.
        self._turn_sub_actions = 0

    def _advance_to_next_agent(self) -> None:
        """Advance past the current initiative slot to find the next alive
        agent, auto-resolving opponents and calling round-end tick when the
        initiative order wraps back to position 0."""
        n = len(self._initiative_order)
        # Bound generously: a skipped incapacitated agent gets a save attempt
        # each time its slot comes round (tick self_turn_end below), so the
        # condition wears off within a few visits — but the loop must be able to
        # run several rounds of opponent turns + saves before giving up. n*3 was
        # too tight: an agent paralyzed on round 1 (opponent won initiative and
        # cast hold/stun) could exhaust it before saving, fall through to the
        # line-below default, and reach step() with no resources entry
        # (KeyError). n*64 covers any realistic incapacitation duration while
        # still terminating.
        for _ in range(n * 64):
            was_last = (self._initiative_pos == n - 1)
            self._initiative_pos = (self._initiative_pos + 1) % n
            if was_last:
                self._end_of_round_tick()
            cid = self._initiative_order[self._initiative_pos]
            c = self.ws.characters[cid]
            # Stop early if the match is already decided — nothing left to do.
            if (all(not self.ws.characters[a].is_alive() for a in self._agent_ids)
                    or all(not self.ws.characters[o].is_alive() for o in self._opp_ids)):
                break
            # Dying PC（0 HP，死亡豁免中）：輪到它的先攻位時擲一次死亡豁免
            # （與 game.py 敘事循環同一套引擎規則）。nat 20 以 1 HP 回神，
            # RAW 上當回合即可行動。
            if cid in self._agent_ids and c.is_dying():
                roll_death_save(c)
                if not c.is_alive():
                    # A dying PC's turn = its death save; the slot still ends
                    # a turn, so legendary creatures may react (5e RAW).
                    run_legendary_actions(self.ws, cid, self.ws.combat.round_number)
                    continue
            if cid in self._agent_ids and c.is_alive():
                # Incapacitated (status.INCAPACITATING_STATUSES): the agent
                # cannot act — skip its whole turn instead of handing control
                # to the policy. The opponent path (_run_opponent_turn) and the
                # scripted BC collector both already skip these turns; the
                # agent path did NOT, so a paralyzed agent was stepped and the
                # policy emitted actions that execute_action bounces with ERROR
                # — wasting up to _MAX_SUB_ACTIONS_PER_TURN sub-actions and
                # feeding the trainer garbage transitions. "Incapacitated
                # cannot act" is the game rule, not a per-class tactic.
                if c.is_incapacitated():
                    tick_status_effects(c, "self_turn_start", self.ws.combat.round_number)
                    tick_terrain_damage(c, self.ws.combat.battlefield)
                    tick_aura_damage(c, self.ws, self.ws.combat.round_number)
                    tick_status_effects(c, "self_turn_end", self.ws.combat.round_number)
                    run_legendary_actions(self.ws, cid, self.ws.combat.round_number)
                    continue
                self._current_agent_id = cid
                self._begin_turn(cid)
                return
            if cid in self._opp_ids and c.is_alive():
                self._run_opponent_turn(cid)
        # Fallthrough: the match is over (someone's whole side is down) or no
        # actionable agent remains. Point at an agent slot; if it is somehow
        # alive but never got a turn this episode, give it a proper begin_turn so
        # step()'s resource lookup can't KeyError.
        self._current_agent_id = self._agent_ids[0]
        if (self._current_agent_id not in self._agent_resources
                and self.ws.characters[self._current_agent_id].is_alive()):
            self._begin_turn(self._current_agent_id)

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        if self.ws is None:
            raise RuntimeError("call reset() before step()")
        self._step_count += 1
        cid = self._current_agent_id
        agent = self.ws.characters[cid]
        # Defensive: a fallthrough in _advance_to_next_agent (match already
        # decided, or an agent that never got a turn because it was incapacitated
        # from round 1) can leave no resources entry. Treat it as an exhausted
        # turn so the step resolves to a terminal/empty no-op instead of crashing.
        resources = self._agent_resources.get(
            cid, {"action": 0, "bonus_action": 0, "movement": 0.0})

        action_dict = decode_action(action, self.ws, cid)
        # Distance to nearest living enemy BEFORE acting — used to detect a MOVE
        # that changed nothing tactically (wasted-movement penalty below).
        # decode_action does not move the agent, so this is the pre-move state.
        enemy_dist_before = (
            self._nearest_enemy_dist(agent)
            if (action_dict is not None and action_dict.get("type") == "MOVE")
            else None)

        wasted_move = False
        result: dict[str, Any] | None = None
        if action_dict is not None:
            result = execute_action(action_dict, self.ws)
            if result.get("type") == "ERROR":
                resources["action"] = 0
            else:
                consume_resources(resources, action_dict, result)
                if action_dict.get("type") == "MOVE":
                    if (result.get("distance", 0) < 0.01
                            or resources["movement"] < 0.5):
                        resources["movement"] = 0.0
                    # A move that didn't change engagement range with the nearest
                    # enemy by WASTED_MOVE_THRESH metres did nothing useful.
                    if enemy_dist_before is not None:
                        d_after = self._nearest_enemy_dist(agent)
                        if (d_after is not None
                                and abs(d_after - enemy_dist_before)
                                    < WASTED_MOVE_THRESH):
                            wasted_move = True
                # Limited-use deduction moved INTO execute_action (the shared
                # funnel) — it was missing from the opponent/BC paths here,
                # giving them unlimited action_surge/rage. Do NOT re-add a
                # driver-level copy: it would double-deduct.

        # Turn ends when the agent chooses to end (None / end slot) OR all
        # resources are spent OR the sub-action cap is hit. This MIRRORS the
        # scripted opponent's _run_opponent_turn loop. The old rule ended the
        # turn the instant `action` hit 0, silently discarding any leftover
        # bonus action and movement — a ~12-point win-rate handicap unique to
        # the agent role (measured: expert-vs-expert agent_win rose 41%→54%
        # once this matched the opponent path). It also created a BC train/eval
        # mismatch: BC demonstrations are full multi-sub-action turns, but the
        # truncated env never let the policy execute the post-action sub-moves
        # it was cloned to make.
        self._turn_sub_actions = getattr(self, "_turn_sub_actions", 0) + 1
        all_resources_spent = (
            resources["action"] <= 0
            and resources["bonus_action"] <= 0
            and resources["movement"] <= 1e-6
        )
        turn_done = (
            action_dict is None
            or all_resources_spent
            or self._turn_sub_actions >= _MAX_SUB_ACTIONS_PER_TURN
        )
        if turn_done:
            tick_status_effects(agent, "self_turn_end", self.ws.combat.round_number)
            # Legendary actions (Wave 3): bosses react at the end of another
            # creature's turn. No legendary creature in play → no-op, no RNG.
            run_legendary_actions(self.ws, cid, self.ws.combat.round_number)
            self._advance_to_next_agent()

        opps_dead = all(not self.ws.characters[oid].is_alive() for oid in self._opp_ids)
        team_dead = all(not self.ws.characters[aid].is_alive() for aid in self._agent_ids)
        terminated = opps_dead or team_dead
        truncated = (not terminated and self._step_count >= self._max_steps)

        reward = 0.0
        # Cost of acting: every executed (non-end) sub-action pays the flat
        # per-action cost. `action_dict is None` is the end-turn choice and is
        # free. (Default 0 — measured to harm; see ACTION_STEP_COST note.)
        if self._action_step_cost and action_dict is not None:
            reward -= self._action_step_cost
        # Wasted-movement cost: a MOVE that didn't change engagement range with
        # the nearest enemy did nothing useful. Charged only to the agent (this
        # reward path is agent-only) and only to genuinely no-op moves, so it
        # leaves attacks / surge / heal and real closing/kiting moves untouched.
        if self._wasted_move_cost and wasted_move:
            reward -= self._wasted_move_cost
        if terminated:
            # A mutual kill (opps_dead AND team_dead in the same step — e.g. a
            # terrain/aura DOT drops the agent the same step it finishes the
            # opp) is NOT a win: eval_v2 requires the agent to survive. Rewarding
            # it +5 would train suicidal damage-trades. Win only if opps die and
            # the agent side is still standing.
            reward = 5.0 if (opps_dead and not team_dead) else -5.0
        elif truncated:
            reward = -10.0

        # Potential-based shaping: dense per-step HP-trade signal on top of the
        # sparse terminal reward. Policy-invariant, so it can't change the
        # optimal policy — only accelerates learning out of the BC basin.
        if self._reward_shaping:
            cur_phi = self._potential()
            reward += self._reward_shaping * (cur_phi - self._prev_phi)
            self._prev_phi = cur_phi

        return (
            build_obs(self.ws, self._current_agent_id, self.resources),
            float(reward), terminated, truncated,
            {"action_result": result},
        )

    def _run_opponent_turn(self, opp_id: str) -> None:
        opp = self.ws.characters[opp_id]
        opp.reaction_used = False
        opp.leveled_spell_cast_this_turn = False
        tick_status_effects(opp, "self_turn_start", self.ws.combat.round_number)
        tick_terrain_damage(opp, self.ws.combat.battlefield)
        tick_aura_damage(opp, self.ws, self.ws.combat.round_number)
        # Incapacitated (status.INCAPACITATING_STATUSES): skip the entire turn.
        # Otherwise the policy will spam invalid actions that execute_action
        # bounces with ERROR, wasting _MAX_SUB_ACTIONS_PER_TURN iterations.
        if opp.is_incapacitated():
            tick_status_effects(opp, "self_turn_end", self.ws.combat.round_number)
            run_legendary_actions(self.ws, opp_id, self.ws.combat.round_number)
            return
        resources = {"action": 1, "bonus_action": 1, "movement": MOVE_BUDGET_M}
        policy = self._opp_policies[opp_id]
        for _ in range(_MAX_SUB_ACTIONS_PER_TURN):
            if not opp.is_alive():
                break
            decision = policy.decide(
                opp_id, opp, self.ws, resources, self.ws.combat.round_number)
            if decision.fled or decision.action is None:
                break
            r = execute_action(decision.action, self.ws)
            if r.get("type") != "ERROR":
                consume_resources(resources, decision.action, r)
            if decision.ended:
                break
            if (resources["action"] <= 0 and resources["bonus_action"] <= 0
                    and resources["movement"] <= 1e-6):
                break
        tick_status_effects(opp, "self_turn_end", self.ws.combat.round_number)
        run_legendary_actions(self.ws, opp_id, self.ws.combat.round_number)

    def _end_of_round_tick(self) -> None:
        cs = self.ws.combat
        for cid in cs.initiative_order:
            c = self.ws.characters.get(cid)
            if c and c.is_alive():
                tick_status_effects(c, "round_end", cs.round_number)
        cs.round_number += 1
