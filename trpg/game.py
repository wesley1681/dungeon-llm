"""Core game flow used by the graphical application."""
import json
import queue
import threading
from dataclasses import dataclass, field
from typing import Any

from .engine.world_state import WorldState
from .engine.combat import (
    execute_action, format_result, make_saving_throw,
    consume_resources, MOVE_BUDGET_M, build_combat_context,
    tick_terrain_damage, tick_aura_damage, roll_death_save,
)
from .engine.combat_policy import CombatPolicy, HeuristicCombatPolicy, HumanInputPolicy
from .engine.quests import apply_reward, check_quest_progress, objective_progress_str
from .engine.status import tick_status_effects
from .llm.action_executor import execute_action_commands
from .llm.dialogue_flow import DialogueDirector, CheckResult


# ── Event types ───────────────────────────────────────────────────────────────

@dataclass
class TagResult:
    """Tag execution finished. ok = executed results, errors = bad formats."""
    ok: list[str]
    errors: list[str]

@dataclass
class StreamChunk:
    """One streaming token from an LLM agent."""
    source: str
    text: str
    thinking: bool = False
    actor: str = ""

@dataclass
class ActionResult:
    """A combat action was resolved."""
    actor: str
    summary: str
    debug: str = ""
    valid: bool = True

@dataclass
class RoundStart:
    number: int

@dataclass
class CombatStart:
    order: list[str]

@dataclass
class CombatEnd:
    loot: list[str]

@dataclass
class ExplorationPrompt:
    kaine: Any
    gm_text: str
    # Other PCs' remarks this turn, keyed by display name. Empty when no
    # other PCs spoke before the human's slot (eg. solo party).
    prior_remarks: dict[str, str] = field(default_factory=dict)

@dataclass
class CombatPrompt:
    kaine: Any
    enemies: dict[str, str]
    info_text: str = ""   # pre-formatted position / distance / weapon range block
    ctx: Any = None       # engine CombatContext for battlefield/skill/target rendering

@dataclass
class ConversationPrompt:
    npc_name: str
    kaine: Any
    attitude_label: str = ""

@dataclass
class StatusMessage:
    text: str

@dataclass
class QuestComplete:
    title: str
    giver_name: str

@dataclass
class GameOver:
    reason: str

@dataclass
class ConversationOutcome:
    game_over: bool = False
    skip_next_gm: bool = False
    triggered_combat: bool = False


_STOP = object()

# ── Combat turn structure ─────────────────────────────────────────────────────

# Safety cap: max sub-actions per character per round (prevents runaway loops)
_MAX_SUB_ACTIONS = 5


def format_kaine_combat_info(kaine, ctx) -> str:
    """Pre-format the combat status block shown to the human player.

    Reads everything from a CombatContext (built by engine.combat) so the
    function has no GameSession dependency.
    """
    from .engine.combat import CombatContext, MOVE_BUDGET_M
    resources = ctx.resources if isinstance(ctx, CombatContext) else {"action": 1, "movement": MOVE_BUDGET_M}
    action_status = "可用" if resources.get("action", 0) > 0 else "已用完"
    move_left = resources.get("movement", 0.0)

    weapon_parts = []
    for w in kaine.weapons:
        if w.range_type == "近戰":
            weapon_parts.append(f"{w.name}（近戰 {w.range_normal:.1f}m）")
        else:
            weapon_parts.append(
                f"{w.name}（遠程 {w.range_normal:.0f}m / 最大 {w.range_long:.0f}m）"
            )
    weapons_line = "、".join(weapon_parts) or "無武器"

    # enemies_str pre-formatted in ctx already includes positions & dodging
    enemies_block = ctx.enemies_str if ctx.enemies_str != "無" else "  - 無"
    if enemies_block != "  - 無":
        # Convert "A、B" → bullet list
        items = enemies_block.split("、")
        enemies_block = "\n".join(f"  - {it}" for it in items)

    conc_line = f"專注：{kaine.concentrating_on}\n" if kaine.concentrating_on else ""
    return (
        f"你的座標：({kaine.position.x:.1f}, {kaine.position.y:.1f})m\n"
        f"剩餘資源：動作 {action_status}、移動 {move_left:.1f}m\n"
        f"{conc_line}"
        f"你的武器：{weapons_line}\n"
        f"敵人：\n{enemies_block}\n"
        f"可選：攻擊、移動、閃避、躲藏、脫身、用道具、招式 <id> [args]；輸入「結束」或「end」結束本回合"
    )


# ── GameSession ───────────────────────────────────────────────────────────────

class GameSession:
    def __init__(self, world_state: WorldState, gm, tag_agent,
                 player_agents: dict = None,
                 npc_agents: dict = None,
                 policies: dict[str, CombatPolicy] | None = None,
                 default_combat_policy: CombatPolicy | None = None):
        self.world_state = world_state
        self.gm          = gm
        self.tag_agent   = tag_agent
        self.player_agents = player_agents or {}
        self.npc_agents  = npc_agents or {}

        # Jev-backed conversation decisions share TagAgent's connection/config.
        self.dialogue = DialogueDirector.from_agent(tag_agent, world_state)

        self._events    : queue.Queue = queue.Queue()
        self._player_in : queue.Queue = queue.Queue()
        self._stop_flag = threading.Event()
        # The prompt currently waiting for a human answer.  Meta-commands such
        # as /旁白 are handled on the UI thread and never enter _player_in, so
        # they must repeat this prompt after reporting their status; otherwise
        # front-ends wait forever (or leave their input widget disabled).
        self._pending_prompt = None

        # Combat decisions go through a policy per character. Human-controlled
        # characters get a human-input policy. Every other combatant — allies AND monsters
        # — defaults to `default_combat_policy` (the trained general model, a
        # shared NeuralCombatPolicy) when provided, else the scripted heuristic.
        # An explicit per-char entry in `policies` still overrides both.
        self.policies: dict[str, CombatPolicy] = dict(policies or {})
        non_human = default_combat_policy or HeuristicCombatPolicy()
        human_ids = {
            cid for cid, character in world_state.characters.items()
            if character.controller_type == "human"
        }
        # Compatibility for engine/tests that create Characters without the
        # scenario controller metadata.
        if not human_ids and "kaine" in world_state.characters:
            human_ids.add("kaine")
        for cid in world_state.characters:
            if cid in human_ids:
                self.policies.setdefault(cid, HumanInputPolicy(
                    char_id=cid,
                    prompt_fn=self._combat_prompt,
                    error_fn=self._combat_error,
                ))
            else:
                self.policies.setdefault(cid, non_human)

        from .llm.controllers import HumanController, LLMPlayerController, LLMNpcController
        self.controllers = {
            cid: HumanController(cid, self._get_input, self._emit)
            for cid in human_ids
        }
        for cid, agent in self.player_agents.items():
            self.controllers[cid] = LLMPlayerController(agent, self._emit)
        for cid, agent in self.npc_agents.items():
            self.controllers[cid] = LLMNpcController(agent, self._emit)

        # Each pending action retains its canonical actor id. TagAgent sends the
        # whole batch to Jev once and returns Python-built structured commands.
        self._tag_actions: list[dict[str, str]] = []

        # 旁白（GM narration）toggle — single source of truth. When False, the GM's
        # exploration prose and combat flavour are skipped entirely (no LLM call);
        # mechanics (tags, dice, damage, resource costs, ActionResult display) run
        # unchanged and the turn passes straight to the next actor. A testing aid.
        self.narrate = True

    # ── Public API ────────────────────────────────────────────────────────────

    # Meta-commands that toggle 旁白 rather than becoming a game action. Defined
    # once here so all GUI input routes get the same toggle for free.
    _NARRATION_CMDS = {"/旁白", "/narration", "/narr", "旁白"}

    def submit_player_input(self, text: str) -> None:
        if self._maybe_toggle_narration(text):
            return          # consumed as a meta-command, not queued as an action
        self._pending_prompt = None
        self._player_in.put(text)

    def _maybe_toggle_narration(self, text: str) -> bool:
        """Intercept the 旁白 on/off meta-command. Returns True if the input was a
        narration command (flipped the flag, not enqueued). `/旁白 on|off` sets it
        explicitly; a bare `/旁白` toggles. Mechanics are unaffected either way."""
        parts = text.strip().lower().split()
        if not parts or parts[0] not in self._NARRATION_CMDS:
            return False
        arg = parts[1] if len(parts) > 1 else ""
        if arg in ("on", "開", "開啟"):
            self.narrate = True
        elif arg in ("off", "關", "關閉"):
            self.narrate = False
        else:
            self.narrate = not self.narrate
        self._emit(StatusMessage(
            f"旁白已{'開啟' if self.narrate else '關閉'}（機制照常運作）"))
        if self._pending_prompt is not None:
            self._emit(self._pending_prompt)
        return True

    def next_event(self, block: bool = True, timeout: float | None = None):
        try:
            return self._events.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._stop_flag.set()
        self._player_in.put(_STOP)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _emit(self, event) -> None:
        if isinstance(event, (ExplorationPrompt, CombatPrompt, ConversationPrompt)):
            self._pending_prompt = event
        self._events.put(event)

    def _get_input(self) -> str | None:
        val = self._player_in.get()
        if val is _STOP or self._stop_flag.is_set():
            return None
        return val

    def _combat_prompt(self, actor, ctx) -> str | None:
        """HumanInputPolicy callback: emit CombatPrompt + block for input."""
        self._emit(CombatPrompt(
            kaine=actor, enemies=ctx.enemies,
            info_text=format_kaine_combat_info(actor, ctx),
            ctx=ctx,
        ))
        return self._get_input()

    def _combat_error(self, message: str) -> None:
        """HumanInputPolicy callback: relay parse errors to the UI."""
        self._emit(StatusMessage(f"指令無效：{message}"))

    def _alive_enemies(self) -> dict[str, str]:
        ws = self.world_state
        if ws.dungeon_map:
            room_ids = set(ws.dungeon_map.current_room.npc_ids)
            return {cid: c.name for cid, c in ws.characters.items()
                    if c.is_npc and c.is_alive() and c.attitude == 0 and cid in room_ids}
        return {cid: c.name for cid, c in ws.characters.items()
                if c.is_npc and c.is_alive() and c.attitude == 0}

    def _alive_human_pcs(self) -> dict[str, str]:
        """PCs that are not yet permanently dead (hp > 0 OR still making death
        saves). GameOver only when ALL PCs are is_dead()."""
        return {cid: c.name for cid, c in self.world_state.characters.items()
                if not c.is_npc and not c.is_dead()}

    def _run_legendary_phase(self, ended_cid: str, round_num: int) -> None:
        """Legendary actions at the end of `ended_cid`'s turn (Wave 3), plus
        narration of any world-level events (death throes) queued meanwhile."""
        from .engine.combat_policy import run_legendary_actions
        ws = self.world_state
        for ev in run_legendary_actions(ws, ended_cid, round_num):
            note = (f"⚡ 傳奇行動（花費 {ev['cost']}，剩餘 {ev['remaining']}）："
                    + format_result(ev["action"], ev["result"],
                                    ev["actor_name"]))
            self._emit(ActionResult(ev["actor_name"], note,
                                    "LEGENDARY_ACTION", valid=True))
            ws.log_event("system", note)
        events = getattr(ws, "pending_events", None)
        if events:
            for ev in events:
                if ev.get("type") == "DEATH_THROES":
                    note = (f"💥 {ev['source_name']} 死亡爆炸 → "
                            f"{ev['target_name']} 受 {ev['damage']} 點"
                            f"{ev['damage_type']}傷害")
                    self._emit(ActionResult(ev["source_name"], note,
                                            "DEATH_THROES", valid=True))
                    ws.log_event("system", note)
            events.clear()

    def _alive_party(self) -> dict[str, str]:
        """Alive party members that actually participate in combat.

        Used as the target pool for hostile NPC turns and for the GameOver
        wipe check at the party level.
        """
        ws = self.world_state
        return {cid: c.name for cid, c in ws.characters.items()
                if c.is_alive() and ws.is_combat_ally(cid)}

    def _last_gm_narration(self) -> str:
        """Most recent GM narration entry — used as scene context for conversation."""
        for e in reversed(self.world_state.narrative_log):
            if e["speaker"] == "gm":
                return e["text"]
        return ""

    # ── Game loop ─────────────────────────────────────────────────────────────

    def _run(self) -> None:
        skip_gm = False
        while not self._stop_flag.is_set():
            done, skip_gm = self._exploration_turn(skip_gm=skip_gm)
            if done:
                return

    def _exploration_turn(self, skip_gm: bool = False) -> tuple[bool, bool]:
        """Run one exploration turn.

        Returns (done, skip_gm_next_turn):
          done — True if game should end (GameOver / quit)
          skip_gm_next_turn — True if the next turn should skip GM narration
                              (e.g. after conversation ended in combat)
        """
        ws = self.world_state

        # ── TagAgent ─────────────────────────────────────────────────────────
        def tag_chunk(c, thinking=False):
            self._emit(StreamChunk("tag", c, thinking))

        commands = self.tag_agent.generate_commands(
            self._tag_actions, on_chunk=tag_chunk,
        )
        ok, errors = execute_action_commands(
            commands, ws, log_to_narrative=True,
        )
        self._emit(TagResult(ok, errors))
        self._check_quests()
        # ── Conversation? (NpcAgent handles the opening; GM is skipped) ──────
        if ws.pending_conversation:
            npc_id = ws.pending_conversation
            ws.pending_conversation = ""
            if npc_id in self.npc_agents:
                outcome = self._run_conversation(npc_id)
                if outcome.game_over:
                    return True, False
                return False, outcome.skip_next_gm

        # ── GM narrative ─────────────────────────────────────────────────────
        # skip_gm: transient (after conversation→combat). self.narrate: the 旁白
        # mode toggle. Either one collapses this to an empty, unlogged narration.
        if skip_gm or not self.narrate:
            gm_text = ""
        else:
            def _gm_chunk(c, thinking=False):
                self._emit(StreamChunk("gm", c, thinking))

            gm_text = self.gm.generate(
                tag_results=ok or None,
                on_chunk=_gm_chunk,
            )
            ws.log_event("gm", gm_text)
            ws.event_log.append(f"GM: {gm_text}")

        # ── Combat? ───────────────────────────────────────────────────────────
        if ws.combat and ws.combat.active:
            combat_log = self._run_combat()
            if self._stop_flag.is_set():
                return True, False
            ws.log_event("system", "戰鬥結束" if combat_log else "（戰鬥結束）")
            self._tag_actions = []
            return False, False

        # ── Death check ───────────────────────────────────────────────────────
        for char in ws.characters.values():
            if not char.is_npc and char.is_dead():
                self._emit(GameOver(f"{char.name} 死亡！遊戲結束。"))
                return True, False

        # ── PC turns ──────────────────────────────────────────────────────────
        # Iterate ws.pc_ids in order. Each controller decides how to fill the
        # slot (LLM PC streams generated text, HumanController emits
        # ExplorationPrompt and blocks for input). Following/joined NPCs in
        # party_ids but NOT in pc_ids are silent automatically.
        prior_remarks: dict[str, str] = {}
        tag_actions: list[dict[str, str]] = []
        for cid in ws.pc_ids:
            char = ws.characters.get(cid)
            ctrl = self.controllers.get(cid)
            if char is None or ctrl is None or not char.is_alive():
                continue
            output = ctrl.take_exploration_turn(
                char, gm_text=gm_text, prior_remarks=prior_remarks,
            )
            if output.quit:
                self._emit(GameOver("冒險結束。再見！"))
                return True, False
            if not output.text:
                continue
            ws.log_event(cid, output.text)
            ws.event_log.append(f"{char.name}：{output.text}")
            prior_remarks[char.name] = output.text
            tag_actions.append({"actor_id": cid, "text": output.text})
        self._tag_actions = tag_actions
        return False, False

    # ── Combat ────────────────────────────────────────────────────────────────

    def _run_combat(self) -> list[str]:
        ws     = self.world_state
        combat = ws.combat
        log: list[str] = []

        self._emit(CombatStart(
            [ws.characters[c].name for c in combat.initiative_order
             if c in ws.characters]
        ))

        while not self._stop_flag.is_set():
            enemies = self._alive_enemies()
            pcs     = self._alive_human_pcs()

            if not enemies:
                ws.combat.active = False
                # Tick combat_end on all surviving combatants
                for ccid in combat.initiative_order:
                    c = ws.characters.get(ccid)
                    if c and c.is_alive():
                        tick_status_effects(c, "combat_end", combat.round_number)
                # Cleanup: remove dead follower NPCs from party
                for npc_id in list(ws.party_ids):
                    c = ws.characters.get(npc_id)
                    if c and c.is_npc and not c.is_alive():
                        ws.party_ids.remove(npc_id)
                        if npc_id in ws.combat_party_ids:
                            ws.combat_party_ids.remove(npc_id)
                        ws.follow_targets.pop(npc_id, None)
                        agent = self.npc_agents.get(npc_id)
                        if agent is not None:
                            agent.in_party = False
                loot = []
                if ws.dungeon_map:
                    room = ws.dungeon_map.current_room
                    room.cleared = True
                    loot = room.loot_names()
                self._emit(CombatEnd(loot))
                self._check_quests()
                return log

            if not pcs:
                ws.combat.active = False
                self._emit(GameOver("全員倒下！遊戲結束。"))
                return log

            combat.round_number += 1
            self._emit(RoundStart(combat.round_number))

            for cid in combat.initiative_order:
                if self._stop_flag.is_set():
                    return log
                char = ws.characters.get(cid)
                if char is None or char.is_dead():
                    continue

                # ── Dying PC: death saving throw (shared engine rules) ────────
                if char.is_dying():
                    save = roll_death_save(char)
                    if save["outcome"] != "skipped_stable":
                        self._emit(ActionResult(char.name, save["note"],
                                                "DEATH_SAVE", valid=True))
                        ws.log_event("system", save["note"])
                    self._run_legendary_phase(cid, combat.round_number)
                    continue

                # ── Normal turn ───────────────────────────────────────────────
                # Phase: start of this character's turn
                tick_status_effects(char, "self_turn_start", combat.round_number)
                char.reaction_used = False
                terrain_dmg = tick_terrain_damage(char, combat.battlefield)
                if terrain_dmg > 0:
                    note = f"{char.name} 因危險地形受到 {terrain_dmg} 點傷害（HP {char.hp}/{char.max_hp}）"
                    self._emit(ActionResult(char.name, note, "TERRAIN", valid=True))
                    ws.log_event("system", note)
                    if char.is_dead():
                        continue
                # Hostile auras + status-tick damage (spirit guardians,
                # swallowed acid). Every other driver (env_v2 / sandbox /
                # bc_collect) already ran this at turn start; the campaign
                # loop was the one place missing it.
                for ev in tick_aura_damage(char, ws, combat.round_number):
                    src = ev.get("source_name") or ev.get("status_name", "")
                    if ev.get("type") == "FRIGHTFUL_PRESENCE":
                        note = (f"{char.name} 面對 {src} 的恐懼威壓"
                                f"（WIS DC{ev['save_dc']}）："
                                + ("豁免成功，本場免疫"
                                   if ev["save_success"] else "陷入恐懼"))
                    else:
                        note = (f"{char.name} 受到 {src} 的持續傷害 {ev['damage']} 點"
                                f"（HP {char.hp}/{char.max_hp}）")
                    self._emit(ActionResult(char.name, note, "TICK", valid=True))
                    ws.log_event("system", note)
                if char.is_dead():
                    continue

                # Per-turn resource budget. Sub-actions decrement these.
                resources = {"action": 1, "bonus_action": 1, "movement": MOVE_BUDGET_M}
                stop_round = self._take_combat_turn(cid, char, resources, combat.round_number, log)

                # Phase: end of this character's turn
                tick_status_effects(char, "self_turn_end", combat.round_number)
                self._run_legendary_phase(cid, combat.round_number)
                if stop_round == "quit":
                    return log
                if not self._alive_enemies() or not self._alive_human_pcs():
                    break

            # Phase: end of round (after initiative cycle)
            for ccid in combat.initiative_order:
                c = ws.characters.get(ccid)
                if c and c.is_alive():
                    tick_status_effects(c, "round_end", combat.round_number)
            # Mid-combat follower cleanup: dead NPCs leave the party immediately,
            # so is_party_ally() returns false for them in the next round.
            for npc_id in list(ws.party_ids):
                c = ws.characters.get(npc_id)
                if c and c.is_npc and not c.is_alive():
                    ws.party_ids.remove(npc_id)
                    if npc_id in ws.combat_party_ids:
                        ws.combat_party_ids.remove(npc_id)
                    ws.follow_targets.pop(npc_id, None)
                    agent = self.npc_agents.get(npc_id)
                    if agent is not None:
                        agent.in_party = False

        return log

    def _take_combat_turn(self, cid: str, char, resources: dict,
                          round_num: int, log: list) -> str:
        """Run a multi-step combat turn for one character via its CombatPolicy.

        Loop ends on policy returning ended/fled, resources exhausted, safety
        cap, or combat end. Returns "quit" if the stop flag is set, "" otherwise.
        """
        ws = self.world_state
        policy = self.policies.get(cid)
        if policy is None:
            return ""

        for _ in range(_MAX_SUB_ACTIONS):
            if self._stop_flag.is_set():
                return "quit"
            if not char.is_alive():
                return ""
            if not self._alive_enemies() or not self._alive_human_pcs():
                return ""

            decision = policy.decide(cid, char, ws, resources, round_num)

            if decision.fled:
                self._handle_flee(cid, char)
                return ""

            result_text = ""
            if decision.action is not None:
                result_text = self._execute_sub_action(
                    cid, char, decision.action, resources
                )

            if result_text:
                log.append(result_text)
                if self.narrate:   # 旁白 off → skip flavour; mechanics already emitted
                    self.gm.combat_narrate(
                        result_text,
                        on_chunk=lambda c, thinking=False: self._emit(StreamChunk("narrate", c)),
                    )

            if decision.ended:
                return ""
            if resources.get("action", 0) <= 0 and resources.get("movement", 0) <= 1e-6:
                return ""
        return ""

    def _handle_flee(self, cid: str, char) -> None:
        execute_action_commands(
            [{"action": "flee", "actor_id": cid}], self.world_state,
        )
        summary = f"{char.name} 逃離了戰鬥"
        self._emit(ActionResult(char.name, summary, "FLEE", valid=True))
        self.world_state.log_event("system", summary)
        agent = self.npc_agents.get(cid)
        if agent is not None:
            agent.in_party = False

    def _execute_sub_action(self, cid, char, action: dict, resources) -> str:
        """Validate → execute → emit. Returns result_text ("" on rejection).

        The action is a structured dict from a CombatPolicy. Engine ERRORs and
        resource-exhausted cases are emitted as invalid ActionResults; the
        policy can choose to react on its next decide() call, so no controller
        retry loop is needed.
        """
        ws = self.world_state
        debug = json.dumps(action, ensure_ascii=False)

        if "action" in action.get("consumes", []) and resources.get("action", 0) <= 0:
            reason = "本回合動作已用完"
            self._emit(ActionResult(char.name, f"{char.name}：{reason}", debug, valid=False))
            return ""

        result = execute_action(action, ws)
        if result.get("type") == "ERROR":
            reason = result["message"]
            self._emit(ActionResult(char.name, f"{char.name}：{reason}", debug, valid=False))
            return ""

        summary = format_result(action, result, char.name)
        self._emit(ActionResult(char.name, summary, debug, valid=True))
        ws.log_event("system", summary)
        consume_resources(resources, action, result)
        # No-op-move guard — MIRROR of env_v2._step_agent (env_v2.py ~590-593),
        # the turn-loop the combat model was TRAINED against. A MOVE that travels
        # ~0m or leaves <0.5m budget ends movement for the turn, so the model
        # can't spin on "move to my own cell" wasting its whole turn. The RL env
        # applies this in its own turn loop; the narrative loop (this method) is
        # a separate turn loop and must replicate it to present the model the
        # same resource dynamics. (Canonical source: env_v2.py:590-593.)
        if action.get("type") == "MOVE" and (
                result.get("distance", 0) < 0.01
                or resources.get("movement", 0) < 0.5):
            resources["movement"] = 0.0
        return summary

    def _check_quests(self) -> None:
        """Check for newly-completed quests; emit events and push notes to log."""
        completed = check_quest_progress(self.world_state)
        for q in completed:
            giver = self.world_state.characters.get(q.giver_id)
            giver_name = giver.name if giver else q.giver_id
            self._emit(QuestComplete(q.title, giver_name))
            note = f"【系統通知：任務「{q.title}」已達成完成條件，可回去找 {giver_name} 回報】"
            self.world_state.log_event("system", note)

    # ── NPC Conversation ──────────────────────────────────────────────────────

    def _run_conversation(self, npc_id: str) -> ConversationOutcome:
        """Multi-turn NPC conversation. Returns ConversationOutcome flags for the main loop.

        Conversation lines are pushed to the unified narrative_log. NPC speech
        flows through LLMNpcController.take_npc_opening/take_npc_response; PC
        speech flows through each pc_ids controller's take_conversation_turn
        (LLM PCs may choose [SILENT]; humans block for input).
        """
        outcome   = ConversationOutcome()
        ws        = self.world_state
        npc_agent = self.npc_agents[npc_id]
        npc_ctrl  = self.controllers[npc_id]
        npc_char  = ws.characters[npc_id]

        # The player input that opened this conversation (e.g. "走向老柯說話")
        # has now been fully consumed by the talk command that brought us here. Clear
        # it immediately so that leaving/quitting — which take early-return paths
        # out of the loop below — can't leave it queued for TagAgent to
        # re-classify into another talk command next turn (== you can never leave).
        self._tag_actions = []

        # Approach cue: re-entry uses a different opening prompt
        prior = [e for e in ws.narrative_log if e["speaker"] == npc_id]
        if prior:
            ws.log_event("system", f"（冒險者再度向 {npc_char.name} 走來）")
        else:
            ws.log_event("system", f"（冒險者向 {npc_char.name} 走近，看著他）")

        # ── NPC opening ───────────────────────────────────────────────────────
        npc_text = npc_ctrl.take_npc_opening(npc_char)
        ws.log_event(npc_id, npc_text)
        if self._apply_npc_dialogue_actions(npc_text, npc_agent, npc_char):
            outcome.skip_next_gm = True
            outcome.triggered_combat = True
            return outcome
        while not self._stop_flag.is_set():
            # ── Each PC takes a conversation turn (LLM may go silent, human
            #     blocks until input or "離開") ───────────────────────────────
            pc_lines: list[str] = []
            pc_actions: list[dict[str, str]] = []
            for cid in ws.pc_ids:
                char = ws.characters.get(cid)
                ctrl = self.controllers.get(cid)
                if char is None or ctrl is None or not char.is_alive():
                    continue
                output = ctrl.take_conversation_turn(
                    char, npc_char, npc_agent.attitude_label,
                )
                if output.quit:
                    outcome.game_over = True
                    return outcome
                if output.leave:
                    npc_agent.pending_join_offer = None
                    self._emit(StatusMessage(f"你結束了與 {npc_char.name} 的對話。"))
                    ws.log_event("system", f"（{char.name} 離開了）")
                    return outcome
                if output.silent or not output.text:
                    continue
                ws.log_event(cid, output.text)
                pc_lines.append(f"{char.name}：{output.text}")
                pc_actions.append({
                    "actor_id": cid,
                    "actor_name": char.name,
                    "text": output.text,
                })

            if not pc_lines:
                # All PCs silent (rare — only happens when no human PC is
                # present and every LLM PC emitted [SILENT]). Loop back so
                # the NPC isn't asked to respond to silence.
                continue

            # ── 3-stage conversation resolution (dialogue_flow) ───────────────
            combined = "\n".join(pc_lines)
            player_started_combat = False
            for player_action in pc_actions:
                player_events = self.dialogue.judge_player_actions(
                    player_action, npc_agent, ws)
                player_events = self.dialogue.resolve_dialogue_actions(
                    player_action["text"], player_events, ws)
                if self._apply_player_dialogue_actions(
                        player_events, npc_agent, npc_char):
                    player_started_combat = True
                    break
            if player_started_combat:
                outcome.skip_next_gm = True
                outcome.triggered_combat = True
                break

            # Stage 1: does this need a check? which / who / DC?
            plan = self.dialogue.decide_check(pc_actions, npc_agent, ws)
            result = None
            if plan.check:
                actor = ws.characters.get(plan.actor) or ws.characters.get("kaine")
                success, total = make_saving_throw(actor, "CHA", plan.dc)
                result = CheckResult(plan.check, success, total, plan.dc, plan.actor)
                note = (f"【{plan.check}檢定：{actor.name} 擲出 {total} vs DC {plan.dc}"
                        f" → {'成功' if success else '失敗'}】")
                ws.log_event("system", note)
                self._emit(StreamChunk("npc_talk", f"\n{note}\n", actor="系統"))
                npc_agent.pending_directives.append(
                    f"玩家的{result.check}檢定{'成功' if result.success else '失敗'}。"
                    "請依照這個結果與你的個性自然回應；"
                    "檢定失敗時不要做出玩家想要的讓步。"
                )

            # ── NPC response ──────────────────────────────────────────────────
            npc_text = npc_ctrl.take_npc_response(npc_char)
            ws.log_event(npc_id, npc_text)
            if self._apply_npc_dialogue_actions(
                    npc_text, npc_agent, npc_char,
                    target_id=pc_actions[-1]["actor_id"]):
                outcome.skip_next_gm = True
                outcome.triggered_combat = True
                break

            # Narrative objectives may be completed by the NPC's actual reply.
            self._apply_quest_events(
                self.dialogue.judge_completion(combined, npc_text, ws))

        # Conversation finished. GM reads everything from the log on next turn;
        # TagAgent's queue was already cleared on entry (see top of method).
        return outcome

    def _apply_npc_dialogue_actions(self, npc_text: str, npc_agent,
                                    npc_char, target_id: str | None = None) -> bool:
        """Classify, resolve, and apply actions explicitly spoken by an NPC.

        Returns True only when an attack started combat.
        """
        ws = self.world_state
        actions = self.dialogue.judge_npc_actions(
            npc_text, npc_agent, ws, target_id=target_id)
        npc_agent.last_dialogue_actions = (
            [event.get("event") for event in actions] or ["none"])
        events = self.dialogue.resolve_dialogue_actions(npc_text, actions, ws)
        npc_agent.last_dialogue_events = events
        return self._apply_dialogue_actions(events, npc_agent, npc_char)

    def _apply_dialogue_actions(self, events: list[dict], npc_agent,
                                npc_char) -> bool:
        """Apply directional dialogue events for either speaker."""
        ws = self.world_state
        for event in events:
            kind = event.get("event")
            if kind == "attack":
                npc_agent.pending_join_offer = None
                actor_id = event.get("actor_id", "")
                target_id = event.get("target_id", "")
                actor = ws.characters.get(actor_id)
                target = ws.characters.get(target_id)
                actor_name = actor.name if actor else event.get("actor_id", "角色")
                target_name = target.name if target else event.get("target_id", "對方")
                note = f"（{actor_name} 對 {target_name} 發動攻擊）"
                ws.log_event("system", note)
                self._emit(StreamChunk("npc_talk", f"\n{note}\n", actor="系統"))
                ok, errors = execute_action_commands([{
                    "action": "attack",
                    "actor_id": actor_id,
                    "target_id": target_id,
                }], ws)
                self._emit(TagResult(ok, errors))
                hostile_id = (
                    actor_id if actor and actor.is_npc
                    else target_id if target and target.is_npc
                    else ""
                )
                hostile_agent = self.npc_agents.get(hostile_id)
                if ok and hostile_agent is not None:
                    hostile_agent.in_party = False
                self._check_quests()
                return bool(ok)
            if kind in {"invite_to_party", "request_to_join"}:
                actor_id = event.get("actor_id", "")
                target_id = event.get("target_id", "")
                actor = ws.characters.get(actor_id)
                target = ws.characters.get(target_id)
                if kind == "invite_to_party":
                    valid_direction = (
                        actor is not None and not actor.is_npc
                        and target is not None and target.is_npc
                    )
                    candidate_id = target_id if valid_direction else ""
                else:
                    valid_direction = (
                        actor is not None and actor.is_npc
                        and target is not None and not target.is_npc
                    )
                    candidate_id = actor_id if valid_direction else ""
                if candidate_id and candidate_id not in ws.party_ids:
                    npc_agent.pending_join_offer = {
                        "kind": kind,
                        "proposer_id": actor_id,
                        "responder_id": target_id,
                        "candidate_id": candidate_id,
                    }
                continue
            if kind == "give":
                item = event.get("item")
                giver_id = event.get("actor_id", "")
                recipient_id = event.get("target_id", "")
                if item and giver_id and recipient_id:
                    ok, errors = execute_action_commands([{
                        "action": "give",
                        "actor_id": giver_id,
                        "target_id": recipient_id,
                        "item": item,
                    }], ws)
                    self._emit(TagResult(ok, errors))
                continue
            if kind == "offer_quest":
                quest = ws.quests.get(event.get("id"))
                if (quest is not None
                        and quest.giver_id == event.get("actor_id")
                        and quest.status == "inactive"):
                    quest.offered = True
                continue
            if kind == "join_accept":
                candidate_id = (event.get("candidate_id")
                                or (npc_agent.pending_join_offer or {}).get(
                                    "candidate_id", ""))
                candidate = ws.characters.get(candidate_id)
                if not candidate or not candidate.is_npc:
                    continue
                if candidate_id not in ws.party_ids:
                    ws.party_ids.append(candidate_id)
                if candidate_id not in ws.combat_party_ids:
                    ws.combat_party_ids.append(candidate_id)
                candidate_agent = self.npc_agents.get(candidate_id)
                if candidate_agent:
                    candidate_agent.in_party = True
                    candidate_agent.pending_join_offer = None
                npc_agent.pending_join_offer = None
                note = f"（{candidate.name} 加入了隊伍）"
                ws.log_event("system", note)
                self._emit(StreamChunk("npc_talk", f"\n{note}\n", actor="系統"))
            elif kind == "join_reject":
                npc_agent.pending_join_offer = None
            elif kind == "quest_accept":
                self._apply_quest_events([event])
            elif kind == "quest_turnin":
                self._apply_quest_events([event])
            # quest_reject intentionally preserves Quest.offered: the NPC does
            # not repeat the full pitch, and the player may change their mind.
        return False

    def _apply_player_dialogue_actions(self, events: list[dict], npc_agent,
                                       npc_char) -> bool:
        return self._apply_dialogue_actions(events, npc_agent, npc_char)

    def _apply_quest_events(self, events: list[dict]) -> None:
        """Execute accepted, completed, or turned-in quest events."""
        ws = self.world_state
        for ev in events:
            kind = ev.get("event")
            qid = ev.get("id")
            if kind not in {"quest_accept", "quest_complete", "quest_turnin"} or not qid:
                continue
            quest = ws.quests.get(qid)
            ok: list[str] = []
            errors: list[str] = []
            if quest is None:
                errors.append(f"找不到任務：{qid}")
            elif kind == "quest_accept":
                if quest.status == "active":
                    errors.append(f"任務「{quest.title}」已在進行中")
                elif quest.status in {"completed", "turned_in"}:
                    errors.append(f"任務「{quest.title}」已完成")
                else:
                    quest.status = "active"
                    ok.append(f"接受任務：{quest.title}")
            elif kind == "quest_complete":
                if quest.status != "active":
                    errors.append(
                        f"任務「{quest.title}」目前不是進行中（{quest.status}），無法標記完成")
                else:
                    quest.status = "completed"
                    ok.append(f"任務目標達成：{quest.title}")
            elif quest.status != "completed":
                errors.append(
                    f"任務「{quest.title}」尚未達成完成條件（目前 {quest.status}）")
            else:
                lines = apply_reward(quest, ws)
                info = (quest.reward or {}).get("info") or []
                giver_agent = self.npc_agents.get(quest.giver_id)
                if info and giver_agent is not None:
                    giver_agent._secrets.extend(info)
                    giver_agent.pending_reveal = list(info)
                quest.status = "turned_in"
                suffix = "；" + "，".join(lines) if lines else ""
                ok.append(f"完成任務：{quest.title}{suffix}")
            self._emit(TagResult(ok, errors))
            for line in ok:
                self._emit(StreamChunk("npc_talk", f"\n【{line}】\n", actor="系統"))
        self._check_quests()
        return None
