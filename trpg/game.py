"""Core game engine — all game logic lives here.

Both cli.py (terminal) and web.py (Gradio) consume events from GameSession
and submit player input via submit_player_input(). Neither contains game logic.
"""
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
from .engine.quests import check_quest_progress, objective_progress_str
from .engine.status import tick_status_effects
from .llm.tag_parser import execute_all_tags, set_npc_agent_registry
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
    ctx: Any = None       # engine CombatContext — lets a GUI front-end render the
                          # battlefield / skills / targets (CLI ignores it, uses info_text)

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
    def __init__(self, world_state: WorldState, gm, tag_agent, thor_agent,
                 npc_agents: dict = None,
                 policies: dict[str, CombatPolicy] | None = None,
                 default_combat_policy: CombatPolicy | None = None):
        self.world_state = world_state
        self.gm          = gm
        self.tag_agent   = tag_agent
        self.thor_agent  = thor_agent
        self.npc_agents  = npc_agents or {}
        set_npc_agent_registry(self.npc_agents)

        # Conversation resolution pipeline (focused LLM stages: decide_check →
        # arrange_events → judge_completion). Built from tag_agent's connection so
        # no extra wiring through bootstrap/cli. Inert if tag_agent is a stub.
        self.dialogue = DialogueDirector.from_agent(tag_agent, world_state)

        self._events    : queue.Queue = queue.Queue()
        self._player_in : queue.Queue = queue.Queue()
        self._stop_flag = threading.Event()
        # The prompt currently waiting for a human answer.  Meta-commands such
        # as /旁白 are handled on the UI thread and never enter _player_in, so
        # they must repeat this prompt after reporting their status; otherwise
        # front-ends wait forever (or leave their input widget disabled).
        self._pending_prompt = None

        # Combat decisions go through a policy per character. 凱恩 (the human PC)
        # gets the human-input policy. Every other combatant — allies AND monsters
        # — defaults to `default_combat_policy` (the trained general model, a
        # shared NeuralCombatPolicy) when provided, else the scripted heuristic.
        # An explicit per-char entry in `policies` still overrides both.
        self.policies: dict[str, CombatPolicy] = dict(policies or {})
        non_human = default_combat_policy or HeuristicCombatPolicy()
        human_policy = HumanInputPolicy(
            char_id="kaine",
            prompt_fn=self._combat_prompt,
            error_fn=self._combat_error,
        )
        for cid in world_state.characters:
            if cid == "kaine":
                self.policies.setdefault(cid, human_policy)
            else:
                self.policies.setdefault(cid, non_human)

        from .llm.controllers import HumanController, LLMPlayerController, LLMNpcController
        self.controllers = {
            "kaine": HumanController("kaine", self._get_input, self._emit),
            "thor": LLMPlayerController(thor_agent, self._emit),
        }
        for cid, agent in self.npc_agents.items():
            self.controllers[cid] = LLMNpcController(agent, self._emit)

        # TagAgent still receives raw action strings (it's stateless and doesn't read log).
        self._tag_actions: list[str] = []

        # 旁白（GM narration）toggle — single source of truth. When False, the GM's
        # exploration prose and combat flavour are skipped entirely (no LLM call);
        # mechanics (tags, dice, damage, resource costs, ActionResult display) run
        # unchanged and the turn passes straight to the next actor. A testing aid.
        self.narrate = True

    # ── Public API ────────────────────────────────────────────────────────────

    # Meta-commands that toggle 旁白 rather than becoming a game action. Defined
    # once here so every front-end (cli / desktop / web all route raw input
    # through submit_player_input) gets the same toggle for free.
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
        """All alive party members (human PCs + follower NPCs).

        Used as the target pool for hostile NPC turns and for the GameOver
        wipe check at the party level.
        """
        ws = self.world_state
        return {cid: c.name for cid, c in ws.characters.items()
                if c.is_alive() and ws.is_party_ally(cid)}

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
        tag_raw = self.tag_agent.generate_tags(
            self._tag_actions,
            on_chunk=lambda c, thinking=False: self._emit(StreamChunk("tag", c, thinking)),
        )
        ok, errors = execute_all_tags(tag_raw, ws, log_to_narrative=True)
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
        # ExplorationPrompt and blocks for input). Followers — recruited NPCs
        # in party_ids but NOT in pc_ids — are silent automatically.
        prior_remarks: dict[str, str] = {}
        tag_actions: list[str] = []
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
            tag_actions.append(f"{char.name}：{output.text}")
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
        from .llm.tag_parser import execute_all_tags
        ok, _ = execute_all_tags(f"[FLEE: {cid}]", self.world_state)
        summary = f"{char.name} 逃離了戰鬥"
        self._emit(ActionResult(char.name, summary, "FLEE", valid=True))
        self.world_state.log_event("system", summary)
        if cid in self.world_state.party_ids:
            self.world_state.party_ids.remove(cid)
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
        # has now been fully consumed by the [TALK] that brought us here. Clear
        # it immediately so that leaving/quitting — which take early-return paths
        # out of the loop below — can't leave it queued for TagAgent to
        # re-classify into another [TALK] next turn (== you can never leave).
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
        # The offer is usually voiced HERE (the NPC's quest-offer rule fires on its
        # first utterance at 戒備+ attitude, which is the opening). Mark it now so the
        # player can accept on their very first reply — otherwise accept only unlocks
        # one exchange late (offered was previously set only after in-loop responses).
        self._mark_quests_offered(npc_agent)

        while not self._stop_flag.is_set():
            # ── Each PC takes a conversation turn (LLM may go silent, human
            #     blocks until input or "離開") ───────────────────────────────
            pc_lines: list[str] = []
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
                    self._emit(StatusMessage(f"你結束了與 {npc_char.name} 的對話。"))
                    ws.log_event("system", f"（{char.name} 離開了）")
                    return outcome
                if output.silent or not output.text:
                    continue
                ws.log_event(cid, output.text)
                pc_lines.append(f"{char.name}：{output.text}")

            if not pc_lines:
                # All PCs silent (rare — only happens when no human PC is
                # present and every LLM PC emitted [SILENT]). Loop back so
                # the NPC isn't asked to respond to silence.
                continue

            # ── 3-stage conversation resolution (dialogue_flow) ───────────────
            combined = "\n".join(pc_lines)

            # Stage 1: does this need a check? which / who / DC?
            plan = self.dialogue.decide_check(combined, npc_agent, ws)
            result = None
            if plan.check:
                actor = ws.characters.get(plan.actor) or ws.characters.get("kaine")
                success, total = make_saving_throw(actor, "CHA", plan.dc)
                result = CheckResult(plan.check, success, total, plan.dc, plan.actor)
                note = (f"【{plan.check}檢定：{actor.name} 擲出 {total} vs DC {plan.dc}"
                        f" → {'成功' if success else '失敗'}】")
                ws.log_event("system", note)
                self._emit(StreamChunk("npc_talk", f"\n{note}\n", actor="系統"))

            # Stage 2: arrange events BEFORE the NPC speaks, so its reply reflects
            # them. Non-quest (reveal/give/suggest_join/attitude/attack) + the
            # player-input-driven quest events accept/turn-in (gated by quest state).
            events = self.dialogue.arrange_events(combined, result, npc_agent, ws)
            quest_evs = [e for e in events
                         if e.get("event") in ("quest_accept", "quest_turnin")]
            other_evs = [e for e in events if e not in quest_evs]
            if self._apply_dialogue_events(other_evs, npc_agent, npc_char, result):
                outcome.skip_next_gm = True   # an attack event started combat
                outcome.triggered_combat = True
                break
            self._apply_quest_events(quest_evs)   # accept / turn-in (pre-response)

            # ── NPC response ──────────────────────────────────────────────────
            npc_text = npc_ctrl.take_npc_response(npc_char)
            ws.log_event(npc_id, npc_text)

            # After the reply: (a) mark this NPC's quests as offered so accept
            # becomes available NEXT turn (structurally blocks same-turn accept);
            # (b) run the dedicated completion judge over accepted quests, which
            # needs the NPC's actual reply (e.g. "boss agreed to stop").
            self._mark_quests_offered(npc_agent)
            self._apply_quest_events(
                self.dialogue.judge_completion(combined, npc_text, ws))

            # ── Recruit decision — handle JOIN / DECLINE ──────────────────────
            if npc_agent.recruit_decision == "join":
                if npc_id not in ws.party_ids:
                    ws.party_ids.append(npc_id)
                    npc_agent.in_party = True
                    note = f"（{npc_char.name} 加入了隊伍）"
                    ws.log_event("system", note)
                    self._emit(StreamChunk("npc_talk", f"\n{note}\n", actor="系統"))
            elif npc_agent.recruit_decision == "decline":
                note = f"（{npc_char.name} 婉拒了邀請）"
                ws.log_event("system", note)
                self._emit(StreamChunk("npc_talk", f"\n{note}\n", actor="系統"))
            npc_agent.recruit_decision = ""   # defensive — already reset in generate()

            # ── NPC chose to attack or flee — end conversation, execute ───────
            if npc_agent.pending_action:
                action = npc_agent.pending_action
                npc_agent.pending_action = ""
                tag = "ATTACK_NPC" if action == "attack" else "FLEE"
                ok, errors = execute_all_tags(f"[{tag}: {npc_id}]", ws)
                self._emit(TagResult(ok, errors))
                self._check_quests()
                if action == "attack":
                    outcome.skip_next_gm = True
                    outcome.triggered_combat = True
                ws.log_event("system",
                    f"（{npc_char.name} {'發動攻擊' if action == 'attack' else '逃離現場'}）")
                break

        # Conversation finished. GM reads everything from the log on next turn;
        # TagAgent's queue was already cleared on entry (see top of method).
        return outcome

    def _apply_dialogue_events(self, events: list[dict], npc_agent, npc_char,
                               result) -> bool:
        """Execute Stage-2 (non-quest) events from the event arranger. Returns
        True if combat started (caller ends the conversation).

        reveal / give / suggest_join are CONCESSIONS — only applied when a check
        was passed this round. attitude / attack are unconditional. Concessions
        set per-turn directives on the NPC so its reply reflects them; the actual
        world-state mutation reuses the existing tag_parser dispatchers.
        """
        ws = self.world_state
        npc_id = npc_agent.char_id
        won = bool(result and result.success)
        for ev in events:
            kind = ev.get("event")
            if kind == "attack":
                note = f"（{npc_char.name} 與冒險者爆發衝突）"
                ws.log_event("system", note)
                self._emit(StreamChunk("npc_talk", f"\n{note}\n", actor="系統"))
                ok, errors = execute_all_tags(f"[ATTACK_NPC: {npc_id}]", ws)
                self._emit(TagResult(ok, errors))
                self._check_quests()
                return True
            if kind == "attitude":
                try:
                    delta = int(ev.get("delta", 0) or 0)
                except (TypeError, ValueError):
                    delta = 0
                if delta:
                    old = npc_char.attitude
                    npc_char.attitude = max(0, min(4, old + delta))
                    if npc_char.attitude != old:
                        note = f"（{npc_char.name} 的態度：{old} → {npc_char.attitude}）"
                        ws.log_event("system", note)
                        self._emit(StreamChunk("npc_talk", f"\n{note}\n", actor="系統"))
                continue
            # ── concessions below: only granted on a passed check ──
            if not won:
                continue
            if kind == "reveal":
                secrets = getattr(npc_agent, "_secrets", []) or []
                said = []
                for i in ev.get("indices", []) or []:
                    if isinstance(i, int) and 0 <= i < len(secrets):
                        s = secrets[i]
                        if s not in npc_agent.revealed:
                            npc_agent.revealed.append(s)
                        said.append(s)
                if said:
                    npc_agent.pending_directives.append(
                        "如實把以下你知道的情報告訴對方：" + "；".join(said))
            elif kind == "give":
                item = ev.get("item", "")
                to = ev.get("to", "kaine")
                if to not in ws.characters or ws.characters[to].is_npc:
                    to = "kaine"
                ok, errors = execute_all_tags(f"[GIVE: {npc_id} {to} {item}]", ws)
                self._emit(TagResult(ok, errors))
                if ok:
                    npc_agent.pending_directives.append(
                        f"你願意把 {item} 交給對方，說話時自然地表達這個舉動。")
            elif kind == "suggest_join":
                npc_agent.pending_join_decision = True
                npc_agent.pending_directives.append(
                    "你覺得對方說得很有道理、很有說服力，開始認真考慮是否加入他們一起冒險。")
        return False

    _QUEST_EVENT_TAGS = {
        "quest_accept":   "QUEST_ACCEPT",
        "quest_complete": "QUEST_COMPLETE",
        "quest_turnin":   "QUEST_TURNIN",
    }

    def _apply_quest_events(self, events: list[dict]) -> None:
        """Execute quest events (accept / turn-in from stage 2, complete from the
        dedicated completion judge). Shared applier — event→tag→dispatcher."""
        ws = self.world_state
        for ev in events:
            tag = self._QUEST_EVENT_TAGS.get(ev.get("event"))
            qid = ev.get("id")
            if not tag or not qid:
                continue
            ok, errors = execute_all_tags(f"[{tag}: {qid}]", ws)
            self._emit(TagResult(ok, errors))
            for line in ok:
                self._emit(StreamChunk("npc_talk", f"\n【{line}】\n", actor="系統"))
        self._check_quests()

    def _mark_quests_offered(self, npc_agent) -> None:
        """After an NPC utterance (opening OR in-loop response), flag its still-
        inactive quests as offered so quest_accept becomes available. The NPC's
        quest-offer prompt section requires it to voice the request when
        attitude >= 戒備(1), so an utterance at that attitude means the offer is now
        on the table. Called after the opening (the offer is usually voiced there)
        and after each response (covers an NPC that warms up mid-conversation).
        The offer and the player's accept are always separate LLM calls, so this
        stays the structural guard against same-turn false-accept."""
        if npc_agent.attitude < 1:
            return
        npc_id = npc_agent.char_id
        for q in self.world_state.quests.values():
            if q.giver_id == npc_id and q.status == "inactive":
                q.offered = True
