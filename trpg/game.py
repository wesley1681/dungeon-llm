"""Core game engine — all game logic lives here.

Both cli.py (terminal) and web.py (Gradio) consume events from GameSession
and submit player input via submit_player_input(). Neither contains game logic.
"""
import json
import re
import queue
import threading
from dataclasses import dataclass, field
from typing import Any

from .engine.world_state import WorldState
from .engine.combat import (
    execute_action, format_result, make_saving_throw,
    consume_resources, MOVE_BUDGET_M, build_combat_context,
)
from .engine.combat_policy import CombatPolicy, HeuristicCombatPolicy
from .engine.quests import check_quest_progress, objective_progress_str
from .engine.status import tick_status_effects
from .llm.tag_parser import execute_all_tags, set_npc_agent_registry


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
    aria: Any
    gm_text: str
    # Other PCs' remarks this turn, keyed by display name. Empty when no
    # other PCs spoke before the human's slot (eg. solo party).
    prior_remarks: dict[str, str] = field(default_factory=dict)

@dataclass
class CombatPrompt:
    aria: Any
    enemies: dict[str, str]
    info_text: str = ""   # pre-formatted position / distance / weapon range block

@dataclass
class ConversationPrompt:
    npc_name: str
    aria: Any
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


def _describe_action(action: dict) -> str:
    """Synthesize the player-description string format_result expects.

    Combat decisions are made by CombatPolicy and arrive as structured dicts;
    this rebuilds a short natural-language label so the narrator's headline
    line ("【索爾的行動】我用長劍攻擊地精") still reads correctly.
    """
    t = action.get("type", "")
    if t == "ATTACK":
        return f"我用 {action.get('weapon', '武器')} 攻擊 {action.get('target', '敵人')}"
    if t == "MOVE":
        if action.get("target"):
            return f"我朝 {action['target']} 移動"
        tp = action.get("target_position")
        if tp is not None:
            return f"我移動到 ({tp[0]:.1f}, {tp[1]:.1f})"
        d = action.get("direction")
        if d == "advance":
            return "我前進"
        if d == "retreat":
            return "我後退"
        return "我移動"
    if t == "SPELL":
        return f"我施展 {action.get('spell_name', '法術')}"
    if t == "DODGE":
        return "我閃避"
    if t == "HIDE":
        return "我躲藏"
    if t == "USE_ITEM":
        return f"我使用 {action.get('item', '道具')}"
    if t == "AOE":
        return f"我投擲 {action.get('item', '物品')}"
    return f"我執行 {t}"


def format_aria_combat_info(aria, ctx) -> str:
    """Pre-format the combat status block shown to the human player.

    Reads everything from a CombatContext (built by engine.combat) so the
    function has no GameSession dependency.
    """
    from .engine.combat import CombatContext, MOVE_BUDGET_M
    resources = ctx.resources if isinstance(ctx, CombatContext) else {"action": 1, "movement": MOVE_BUDGET_M}
    action_status = "可用" if resources.get("action", 0) > 0 else "已用完"
    move_left = resources.get("movement", 0.0)

    weapon_parts = []
    for w in aria.weapons:
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

    return (
        f"你的座標：({aria.position.x:.1f}, {aria.position.y:.1f})m\n"
        f"剩餘資源：動作 {action_status}、移動 {move_left:.1f}m\n"
        f"你的武器：{weapons_line}\n"
        f"敵人：\n{enemies_block}\n"
        f"可選：攻擊、移動、閃避、用道具；輸入「結束」或「end」結束本回合"
    )


# ── GameSession ───────────────────────────────────────────────────────────────

class GameSession:
    def __init__(self, world_state: WorldState, gm, tag_agent, thor_agent,
                 npc_agents: dict = None,
                 policies: dict[str, CombatPolicy] | None = None):
        self.world_state = world_state
        self.gm          = gm
        self.tag_agent   = tag_agent
        self.thor_agent  = thor_agent
        self.npc_agents  = npc_agents or {}
        set_npc_agent_registry(self.npc_agents)

        # Combat decisions go through a policy per character. Anyone without an
        # explicit policy falls back to the scripted heuristic — this is the
        # placeholder until a trained RL policy plugs in. Aria (human PC) also
        # uses the heuristic for now; we'll swap in a structured-input policy
        # once the UI exposes action choices instead of free text.
        self.policies: dict[str, CombatPolicy] = dict(policies or {})
        default_policy = HeuristicCombatPolicy()
        for cid in world_state.characters:
            self.policies.setdefault(cid, default_policy)

        self._events    : queue.Queue = queue.Queue()
        self._player_in : queue.Queue = queue.Queue()
        self._stop_flag = threading.Event()

        from .llm.controllers import HumanController, LLMPlayerController, LLMNpcController
        self.controllers = {
            "aria": HumanController("aria", self._get_input, self._emit),
            "thor": LLMPlayerController(thor_agent, self._emit),
        }
        for cid, agent in self.npc_agents.items():
            self.controllers[cid] = LLMNpcController(agent, self._emit)

        # TagAgent still receives raw action strings (it's stateless and doesn't read log).
        self._tag_actions: list[str] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def submit_player_input(self, text: str) -> None:
        self._player_in.put(text)

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
        self._events.put(event)

    def _get_input(self) -> str | None:
        val = self._player_in.get()
        if val is _STOP or self._stop_flag.is_set():
            return None
        return val

    def _alive_enemies(self) -> dict[str, str]:
        ws = self.world_state
        if ws.dungeon_map:
            room_ids = set(ws.dungeon_map.current_room.npc_ids)
            return {cid: c.name for cid, c in ws.characters.items()
                    if c.is_npc and c.is_alive() and c.attitude == 0 and cid in room_ids}
        return {cid: c.name for cid, c in ws.characters.items()
                if c.is_npc and c.is_alive() and c.attitude == 0}

    def _alive_human_pcs(self) -> dict[str, str]:
        """Strict human-controlled PCs only — used for the 'did the party wipe?'
        GameOver check. Follower NPC death does NOT trigger GameOver."""
        return {cid: c.name for cid, c in self.world_state.characters.items()
                if not c.is_npc and c.is_alive()}

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
        if skip_gm:
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
            if not char.is_npc and not char.is_alive():
                self._emit(GameOver(f"{char.name} 倒下了！遊戲結束。"))
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
                if not char or not char.is_alive():
                    continue

                # Phase: start of this character's turn
                tick_status_effects(char, "self_turn_start", combat.round_number)

                # Per-turn resource budget. Sub-actions decrement these.
                resources = {"action": 1, "movement": MOVE_BUDGET_M}
                stop_round = self._take_combat_turn(cid, char, resources, combat.round_number, log)

                # Phase: end of this character's turn
                tick_status_effects(char, "self_turn_end", combat.round_number)
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

        description = _describe_action(action)
        summary = format_result(description, result, char.name)
        self._emit(ActionResult(char.name, summary, debug, valid=True))
        ws.log_event("system", summary)
        consume_resources(resources, action, result)
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

        # Approach cue: re-entry uses a different opening prompt
        prior = [e for e in ws.narrative_log if e["speaker"] == npc_id]
        if prior:
            ws.log_event("system", f"（冒險者再度向 {npc_char.name} 走來）")
        else:
            ws.log_event("system", f"（冒險者向 {npc_char.name} 走近，看著他）")

        # ── NPC opening ───────────────────────────────────────────────────────
        npc_text = npc_ctrl.take_npc_opening(npc_char)
        ws.log_event(npc_id, npc_text)

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

            # ── Social skill check on combined PC speech this round ───────────
            combined = "\n".join(pc_lines)
            check = self._run_social_check(combined, npc_agent, npc_id)
            if check and check[0] == "attack":
                ws.log_event("system", f"（冒險者 對 {npc_char.name} 發動攻擊）")
                self._emit(StreamChunk("npc_talk",
                    f"\n（冒險者 對 {npc_char.name} 發動攻擊）\n", actor="系統"))
                ok, errors = execute_all_tags(f"[ATTACK_NPC: {npc_id}]", ws)
                self._emit(TagResult(ok, errors))
                self._check_quests()
                outcome.skip_next_gm = True
                outcome.triggered_combat = True
                break
            if check and check[0] == "social":
                social_note = check[1]
                ws.log_event("system", social_note)
                self._emit(StreamChunk("npc_talk", f"\n{social_note}\n", actor="系統"))

            # ── NPC response ──────────────────────────────────────────────────
            npc_text = npc_ctrl.take_npc_response(npc_char)
            ws.log_event(npc_id, npc_text)

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
        # TagAgent doesn't need conversation actions queued.
        self._tag_actions = []
        return outcome

    _SOCIAL_RE     = re.compile(r'\[SOCIAL:\s*(\w+)\s+<?(\w+)>?(?:\s+DC\d+)?\]', re.IGNORECASE)
    _ATTACK_NPC_RE = re.compile(r'\[ATTACK_NPC:\s*<?(\w+)>?\]', re.IGNORECASE)
    _QUEST_RE      = re.compile(r'\[QUEST_(ACCEPT|TURNIN):\s*<?(\w+)>?\]', re.IGNORECASE)
    _RECRUIT_RE    = re.compile(r'\[RECRUIT:\s*<?(\w+)>?\]', re.IGNORECASE)

    _SOCIAL_NOTES = {
        ("intimidate", True):  "恐嚇成功（{t} vs DC{dc}）。你被迫說出你知道的秘密，但內心充滿恐懼與怨恨。",
        ("intimidate", False): "恐嚇失敗（{t} vs DC{dc}）。對方的威嚇沒有奏效，你反而更加抵觸。",
        ("persuade",   True):  "說服成功（{t} vs DC{dc}）。你被對方說服，願意配合他的請求。",
        ("persuade",   False): "說服失敗（{t} vs DC{dc}）。你沒有被說服，維持原本立場。",
        ("deceive",    True):  "欺騙成功（{t} vs DC{dc}）。你相信了對方的話，放下了部分戒心。",
        ("deceive",    False): "欺騙失敗（{t} vs DC{dc}）。你察覺對方在說謊，感到憤怒與不信任。",
    }

    def _run_social_check(self, combined_input: str, npc_agent, npc_id: str = "") -> tuple | None:
        try:
            return self.__run_social_check(combined_input, npc_agent, npc_id)
        except Exception as e:
            self._emit(StatusMessage(f"（社交偵測失敗：{e}）"))
            return None

    def __run_social_check(self, combined_input: str, npc_agent, npc_id: str = "") -> tuple | None:
        ws = self.world_state
        tag_raw = self.tag_agent.generate_conversation_tags(combined_input, npc_id)

        if self._ATTACK_NPC_RE.search(tag_raw):
            return ("attack", None)

        qm = self._QUEST_RE.search(tag_raw)
        if qm:
            kind = qm.group(1).upper()
            qid  = qm.group(2)
            ok, errors = execute_all_tags(f"[QUEST_{kind}: {qid}]", ws)
            self._emit(TagResult(ok, errors))
            self._check_quests()
            note = "；".join(ok) or "；".join(errors) or "（任務動作無回報）"
            return ("social", f"【系統判定：{note}】")

        rm = self._RECRUIT_RE.search(tag_raw)
        if rm:
            target_id = rm.group(1)
            ok, errors = execute_all_tags(f"[RECRUIT: {target_id}]", ws)
            self._emit(TagResult(ok, errors))
            note = "；".join(ok) or "；".join(errors) or "（招募動作無回報）"
            return ("social", f"【系統判定：{note}】")

        m = self._SOCIAL_RE.search(tag_raw)
        if not m:
            return None

        social_type = m.group(1).lower()
        char_id     = m.group(2).lower()
        char        = ws.characters.get(char_id)
        if not char or social_type not in ("intimidate", "persuade", "deceive"):
            return None

        dc = npc_agent.estimate_dc(social_type, combined_input)
        success, total = make_saving_throw(char, "CHA", dc)

        if social_type == "intimidate":
            npc_agent.attitude = max(0, npc_agent.attitude - 1)
            if success:
                npc_agent.force_reveal = "intimidate"
        elif social_type == "persuade":
            if success:
                npc_agent.force_reveal = "persuade"
        elif social_type == "deceive":
            if success:
                npc_agent.force_reveal = "deceive"
            else:
                npc_agent.attitude = max(0, npc_agent.attitude - 1)

        npc_agent._skip_marker = True

        template = self._SOCIAL_NOTES.get((social_type, success), "")
        note = template.format(t=total, dc=dc)
        return ("social", f"【系統判定：{note}】")
