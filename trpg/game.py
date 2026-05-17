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
from .engine.combat import execute_action, format_result, make_saving_throw
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
    thor_text: str

@dataclass
class CombatPrompt:
    aria: Any
    enemies: dict[str, str]
    info_text: str = ""   # pre-formatted position / distance / weapon range block

@dataclass
class ConversationPrompt:
    npc_id: str
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


_STOP = object()

# ── Combat turn structure ─────────────────────────────────────────────────────

_END_RE = re.compile(r'<\s*END\s*>', re.IGNORECASE)
# Action types that consume the per-turn "action" slot
_ACTION_CONSUMING_TYPES = {"ATTACK", "AOE", "USE_ITEM", "ROLL", "DODGE", "HIDE"}
# Inputs from human player that mean "end my turn"
_ARIA_END_INPUTS = {"end", "結束", "結束回合", "我這就好", "我這回合到這"}
# Safety cap: max sub-actions per character per round (prevents runaway loops)
_MAX_SUB_ACTIONS = 5


def _strip_end_marker(text: str) -> tuple[str, bool]:
    """Return (cleaned_text, ended): strips <END> and reports whether it was present."""
    ended = bool(_END_RE.search(text))
    cleaned = _END_RE.sub("", text).strip()
    return cleaned, ended


# ── GameSession ───────────────────────────────────────────────────────────────

class GameSession:
    def __init__(self, world_state: WorldState, gm, tag_agent, thor_agent, arbiter,
                 npc_agents: dict = None):
        self.world_state = world_state
        self.gm          = gm
        self.tag_agent   = tag_agent
        self.thor_agent  = thor_agent
        self.arbiter     = arbiter
        self.npc_agents  = npc_agents or {}
        set_npc_agent_registry(self.npc_agents)

        self._events    : queue.Queue = queue.Queue()
        self._player_in : queue.Queue = queue.Queue()
        self._stop_flag = threading.Event()

        # TagAgent still receives raw action strings (it's stateless and doesn't read log).
        self._tag_actions: list[str] = []
        # When a conversation ends with the player attacking, skip the next GM
        # narration (combat narration handles it) so the GM doesn't hallucinate
        # the kill before combat resolves it.
        self._skip_next_gm: bool = False

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

    def _alive_pcs(self) -> dict[str, str]:
        """Strict PCs only — used for the 'did the party wipe?' GameOver check."""
        return {cid: c.name for cid, c in self.world_state.characters.items()
                if not c.is_npc and c.is_alive()}

    def _alive_side_b(self) -> dict[str, str]:
        """Side B = PCs + follower NPCs that are still alive.

        Used as the target pool for hostile NPC turns (they may attack anyone
        on the party's side).
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
        while not self._stop_flag.is_set():
            if self._exploration_turn():
                return

    def _exploration_turn(self) -> bool:
        ws = self.world_state

        # ── TagAgent ─────────────────────────────────────────────────────────
        tag_raw = self.tag_agent.generate_tags(
            self._tag_actions,
            on_chunk=lambda c, thinking=False: self._emit(StreamChunk("tag", c, thinking)),
        )
        ok, errors = execute_all_tags(tag_raw, ws)
        self._emit(TagResult(ok, errors))
        for line in ok:
            ws.log_event("system", line)
        self._check_quests()

        # ── Conversation? (NpcAgent handles the opening; GM is skipped) ──────
        if ws.pending_conversation:
            npc_id = ws.pending_conversation
            ws.pending_conversation = ""
            if npc_id in self.npc_agents:
                over = self._run_conversation(npc_id)
                if over:
                    return True
                return False

        # ── GM narrative ─────────────────────────────────────────────────────
        if self._skip_next_gm:
            self._skip_next_gm = False
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
                return True
            ws.log_event("system", "戰鬥結束" if combat_log else "（戰鬥結束）")
            self._tag_actions = []
            return False

        # ── Death check ───────────────────────────────────────────────────────
        for char in ws.characters.values():
            if not char.is_npc and not char.is_alive():
                self._emit(GameOver(f"{char.name} 倒下了！遊戲結束。"))
                return True

        # ── Thor ──────────────────────────────────────────────────────────────
        def _thor_chunk(c, thinking=False):
            self._emit(StreamChunk("thor", c, thinking))

        thor_text = self.thor_agent.generate(on_chunk=_thor_chunk)
        ws.log_event("thor", thor_text)
        ws.event_log.append(f"索爾：{thor_text}")

        # ── Player prompt ─────────────────────────────────────────────────────
        self._emit(ExplorationPrompt(ws.characters["aria"], gm_text, thor_text))
        player_input = self._get_input()

        if player_input is None or player_input.lower() == "quit":
            self._emit(GameOver("冒險結束。再見！"))
            return True

        ws.log_event("aria", player_input)
        ws.event_log.append(f"凱恩：{player_input}")
        self._tag_actions = [f"索爾：{thor_text}", f"凱恩：{player_input}"]
        return False

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
            pcs     = self._alive_pcs()

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
                resources = {"action": 1, "bonus_action": 1, "movement": 9.0}
                stop_round = self._take_combat_turn(cid, char, resources, combat.round_number, log)

                # Phase: end of this character's turn
                tick_status_effects(char, "self_turn_end", combat.round_number)
                if stop_round == "quit":
                    return log
                if not self._alive_enemies() or not self._alive_pcs():
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
        """Run a multi-step combat turn for one character.

        The character may issue multiple sub-actions (move + attack + dodge etc.)
        within their turn budget. Loop ends when:
          - the character emits the <END> marker / Aria types "結束"
          - all resources exhausted
          - safety cap (_MAX_SUB_ACTIONS) reached
          - combat ends (someone wiped)
        Returns "quit" to signal the outer loop to stop, "" otherwise.
        """
        ws = self.world_state
        for _ in range(_MAX_SUB_ACTIONS):
            if self._stop_flag.is_set():
                return "quit"
            if not char.is_alive():
                return ""
            if not self._alive_enemies() or not self._alive_pcs():
                return ""

            if char.is_npc:
                targets = self._alive_enemies() if ws.is_party_ally(cid) else self._alive_side_b()
                result_text, ended = self._npc_sub_action(cid, char, targets, resources)
            elif cid == "thor":
                result_text, ended = self._thor_sub_action(char, resources, round_num)
            elif cid == "aria":
                outcome = self._aria_sub_action(char, resources)
                if outcome == "quit":
                    return "quit"
                result_text, ended = outcome
            else:
                return ""

            if result_text:
                log.append(result_text)
                brief = (
                    "用一句（30字以內）繁體中文敘述此戰鬥結果，"
                    "直接輸出敘事，不加格式欄位：\n" + result_text
                )
                self.gm.combat_narrate(
                    brief,
                    on_chunk=lambda c, thinking=False: self._emit(StreamChunk("narrate", c)),
                )

            if ended:
                return ""
            # Auto-end if main resources exhausted (bonus_action ignored —
            # not yet wired to any action type)
            if (resources.get("action", 0) <= 0
                and resources.get("movement", 0) <= 1e-6):
                return ""
        return ""

    def _check_quests(self) -> None:
        """Check for newly-completed quests; emit events and push notes to log."""
        completed = check_quest_progress(self.world_state)
        for q in completed:
            giver = self.world_state.characters.get(q.giver_id)
            giver_name = giver.name if giver else q.giver_id
            self._emit(QuestComplete(q.title, giver_name))
            note = f"【系統通知：任務「{q.title}」已達成完成條件，可回去找 {giver_name} 回報】"
            self.world_state.log_event("system", note)

    def _build_npc_combat_context(self, cid: str, char) -> tuple[str, str, str]:
        """Build (weapons, allies, enemies) strings for an NPC's combat prompt.

        Sides are determined by party membership; entries include positions
        and distances relative to the acting NPC so the LLM can decide whether
        to attack from here, move closer, or hold back.
        """
        ws = self.world_state
        weapon_parts = []
        for w in char.weapons:
            if w.range_type == "近戰":
                weapon_parts.append(f"{w.name}（近戰，伸手 {w.range_normal:.1f}m）")
            else:
                weapon_parts.append(
                    f"{w.name}（遠程，正常 {w.range_normal:.0f}m / 最大 {w.range_long:.0f}m）"
                )
        weapons_str = "、".join(weapon_parts) or "無武器（徒手）"

        room = ws.dungeon_map.current_room if ws.dungeon_map else None
        room_ids = set(room.npc_ids) if room else set(ws.characters.keys())
        is_party_npc = ws.is_party_ally(cid)

        def _entry(other) -> str:
            d = abs(other.position - char.position)
            dodging = "（閃避中）" if other.has_status("dodging") else ""
            return f"{other.name} HP {other.hp}/{other.max_hp}，位置 {other.position:.1f}m（距你 {d:.1f}m）{dodging}"

        ally_parts, enemy_parts = [], []
        for oid, other in ws.characters.items():
            if oid == cid or not other.is_alive():
                continue
            other_in_party = ws.is_party_ally(oid)
            other_hostile = other.is_npc and other.attitude == 0 and oid in room_ids
            entry = _entry(other)
            if is_party_npc:
                if other_in_party:
                    ally_parts.append(entry)
                elif other_hostile:
                    enemy_parts.append(entry)
            else:
                if other_hostile:
                    ally_parts.append(entry)
                elif other_in_party:
                    enemy_parts.append(entry)
        return (
            weapons_str,
            "、".join(ally_parts) or "無",
            "、".join(enemy_parts) or "無",
        )

    def _npc_sub_action(self, cid: str, char, targets: dict,
                        resources: dict) -> tuple[str, bool]:
        """One NPC sub-action. Returns (result_text, ended)."""
        ws = self.world_state
        npc_agent = self.npc_agents.get(cid)
        actor = char.name

        if not npc_agent:
            self._emit(ActionResult(actor, f"{actor} 無 NpcAgent，跳過行動", "", valid=False))
            return ("", True)

        weapons_str, allies_str, enemies_str = self._build_npc_combat_context(cid, char)
        desc, fled, ended = npc_agent.combat_action(
            weapons_str, allies_str, enemies_str,
            resources=resources,
            on_chunk=lambda c, thinking=False: self._emit(StreamChunk("npc", c, actor=actor)),
        )

        if fled:
            from .llm.tag_parser import execute_all_tags
            ok, _ = execute_all_tags(f"[FLEE: {cid}]", ws)
            summary = f"{actor} 逃離了戰鬥"
            self._emit(ActionResult(actor, summary, "FLEE", valid=True))
            ws.log_event("system", summary)
            if cid in ws.party_ids:
                ws.party_ids.remove(cid)
                npc_agent.in_party = False
            return (summary, True)

        # Empty desc (LLM only output <END>): just end the turn
        if not desc.strip():
            return ("", ended)

        action = self.arbiter.parse(
            player_action=desc, actor_id=cid, actor_name=char.name,
            available_targets=targets, resources=resources, actor_char=char,
            world_state=ws,
        )
        debug = json.dumps(action, ensure_ascii=False)
        if not action.get("valid"):
            self._emit(ActionResult(char.name, f"無效行動：{action.get('reason')}", debug, valid=False))
            return ("", ended)

        a_type = action.get("type", "").upper()
        if a_type in _ACTION_CONSUMING_TYPES and resources.get("action", 0) <= 0:
            self._emit(StatusMessage(f"{char.name} 本回合動作已用完，跳過此 {a_type}"))
            return ("", ended)

        result  = execute_action(action, ws)
        if result.get("type") == "ERROR":
            self._emit(ActionResult(char.name, f"{char.name}：{result['message']}", debug, valid=False))
            return ("", ended)
        summary = format_result(desc, result, char.name)
        self._emit(ActionResult(char.name, summary, debug, valid=True))
        ws.log_event("system", summary)
        self._consume_resources(resources, a_type, result)
        return (summary, ended)

    def _consume_resources(self, resources: dict, a_type: str, result: dict) -> None:
        """Decrement per-turn budget based on what was just executed."""
        if a_type in _ACTION_CONSUMING_TYPES:
            resources["action"] = 0
        elif a_type == "MOVE":
            dist = result.get("distance", 0)
            resources["movement"] = max(0.0, resources.get("movement", 0.0) - dist)

    def _thor_sub_action(self, char, resources: dict,
                         round_num: int) -> tuple[str, bool]:
        """One Thor sub-action. Returns (result_text, ended)."""
        ws = self.world_state
        enemies = self._alive_enemies()

        room_enemies = (ws.dungeon_map.current_room.alive_enemies(ws.characters).values()
                        if ws.dungeon_map else [])
        enemies_lines = []
        for c in room_enemies:
            d = abs(c.position - char.position)
            dodging = "（閃避中）" if c.has_status("dodging") else ""
            enemies_lines.append(f"{c.name}（HP {c.hp}/{c.max_hp}，距你 {d:.1f}m{dodging}）")
        enemies_str = "、".join(enemies_lines) or "、".join(enemies.values())

        aria_char = ws.characters.get("aria")
        aria_hp   = f"{aria_char.hp}/{aria_char.max_hp}" if aria_char else "?"
        weapon_parts = []
        for w in char.weapons:
            if w.range_type == "近戰":
                weapon_parts.append(f"{w.name}（近戰 {w.range_normal:.1f}m）")
            else:
                weapon_parts.append(f"{w.name}（遠程 {w.range_normal:.0f}m）")
        weapons_str = "、".join(weapon_parts) or "無武器"

        action_status = "可用" if resources.get("action", 0) > 0 else "已用完"
        move_left = resources.get("movement", 0.0)
        nudge = (
            f"【戰鬥回合 {round_num}】\n"
            f"你的位置：{char.position:.1f}m\n"
            f"剩餘資源：動作 {action_status}、移動 {move_left:.1f}m\n"
            f"HP：{char.hp}/{char.max_hp}　凱恩 HP：{aria_hp}\n"
            f"武器：{weapons_str}\n"
            f"敵人：{enemies_str}\n"
            f"做一個 sub-action（攻擊 / 移動 / 閃避）；想結束本回合就在訊息結尾加 <END>。\n"
            f"例：「我衝向哥布林。」不加 <END>→系統會問你下一步；"
            f"「我用長劍砍他。<END>」→ 砍完直接結束。"
        )
        desc = self.thor_agent.generate(
            nudge=nudge,
            on_chunk=lambda c, thinking=False: self._emit(StreamChunk("thor_combat", c)),
        )
        desc, ended = _strip_end_marker(desc)

        if not desc.strip():
            return ("", ended)

        action = self.arbiter.parse(
            player_action=desc, actor_id="thor", actor_name=char.name,
            available_targets=enemies, resources=resources, actor_char=char,
            world_state=ws,
        )
        debug = json.dumps(action, ensure_ascii=False)
        if not action.get("valid"):
            self._emit(ActionResult(char.name, f"無效行動：{action.get('reason')}", debug, valid=False))
            return ("", ended)

        a_type = action.get("type", "").upper()
        if a_type in _ACTION_CONSUMING_TYPES and resources.get("action", 0) <= 0:
            self._emit(StatusMessage(f"{char.name} 本回合動作已用完"))
            return ("", ended)

        result  = execute_action(action, ws)
        if result.get("type") == "ERROR":
            self._emit(ActionResult(char.name, f"{char.name}：{result['message']}", debug, valid=False))
            return ("", ended)
        summary = format_result(desc, result, char.name)
        self._emit(ActionResult(char.name, summary, debug, valid=True))
        ws.log_event("system", summary)
        self._consume_resources(resources, a_type, result)
        return (summary, ended)

    def _aria_combat_info(self, aria, resources: dict | None = None) -> str:
        """Pre-format the combat status block shown to the human player.

        Includes Aria's own position, remaining resources, weapon ranges,
        and each visible enemy's position + distance + dodge status.
        """
        ws = self.world_state
        if resources is None:
            resources = {"action": 1, "bonus_action": 1, "movement": 9.0}
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

        enemy_lines = []
        room_enemies = (ws.dungeon_map.current_room.alive_enemies(ws.characters).values()
                        if ws.dungeon_map else [])
        for e in room_enemies:
            d = abs(e.position - aria.position)
            dodging = "（閃避中）" if e.has_status("dodging") else ""
            enemy_lines.append(
                f"  - {e.name} HP {e.hp}/{e.max_hp}，位置 {e.position:.1f}m（距你 {d:.1f}m）{dodging}"
            )
        enemies_block = "\n".join(enemy_lines) or "  - 無"

        return (
            f"你的位置：{aria.position:.1f}m\n"
            f"剩餘資源：動作 {action_status}、移動 {move_left:.1f}m\n"
            f"你的武器：{weapons_line}\n"
            f"敵人：\n{enemies_block}\n"
            f"可選：攻擊、移動、閃避、用道具；輸入「結束」或「end」結束本回合"
        )

    def _aria_sub_action(self, char, resources: dict):
        """One Aria sub-action. Returns (result_text, ended) tuple, or "quit"."""
        ws   = self.world_state
        aria = ws.characters["aria"]
        while not self._stop_flag.is_set():
            enemies = self._alive_enemies()
            self._emit(CombatPrompt(aria, enemies, info_text=self._aria_combat_info(aria, resources)))
            human_input = self._get_input()

            if human_input is None:
                return "quit"
            stripped = human_input.strip()
            if not stripped:
                continue
            if stripped.lower() == "quit":
                self._emit(GameOver("冒險結束。再見！"))
                return "quit"
            if stripped.lower() in _ARIA_END_INPUTS:
                return ("", True)

            action = self.arbiter.parse(
                player_action=human_input, actor_id="aria", actor_name=aria.name,
                available_targets=enemies, resources=resources, actor_char=aria,
                world_state=ws,
            )
            debug = json.dumps(action, ensure_ascii=False)
            if not action.get("valid"):
                self._emit(StatusMessage(
                    f"❌ {action.get('reason', '無效行動')}\n"
                    f"💡 {action.get('suggestion', '')}"
                ))
                continue

            a_type = action.get("type", "").upper()
            if a_type in _ACTION_CONSUMING_TYPES and resources.get("action", 0) <= 0:
                self._emit(StatusMessage("動作已用完，請改用移動或輸入「結束」結束回合"))
                continue

            result  = execute_action(action, ws)
            if result.get("type") == "ERROR":
                self._emit(StatusMessage(f"❌ {result.get('message')}"))
                continue
            summary = format_result(human_input, result, aria.name)
            self._emit(ActionResult(aria.name, summary, debug, valid=True))
            ws.log_event("system", summary)
            ws.event_log.append(f"凱恩：{human_input}")
            self._consume_resources(resources, a_type, result)
            return (summary, False)

        return "quit"

    # ── NPC Conversation ──────────────────────────────────────────────────────

    def _run_conversation(self, npc_id: str) -> bool:
        """Multi-turn NPC conversation. Returns True if game should end.

        Conversation lines are pushed to the unified narrative_log; the NPC,
        Thor, and Aria all read their filtered view of it.
        """
        ws        = self.world_state
        npc_agent = self.npc_agents[npc_id]
        npc_char  = ws.characters[npc_id]
        aria_name = ws.characters["aria"].name
        thor_name = ws.characters["thor"].name

        # Approach cue: re-entry uses a different opening prompt
        prior = [e for e in ws.narrative_log if e["speaker"] == npc_id]
        if prior:
            ws.log_event("system", f"（冒險者再度向 {npc_char.name} 走來）")
        else:
            ws.log_event("system", f"（冒險者向 {npc_char.name} 走近，看著他）")

        # ── NPC opening (skip attitude marker — no real interaction yet) ──────
        npc_agent._skip_marker = True
        npc_text = npc_agent.generate(
            on_chunk=lambda c, thinking=False: self._emit(
                StreamChunk("npc_talk", c, actor=npc_char.name)
            ),
        )
        ws.log_event(npc_id, npc_text)

        while not self._stop_flag.is_set():
            # ── Thor sees the conversation so far, decides whether to speak ───
            def _thor_chunk(c, thinking=False):
                self._emit(StreamChunk("thor", c, thinking))

            nudge = (
                f"現在輪到你（{thor_name}）。"
                "如果你有話要說，直接說出來；如果選擇保持沉默，輸出 [SILENT]。"
            )
            thor_text = self.thor_agent.generate(nudge=nudge, on_chunk=_thor_chunk)
            spoke = "[SILENT]" not in thor_text.upper() and thor_text.strip() != ""
            if spoke:
                ws.log_event("thor", thor_text.strip())

            # ── Player input ──────────────────────────────────────────────────
            self._emit(ConversationPrompt(
                npc_id, npc_char.name, ws.characters["aria"],
                attitude_label=npc_agent.attitude_label,
            ))
            player_input = self._get_input()

            if player_input is None:
                return True
            if player_input.lower() in ("離開", "結束", "quit"):
                self._emit(StatusMessage(f"你結束了與 {npc_char.name} 的對話。"))
                ws.log_event("system", f"（{aria_name} 離開了）")
                break

            ws.log_event("aria", player_input)

            # ── Social skill check (after both Thor + Aria have spoken) ───────
            combined = (
                f"{thor_name}：{thor_text.strip()}\n凱恩：{player_input}"
                if spoke
                else f"凱恩：{player_input}"
            )
            check = self._run_social_check(combined, npc_agent, npc_id)
            if check and check[0] == "attack":
                ws.log_event("system", f"（{aria_name} 對 {npc_char.name} 發動攻擊）")
                self._emit(StreamChunk("npc_talk",
                    f"\n（{aria_name} 對 {npc_char.name} 發動攻擊）\n", actor="系統"))
                ok, errors = execute_all_tags(f"[ATTACK_NPC: {npc_id}]", ws)
                self._emit(TagResult(ok, errors))
                self._check_quests()
                self._skip_next_gm = True
                break
            if check and check[0] == "social":
                social_note = check[1]
                ws.log_event("system", social_note)
                self._emit(StreamChunk("npc_talk", f"\n{social_note}\n", actor="系統"))

            # ── NPC reads log, responds ───────────────────────────────────────
            npc_text = npc_agent.generate(
                on_chunk=lambda c, thinking=False: self._emit(
                    StreamChunk("npc_talk", c, actor=npc_char.name)
                ),
            )
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
                    self._skip_next_gm = True
                ws.log_event("system",
                    f"（{npc_char.name} {'發動攻擊' if action == 'attack' else '逃離現場'}）")
                break

        # Conversation finished. GM reads everything from the log on next turn;
        # TagAgent doesn't need conversation actions queued.
        self._tag_actions = []
        return False

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
