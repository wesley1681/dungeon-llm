import json
import re
import pathlib
from ..engine.character import Character
from ..engine.quests import Quest, objective_progress_str
from .backend import stream_chat
from .log_render import render_messages, render_script
from .social_dc_agent import SocialDcAgent

_DEBUG_DIR = pathlib.Path(__file__).parent.parent / "debug"
_DEBUG_DIR.mkdir(exist_ok=True)

_ATTITUDE_LABELS = ["敵意", "戒備", "中立", "友好", "信任"]

# Core body is mode-agnostic — identity / personality / tactics-of-the-moment
# / knowledge boundary. Both conversation and combat builds start from this.
_CORE_BODY_TEMPLATE = """你正在扮演 {name}。{personality}

{tactics_section}## 知識邊界（鐵則，凌駕一切）
你**只**知道下列來源的事：
1. 上方角色背景與個性所述
2.「你知道的情報」清單（如有）
3. 之前對話中你親自說過或聽過的事

對超出此範圍的問題（其他大陸的傳說、未來的事、神級秘密、玩家編造的離奇情境），
誠實回答「我不知道」「沒聽過」「這超出我的見識」。
**絕對禁止編造**你不該知道的事，即使對方在威脅你也一樣——
威脅只能讓你說出你**真的知道**的事，威脅變不出你不知道的事。
"""

# Conversation-only extras: dialogue format rules, volatile state (attitude,
# pending reveals, recruit, secrets, quests, action threshold), attitude
# marker grammar. Combat mode skips this entire block.
_CONVERSATION_EXTRAS_TEMPLATE = """## 對話規則
- 用第一人稱繁體中文回應，控制在 60 字以內
- 直接輸出對話內容，不加旁白或括號說明
- 只扮演自己，不描述其他角色的行為或感受

## 當前態度：{attitude_label}（{attitude}/4）

{pending_section}{recruit_section}{secrets_section}{quest_section}{action_section}## 態度標記（回應的最後另起一行，只能是 [+] [=] [-] 三選一，禁止加任何描述文字）
[+] = 對方更有好感或信任
[=] = 沒明顯變化
[-] = 更警戒、恐懼或反感
範例輸出結尾：「……我會跟著你們。\n[+]」（只寫符號本體，不要寫「這次互動讓你...」這類描述）
"""

_ACTION_SECTION = """## 行動選擇（你已經到達敵意，根據自己的性格決定）
若你決定不再忍受、動手攻擊對方，在回應結尾加 <ATTACK>
若你決定逃離現場，在回應結尾加 <FLEE>
若你只想繼續用言語抗拒、拒絕配合，不加任何標記

"""

_ACTION_RE = re.compile(r'<\s*(ATTACK|FLEE)\s*>', re.IGNORECASE)

_SECRETS_UNLOCKED = """## 你知道的情報（態度已足夠，對方問起時如實回答）
{secrets}
"""

_FORCE_DETAILS = {
    "intimidate": {
        "heading": "## 對方威嚇成功（你選擇屈服）",
        "intro":   "你害怕了，不敢再硬撐——你決定回答對方剛才提出的具體問題/要求。",
    },
    "persuade": {
        "heading": "## 對方說服成功（你被打動了）",
        "intro":   "你覺得對方說的有道理，願意配合——你決定回答對方剛才提出的具體問題/要求。",
    },
    "deceive": {
        "heading": "## 對方騙過你（你完全相信他）",
        "intro":   "你被對方的說詞騙倒，把他當成可信任的對象——你願意如實回答他剛才的問題/要求。",
    },
}

_SECRETS_FORCE = """{heading}
{intro}

回應規則：
- 鎖定對方在對話最後一段提出的**具體問題或要求**
- 從下方「你知道的情報」清單中找出**所有與對方提問主題相關**的條目，都坦白告訴對方
- 與提問主題**無關**的情報，不要主動倒出來
- 若整個問題都超出你的知識範圍 → 誠實說「我不知道」「沒聽過」「我哪會知道這個」
- **絕對不要**因為對方很兇／話術巧妙就編造你不知道的事

{secrets}
"""

_SECRETS_LOCKED = """## 你知道的情報
你隱約知道一些事，但現在還不夠信任對方，不願透露細節。
"""

_PENDING_REVEAL = """## 這一輪你必須親口告訴對方以下事項（劇情指示，不可省略）
{items}

把這些事完整、自然地說出來，作為對對方剛才行為的回應。
可以加上感情、語氣、自己的想法（緊張、感激、害怕都可），但**內容必須完整傳達**——
玩家必須從你的回應中聽到這些事的關鍵內容。

"""

_QUEST_OFFER = """## 你有事想拜託對方（重要任務指示）
{description}

行為規則：
- 若你的態度為「戒備」「中立」「友好」「信任」其中之一，**這次回應必須**主動把這個請求說出口，不要再等對方先問
- 開口時要具體說出你想要什麼（「替我採三朵月光草」），並承諾回報
- 對方明確答應後，本次任務由系統判定，你不必再重複拜託
- 只有在你的態度為「敵意」、或對方剛剛拔武器威脅、辱罵你時，才暫時不提

"""

_QUEST_ACTIVE = """## 你已經拜託的事
你拜託對方：{description}
目前進度：{progress}。對方提及這件事時自然回應，不要重複請託。

"""

_QUEST_COMPLETED = """## 對方已完成的請託
對方已替你完成：{description}
若對方提起、回報、或詢問獎勵，立刻表達感謝並交付承諾的報酬與情報。

"""

_MARKER_RE = re.compile(r'\[\s*([+\-=])\s*\]')
_RECRUIT_RE = re.compile(r'<\s*(JOIN|DECLINE)\s*>', re.IGNORECASE)

_RECRUIT_SECTION = """## 對方邀你加入冒險（重要決定）
對方邀請你跟著他們一起冒險。根據你的個性、當前狀況、對對方的信任程度，
決定要不要去。決定後在回應**結尾另起一行**只能輸出以下兩種標記：
<JOIN>   = 願意加入
<DECLINE> = 拒絕，留在原地

回應內容簡短說明你的想法/原因，再加上標記。

"""

class NpcAgent:
    def __init__(self, model: str, char_id: str, character: Character,
                 personality: str,
                 base_url: str, backend: str, world_state, options: dict = None,
                 tactics: str = "",
                 combat_tactics: str = "",
                 combat_reasoning: bool = False,
                 secrets: list[str] = None, reveal_threshold: int = 3,
                 quests: list[Quest] = None, api_key: str = None):
        self.model            = model
        self.char_id          = char_id
        self.char             = character
        self.base_url         = base_url
        self.backend          = backend
        self.api_key          = api_key
        self.world_state      = world_state
        # Reasoning agents need budget for <think>…</think> + action; plain
        # NPCs stay slim. 500 was tight enough that a single verbose model
        # would hit the cap mid-think and leak reasoning fragments into the
        # arbiter — 1024 gives breathing room. Plain NPCs also got truncated
        # mid-prose at 150 (LLMs writing 100+ char monologues run past it),
        # so default is now 300. Explicit `options=` still overrides.
        default_predict = 1024 if combat_reasoning else 300
        self.options          = options or {"temperature": 0.85, "num_predict": default_predict}
        self._personality     = personality
        # Routed by mode: _system() (conversation) reads _tactics; combat
        # callers read .combat_tactics directly (exposed as public attr so
        # LLMNpcController can compose the combat nudge, mirroring how
        # LLMPlayerController reads PlayerAgent.combat_tactics).
        self._tactics         = tactics
        self.combat_tactics   = combat_tactics
        # When True, LLMNpcController appends a chain-of-thought instruction to
        # the combat nudge (think inside <think>...</think>, then action). The
        # controller strips the think block before passing to the arbiter.
        # Opt-in: small grunts stay fast/cheap; bosses & casters get reasoning.
        self.combat_reasoning = combat_reasoning
        self._secrets         = secrets or []
        self._reveal_threshold = reveal_threshold
        self._quests          = quests or []
        self.force_reveal: str = ""                     # "" / "intimidate" / "persuade" / "deceive"
        self.pending_reveal: list[str] = []             # quest reward lines NPC MUST say this turn
        self._skip_marker: bool = False                 # suppress [+/-] when social check ran this turn
        self.pending_action: str = ""                   # "attack" / "flee"
        self.pending_join_decision: bool = False        # one-shot: NPC must answer join/decline this turn
        self.recruit_decision: str = ""                 # "" / "join" / "decline" — parsed from this turn's output
        self.in_party: bool = False                     # mirror of (char_id in ws.party_ids)
        self._social_dc = SocialDcAgent(model, base_url, backend, api_key=api_key)

    def _quest_section(self) -> str:
        parts: list[str] = []
        for q in self._quests:
            if q.status == "inactive":
                parts.append(_QUEST_OFFER.format(description=q.description))
            elif q.status == "active":
                prog = objective_progress_str(q, self.world_state) if self.world_state else ""
                parts.append(_QUEST_ACTIVE.format(description=q.description, progress=prog or "未知"))
            elif q.status == "completed":
                parts.append(_QUEST_COMPLETED.format(description=q.description))
        return "".join(parts)

    def _system(self) -> str:
        """Conversation-mode body: core (identity/personality/general tactics/
        knowledge boundary) + conversation extras (dialogue rules / attitude
        / secrets / quests / recruit / attitude marker grammar).
        """
        if self._secrets:
            formatted = "\n".join(f"- {s}" for s in self._secrets)
            if self.force_reveal in _FORCE_DETAILS:
                d = _FORCE_DETAILS[self.force_reveal]
                secrets_section = _SECRETS_FORCE.format(
                    heading=d["heading"], intro=d["intro"], secrets=formatted,
                )
            elif self.attitude >= self._reveal_threshold:
                secrets_section = _SECRETS_UNLOCKED.format(secrets=formatted)
            else:
                secrets_section = _SECRETS_LOCKED
        else:
            secrets_section = ""

        if self.pending_reveal:
            items = "\n".join(f"- {line}" for line in self.pending_reveal)
            pending_section = _PENDING_REVEAL.format(items=items)
        else:
            pending_section = ""

        action_section = _ACTION_SECTION if self.attitude == 0 else ""
        recruit_section = _RECRUIT_SECTION if self.pending_join_decision else ""
        tactics_section = (self._tactics.rstrip() + "\n\n") if self._tactics else ""

        core = _CORE_BODY_TEMPLATE.format(
            name=self.char.name,
            personality=self._personality,
            tactics_section=tactics_section,
        )
        extras = _CONVERSATION_EXTRAS_TEMPLATE.format(
            attitude_label=_ATTITUDE_LABELS[self.attitude],
            attitude=self.attitude,
            pending_section=pending_section,
            recruit_section=recruit_section,
            secrets_section=secrets_section,
            quest_section=self._quest_section(),
            action_section=action_section,
        )
        return core.rstrip() + "\n\n" + extras

    def _combat_body(self) -> str:
        """Combat-mode body: just the core (identity / personality / combat
        tactics / knowledge boundary). No conversation state — controllers
        compose situation + action menu via nudge."""
        tactics_section = (self.combat_tactics.rstrip() + "\n\n") if self.combat_tactics else ""
        return _CORE_BODY_TEMPLATE.format(
            name=self.char.name,
            personality=self._personality,
            tactics_section=tactics_section,
        )

    def generate(self, nudge: str = "", combat: bool = False, on_chunk=None) -> str:
        """Generate an NPC response.

        combat=False (default): conversation mode. Body includes attitude /
            secrets / quests / recruit / attitude-marker rules. After the LLM
            replies, JOIN/DECLINE/[+/=/-]/<ATTACK>/<FLEE> markers are parsed
            out and used to update internal state; cleaned text is returned.
        combat=True: slim body (identity + personality + combat tactics +
            knowledge boundary only). No marker parsing — caller
            (LLMNpcController) is responsible for stripping <FLEE>/<END>.

        nudge: per-call instruction appended after the body. Both callers
            (game.py conversation, LLMNpcController combat) use this to pass
            the situation-specific call-to-action.
        """
        history_msgs = render_messages(self.world_state, self.char_id)
        if not history_msgs and not combat:
            history_msgs = [{"role": "user", "content": "（冒險者向你走近，看著你）"}]

        body = self._combat_body() if combat else self._system()
        if nudge:
            body = body.rstrip() + "\n\n" + nudge

        messages = history_msgs + [{"role": "user", "content": body}]
        debug_name = (f"npc_{self.char.name}_combat_context.json" if combat
                      else f"npc_{self.char.name}_context.json")
        (_DEBUG_DIR / debug_name).write_text(
            json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8", errors="replace"
        )

        buf: list[str] = []
        def _on_chunk(chunk, thinking=False):
            if not thinking:
                buf.append(chunk)
                if on_chunk:
                    on_chunk(chunk, thinking=False)

        full = stream_chat(
            self.base_url, self.model, messages, self.options,
            think=False, on_chunk=_on_chunk, backend=self.backend, timeout=60,
            api_key=self.api_key,
        )

        if combat:
            # Combat output is returned raw — controller parses <FLEE>/<END>.
            return full

        # ── Conversation mode: reset one-shots and parse markers ──────────────
        self.force_reveal   = ""
        self.pending_reveal = []
        self.pending_join_decision = False
        self.recruit_decision = ""

        # Parse and strip action marker (only meaningful when attitude == 0)
        am = _ACTION_RE.search(full)
        if am:
            self.pending_action = am.group(1).lower()
            full = _ACTION_RE.sub("", full).strip()

        # Parse and strip recruit decision marker (<JOIN> / <DECLINE>)
        rm = _RECRUIT_RE.search(full)
        if rm:
            self.recruit_decision = rm.group(1).lower()
            full = _RECRUIT_RE.sub("", full).strip()

        # Parse and strip attitude marker (suppress if social check ran this turn).
        matches = list(_MARKER_RE.finditer(full))
        if matches:
            last = matches[-1]
            if not self._skip_marker:
                marker = last.group(1)
                if marker == "+" and self.char.attitude < 4:
                    self.char.attitude += 1
                elif marker == "-" and self.char.attitude > 0:
                    self.char.attitude -= 1
            self._skip_marker = False
            full = full[:last.start()].rstrip()

        return full

    @property
    def attitude(self) -> int:
        return self.char.attitude

    @attitude.setter
    def attitude(self, value: int) -> None:
        self.char.attitude = value

    def estimate_dc(self, social_type: str, attempt_text: str) -> int:
        """Delegate DC estimation to SocialDcAgent with full NPC context."""
        return self._social_dc.estimate(
            social_type=social_type,
            attempt_text=attempt_text,
            personality=self._personality,
            attitude=self.attitude,
            attitude_label=_ATTITUDE_LABELS[self.attitude],
            conv_log_text=render_script(self.world_state, self.char_id),
        )

    @property
    def attitude_label(self) -> str:
        return _ATTITUDE_LABELS[self.attitude]
