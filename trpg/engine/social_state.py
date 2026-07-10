"""非戰鬥（社會層）state 容器。

設計來源＝根目錄 intent_table.md / consequence_table.md。
所有變更必須經 consequences.commit_bundle 落帳（Ledger）——
這些 dataclass 只是「現值快照」，歷史在帳本裡。
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional

from .ledger import Ledger


@dataclass
class KnowledgeItem:
    fact_id: str
    text: str
    dc: int = 12                       # 資訊隱藏程度（裁決器 lookup 來源）
    min_disposition: Optional[int] = None  # 態度達標即可自然揭露
    revealed: bool = False


@dataclass
class Belief:
    content: str
    flaw: dict = field(default_factory=dict)   # predicates.py 謂詞（破綻條件）
    source: str = ""                            # 誰種下的


@dataclass
class NPCCard:
    npc_id: str
    name: str
    goals: list = field(default_factory=list)
    disposition: dict = field(default_factory=dict)   # target_id("party"/npc_id) → -5..+5
    knowledge: dict = field(default_factory=dict)     # fact_id → KnowledgeItem
    beliefs: list = field(default_factory=list)       # list[Belief]
    red_lines: list = field(default_factory=list)
    faction_id: Optional[str] = None
    insight: int = 12   # 對抗唬騙的 DC（裁決器 lookup 來源）
    will: int = 12      # 對抗威嚇/說服的 DC
    recruited: Optional[dict] = None   # {"role": str, "loyalty": int} — RECRUIT 寫入
    removed: str = ""   # "" | "死亡" | "俘虜" | "驅離" — ENTITY_REMOVE 寫入


@dataclass
class Faction:
    faction_id: str
    name: str
    reputation: int = 0   # 對隊伍名聲 -10..+10


@dataclass
class Clock:
    clock_id: str
    name: str
    remaining_days: float
    on_expire: list = field(default_factory=list)  # 模板束 [{"template","params"}]
    owner: str = "world"
    expired: bool = False


@dataclass
class StandingRule:
    rule_id: str
    owner: str
    trigger: dict          # {"kind":"periodic","interval_days":1.0} 或 {"kind":"conditional","predicate":{...}}
    effects: list          # 模板束
    cancel: Optional[dict] = None      # 謂詞；真→停用
    last_fired_day: Optional[float] = None
    active: bool = True


@dataclass
class Promise:
    promise_id: str
    a: str
    b: str
    content: str
    due: Optional[dict] = None   # 謂詞（期限/破綻）
    status: str = "open"         # open | kept | broken | orphaned


@dataclass
class SocialState:
    npc_cards: dict = field(default_factory=dict)
    factions: dict = field(default_factory=dict)
    clocks: dict = field(default_factory=dict)
    standing_rules: dict = field(default_factory=dict)
    promises: dict = field(default_factory=dict)
    party_knowledge: dict = field(default_factory=dict)   # 黨帳本 fact_id → KnowledgeItem
    aggregates: dict = field(default_factory=dict)        # "變數:區域" → 0..10
    accounts: dict = field(default_factory=dict)          # entity_id → 金錢（不進 Character）
    conditions: dict = field(default_factory=dict)        # entity_id → [{"status","until_min"}]
    object_states: dict = field(default_factory=dict)     # object_id → 狀態
    object_dcs: dict = field(default_factory=dict)        # object_id → 強度 DC
    spawned_locations: dict = field(default_factory=dict) # 神諭長出的地點（DungeonMap 橋接前的落點）
    flags: set = field(default_factory=set)
    resources: dict = field(default_factory=dict)         # (entity_id, 資源名) → int
    time_minutes: int = 0
    ledger: Ledger = field(default_factory=Ledger)

    @property
    def day(self) -> float:
        return self.time_minutes / 1440.0

    # ── 序列化（社會層自足；戰鬥層 Character 序列化不在此範圍）──────────
    def to_dict(self) -> dict:
        return {
            "npc_cards": {k: asdict(v) for k, v in self.npc_cards.items()},
            "factions": {k: asdict(v) for k, v in self.factions.items()},
            "clocks": {k: asdict(v) for k, v in self.clocks.items()},
            "standing_rules": {k: asdict(v) for k, v in self.standing_rules.items()},
            "promises": {k: asdict(v) for k, v in self.promises.items()},
            "party_knowledge": {k: asdict(v) for k, v in self.party_knowledge.items()},
            "aggregates": dict(self.aggregates),
            "accounts": dict(self.accounts),
            "conditions": {k: [dict(c) for c in v] for k, v in self.conditions.items()},
            "object_states": dict(self.object_states),
            "object_dcs": dict(self.object_dcs),
            "spawned_locations": dict(self.spawned_locations),
            "flags": sorted(self.flags),
            "resources": {f"{k[0]}|{k[1]}": v for k, v in self.resources.items()},
            "time_minutes": self.time_minutes,
            "ledger": self.ledger.to_dicts(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SocialState":
        s = cls()
        s.npc_cards = {k: _card_from_dict(v) for k, v in d["npc_cards"].items()}
        s.factions = {k: Faction(**v) for k, v in d["factions"].items()}
        s.clocks = {k: Clock(**v) for k, v in d["clocks"].items()}
        s.standing_rules = {k: StandingRule(**v) for k, v in d["standing_rules"].items()}
        s.promises = {k: Promise(**v) for k, v in d["promises"].items()}
        s.party_knowledge = {k: KnowledgeItem(**v) for k, v in d["party_knowledge"].items()}
        s.aggregates = dict(d["aggregates"])
        s.accounts = dict(d["accounts"])
        s.conditions = {k: [dict(c) for c in v] for k, v in d["conditions"].items()}
        s.object_states = dict(d["object_states"])
        s.object_dcs = dict(d["object_dcs"])
        s.spawned_locations = dict(d["spawned_locations"])
        s.flags = set(d["flags"])
        s.resources = {tuple(k.split("|", 1)): v for k, v in d["resources"].items()}
        s.time_minutes = d["time_minutes"]
        s.ledger = Ledger.from_dicts(d["ledger"])
        return s


def _card_from_dict(d: dict) -> NPCCard:
    d = dict(d)
    d["knowledge"] = {k: KnowledgeItem(**v) for k, v in d["knowledge"].items()}
    d["beliefs"] = [Belief(**b) for b in d["beliefs"]]
    return NPCCard(**d)
