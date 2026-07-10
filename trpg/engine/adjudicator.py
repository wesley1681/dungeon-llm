"""三分裁決器——骰由引擎擲、DC 由引擎 lookup（intent_table 前置規則 2/7）。

第三輪對抗性壓測的核心教訓：「系統唯一忘了不信任的元件是裁決者本身」。
所以：玩家/GM 自報骰值無效（骰只在這裡擲）；DC 能 lookup 就 lookup，
lookup 不到來源時裁決器回 impossible——迫使呼叫端先把 DC 錨到 state
（物件強度/資訊隱藏度/NPC 洞察意志），LLM 只剩「挑哪個來源」的裁量。
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional

from .dice import roll_d20

LIKELIHOOD_DC = {"很可能": 6, "五五波": 11, "不太可能": 16}


@dataclass(frozen=True)
class Ruling:
    kind: str                      # "auto" | "check" | "oracle" | "impossible"
    success: bool
    reason: str = ""
    dc: Optional[int] = None
    roll: Optional[int] = None

    def as_dict(self) -> dict:
        return asdict(self)


def rule_auto(reason: str) -> Ruling:
    return Ruling(kind="auto", success=True, reason=reason)


def rule_impossible(reason: str) -> Ruling:
    return Ruling(kind="impossible", success=False, reason=reason)


def lookup_dc(world, source: tuple) -> Optional[int]:
    """DC 的既定來源（表一「對抗方/DC 來源」欄的機器面）。"""
    s = world.social
    kind = source[0]
    if kind == "npc_insight":
        card = s.npc_cards.get(source[1])
        return card.insight if card else None
    if kind == "npc_will":
        card = s.npc_cards.get(source[1])
        return card.will if card else None
    if kind == "object":
        return s.object_dcs.get(source[1])
    if kind == "info":
        card = s.npc_cards.get(source[1])
        if card and source[2] in card.knowledge:
            return card.knowledge[source[2]].dc
        return None
    return None


def rule_check(world, *, dc: Optional[int] = None, dc_from: Optional[tuple] = None,
               modifier: int = 0, mode: str = "normal", reason: str = "") -> Ruling:
    if dc is None and dc_from is not None:
        dc = lookup_dc(world, dc_from)
    if dc is None:
        return rule_impossible(f"DC 無來源可查（{reason}）——先把對抗方錨到 state")
    r = roll_d20(mode) + modifier
    return Ruling(kind="check", success=r >= dc, dc=dc, roll=r, reason=reason)


def rule_oracle(world, *, likelihood: str, reason: str = "") -> Ruling:
    dc = LIKELIHOOD_DC.get(likelihood)
    if dc is None:
        return rule_impossible(f"未知機率評估 {likelihood!r}（{sorted(LIKELIHOOD_DC)}）")
    r = roll_d20()
    return Ruling(kind="oracle", success=r >= dc, dc=dc, roll=r, reason=reason)
