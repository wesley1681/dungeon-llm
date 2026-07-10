"""引擎可評估謂詞——破綻條件/期限/常駐規則觸發的機器面。

表二通用規則：「所有條件式參數必須是引擎可評估、且指向可達 state 的謂詞；
不可證偽/恆假的條件拒絕 commit。」validate_predicate 是那條規則的實作：
kind 必須註冊、參數齊全、引用的實體存在。新增 kind＝在 PREDICATE_KINDS
登記＋在 _EVALUATORS 補一個函式，兩處都不改既有項（append-only）。
"""
from __future__ import annotations

# kind → 必要參數名
PREDICATE_KINDS: dict[str, tuple] = {
    "clock_expired": ("clock_id",),
    "time_after": ("minutes",),
    "entity_removed": ("entity_id",),
    "disposition_at_least": ("npc_id", "target", "value"),
    "flag": ("flag_id",),
}


def validate_predicate(pred: dict, world) -> list[str]:
    errs: list[str] = []
    kind = pred.get("kind")
    if kind not in PREDICATE_KINDS:
        return [f"未知謂詞 kind: {kind!r}（合法：{sorted(PREDICATE_KINDS)}）"]
    for p in PREDICATE_KINDS[kind]:
        if p not in pred:
            errs.append(f"謂詞 {kind} 缺參數 {p}")
    if errs:
        return errs
    s = world.social
    if kind == "clock_expired" and pred["clock_id"] not in s.clocks:
        errs.append(f"謂詞引用不存在的時鐘 {pred['clock_id']!r}")
    if kind in ("entity_removed", "disposition_at_least"):
        eid = pred.get("entity_id") or pred.get("npc_id")
        if eid not in s.npc_cards and eid not in world.characters:
            errs.append(f"謂詞引用不存在的實體 {eid!r}")
    return errs


def _eval_clock_expired(pred, world):
    c = world.social.clocks.get(pred["clock_id"])
    return bool(c and c.expired)


def _eval_time_after(pred, world):
    return world.social.time_minutes > pred["minutes"]


def _eval_entity_removed(pred, world):
    card = world.social.npc_cards.get(pred["entity_id"])
    return bool(card and card.removed)


def _eval_disposition_at_least(pred, world):
    card = world.social.npc_cards.get(pred["npc_id"])
    if card is None:
        return False
    return card.disposition.get(pred["target"], 0) >= pred["value"]


def _eval_flag(pred, world):
    return pred["flag_id"] in world.social.flags


_EVALUATORS = {
    "clock_expired": _eval_clock_expired,
    "time_after": _eval_time_after,
    "entity_removed": _eval_entity_removed,
    "disposition_at_least": _eval_disposition_at_least,
    "flag": _eval_flag,
}


def evaluate_predicate(pred: dict, world) -> bool:
    kind = pred.get("kind")
    fn = _EVALUATORS.get(kind)
    if fn is None:
        return False   # 不可評估＝永不為真（validate 應已擋下）
    return fn(pred, world)
