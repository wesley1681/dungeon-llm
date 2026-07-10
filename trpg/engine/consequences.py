"""後果模板註冊表——GM 提案的唯一出口，engine 是唯一執行者。

與 ABILITY_REGISTRY 同構（abilities.py）：模板=schema、參數帶=合法欄位、
_register 填表。tag_parser._dispatch 的 if/elif 派發最終遷移到這裡
（該遷移屬 LLM 整合 plan）。

commit_bundle 是唯一入口：先驗證整束、再逐一執行並落帳（Ledger）。
半束落地＝state 汙染，所以驗證失敗時整束報廢——這是原子性契約。
規格：根目錄 consequence_table.md（19 模板 + 通用驗證規則）。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class ConsequenceTemplate:
    template_id: str
    # validate(params, world, ctx) -> list[str]  空列表=合法
    validate: Callable
    # execute(params, world, ctx) -> str  回傳事實句（給 GM 敘事的素材）
    execute: Callable


CONSEQUENCE_REGISTRY: dict[str, ConsequenceTemplate] = {}


def _register(t: ConsequenceTemplate) -> ConsequenceTemplate:
    CONSEQUENCE_REGISTRY[t.template_id] = t
    return t


def commit_bundle(world, bundle: list[dict], *, actor: str,
                  ruling: Optional[dict] = None) -> tuple[bool, list[str]]:
    """驗證→執行→落帳。回 (ok, messages)：ok=False 時 messages 是拒絕原因
    （回注給 GM 重寫提案），ok=True 時是每個模板的事實句。"""
    ctx = {"actor": actor, "ruling": ruling}
    errors: list[str] = []
    for inst in bundle:
        tid = inst.get("template")
        t = CONSEQUENCE_REGISTRY.get(tid)
        if t is None:
            errors.append(f"模板 {tid!r} 未註冊")
            continue
        errors.extend(t.validate(inst.get("params", {}), world, ctx))
    if errors:
        return False, errors

    facts: list[str] = []
    for inst in bundle:
        t = CONSEQUENCE_REGISTRY[inst["template"]]
        params = inst.get("params", {})
        fact = t.execute(params, world, ctx)
        facts.append(fact if isinstance(fact, str) else "")
        world.social.ledger.append(t.template_id, params, actor=actor,
                                   time_minutes=world.social.time_minutes,
                                   ruling=ruling)
    return True, facts


# ═══════════════════════════════════════════════════════════════════════════
# 模板實作（依 consequence_table.md；分組追加，_register 即生效）
# ═══════════════════════════════════════════════════════════════════════════

from .predicates import validate_predicate
from .social_state import Belief, KnowledgeItem, Promise

_DELTA_WINDOW_MIN = 1440   # 同軸遞減的回看窗（1 遊戲日）


def _effective_delta(world, ctx, template_id: str, axis: dict, delta: int) -> int:
    """同軸遞減：窗內每筆同軸 commit 使 |delta| 遞減 1，最低 0（表二通用規則）。"""
    n = world.social.ledger.count_axis(
        template_id, ctx["actor"], axis,
        since_minutes=_DELTA_WINDOW_MIN, now_minutes=world.social.time_minutes)
    mag = max(0, abs(delta) - n)
    return mag if delta >= 0 else -mag


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _require_card(world, npc_id):
    return world.social.npc_cards.get(npc_id)


# ── DISPOSITION ──────────────────────────────────────────────────────────────

def _v_disposition(p, world, ctx):
    errs = []
    if _require_card(world, p.get("observer")) is None:
        errs.append(f"DISPOSITION: 觀察方 {p.get('observer')!r} 沒有 NPC 卡")
    if not isinstance(p.get("delta"), int) or not (1 <= abs(p["delta"]) <= 3):
        errs.append("DISPOSITION: delta 必須是 ±1~3 的整數")
    tgt = p.get("target", "party")
    if tgt != "party" and _require_card(world, tgt) is None and tgt not in world.characters:
        errs.append(f"DISPOSITION: 對象 {tgt!r} 不存在")
    return errs


def _x_disposition(p, world, ctx):
    card = world.social.npc_cards[p["observer"]]
    tgt = p.get("target", "party")
    eff = _effective_delta(world, ctx, "DISPOSITION",
                           {"observer": p["observer"], "target": tgt}, p["delta"])
    old = card.disposition.get(tgt, 0)
    card.disposition[tgt] = _clamp(old + eff, -5, 5)
    if eff == 0:
        return f"{card.name} 對 {tgt} 的態度不再動搖（邊際效應遞減）"
    return f"{card.name} 對 {tgt} 的態度 {old:+d} → {card.disposition[tgt]:+d}"


_register(ConsequenceTemplate("DISPOSITION", _v_disposition, _x_disposition))


# ── BELIEF_FLAG ──────────────────────────────────────────────────────────────

def _v_belief(p, world, ctx):
    errs = []
    if _require_card(world, p.get("holder")) is None:
        errs.append(f"BELIEF_FLAG: 對象 {p.get('holder')!r} 沒有 NPC 卡")
    if not p.get("content"):
        errs.append("BELIEF_FLAG: content 必填")
    flaw = p.get("flaw")
    if not isinstance(flaw, dict):
        errs.append("BELIEF_FLAG: 破綻條件必填且須為謂詞")
    else:
        errs.extend(validate_predicate(flaw, world))   # 壓測漏洞2：可評估、可達
    return errs


def _x_belief(p, world, ctx):
    card = world.social.npc_cards[p["holder"]]
    card.beliefs.append(Belief(content=p["content"], flaw=dict(p["flaw"]),
                               source=ctx["actor"]))
    return f"{card.name} 相信了：{p['content']}"


_register(ConsequenceTemplate("BELIEF_FLAG", _v_belief, _x_belief))


# ── REVEAL ───────────────────────────────────────────────────────────────────

def _v_reveal(p, world, ctx):
    errs = []
    fid = p.get("fact_id")
    if not fid:
        return ["REVEAL: fact_id 必填"]
    src = p.get("source", "card")
    if src == "card":
        card = _require_card(world, p.get("npc_id"))
        if card is None or fid not in card.knowledge:
            errs.append(f"REVEAL: 知識 {fid!r} 不存在於 {p.get('npc_id')!r} 的卡上（不可無中生有）")
    elif src == "oracle":
        if not p.get("text"):
            errs.append("REVEAL(oracle): text 必填")
        if fid in world.social.party_knowledge or world.social.ledger.has_prior(
                "REVEAL", where=lambda c: c.params.get("fact_id") == fid):
            errs.append(f"REVEAL(oracle): {fid!r} 已是 canon，不可重擲")
    else:
        errs.append(f"REVEAL: 未知 source {src!r}")
    return errs


def _x_reveal(p, world, ctx):
    fid = p["fact_id"]
    if p.get("source", "card") == "card":
        item = world.social.npc_cards[p["npc_id"]].knowledge[fid]
        item.revealed = True
        world.social.party_knowledge[fid] = item
        return f"隊伍得知：{item.text}"
    item = KnowledgeItem(fact_id=fid, text=p["text"], revealed=True)
    world.social.party_knowledge[fid] = item
    return f"（神諭）隊伍得知：{item.text}"


_register(ConsequenceTemplate("REVEAL", _v_reveal, _x_reveal))


# ── PROMISE_FLAG ─────────────────────────────────────────────────────────────

def _v_promise(p, world, ctx):
    errs = []
    if not p.get("promise_id") or not p.get("content"):
        errs.append("PROMISE_FLAG: promise_id 與 content 必填")
    if p.get("promise_id") in world.social.promises:
        errs.append(f"PROMISE_FLAG: {p.get('promise_id')!r} 已存在")
    for side in ("a", "b"):
        eid = p.get(side)
        if eid != "party" and _require_card(world, eid) is None and eid not in world.characters:
            errs.append(f"PROMISE_FLAG: 當事方 {eid!r} 不存在")
    due = p.get("due")
    if due is not None:
        errs.extend(validate_predicate(due, world))
    return errs


def _x_promise(p, world, ctx):
    pr = Promise(promise_id=p["promise_id"], a=p["a"], b=p["b"],
                 content=p["content"], due=p.get("due"))
    world.social.promises[pr.promise_id] = pr
    return f"約定成立：{p['a']} ↔ {p['b']}——{p['content']}"


_register(ConsequenceTemplate("PROMISE_FLAG", _v_promise, _x_promise))


# ── REP_FACTION ──────────────────────────────────────────────────────────────

def _v_rep(p, world, ctx):
    errs = []
    if p.get("faction_id") not in world.social.factions:
        errs.append(f"REP_FACTION: 陣營 {p.get('faction_id')!r} 不存在")
    if not isinstance(p.get("delta"), int) or not (1 <= abs(p["delta"]) <= 3):
        errs.append("REP_FACTION: delta 必須是 ±1~3 的整數")
    return errs


def _x_rep(p, world, ctx):
    f = world.social.factions[p["faction_id"]]
    eff = _effective_delta(world, ctx, "REP_FACTION",
                           {"faction_id": p["faction_id"]}, p["delta"])
    old = f.reputation
    f.reputation = _clamp(old + eff, -10, 10)
    return f"{f.name} 對隊伍的名聲 {old:+d} → {f.reputation:+d}"


_register(ConsequenceTemplate("REP_FACTION", _v_rep, _x_rep))
