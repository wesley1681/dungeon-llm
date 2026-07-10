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


# ── OBJ_STATE / TRANSACT / RESOURCE / CONDITION ─────────────────────────────

# 物件狀態機：通用合法轉移（之後可按物件類別細化；未列出的轉移一律非法）
OBJ_TRANSITIONS = {
    ("完好", "損壞"), ("損壞", "毀壞"), ("完好", "毀壞"), ("損壞", "完好"),
    ("鎖上", "開啟"), ("開啟", "鎖上"),
    ("隱藏", "可見"), ("可見", "隱藏"),
    ("開啟", "下毒"), ("完好", "下毒"),
}

RESOURCE_CATALOG = {"hp", "補給", "彈藥"}

SOCIAL_CONDITIONS = {"中毒", "受傷", "迷醉", "昏睡", "束縛", "通緝", "受詛咒",
                     "被判有罪", "偽裝"}


def _v_obj_state(p, world, ctx):
    oid, new = p.get("object_id"), p.get("new_state")
    cur = world.social.object_states.get(oid)
    if cur is None:
        return [f"OBJ_STATE: 物件 {oid!r} 未登記（不可無中生有——先 SPAWN）"]
    if (cur, new) not in OBJ_TRANSITIONS:
        return [f"OBJ_STATE: {cur} → {new} 不是合法轉移"]
    return []


def _x_obj_state(p, world, ctx):
    old = world.social.object_states[p["object_id"]]
    world.social.object_states[p["object_id"]] = p["new_state"]
    return f"{p['object_id']}：{old} → {p['new_state']}"


_register(ConsequenceTemplate("OBJ_STATE", _v_obj_state, _x_obj_state))


def _gear_of(world, eid):
    c = world.characters.get(eid)
    return c.gear if c is not None else None


def _v_transact(p, world, ctx):
    kind = p.get("kind")
    if kind == "money":
        amt = p.get("amount")
        if not isinstance(amt, int) or amt <= 0:
            return ["TRANSACT: amount 必須是正整數"]
        if world.social.accounts.get(p.get("src"), 0) < amt:
            return [f"TRANSACT: {p.get('src')!r} 持有金錢不足（轉出方必須確實持有）"]
        return []
    if kind == "item":
        gear = _gear_of(world, p.get("src"))
        if gear is None or p.get("item") not in gear:
            return [f"TRANSACT: {p.get('src')!r} 未持有 {p.get('item')!r}"]
        if _gear_of(world, p.get("dst")) is None:
            return [f"TRANSACT: 收受方 {p.get('dst')!r} 不存在"]
        return []
    return [f"TRANSACT: 未知 kind {kind!r}"]


def _x_transact(p, world, ctx):
    if p["kind"] == "money":
        s = world.social.accounts
        s[p["src"]] = s.get(p["src"], 0) - p["amount"]
        s[p["dst"]] = s.get(p["dst"], 0) + p["amount"]
        return f"{p['src']} 付給 {p['dst']} {p['amount']} 金"
    _gear_of(world, p["src"]).remove(p["item"])
    _gear_of(world, p["dst"]).append(p["item"])
    return f"{p['item']} 由 {p['src']} 轉手給 {p['dst']}"


_register(ConsequenceTemplate("TRANSACT", _v_transact, _x_transact))


def _v_resource(p, world, ctx):
    if p.get("resource") not in RESOURCE_CATALOG:
        return [f"RESOURCE: {p.get('resource')!r} 不在資源目錄 {sorted(RESOURCE_CATALOG)}"]
    if not isinstance(p.get("delta"), int):
        return ["RESOURCE: delta 必須是整數"]
    if p["resource"] == "hp" and p.get("entity") not in world.characters:
        return [f"RESOURCE: 實體 {p.get('entity')!r} 不存在"]
    return []


def _x_resource(p, world, ctx):
    if p["resource"] == "hp":
        c = world.characters[p["entity"]]
        old = c.hp
        c.hp = max(0, min(c.max_hp, c.hp + p["delta"]))
        return f"{c.name} HP {old} → {c.hp}"
    key = (p["entity"], p["resource"])
    old = world.social.resources.get(key, 0)
    world.social.resources[key] = max(0, old + p["delta"])
    return f"{p['entity']} 的{p['resource']} {old} → {world.social.resources[key]}"


_register(ConsequenceTemplate("RESOURCE", _v_resource, _x_resource))


def _v_condition(p, world, ctx):
    errs = []
    if p.get("status") not in SOCIAL_CONDITIONS:
        errs.append(f"CONDITION: {p.get('status')!r} 不在狀態目錄")
    op = p.get("op", "+")
    if op not in ("+", "-"):
        errs.append("CONDITION: op 必須是 '+' 或 '-'")
    if op == "-":
        cur = world.social.conditions.get(p.get("entity"), [])
        if not any(c["status"] == p.get("status") for c in cur):
            errs.append(f"CONDITION: {p.get('entity')!r} 身上沒有 {p.get('status')!r}（−解需狀態在身）")
    return errs


def _x_condition(p, world, ctx):
    lst = world.social.conditions.setdefault(p["entity"], [])
    if p.get("op", "+") == "+":
        lst.append({"status": p["status"], "until_min": p.get("until_min")})
        return f"{p['entity']} 獲得狀態：{p['status']}"
    lst[:] = [c for c in lst if c["status"] != p["status"]]
    return f"{p['entity']} 解除狀態：{p['status']}"


_register(ConsequenceTemplate("CONDITION", _v_condition, _x_condition))


# ── TIME_ADVANCE / CLOCK_MODIFY / CLOCK_SPAWN ───────────────────────────────

def _v_time_advance(p, world, ctx):
    m = p.get("minutes")
    if not isinstance(m, int) or m <= 0:
        return ["TIME_ADVANCE: minutes 必須是正整數"]
    return []


def _x_time_advance(p, world, ctx):
    from .social_time import advance_time
    facts = advance_time(world, p["minutes"])
    hours = p["minutes"] / 60
    lead = f"時間推進 {hours:.1f} 小時"
    return "；".join([lead] + facts) if facts else lead


_register(ConsequenceTemplate("TIME_ADVANCE", _v_time_advance, _x_time_advance))


def _v_clock_modify(p, world, ctx):
    errs = []
    clock = world.social.clocks.get(p.get("clock_id"))
    if clock is None:
        return [f"CLOCK_MODIFY: 時鐘 {p.get('clock_id')!r} 不存在"]
    d = p.get("delta_days")
    if not isinstance(d, (int, float)) or not (0 < abs(d) <= 2):
        errs.append("CLOCK_MODIFY: |delta_days| 必須 ≤ 2 且非零")
    # 壓測漏洞4：延後（+）自己擁有的時鐘須付代價＝附成功的 check 裁決
    if isinstance(d, (int, float)) and d > 0 and clock.owner == ctx["actor"]:
        r = ctx.get("ruling") or {}
        if not (r.get("kind") == "check" and r.get("success")):
            errs.append("CLOCK_MODIFY: 延後自己擁有的時鐘須通過檢定（無阻力≠自動成功的例外）")
    return errs


def _x_clock_modify(p, world, ctx):
    c = world.social.clocks[p["clock_id"]]
    old = c.remaining_days
    c.remaining_days = max(0.0, c.remaining_days + p["delta_days"])
    return f"時鐘「{c.name}」{old:.1f} 天 → {c.remaining_days:.1f} 天"


_register(ConsequenceTemplate("CLOCK_MODIFY", _v_clock_modify, _x_clock_modify))


_FORBIDDEN_IN_EXPIRE = {"TIME_ADVANCE", "STANDING_RULE"}


def _v_clock_spawn(p, world, ctx):
    errs = []
    cid = p.get("clock_id")
    if not cid or not p.get("name"):
        errs.append("CLOCK_SPAWN: clock_id 與 name 必填")
    if cid in world.social.clocks:
        errs.append(f"CLOCK_SPAWN: {cid!r} 已存在")
    if not isinstance(p.get("days"), (int, float)) or p["days"] < 0:
        errs.append("CLOCK_SPAWN: days 必須 ≥ 0")
    for inst in p.get("on_expire", []):
        t = inst.get("template")
        if t in _FORBIDDEN_IN_EXPIRE:
            errs.append(f"CLOCK_SPAWN: 到期束不得含 {t}")
        if t == "CLOCK_SPAWN" and inst.get("params", {}).get("clock_id") == cid:
            errs.append("CLOCK_SPAWN: 到期束不得重生自身（不朽時鐘）")
        if t == "CLOCK_MODIFY" and inst.get("params", {}).get("clock_id") == cid:
            errs.append("CLOCK_SPAWN: 到期束不得修改自身")
        if t not in CONSEQUENCE_REGISTRY:
            errs.append(f"CLOCK_SPAWN: 到期束含未註冊模板 {t!r}")
    return errs


def _x_clock_spawn(p, world, ctx):
    from .social_state import Clock
    world.social.clocks[p["clock_id"]] = Clock(
        clock_id=p["clock_id"], name=p["name"], remaining_days=float(p["days"]),
        on_expire=list(p.get("on_expire", [])), owner=p.get("owner", ctx["actor"]))
    return f"新時鐘：{p['name']}（{p['days']} 天）"


_register(ConsequenceTemplate("CLOCK_SPAWN", _v_clock_spawn, _x_clock_spawn))
