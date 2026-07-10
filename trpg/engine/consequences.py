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


# ── STANDING_RULE ────────────────────────────────────────────────────────────

_FORBIDDEN_IN_RULE = {"STANDING_RULE", "CLOCK_SPAWN", "TIME_ADVANCE"}


def _v_standing_rule(p, world, ctx):
    errs = []
    if not p.get("rule_id"):
        errs.append("STANDING_RULE: rule_id 必填")
    if p.get("rule_id") in world.social.standing_rules:
        errs.append(f"STANDING_RULE: {p.get('rule_id')!r} 已存在")
    trig = p.get("trigger", {})
    kind = trig.get("kind")
    if kind == "periodic":
        if not isinstance(trig.get("interval_days"), (int, float)) or trig["interval_days"] <= 0:
            errs.append("STANDING_RULE: periodic 須 interval_days > 0")
    elif kind == "conditional":
        pred = trig.get("predicate")
        if not isinstance(pred, dict):
            errs.append("STANDING_RULE: conditional 須附 predicate")
        else:
            errs.extend(validate_predicate(pred, world))
    else:
        errs.append(f"STANDING_RULE: 未知 trigger kind {kind!r}")
    if p.get("cancel") is not None:
        errs.extend(validate_predicate(p["cancel"], world))
    for inst in p.get("effects", []):
        t = inst.get("template")
        if t in _FORBIDDEN_IN_RULE:
            errs.append(f"STANDING_RULE: 效果束不得含 {t}（防自激迴圈）")
        elif t not in CONSEQUENCE_REGISTRY:
            errs.append(f"STANDING_RULE: 效果束含未註冊模板 {t!r}")
    if not p.get("effects"):
        errs.append("STANDING_RULE: effects 不得為空")
    return errs


def _x_standing_rule(p, world, ctx):
    from .social_state import StandingRule
    world.social.standing_rules[p["rule_id"]] = StandingRule(
        rule_id=p["rule_id"], owner=p.get("owner", ctx["actor"]),
        trigger=dict(p["trigger"]), effects=list(p["effects"]),
        cancel=p.get("cancel"), last_fired_day=world.social.day)
    return f"常駐規則成立：{p['rule_id']}"


_register(ConsequenceTemplate("STANDING_RULE", _v_standing_rule, _x_standing_rule))


# ── SPAWN / ENTITY_REMOVE / RECRUIT / TRAVEL / AGGREGATE_STATE ──────────────

SPAWN_SOURCES = {"oracle", "craft", "recruit_org", "collective"}
AGGREGATE_VARS = {"物價", "供給", "民心", "治安"}
AGGREGATE_DEFAULT = 5


def _v_spawn(p, world, ctx):
    errs = []
    kind, eid = p.get("kind"), p.get("entity_id")
    if p.get("source") not in SPAWN_SOURCES:
        errs.append(f"SPAWN: 來源 {p.get('source')!r} 不合法（{sorted(SPAWN_SOURCES)}）")
    if not eid:
        errs.append("SPAWN: entity_id 必填")
    if kind == "npc":
        if eid in world.social.npc_cards:
            errs.append(f"SPAWN: NPC {eid!r} 已存在")
        if not p.get("name"):
            errs.append("SPAWN(npc): name 必填")
    elif kind == "organization":
        if eid in world.social.factions:
            errs.append(f"SPAWN: 組織 {eid!r} 已存在")
    elif kind == "item":
        dst = p.get("dst")
        if dst is not None and dst not in world.characters:
            errs.append(f"SPAWN(item): 持有者 {dst!r} 不存在")
    elif kind == "location":
        if eid in world.social.spawned_locations:
            errs.append(f"SPAWN: 地點 {eid!r} 已存在")
    else:
        errs.append(f"SPAWN: 未知 kind {kind!r}")
    return errs


def _x_spawn(p, world, ctx):
    from .social_state import NPCCard, Faction
    kind, eid = p["kind"], p["entity_id"]
    if kind == "npc":
        world.social.npc_cards[eid] = NPCCard(
            npc_id=eid, name=p["name"], goals=list(p.get("goals", [])),
            disposition={"party": p.get("disposition", 0)},
            insight=p.get("insight", 12), will=p.get("will", 12),
            faction_id=p.get("faction_id"))
        return f"新人物登場：{p['name']}"
    if kind == "organization":
        world.social.factions[eid] = Faction(faction_id=eid, name=p.get("name", eid))
        return f"新組織成立：{p.get('name', eid)}"
    if kind == "item":
        dst = p.get("dst")
        if dst is not None:
            world.characters[dst].gear.append(eid)
            return f"{dst} 獲得了 {eid}"
        world.social.object_states[eid] = p.get("state", "完好")
        return f"出現了 {eid}"
    world.social.spawned_locations[eid] = {"name": p.get("name", eid),
                                           "desc": p.get("desc", "")}
    return f"發現新地點：{p.get('name', eid)}"


_register(ConsequenceTemplate("SPAWN", _v_spawn, _x_spawn))


def _v_entity_remove(p, world, ctx):
    errs = []
    card = _require_card(world, p.get("entity_id"))
    if card is None:
        return [f"ENTITY_REMOVE: {p.get('entity_id')!r} 沒有 NPC 卡"]
    if card.removed:
        errs.append(f"ENTITY_REMOVE: {card.name} 已被移除")
    if p.get("way") not in ("死亡", "俘虜", "驅離"):
        errs.append("ENTITY_REMOVE: way 必須是 死亡/俘虜/驅離")
    r = ctx.get("ruling") or {}
    if not (r.get("kind") == "check" and r.get("success")):
        errs.append("ENTITY_REMOVE: 須附成功的檢定裁決（無助前提由裁決層背書；"
                    "清醒有備者必須走 START_COMBAT）")
    return errs


def _x_entity_remove(p, world, ctx):
    s = world.social
    card = s.npc_cards[p["entity_id"]]
    card.removed = p["way"]
    # 外向引用處置（壓測漏洞7）：己方常駐規則停用；雙向承諾轉 orphaned（不蒸發）
    for rule in s.standing_rules.values():
        if rule.owner == p["entity_id"]:
            rule.active = False
    for pr in s.promises.values():
        if pr.status == "open" and p["entity_id"] in (pr.a, pr.b):
            pr.status = "orphaned"
    if p.get("leave_corpse") and p["way"] == "死亡":
        s.object_states[f"屍體:{card.name}"] = "可見"
    return f"{card.name} 被{p['way']}，退出社會層"


_register(ConsequenceTemplate("ENTITY_REMOVE", _v_entity_remove, _x_entity_remove))


def _v_recruit(p, world, ctx):
    errs = []
    card = _require_card(world, p.get("entity_id"))
    if card is None:
        return [f"RECRUIT: {p.get('entity_id')!r} 沒有 NPC 卡"]
    if card.removed:
        errs.append(f"RECRUIT: {card.name} 已被移除")
    if card.recruited:
        errs.append(f"RECRUIT: {card.name} 已在名冊上")
    if not isinstance(p.get("loyalty"), int) or not (1 <= p["loyalty"] <= 10):
        errs.append("RECRUIT: loyalty 必須是 1~10 整數")
    wage = p.get("wage")
    if wage is not None and (not isinstance(wage.get("amount"), int)
                             or wage["amount"] <= 0
                             or not wage.get("interval_days")):
        errs.append("RECRUIT: wage 須含正整數 amount 與 interval_days")
    return errs


def _x_recruit(p, world, ctx):
    card = world.social.npc_cards[p["entity_id"]]
    card.recruited = {"role": p.get("role", ""), "loyalty": p["loyalty"]}
    wage = p.get("wage")
    if wage:
        from .social_state import StandingRule
        rid = f"wage:{p['entity_id']}"
        world.social.standing_rules[rid] = StandingRule(
            rule_id=rid, owner=ctx["actor"],
            trigger={"kind": "periodic", "interval_days": wage["interval_days"]},
            effects=[{"template": "TRANSACT",
                      "params": {"kind": "money", "amount": wage["amount"],
                                 "src": ctx["actor"], "dst": p["entity_id"]}}],
            last_fired_day=world.social.day)
    return f"{card.name} 加入名冊（{p.get('role', '')}）"


_register(ConsequenceTemplate("RECRUIT", _v_recruit, _x_recruit))


def _v_travel(p, world, ctx):
    dest = p.get("dest")
    if dest in world.social.spawned_locations:
        return []
    dmap = world.dungeon_map
    if dmap is None or dest not in getattr(dmap, "rooms", {}):
        return [f"TRAVEL: 目的地 {dest!r} 不存在"]
    mode = p.get("mode", "徒步")
    if mode == "徒步" and dest not in dmap.current_room.exits.values():
        return [f"TRAVEL: {dest!r} 與當前位置不連通（徒步）"]
    if mode == "已知節點" and not dmap.rooms[dest].visited:
        return [f"TRAVEL: {dest!r} 不是已知節點（傳送需已知或視線）"]
    return []


def _x_travel(p, world, ctx):
    dest = p["dest"]
    dmap = world.dungeon_map
    fact = f"隊伍抵達 {dest}"
    if dmap is not None and dest in getattr(dmap, "rooms", {}):
        dmap.current_room_id = dest
        dmap.current_room.visited = True
        fact = f"隊伍抵達 {dmap.current_room.name}"
    else:
        world.social.flags.add(f"at:{dest}")
    hours = p.get("hours", 0)
    if hours:
        from .social_time import advance_time
        advance_time(world, int(hours * 60))
        fact += f"（耗時 {hours} 小時）"
    return fact


_register(ConsequenceTemplate("TRAVEL", _v_travel, _x_travel))


def _v_aggregate(p, world, ctx):
    errs = []
    if p.get("var") not in AGGREGATE_VARS:
        errs.append(f"AGGREGATE_STATE: {p.get('var')!r} 不在聚合變數目錄")
    if not p.get("region"):
        errs.append("AGGREGATE_STATE: region 必填")
    if not isinstance(p.get("delta"), int) or not (1 <= abs(p["delta"]) <= 3):
        errs.append("AGGREGATE_STATE: delta 必須是 ±1~3 的整數")
    return errs


def _x_aggregate(p, world, ctx):
    key = f"{p['var']}:{p['region']}"
    eff = _effective_delta(world, ctx, "AGGREGATE_STATE",
                           {"var": p["var"], "region": p["region"]}, p["delta"])
    old = world.social.aggregates.get(key, AGGREGATE_DEFAULT)
    world.social.aggregates[key] = _clamp(old + eff, 0, 10)   # 絕對上下限（壓測漏洞3）
    return f"{p['region']}的{p['var']}：{old} → {world.social.aggregates[key]}"


_register(ConsequenceTemplate("AGGREGATE_STATE", _v_aggregate, _x_aggregate))
