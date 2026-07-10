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
