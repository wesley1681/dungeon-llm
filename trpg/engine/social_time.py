"""遊戲內時間推進——時鐘 tick 與到期觸發。

TIME_ADVANCE 是時間唯一入口（表二）；到期束以 actor="engine" 落帳。
級聯護欄（第三輪壓測漏洞4）：到期束可能又生時鐘/推時間，深度超過
MAX_CASCADE_DEPTH 的觸發直接截斷（時鐘留著 expired=True，束不執行）。
常駐規則評估掛在每次 advance_time 尾端。
"""
from __future__ import annotations

MAX_CASCADE_DEPTH = 3
MAX_FIRES_PER_RULE = 3


def advance_time(world, minutes: int, _depth: int = 0) -> list[str]:
    s = world.social
    s.time_minutes += minutes
    days = minutes / 1440.0
    facts: list[str] = []
    from .consequences import commit_bundle   # 局部 import 避免循環
    for clock in list(s.clocks.values()):
        if clock.expired:
            continue
        clock.remaining_days -= days
        if clock.remaining_days <= 0:
            clock.remaining_days = 0.0
            clock.expired = True
            facts.append(f"時鐘到期：{clock.name}")
            if clock.on_expire and _depth < MAX_CASCADE_DEPTH:
                ok, msgs = commit_bundle(world, clock.on_expire, actor="engine")
                facts.extend(msgs if ok else [f"（到期束被拒：{'；'.join(msgs)}）"])
    facts.extend(_evaluate_standing_rules(world, _depth))
    return facts


def _evaluate_standing_rules(world, _depth: int) -> list[str]:
    if _depth >= MAX_CASCADE_DEPTH:
        return []
    from .consequences import commit_bundle
    from .predicates import evaluate_predicate
    s = world.social
    facts: list[str] = []
    for rule in list(s.standing_rules.values()):
        if not rule.active:
            continue
        if rule.cancel and evaluate_predicate(rule.cancel, world):
            rule.active = False
            facts.append(f"常駐規則取消：{rule.rule_id}")
            continue
        fires = 0
        if rule.trigger["kind"] == "periodic":
            interval = rule.trigger["interval_days"]
            base = rule.last_fired_day if rule.last_fired_day is not None else 0.0
            while base + interval <= s.day and fires < MAX_FIRES_PER_RULE:
                base += interval
                fires += 1
            if fires:
                rule.last_fired_day = base
        else:   # conditional：謂詞真→本次 advance_time 觸發一次
            if evaluate_predicate(rule.trigger["predicate"], world):
                fires = 1
        for _ in range(fires):
            ok, msgs = commit_bundle(world, rule.effects,
                                     actor=f"engine:rule:{rule.rule_id}")
            facts.extend(msgs if ok else [f"（規則 {rule.rule_id} 效果被拒：{'；'.join(msgs)}）"])
    return facts
