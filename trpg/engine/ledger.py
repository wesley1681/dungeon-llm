"""Append-only commit ledger — 社會層狀態變更的唯一事實來源。

每次後果模板執行都落成一筆不可變 Commit。這讓三條規則從「LLM 記憶」
變成機械查詢：intent_table 規則 6（禁回溯 → has_prior）、規則 7 的審計
（Commit.ruling 存 DC/骰值/理由）、consequence_table 同軸遞減（count_axis）。
Ledger 沒有任何刪改方法——修正錯誤靠追加新 commit，不靠塗改歷史。
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class Commit:
    seq: int               # 單調遞增流水號（0 起）
    time_minutes: int      # 落帳時的遊戲內時間
    template_id: str       # 後果模板 id，如 "DISPOSITION"
    params: dict           # 執行時的參數（防禦性深拷貝）
    actor: str             # 提案者："party" / "npc:<id>" / "engine"
    ruling: dict           # 裁決記錄；無裁決時 {"kind": "none"}


class Ledger:
    def __init__(self) -> None:
        self._commits: list[Commit] = []

    def __len__(self) -> int:
        return len(self._commits)

    def __iter__(self):
        return iter(self._commits)

    def append(self, template_id: str, params: dict, *, actor: str,
               time_minutes: int, ruling: Optional[dict] = None) -> Commit:
        c = Commit(
            seq=len(self._commits),
            time_minutes=time_minutes,
            template_id=template_id,
            params=copy.deepcopy(params),
            actor=actor,
            ruling=copy.deepcopy(ruling) if ruling else {"kind": "none"},
        )
        self._commits.append(c)
        return c

    def query(self, template_id: Optional[str] = None,
              actor: Optional[str] = None,
              where: Optional[Callable[[Commit], bool]] = None) -> list[Commit]:
        out = []
        for c in self._commits:
            if template_id is not None and c.template_id != template_id:
                continue
            if actor is not None and c.actor != actor:
                continue
            if where is not None and not where(c):
                continue
            out.append(c)
        return out

    def has_prior(self, template_id: str,
                  where: Optional[Callable[[Commit], bool]] = None) -> bool:
        return bool(self.query(template_id=template_id, where=where))

    def count_axis(self, template_id: str, actor: str, axis: dict, *,
                   since_minutes: int, now_minutes: int) -> int:
        """同一 (actor, template, 軸) 在最近 since_minutes 窗內的落帳次數。
        軸=params 的鍵值子集全等（consequence_table 通用規則「同軸遞減」）。"""
        def in_axis(c: Commit) -> bool:
            if now_minutes - c.time_minutes >= since_minutes:
                return False
            return all(c.params.get(k) == v for k, v in axis.items())
        return len(self.query(template_id=template_id, actor=actor, where=in_axis))

    def to_dicts(self) -> list[dict]:
        return [{"seq": c.seq, "time_minutes": c.time_minutes,
                 "template_id": c.template_id, "params": copy.deepcopy(c.params),
                 "actor": c.actor, "ruling": copy.deepcopy(c.ruling)}
                for c in self._commits]

    @classmethod
    def from_dicts(cls, rows: list[dict]) -> "Ledger":
        led = cls()
        for r in rows:
            led._commits.append(Commit(
                seq=r["seq"], time_minutes=r["time_minutes"],
                template_id=r["template_id"], params=copy.deepcopy(r["params"]),
                actor=r["actor"], ruling=copy.deepcopy(r["ruling"])))
        return led
