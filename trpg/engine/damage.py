"""Canonical damage-type vocabulary (single source of truth).

5e's 13 damage types, in the engine's existing string convention (weapons
use 斬擊/穿刺/鈍擊, spells use 火/冰/光耀/力場 — already in live data).
Everything that keys on a damage type (typed resistance tables, the obs v4
capability-descriptor resist summary) must validate against THIS tuple —
a typo'd type in a monster def should fail loudly at registration, not
silently never match at runtime.

"untyped" / "environment" are deliberately NOT in the list: they are
engine-internal channels (terrain ticks, generic effects) that resistance
tables can never address.

Append-only: the obs descriptor maps these to fixed columns.
"""
from __future__ import annotations

DAMAGE_TYPES: tuple[str, ...] = (
    "斬擊", "穿刺", "鈍擊",            # physical (weapons)
    "火", "冰", "閃電", "雷鳴", "強酸", "毒",   # elemental
    "光耀", "黯蝕", "力場", "精神",      # magical
)
N_DAMAGE_TYPES = len(DAMAGE_TYPES)   # 13
DAMAGE_TYPE_INDEX: dict[str, int] = {t: i for i, t in enumerate(DAMAGE_TYPES)}


def validate_damage_type(dtype: str, context: str = "") -> str:
    """Fail-loud guard for data entries (monster defs, trait params)."""
    if dtype not in DAMAGE_TYPE_INDEX:
        raise ValueError(
            f"unknown damage type {dtype!r}{' in ' + context if context else ''}"
            f" — must be one of {DAMAGE_TYPES}")
    return dtype
