"""Deterministic, weighted ability-score growth shared by every Character.

The profile contains rules, never a character's concrete panel.  Class
weights establish the main direction; race multipliers only bias that
direction.  Each awarded point is rolled independently and recorded so a
saved character never changes when definitions are rebalanced later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import random


STAT_NAMES: tuple[str, ...] = ("STR", "DEX", "CON", "INT", "WIS", "CHA")


@dataclass(frozen=True)
class StatGrowth:
    level: int
    point: int
    stat: str


@dataclass(frozen=True)
class GrowthProfile:
    class_weights: dict[str, float]
    race_multipliers: dict[str, float] = field(default_factory=dict)
    points_per_level: int = 2
    stat_cap: int | None = 20
    enabled: bool = True

    def __post_init__(self) -> None:
        unknown = (set(self.class_weights) | set(self.race_multipliers)) - set(STAT_NAMES)
        if unknown:
            raise ValueError(f"未知六維名稱：{sorted(unknown)}")
        if self.points_per_level < 0:
            raise ValueError("每級成長點數不可為負數")
        if self.stat_cap is not None and self.stat_cap < 1:
            raise ValueError("六維上限至少為 1")

    def normalized_weights(self, available: tuple[str, ...] = STAT_NAMES) -> dict[str, float]:
        """Return class_weight * race_multiplier, normalized to sum to one."""
        raw: dict[str, float] = {}
        for stat in available:
            class_weight = float(self.class_weights.get(stat, 0.0))
            race_multiplier = float(self.race_multipliers.get(stat, 1.0))
            if class_weight < 0 or race_multiplier < 0:
                raise ValueError(f"成長權重不可為負數：{stat}")
            raw[stat] = class_weight * race_multiplier
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("至少需要一個大於 0 的六維成長權重")
        return {stat: weight / total for stat, weight in raw.items()}

    def choose_stat(self, stats, *, seed: int, level: int, point: int) -> str | None:
        """Choose one score with a stable per-level/per-point RNG stream."""
        available = tuple(
            stat for stat in STAT_NAMES
            if self.stat_cap is None or getattr(stats, stat) < self.stat_cap
        )
        if not available:
            return None
        weights = self.normalized_weights(available)
        # Do not use hash(name): Python intentionally randomizes it per process.
        point_seed = int(seed) * 1_000_003 + int(level) * 1_009 + int(point)
        rng = random.Random(point_seed)
        return rng.choices(available, weights=[weights[s] for s in available], k=1)[0]


def allocate_level_growth(stats, profile: GrowthProfile, *, seed: int,
                          level: int) -> list[StatGrowth]:
    """Mutate ``stats`` with one level's gains and return the audit records."""
    if not profile.enabled or profile.points_per_level <= 0:
        return []
    gains: list[StatGrowth] = []
    for point in range(profile.points_per_level):
        stat = profile.choose_stat(stats, seed=seed, level=level, point=point)
        if stat is None:
            break
        setattr(stats, stat, getattr(stats, stat) + 1)
        gains.append(StatGrowth(level=level, point=point, stat=stat))
    return gains
