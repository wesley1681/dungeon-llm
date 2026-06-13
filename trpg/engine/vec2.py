"""2D spatial primitives for the combat engine.

Vec2: immutable position/direction vector with the helpers combat math needs.
Battlefield: bounded rectangular play area with a terrain grid and LoS check.

Origin is at (0, 0); the playable area is [0, width] × [0, height]. A position
outside that rectangle is out of bounds.

Terrain grid cells store a TerrainType:
  NORMAL    — no effect
  DIFFICULT — entering doubles movement cost; doesn't block LoS
  DANGEROUS — same movement cost as DIFFICULT; deals damage at turn start
  BLOCKED   — full wall; blocks movement endpoints and LoS
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum


@dataclass(frozen=True)
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def distance_to(self, other: "Vec2") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Vec2":
        L = self.length()
        if L < 1e-9:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / L, self.y / L)

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, s: float) -> "Vec2":
        return Vec2(self.x * s, self.y * s)

    __rmul__ = __mul__

    def __iter__(self):
        yield self.x
        yield self.y

    @classmethod
    def coerce(cls, value) -> "Vec2":
        """Accept Vec2, (x, y) tuple/list, or scalar (treated as x with y=0)."""
        if isinstance(value, Vec2):
            return value
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return cls(float(value[0]), float(value[1]))
        if isinstance(value, (int, float)):
            return cls(float(value), 0.0)
        raise TypeError(f"cannot coerce {type(value).__name__} to Vec2")


def point_segment_distance(p: "Vec2", a: "Vec2", b: "Vec2") -> float:
    """Shortest distance from point `p` to the segment a→b.

    LINE-shaped effects (breath weapons, lightning bolt) define their area as
    "within width/2 of the caster→endpoint segment" — both the engine's
    affected-target resolution and the policies' friendly-fire checks must use
    this same geometry, so it lives here next to Vec2."""
    ab = b - a
    ab_len2 = ab.x * ab.x + ab.y * ab.y
    if ab_len2 < 1e-12:
        return p.distance_to(a)
    t = ((p.x - a.x) * ab.x + (p.y - a.y) * ab.y) / ab_len2
    t = max(0.0, min(1.0, t))
    return p.distance_to(Vec2(a.x + ab.x * t, a.y + ab.y * t))


class TerrainType(IntEnum):
    """Per-cell terrain. Numeric values are the schema — don't reorder."""
    NORMAL    = 0
    DIFFICULT = 1   # entering doubles movement cost
    DANGEROUS = 2   # entering doubles cost + damages at turn start
    BLOCKED   = 3   # wall: blocks movement endpoints + line of sight


_TERRAIN_MOVE_MULT = {
    TerrainType.NORMAL:    1.0,
    TerrainType.DIFFICULT: 2.0,
    TerrainType.DANGEROUS: 2.0,
    TerrainType.BLOCKED:   float("inf"),   # unreachable
}


@dataclass
class Battlefield:
    """Bounded play area with a 2D terrain grid at `grid_resolution` m / cell.

    `cells[iy][ix]` stores a TerrainType. Defaults to all NORMAL.
    """
    width: float = 30.0
    height: float = 30.0
    grid_resolution: float = 0.5
    cells: list[list[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.cells:
            nx = max(1, int(round(self.width / self.grid_resolution)))
            ny = max(1, int(round(self.height / self.grid_resolution)))
            self.cells = [[TerrainType.NORMAL] * nx for _ in range(ny)]

    # ── Indexing ────────────────────────────────────────────────────────────

    def _cell(self, p: Vec2) -> tuple[int, int]:
        ix = int(p.x / self.grid_resolution)
        iy = int(p.y / self.grid_resolution)
        ix = max(0, min(ix, len(self.cells[0]) - 1))
        iy = max(0, min(iy, len(self.cells) - 1))
        return ix, iy

    def in_bounds(self, p: Vec2) -> bool:
        return 0.0 <= p.x <= self.width and 0.0 <= p.y <= self.height

    # ── Terrain queries ─────────────────────────────────────────────────────

    def terrain_at(self, p: Vec2) -> TerrainType:
        """Returns BLOCKED for out-of-bounds (can't stand there)."""
        if not self.in_bounds(p):
            return TerrainType.BLOCKED
        ix, iy = self._cell(p)
        return TerrainType(self.cells[iy][ix])

    def is_blocked(self, p: Vec2) -> bool:
        """True if this position is a wall or out of bounds."""
        return self.terrain_at(p) == TerrainType.BLOCKED

    def is_difficult(self, p: Vec2) -> bool:
        return self.terrain_at(p) == TerrainType.DIFFICULT

    def is_dangerous(self, p: Vec2) -> bool:
        return self.terrain_at(p) == TerrainType.DANGEROUS

    def terrain_multiplier(self, p: Vec2) -> float:
        """Movement-cost multiplier for ending a move at this cell."""
        return _TERRAIN_MOVE_MULT[self.terrain_at(p)]

    # Backward-compat alias.
    def is_obstacle(self, p: Vec2) -> bool:
        return self.is_blocked(p)

    # ── Terrain editing ─────────────────────────────────────────────────────

    def set_terrain(self, p: Vec2, kind: TerrainType) -> None:
        if not self.in_bounds(p):
            return
        ix, iy = self._cell(p)
        self.cells[iy][ix] = int(kind)

    def add_rect_terrain(self, x0: float, y0: float, x1: float, y1: float,
                         kind: TerrainType) -> None:
        """Paint an axis-aligned rectangle with the given terrain type."""
        lo_x, hi_x = sorted((x0, x1))
        lo_y, hi_y = sorted((y0, y1))
        res = self.grid_resolution
        ix_start = max(0, int(lo_x / res))
        iy_start = max(0, int(lo_y / res))
        ix_end = min(len(self.cells[0]), int(math.ceil(hi_x / res)))
        iy_end = min(len(self.cells), int(math.ceil(hi_y / res)))
        kind_val = int(kind)
        for iy in range(iy_start, iy_end):
            for ix in range(ix_start, ix_end):
                self.cells[iy][ix] = kind_val

    def add_rect_obstacle(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Convenience: paint a BLOCKED rectangle (wall)."""
        self.add_rect_terrain(x0, y0, x1, y1, TerrainType.BLOCKED)

    # ── Line of sight ───────────────────────────────────────────────────────

    def has_line_of_sight(self, a: Vec2, b: Vec2) -> bool:
        """Return True iff segment a→b is clear of BLOCKED cells.

        Difficult and dangerous terrain don't block sight — only walls do.
        Endpoints themselves are NOT tested. Sampled at half the grid
        resolution to avoid skipping over a thin wall.
        """
        if not (self.in_bounds(a) and self.in_bounds(b)):
            return False
        dist = a.distance_to(b)
        if dist < 1e-9:
            return True
        step_m = self.grid_resolution * 0.5
        steps = max(2, int(math.ceil(dist / step_m)))
        for i in range(1, steps):
            t = i / steps
            x = a.x + (b.x - a.x) * t
            y = a.y + (b.y - a.y) * t
            if self.is_blocked(Vec2(x, y)):
                return False
        return True
