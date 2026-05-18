"""2D spatial primitives for the combat engine.

Vec2: immutable position/direction vector with the helpers combat math needs.
Battlefield: bounded rectangular play area with an obstacle grid and line-of-
sight raycast.

Origin is at (0, 0); the playable area is [0, width] × [0, height]. A position
outside that rectangle is out of bounds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


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


@dataclass
class Battlefield:
    """Bounded play area with a 2D obstacle grid.

    Obstacles are stored as a bool grid at `grid_resolution` metre per cell.
    `obstacles[iy][ix] == True` means cell (ix, iy) is blocked.
    """
    width: float = 30.0
    height: float = 30.0
    grid_resolution: float = 0.5
    obstacles: list[list[bool]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.obstacles:
            nx = max(1, int(round(self.width / self.grid_resolution)))
            ny = max(1, int(round(self.height / self.grid_resolution)))
            self.obstacles = [[False] * nx for _ in range(ny)]

    def _cell(self, p: Vec2) -> tuple[int, int]:
        ix = int(p.x / self.grid_resolution)
        iy = int(p.y / self.grid_resolution)
        ix = max(0, min(ix, len(self.obstacles[0]) - 1))
        iy = max(0, min(iy, len(self.obstacles) - 1))
        return ix, iy

    def in_bounds(self, p: Vec2) -> bool:
        return 0.0 <= p.x <= self.width and 0.0 <= p.y <= self.height

    def is_obstacle(self, p: Vec2) -> bool:
        """A point outside the map counts as obstacle (you can't stand there)."""
        if not self.in_bounds(p):
            return True
        ix, iy = self._cell(p)
        return self.obstacles[iy][ix]

    def set_obstacle(self, p: Vec2, blocked: bool = True) -> None:
        if not self.in_bounds(p):
            return
        ix, iy = self._cell(p)
        self.obstacles[iy][ix] = blocked

    def add_rect_obstacle(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Block all cells whose centres fall inside the axis-aligned rect."""
        lo_x, hi_x = sorted((x0, x1))
        lo_y, hi_y = sorted((y0, y1))
        res = self.grid_resolution
        ix_start = max(0, int(lo_x / res))
        iy_start = max(0, int(lo_y / res))
        ix_end = min(len(self.obstacles[0]), int(math.ceil(hi_x / res)))
        iy_end = min(len(self.obstacles), int(math.ceil(hi_y / res)))
        for iy in range(iy_start, iy_end):
            for ix in range(ix_start, ix_end):
                self.obstacles[iy][ix] = True

    def has_line_of_sight(self, a: Vec2, b: Vec2) -> bool:
        """Return True iff segment a→b is clear of obstacles.

        Endpoints themselves are NOT tested (a creature standing on the
        edge of cover can still shoot from it). Sampled at half the grid
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
            if self.is_obstacle(Vec2(x, y)):
                return False
        return True
