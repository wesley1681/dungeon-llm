"""Preset terrain layouts for the sparring sandbox.

Each preset is a function that mutates a 30x30 Battlefield in place.
Default battlefield grid is 0.5m/cell (60x60 cells). Helpers use the
existing Battlefield API — no direct cells[][] writes.
"""
from __future__ import annotations
from collections.abc import Callable

from ..engine.vec2 import Battlefield, TerrainType


def _empty(bf: Battlefield) -> None:
    """No-op. All cells already default to NORMAL."""
    return


def _lava_strip(bf: Battlefield) -> None:
    """A 1m-tall horizontal lava band across the middle (y ∈ [14.5, 15.5])."""
    bf.add_rect_terrain(0.0, 14.5, bf.width, 15.5, TerrainType.DANGEROUS)


def _pillars(bf: Battlefield) -> None:
    """Four 1x1m wall pillars at the corners of an inner 15m square."""
    for cx, cy in ((7.5, 7.5), (22.5, 7.5), (7.5, 22.5), (22.5, 22.5)):
        bf.add_rect_obstacle(cx - 0.5, cy - 0.5, cx + 0.5, cy + 0.5)


def _corridor(bf: Battlefield) -> None:
    """Walls block the top and bottom thirds, forcing a central 3m corridor."""
    # Bottom block: full width, y ∈ [3, 7]
    bf.add_rect_obstacle(0.0, 3.0, bf.width, 7.0)
    # Top block: full width, y ∈ [23, 27]
    bf.add_rect_obstacle(0.0, 23.0, bf.width, 27.0)


def _arena(bf: Battlefield) -> None:
    """1m-thick perimeter walls. Interior is 28x28m."""
    bf.add_rect_obstacle(0.0, 0.0, bf.width, 1.0)               # bottom
    bf.add_rect_obstacle(0.0, bf.height - 1.0, bf.width, bf.height)  # top
    bf.add_rect_obstacle(0.0, 0.0, 1.0, bf.height)              # left
    bf.add_rect_obstacle(bf.width - 1.0, 0.0, bf.width, bf.height)   # right


TERRAIN_PRESETS: dict[str, Callable[[Battlefield], None]] = {
    "empty":      _empty,
    "lava_strip": _lava_strip,
    "pillars":    _pillars,
    "corridor":   _corridor,
    "arena":      _arena,
}
