"""Static wall storage: walls never change during a match, so mirror
`conf.map` into a plain array once and answer "is this spot on a wall"
without touching the engine bindings again for every candidate slot.
"""

from __future__ import annotations

from .. import MAP_SIZE, MapTile, Vec2, point_free
from .cache import _cache


class WallGrid:
    """A static, once-built mirror of `conf.map` for O(1) local wall lookups.

    Walls never change during a match, so there is no reason to keep touching
    `conf.map` every time formation code wants to know "is this candidate spot
    on a wall". `conf.map` is indexed `[x][y]` (per the engine source), and a
    bot has nonzero radius, so a point right at a wall's edge can still read
    as free here but blocked for the engine -- placements are still confirmed
    once against the engine's own `point_free` before being used.
    """

    __slots__ = ("bits", "size")

    def __init__(self, bits: bytearray, size: int) -> None:
        self.bits = bits
        self.size = size

    @classmethod
    def build(cls, conf) -> "WallGrid":
        size = MAP_SIZE
        bits = bytearray(size * size)
        m = conf.map
        for i in range(size):
            row = m[i]
            for j in range(size):
                if row[j] == MapTile.Wall:
                    bits[i * size + j] = 1
        return cls(bits, size)

    def is_wall_rc(self, r: int, c: int) -> bool:
        if r < 0 or c < 0 or r >= self.size or c >= self.size:
            return True
        return self.bits[r * self.size + c] == 1

    def is_wall_at(self, pos: Vec2) -> bool:
        # The engine indexes `map[x][y]` (confirmed in the engine source), and
        # ctypes preserves that nesting for `conf.map`, so `build()` stored
        # bits at `x * size + y` -- look them up the same way.
        return self.is_wall_rc(int(pos.x), int(pos.y))


def _get_wall_grid(conf) -> WallGrid:
    grid = _cache.get("wall_grid")
    if grid is None:
        grid = WallGrid.build(conf)
        _cache["wall_grid"] = grid
    return grid


def _is_slot_free(wall_grid: WallGrid, slot: Vec2) -> bool:
    if wall_grid.is_wall_at(slot):
        return False
    return point_free(slot)
