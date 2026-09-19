"""Corner-hugging avoidance.

`navigate_to` already routes around walls, but it routes *tightly* -- a taut
path grazes a wall's convex corners (the tip of an inverted "L", say) at
exactly the bot's own radius, since that's the shortest route. With limited
turn speed and per-tick incremental steps, a bot asked to pivot that close to
a sharp corner visibly snags/oscillates there instead of sliding past.

This adds a wider, chamfered detour around exactly those corners: the same
convex-corner points the engine's own pathfinder uses (see the engine's
`convex_corners` in `topology.rs`), each carrying the direction the solid
wall quadrant lies in, so a blocked corner can be given a berth by pushing
the aim point out along the opposite (open) diagonal -- a sloped way past
the point instead of a tight pivot on it.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .. import MoveAction, Vec2, move_bot, navigate_to
from .cache import _cache
from .walls import WallGrid

# (corner position, unit direction away from the wall quadrant into open space)
_Corner = Tuple[Vec2, Vec2]


def _find_convex_corners(wall_grid: WallGrid) -> List[_Corner]:
    """Lattice points where exactly one of the four surrounding tiles is a
    wall -- computed once; walls never change mid-match."""
    size = wall_grid.size

    def is_wall(x: int, y: int) -> bool:
        return wall_grid.is_wall_rc(x, y)

    corners: List[_Corner] = []
    for cx in range(size + 1):
        for cy in range(size + 1):
            quadrants = [
                (is_wall(cx - 1, cy - 1), -1.0, -1.0),
                (is_wall(cx, cy - 1), 1.0, -1.0),
                (is_wall(cx - 1, cy), -1.0, 1.0),
                (is_wall(cx, cy), 1.0, 1.0),
            ]
            solid = [q for q in quadrants if q[0]]
            if len(solid) == 1:
                _, sx, sy = solid[0]
                away = Vec2(-sx, -sy).normalize_or_zero()
                corners.append((Vec2(float(cx), float(cy)), away))
    return corners


def _get_corners(wall_grid: WallGrid) -> List[_Corner]:
    corners = _cache.get("wall_corners")
    if corners is None:
        corners = _find_convex_corners(wall_grid)
        _cache["wall_corners"] = corners
    return corners


def _closest_point_on_segment(p: Vec2, a: Vec2, b: Vec2) -> Vec2:
    ab = b - a
    denom = ab.dot(ab)
    if denom < 1e-9:
        return a
    t = (p - a).dot(ab) / denom
    t = max(0.0, min(1.0, t))
    return a + ab * t


def _widen_path(bot_pos: Vec2, target_pos: Vec2, wall_grid: WallGrid,
                 buffer: float, detect_radius: float) -> Vec2:
    """If the straight line from `bot_pos` to `target_pos` passes within
    `buffer` of a wall's convex corner, aim past that corner with `buffer`
    to spare instead of letting `navigate_to` thread the gap at the bot's
    bare radius."""
    corners = _get_corners(wall_grid)
    if not corners:
        return target_pos

    nearest: Optional[_Corner] = None
    nearest_dist = buffer
    detect_radius_sq = detect_radius * detect_radius
    for corner_pos, away in corners:
        if bot_pos.dist_sq(corner_pos) > detect_radius_sq:
            continue
        # Once the bot has actually reached the widened waypoint (within
        # `buffer` of the corner itself), it has already rounded the corner:
        # steering it at that waypoint forever would pin it there, since the
        # straight line from the waypoint onward to `target_pos` still grazes
        # the same corner within `buffer`. Give this a bit of slack past the
        # exact buffer distance -- incremental per-tick movement lands the
        # bot within float noise of `buffer`, and a bare `<= buffer` check
        # flip-flops tick to tick right at that boundary, freezing the bot in
        # place instead of reliably releasing it once it arrives.
        if bot_pos.dist(corner_pos) <= buffer * 1.1:
            continue
        closest = _closest_point_on_segment(corner_pos, bot_pos, target_pos)
        d = corner_pos.dist(closest)
        if d < nearest_dist:
            nearest_dist = d
            nearest = (corner_pos, away)

    if nearest is None:
        return target_pos

    corner_pos, away = nearest
    return corner_pos + away * buffer


def _steer(bot_pos: Vec2, target: Vec2, wall_grid: WallGrid, conf) -> MoveAction:
    """Drop-in replacement for `move_bot(navigate_to(bot_pos, target))` that
    detours around a nearby convex wall corner first, so every bot's
    movement gets the wider berth, not just combat repositioning."""
    buffer = conf.bot.radius * 3.0
    detect_radius = buffer * 4.0
    routed = _widen_path(bot_pos, target, wall_grid, buffer, detect_radius)
    return move_bot(navigate_to(bot_pos, routed))
