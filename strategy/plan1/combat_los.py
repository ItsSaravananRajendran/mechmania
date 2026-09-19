"""Clear-shot checks: `line_of_sight` only knows about walls, but the
engine's own blaster ray also stops at a deposit or the payload -- and
`navigate_to` only routes around walls, not those two. This module fills
both gaps: telling whether a shot is really clear, and where to step to make
it one.
"""

from __future__ import annotations

from typing import Optional, Tuple

from .. import GameState, Vec2, line_of_sight, point_free, point_seg_dist


def _has_clear_shot(a: Vec2, b: Vec2, state: GameState, conf) -> bool:
    """`line_of_sight` only checks walls -- the engine's own blaster ray also
    stops at a deposit or the payload (see `wiki/mechanics.md`), so a target
    standing behind either of those reads as "visible" to `line_of_sight`
    while the real shot would just detonate on the obstacle. Check both."""
    if not line_of_sight(a, b):
        return False
    depot_r = conf.deposit.radius
    if point_seg_dist(state.deposit_me.pos, a, b) < depot_r:
        return False
    if point_seg_dist(state.deposit_other.pos, a, b) < depot_r:
        return False
    if point_seg_dist(state.payload_pos(), a, b) < conf.payload.radius:
        return False
    return True


def _blocking_obstacle(a: Vec2, b: Vec2, state: GameState, conf) -> Optional[Tuple[Vec2, float]]:
    """Which disc obstacle (a deposit or the payload) sits between `a` and
    `b`, if any -- `None` means either nothing blocks, or a wall does (which
    calls for routing around it via `navigate_to`, not side-stepping)."""
    depot_r = conf.deposit.radius
    candidates = [
        (state.deposit_me.pos, depot_r),
        (state.deposit_other.pos, depot_r),
        (state.payload_pos(), conf.payload.radius),
    ]
    blockers = [(pos, r) for pos, r in candidates if point_seg_dist(pos, a, b) < r]
    if not blockers:
        return None
    # If more than one disc happens to intersect the line, the one closest to
    # the shooter is the one actually in the way first.
    return min(blockers, key=lambda pr: a.dist_sq(pr[0]))


def _flank_point(bot_pos: Vec2, target_pos: Vec2, obstacle_pos: Vec2, clearance: float) -> Vec2:
    """A point off to one side of `obstacle_pos`, far enough out
    (`clearance`) to clear it, so a bot blocked by a deposit or the payload
    can step around it instead of standing still with no shot. Prefers
    whichever side the bot is already leaning towards, so it doesn't flip
    sides (and re-block itself) every recompute."""
    direction = (target_pos - bot_pos).normalize_or_zero()
    if direction.norm_sq() < 1e-6:
        direction = Vec2(1.0, 0.0)
    perp = direction.rotate_deg(90.0)
    preferred = 1.0 if (bot_pos - obstacle_pos).dot(perp) >= 0.0 else -1.0
    for side in (preferred, -preferred):
        p = obstacle_pos + perp * (clearance * side)
        if point_free(p):
            return p
    return target_pos
