"""Formation and positioning: line-abreast zones around the payload (and a
spread mining line at the deposit) so bots don't cluster into a single
splash-vulnerable blob, plus the wall-aware fallbacks that keep a blocked
slot from collapsing multiple bots onto the same point.
"""

from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple

from .. import BotState, GameState, Vec2
from .cache import _cache
from .walls import WallGrid, _is_slot_free

# A splash hit only reaches `base_blaster_splash_radius` from where it lands,
# so any two bots farther apart than that can't both be caught by the same
# shot -- but exactly that far apart leaves zero room for float slop or a
# bot's actual position drifting slightly off its target slot between
# recomputes. This is the margin above the literal radius that "just greater
# than splash radius" spacing keeps.
SPLASH_SAFETY_MARGIN = 1.2


def _forward_direction(state: GameState) -> Vec2:
    """Unit vector from our deposit towards the enemy's -- a map-relative
    "which way is the enemy" axis, derived once from live geometry instead of
    assuming a fixed world axis (deposits/goals move between maps, walls don't
    move within one, but neither is a constant across matches)."""
    cached = _cache.get("forward_dir")
    if cached is not None:
        return cached
    delta = state.deposit_other.pos - state.deposit_me.pos
    forward = delta.normalize_or_zero()
    if forward.norm_sq() < 1e-6:
        forward = Vec2(0.0, 1.0)
    _cache["forward_dir"] = forward
    return forward


def _min_safe_spacing(conf) -> float:
    """The minimum distance any two bots' target slots may end up apart:
    just over one splash radius (`SPLASH_SAFETY_MARGIN`), floored by twice
    the bot radius so bots are never asked to stand inside one another.
    This is the hard "one hit can't reach two bots" invariant; `_formation_scale`'s
    `spacing`/`row_gap` are the larger, tactical distances layered on top of it."""
    return max(2.0 * conf.bot.radius, conf.bot.base_blaster_splash_radius * SPLASH_SAFETY_MARGIN)


def _lateral_slot(center: Vec2, perp: Vec2, index: int, count: int, spacing: float) -> Vec2:
    # Every caller already offsets `center` off the literal anchor point
    # (payload/deposit) along `forward` before calling this, so a zero
    # lateral offset for the middle bot of an odd-sized row never lands on
    # anything -- keep the row evenly spaced and centred on `center`.
    offset_units = index - (count - 1) / 2.0
    return center + perp * (offset_units * spacing)


def _resolve_slot(wall_grid: WallGrid, candidate: Vec2, anchor: Vec2) -> Vec2:
    """If `candidate` isn't usable, step it back towards its own zone
    `anchor` instead of jumping straight to a single shared fallback point --
    otherwise every blocked slot this tick (regardless of which zone or which
    bot it belonged to) collapses onto the exact same coordinate, which is
    the clustering formation is supposed to prevent, not cause."""
    if _is_slot_free(wall_grid, candidate):
        return candidate
    for t in (0.75, 0.5, 0.25, 0.1):
        p = anchor + (candidate - anchor) * t
        if _is_slot_free(wall_grid, p):
            return p
    # Every interpolation step was blocked too (a corridor exactly one bot
    # wide, say) -- rather than every such bot in the zone returning the
    # identical, literal `anchor`, keep a hair of each bot's own (distinct)
    # candidate direction so they don't all land on top of each other.
    return anchor + (candidate - anchor) * 0.02


def _formation_scale(conf) -> Tuple[float, float, float, float]:
    """(spacing, min_spacing, max_width, row_gap), shared by every zone.

    Bot/splash radii are tiny compared to `blaster_range` (e.g. 0.25 and 0.3
    against a range of 10), so sizing spacing off them alone collapses a
    formation into a blob well inside one splash radius of itself once you
    account for bot movement/steering slop. Scale off `blaster_range` (the
    distance the fight actually happens at) instead, with the radius
    multiples only as a floor for maps with a very short blaster range.
    """
    spacing = max(4.0 * conf.bot.radius, conf.bot.base_blaster_splash_radius * 6.0,
                  conf.bot.blaster_range * 0.2)
    min_spacing = _min_safe_spacing(conf)
    max_width = max(6.0 * conf.bot.radius, conf.bot.blaster_range * 1.5)
    row_gap = max(2.0 * conf.bot.radius, conf.bot.blaster_range * 0.15)
    return spacing, min_spacing, max_width, row_gap


def _free_anchor(wall_grid: WallGrid, anchor: Vec2, perp: Vec2, step: float, max_tries: int = 6) -> Vec2:
    """If a zone's anchor itself sits on a wall, every bot in that zone hits
    the same wall and every one of `_resolve_slot`'s interpolation steps
    (which all aim back at that same anchor) fails too -- they all fall back
    to the identical, still-blocked anchor point instead of spreading out.
    Nudge the anchor sideways along the formation line until it lands
    somewhere free, before laying bots out around it."""
    if _is_slot_free(wall_grid, anchor):
        return anchor
    for i in range(1, max_tries + 1):
        for sign in (1, -1):
            p = anchor + perp * (sign * step * i)
            if _is_slot_free(wall_grid, p):
                return p
    return anchor


def _arrange_group(
    wall_grid: WallGrid, anchor: Vec2, forward: Vec2, perp: Vec2, group: List[BotState],
    spacing: float, min_spacing: float, max_width: float, row_gap: float,
) -> Tuple[Dict[int, Vec2], Dict[int, Vec2]]:
    """Lay `group` out abreast around `anchor`, but never stretch a single row
    wider than `max_width` -- a long line of 15 bots at a fixed spacing would
    run off the map or into a wall well before it runs out of bots, and every
    slot that lands somewhere illegal collapses back towards `anchor` via
    `_resolve_slot`, re-clustering exactly the bots this is meant to spread.
    Past `max_width`, wrap into further rows stepped back along `forward`
    instead of stretching one row indefinitely."""
    if not group:
        return {}, {}

    row_capacity = max(1, int(max_width / min_spacing) + 1)
    slots: Dict[int, Vec2] = {}
    anchors: Dict[int, Vec2] = {}
    for start in range(0, len(group), row_capacity):
        row = group[start:start + row_capacity]
        row_index = start // row_capacity
        row_center = anchor + forward * (row_index * row_gap)
        row_center = _free_anchor(wall_grid, row_center, perp, spacing)
        row_spacing = spacing if len(row) <= 1 else max(min_spacing, min(spacing, max_width / (len(row) - 1)))
        for i, b in enumerate(row):
            slots[b.id] = _lateral_slot(row_center, perp, i, len(row), row_spacing)
            anchors[b.id] = row_center
    return slots, anchors


def _dedupe_slots(
    slots: Dict[int, Vec2], perp: Vec2, spacing: float, min_distance: float = 0.0,
) -> Dict[int, Vec2]:
    """Last-resort safety net: whatever upstream cause it had (two zones'
    wall-nudged anchors happening to coincide, an anchor and its own bots'
    interpolated fallbacks converging, ...), no two bots should ever end up
    reported at the literal same tile -- that's a free multi-kill for
    whoever's shooting at either of them. Walk bots in a stable order and
    nudge any exact repeat along `perp` until it's distinct.

    `min_distance`, when given, is a stronger check than exact-duplicate: it
    also nudges a slot that's merely *closer than that* to an already-placed
    one -- the "just greater than splash radius" invariant -- since two bots
    a few centimetres apart (not on the identical tile) are just as much a
    free multi-kill as two bots stacked exactly on top of each other."""
    seen: Set[Tuple[float, float]] = set()
    placed: List[Vec2] = []
    result: Dict[int, Vec2] = {}
    for bot_id in sorted(slots.keys()):
        pos = slots[bot_id]
        bump = 1
        key = (round(pos.x, 2), round(pos.y, 2))
        too_close = min_distance > 0.0 and any(pos.dist(p) < min_distance for p in placed)
        while (key in seen or too_close) and bump < 200:
            pos = slots[bot_id] + perp * (spacing * 0.3 * bump)
            key = (round(pos.x, 2), round(pos.y, 2))
            too_close = min_distance > 0.0 and any(pos.dist(p) < min_distance for p in placed)
            bump += 1
        seen.add(key)
        placed.append(pos)
        result[bot_id] = pos
    return result


def _stack_behind(
    wall_grid: WallGrid, front_positions: List[Vec2], forward: Vec2,
    trailing: List[BotState], depth_gap: float,
) -> Tuple[Dict[int, Vec2], Dict[int, Vec2]]:
    """Stack `trailing` bots directly behind `front_positions` (same lateral
    offset, stepped back along `-forward` by multiples of `depth_gap`)
    instead of giving them their own lateral line -- each trailing bot sits
    in the column "shadow" of the bot ahead of it. A shot that hits the
    front bot can't also reach the one behind it once `depth_gap` clears
    splash radius (see `_min_safe_spacing`), and the front bot's hull sits
    on the straight-line ray to it too, since the engine's blaster ray stops
    at the first enemy bot it hits (`wiki/mechanics.md`) -- the front bot
    screens the one(s) behind it both from splash and from direct fire on
    that line. Trailing bots cycle through columns round-robin (`i % len`)
    so extra bots stack deeper rather than piling behind a single column."""
    if not front_positions:
        return {}, {}
    n_cols = len(front_positions)
    raw_slots: Dict[int, Vec2] = {}
    anchors: Dict[int, Vec2] = {}
    for i, bot in enumerate(trailing):
        col = i % n_cols
        depth = (i // n_cols) + 1
        anchor = front_positions[col]
        anchors[bot.id] = anchor
        raw_slots[bot.id] = anchor - forward * (depth_gap * depth)
    resolved = {
        bot_id: _resolve_slot(wall_grid, slot, anchors[bot_id])
        for bot_id, slot in raw_slots.items()
    }
    return resolved, anchors


def _compute_battle_formation(
    battles: List[BotState], payload: Vec2, forward: Vec2, conf, wall_grid: WallGrid
) -> Dict[int, Vec2]:
    """A screen layer a little ahead of the payload (first to take a hit),
    line-abreast so splash can't multi-kill along that line, with the rest
    of the fleet stacked directly *behind* a screen bot in the same column
    (see `_stack_behind`) rather than in their own separate lateral line --
    each screen bot both shields and gets shielded by the bot(s) queued
    behind it, one hit can only reach one bot in a column, and healers
    assigned to heal a screen bot (see `_recompute_assignments`) land in
    that same column too. The screen width is `ceil(sqrt(n))`, so the
    resulting grid comes out roughly square -- as many columns as rows --
    rather than one wide row with a deep tail behind it. Small fleets
    collapse to one spread line -- there aren't enough bots to make a
    screen-plus-column split meaningful."""
    n = len(battles)
    if n == 0:
        return {}

    perp = forward.rotate_deg(90.0)
    spacing, min_spacing, max_width, row_gap = _formation_scale(conf)
    min_safe = _min_safe_spacing(conf)

    # Never anchor a zone directly on `payload` itself -- the dedicated
    # payload-defender bot already sits there (see `_recompute_assignments`),
    # so any formation slot with a zero lateral offset would land exactly on
    # top of it. A small forward nudge keeps every zone visibly its own spot.
    small_center = payload + forward * max(2.0 * conf.bot.radius, conf.bot.blaster_range * 0.15)

    raw_slots: Dict[int, Vec2] = {}
    anchors: Dict[int, Vec2] = {}
    if n <= 3:
        s, a = _arrange_group(wall_grid, small_center, forward, perp, battles, spacing, min_spacing, max_width, row_gap)
        raw_slots.update(s)
        anchors.update(a)
    else:
        # ceil(sqrt(n)) columns means ceil(n / that) rows too -- a square
        # grid (or as close to one as an integer bot count allows), instead
        # of the lopsided "half the fleet in one wide screen row, the other
        # half stacked deep behind it" split a fixed 50/50 would give a
        # large fleet.
        n_screen = max(1, math.ceil(math.sqrt(n)))
        screen_center = payload + forward * max(2.0 * conf.bot.radius, conf.bot.blaster_range * 0.25)

        screen = battles[:n_screen]
        rest = battles[n_screen:]
        s, a = _arrange_group(wall_grid, screen_center, forward, perp, screen, spacing, min_spacing, max_width, row_gap)
        raw_slots.update(s)
        anchors.update(a)

        front_positions = [s[b.id] for b in screen]
        depth_gap = max(row_gap, min_safe)
        s, a = _stack_behind(wall_grid, front_positions, forward, rest, depth_gap)
        raw_slots.update(s)
        anchors.update(a)

    resolved = {
        bot_id: _resolve_slot(wall_grid, slot, anchors[bot_id])
        for bot_id, slot in raw_slots.items()
    }
    return _dedupe_slots(resolved, perp, spacing, min_distance=min_safe)


def _compute_mining_spots(
    extractors: List[BotState], state: GameState, conf, wall_grid: WallGrid
) -> Dict[int, Vec2]:
    """Spread extractors along the deposit's face (perpendicular to the
    deposit-to-payload axis) instead of stacking them shoulder to shoulder on
    one side -- keeps a stray splash from an enemy that reaches the deposit
    from wiping more than one extractor at once."""
    if not extractors:
        return {}

    deposit_pos = state.deposit_me.pos
    forward = _forward_direction(state)
    perp = forward.rotate_deg(90.0)
    # `forward` points from our deposit towards the enemy's -- standing on
    # that side puts extractors in the open, facing exactly the direction
    # fire is likely to come from. The far side puts the deposit disc's own
    # bulk between them and the enemy (a blaster ray stops at a deposit),
    # and mining only needs a clear ray back to the deposit, not a
    # particular side of it.
    base = deposit_pos - forward * (conf.deposit.radius + conf.bot.radius * 1.5)
    spacing = max(4.0 * conf.bot.radius, conf.bot.base_blaster_splash_radius * 6.0)

    n = min(len(extractors), 3)
    result: Dict[int, Vec2] = {}
    for i, ext in enumerate(extractors[:n]):
        slot = _lateral_slot(base, perp, i, n, spacing)
        result[ext.id] = _resolve_slot(wall_grid, slot, base)
    return result
