# `formation.py` — positioning so bots don't cluster

## Problem

Bots never collide with each other (per the mechanics doc), which means
nothing physically stops a fleet from stacking on one tile — and a stacked
fleet is a free multi-kill under blaster splash. Formation is the answer:
compute distinct, spread-out target positions for every bot in a group, wide
enough apart that one splash radius can't reach two of them, while staying
useful (near the payload, near the deposit) and not walking into walls.

## Geometry primitives

- **`_forward_direction(state) -> Vec2`** — unit vector from our deposit to
  the enemy's. Computed once and cached (`_cache["forward_dir"]`) since
  deposits are static for the match. This is the map-relative "which way is
  the enemy" axis everything else in this module orients against, instead of
  assuming a fixed world axis that wouldn't hold across different generated
  maps.
- **`_lateral_slot(center, perp, index, count, spacing) -> Vec2`** — the
  `index`-th of `count` evenly spaced points along `perp`, centered on
  `center`. Every caller already offsets `center` off the literal anchor
  point (payload or deposit) along `forward` before calling this, so the
  middle slot of an odd-sized row landing exactly on `center` is never a
  problem — there's nothing else anchored there to collide with.
- **`_formation_scale(conf) -> (spacing, min_spacing, max_width, row_gap)`**
  — sizes everything off `conf.bot.blaster_range`, not `conf.bot.radius` or
  `base_blaster_splash_radius` alone. Those are tiny by comparison (e.g. 0.25
  and 0.3 against a range of 10); scaling spacing off them collapses a
  "spread" formation into a blob well inside one splash radius of itself
  once bot movement/steering slop is accounted for. The radius-based values
  are only a floor, for maps with an unusually short blaster range.
  `min_spacing` is exactly `_min_safe_spacing(conf)` below.
- **`_min_safe_spacing(conf) -> float`** — the hard "one hit can't reach two
  bots" floor: `base_blaster_splash_radius * SPLASH_SAFETY_MARGIN` (1.2), i.e.
  just over one splash radius, floored by `2 * conf.bot.radius`. This is the
  `min_distance` every dedupe/stacking pass below is checked against; the
  larger `spacing`/`row_gap` values are tactical distances layered on top of
  it, not a replacement for it.

## Wall-safety fallbacks

- **`_resolve_slot(wall_grid, candidate, anchor) -> Vec2`** — if `candidate`
  isn't `_is_slot_free`, step it back toward its own zone's `anchor` in
  decreasing fractions (`0.75, 0.5, 0.25, 0.1`) until one is free. If even
  that fails (a corridor exactly one bot wide, say), returns a point 2% of
  the way from `anchor` toward `candidate` — not `anchor` itself, and not
  `candidate` itself, so that if several bots in the same zone all hit this
  fallback, they don't all collapse onto the identical coordinate.
- **`_free_anchor(wall_grid, anchor, perp, step, max_tries=6) -> Vec2`** — if
  a zone's *anchor* itself sits on a wall (not just an individual candidate
  slot), every bot in that zone would hit the same wall and every one of
  `_resolve_slot`'s interpolation steps (which all aim back at that same
  anchor) would fail too, collapsing the whole zone onto one blocked point.
  This nudges the anchor sideways along the formation line, trying
  `±step, ±2·step, ..., ±6·step`, before anyone is laid out around it.
- **`_dedupe_slots(slots, perp, spacing, min_distance=0.0) -> Dict[bot_id, Vec2]`**
  — the final safety net: walk bots in a stable (sorted-by-id) order, and if
  a slot's position (rounded to 2 decimals) has already been claimed this
  pass, nudge it along `perp` by increasing multiples of `0.3 · spacing`
  until it's distinct. Whatever upstream cause produced a collision (two
  zones' wall-nudged anchors happening to coincide, say), no two bots should
  ever end up reported at the literal same tile. When `min_distance` is
  given (callers pass `_min_safe_spacing(conf)`), the same nudge also
  triggers for a slot merely *closer than that* to an already-placed one --
  not just an exact duplicate — since two bots a few centimetres apart are
  just as much a free multi-kill as two stacked on the identical tile.

## `_arrange_group(wall_grid, anchor, forward, perp, group, spacing, min_spacing, max_width, row_gap)`

Lays `group` out abreast around `anchor`, but caps a single row's width at
`max_width` — a long line of 15 bots at a fixed spacing would run off the map
or into a wall well before it runs out of bots, and every slot that lands
somewhere illegal collapses back toward `anchor`, re-clustering exactly the
bots this is meant to spread. Past `max_width`, later bots wrap into further
rows stepped back along `forward` (`row_capacity = max_width / min_spacing`,
rounded up), instead of stretching one row indefinitely. Returns both the
raw slot per bot and which row-anchor it belongs to (needed by
`_resolve_slot`'s per-zone fallback).

## `_stack_behind(wall_grid, front_positions, forward, trailing, depth_gap) -> (Dict[bot_id, Vec2], Dict[bot_id, Vec2])`

Places `trailing` bots directly behind `front_positions` — same lateral
(`perp`) offset, stepped back along `-forward` by `depth_gap, 2·depth_gap,
...` — cycling through columns round-robin (`i % len(front_positions)`) so
extra bots stack deeper rather than piling behind a single column. This is
the shielding primitive: the engine's blaster ray stops at the first enemy
bot it hits (`wiki/mechanics.md`), so the front bot of a column blocks that
firing line for whoever's queued up behind it, and once `depth_gap` clears
`_min_safe_spacing` a splash hit on the front bot can't reach the one behind
it either. Each trailing slot still goes through `_resolve_slot` against its
column's front position as anchor.

## `_compute_battle_formation(battles, payload, forward, conf, wall_grid) -> Dict[bot_id, Vec2]`

- **Screen** — a little ahead of the payload (`forward * max(2·radius,
  0.25·blaster_range)`), line-abreast, first to take a hit.
- **Rest of the fleet** — stacked directly *behind* a screen bot in the same
  column via `_stack_behind`, at `depth_gap = max(row_gap, _min_safe_spacing(conf))`,
  instead of their own separate lateral stand-off line. Each screen bot both
  shields and is shielded by whoever's queued behind it in its column; a
  healer assigned to heal a screen bot (see `orchestration.md`) lands in
  that same column too, for the same reason.

Screen width is `n_screen = max(1, ceil(sqrt(n)))`, so the number of columns
and the number of rows (`ceil(n / n_screen)`) come out equal, or as close to
it as an integer bot count allows -- a square grid instead of one wide row
with a deep tail behind it. Fleets of 3 or
column split meaningful), anchored a little off `payload` along `forward` so
it doesn't sit on top of the dedicated payload defender (see
`orchestration.md`). Every raw slot is passed through `_resolve_slot` then
`_dedupe_slots(..., min_distance=_min_safe_spacing(conf))` before being
returned.

## `_compute_mining_spots(extractors, state, conf, wall_grid) -> Dict[bot_id, Vec2]`

Spreads up to 3 extractors along the deposit's face, perpendicular to the
deposit↔enemy axis — not stacked shoulder to shoulder, so one splash can't
wipe more than one. Placed on the **`-forward`** side of the deposit (away
from the enemy), so the deposit's own disc sits between the extractors and
incoming fire — a blaster ray stops at a deposit, and mining only needs a
clear ray back to it, not a particular side.

## Edge cases this handles

- **Empty group**: every function returns `{}` immediately.
- **A zone anchor on a wall**: `_free_anchor` relocates it before laying
  bots out.
- **More bots than fit one row at safe spacing**: wraps into additional rows
  rather than shrinking spacing to zero or running off-map.
- **A corridor too narrow for any interpolated fallback**: `_resolve_slot`'s
  final 2%-offset step still keeps colliding bots distinguishable, and
  `_dedupe_slots` catches anything that still coincides.
