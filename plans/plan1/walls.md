# `walls.py` — static wall storage

## Problem

Walls never change during a match, but `conf.map` is a raw ctypes array
(`conf.map[x][y]` per the engine source). Indexing into it directly, or
calling the engine's own `point_free`/`line_of_sight` for every candidate
formation slot, wall-check, or mining spot, re-does the same lookup over and
over across the match for data that was fixed before tick 1.

## Data structure

`WallGrid` mirrors `conf.map` into a flat `bytearray` (one byte per tile, `1`
= wall) at `x * size + y`, built once via `WallGrid.build(conf)`. Lookups are
O(1):

- `is_wall_rc(r, c)` — raw row/column lookup; out-of-bounds counts as a wall
  (so a candidate point that's fallen off the map edge is correctly treated
  as unusable, not silently "free").
- `is_wall_at(pos: Vec2)` — converts world coordinates to the grid, calling
  `is_wall_rc(int(pos.x), int(pos.y))`. **Not** `(pos.y, pos.x)` — the engine
  indexes `map[x][y]` (confirmed in the engine source, `game/config.rs`:
  `// Indexed map[x][y]`), and ctypes preserves that nesting for `conf.map`,
  so `build()` stores bits the same way. Getting this backwards was a real
  bug earlier in this codebase's history: it silently flagged the wrong
  cells as walls, defeating every downstream free-slot check.

`_get_wall_grid(conf)` builds it once and caches the result in the shared
`_cache` dict under `"wall_grid"`; every later call in the match is a dict
lookup.

## `_is_slot_free(wall_grid, slot)`

The combined check formation code actually wants: not on a wall tile *and*
clear per the engine's own `point_free` (which accounts for the bot's own
radius — a point right at a wall's edge can read as free on the coarse grid
but still be blocked once the bot's hull is considered). The grid is a cheap
first filter; `point_free` is the authority for anything actually placed.

## Edge cases this handles

- **Out-of-bounds coordinates** (a computed slot that drifted off the map):
  treated as a wall, not an index error and not "free".
- **Coordinate truncation**: `int(pos.x)` floors rather than rounds, matching
  how the engine's own tile indexing works (tile `(x, y)` covers
  `[x, x+1) × [y, y+1)`).
- **Stale cache across matches**: irrelevant in practice — `_cache` is a
  fresh module-level dict per process/match, and `mm-cli run` starts a new
  process per match.
