# `combat_los.py` — clear-shot checks and side-stepping

## Problem

The engine's `line_of_sight(a, b)` helper only checks walls (`ray_clear_in`
over `conf.map` — confirmed in the engine source). But the actual blaster
ray, per `wiki/mechanics.md`, also stops at **a deposit or the payload**. A
target standing behind either reads as "visible" to `line_of_sight` while
the real shot would just detonate on the obstacle — a wasted shot that still
puts the blaster on cooldown. Symmetrically, `navigate_to` only routes
around walls; it has no idea a deposit or the payload is in the way, so a
bot told to walk straight at a target behind one will walk into it rather
than around it.

## `_has_clear_shot(a, b, state, conf) -> bool`

Layers three checks on top of `line_of_sight`:

1. `line_of_sight(a, b)` — walls.
2. `point_seg_dist(state.deposit_me.pos, a, b) < conf.deposit.radius` — does
   the segment pass through our own deposit's disc?
3. Same for `state.deposit_other.pos`.
4. `point_seg_dist(state.payload_pos(), a, b) < conf.payload.radius` — does
   it pass through the payload?

Any of the four failing means "not clear" — this is what target
prioritization (`targeting.py`) and the fire decision (`__init__.py`) both
gate on, instead of the bare engine `line_of_sight`.

## `_blocking_obstacle(a, b, state, conf) -> Optional[(Vec2, float)]`

Given `_has_clear_shot` already said no and it wasn't a wall, this finds
*which* disc is actually in the way (position + radius), so the caller knows
what to step around. If more than one disc's segment-distance qualifies, it
picks whichever is closest to the shooter — that's the one physically
encountered first.

## `_flank_point(bot_pos, target_pos, obstacle_pos, clearance) -> Vec2`

A point to move to instead of straight at the obstacle: offset
perpendicular to the bot→target line, at `clearance` distance from the
obstacle's center, on whichever side the bot is already closer to (checked
via a dot product against the perpendicular) — that's a stability choice, not
a correctness one: picking "whichever side is already true" means a bot
mid-maneuver doesn't flip sides and re-block itself every time this gets
recomputed. If that side isn't `point_free`, it tries the other side; if
neither is free, it just returns `target_pos` (better to walk toward the
real goal than sit still on an unresolvable geometry).

## Edge cases this handles

- **Multiple obstacles on one line** (rare, but the payload can drift near a
  deposit): `_blocking_obstacle` picks the nearest to the shooter, not an
  arbitrary one.
- **Both flank sides blocked**: falls back to `target_pos` rather than
  returning a point inside a wall.
- **Degenerate direction** (`bot_pos == target_pos`): `_flank_point` defaults
  the travel direction to `Vec2(1.0, 0.0)` rather than dividing by a
  zero-length vector.
