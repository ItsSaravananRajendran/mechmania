# `heal_coverage.py` — squad healer positioning

## Problem

A squad's support healer (see `healer_roles.md`) used to just stand behind
whichever single ally it happened to be actively healing that tick. That
protects one bot well, but leaves the rest of the squad uncovered until
*they're* the lowest-health one -- by which point they've already taken a
hit nobody was in range to answer. This module instead picks where the
healer stands based on the whole squad's positions, not just its current
patient.

## `MAX_ATTACKERS_OUT_OF_HEAL_RANGE = 2`

The target every healer's position is chosen against: at most 2 attackers
in a squad may end up outside `base_heal_range` of their healer. Not a hard
guarantee -- a squad spread wide enough with only one healer to cover it
can't always do better -- but the number every placement decision below
tries not to exceed.

## `_out_of_range_count(anchor, allies, heal_range) -> int`

How many `allies` are farther than `heal_range` from `anchor`. The core
measurement every other function here is built on.

## `_best_heal_anchor(allies, heal_range) -> (Vec2, int)`

Tries each ally's own current position as a candidate "stand here" point,
and keeps whichever leaves the fewest of the rest out of range. Squads in
this strategy are always small (a handful of bots), so checking every
ally-position candidate is a simple, exact solution -- no need for a real
minimum-enclosing-circle solver. Ties keep whichever candidate was checked
first (the order `allies` was given in).

## `_heal_position(allies, forward, conf) -> Vec2`

`_best_heal_anchor`'s point, then nudged a step further back (`-forward`,
`_min_safe_spacing(conf)`) for the same "healer stays off the front line"
reasoning used everywhere else a healer is positioned (`orchestration.md`).
The nudge is applied whenever it keeps the squad's out-of-range count at or
under `max(best_achievable, MAX_ATTACKERS_OUT_OF_HEAL_RANGE)` -- so a squad
that already fits entirely in range keeps fitting after the safety step,
and a squad already at the 2-straggler cap doesn't get pushed to 3 just for
the sake of the nudge; anywhere in between, the nudge still wins since 1 or
2 stragglers is within the target either way.

## How this plugs into `orchestration.md`

`_assign_support_healers` calls `_heal_position` once per squad (using the
same `allies` list already used to pick each healer's heal target) and uses
the result as every healer's `move_target` in that squad, deduped apart via
`formation._dedupe_slots` for squads with more than one healer. Each
healer's `target_id` is still picked independently (lowest-health ally,
same as before) -- coverage positioning and heal targeting are separate
decisions that happen to use the same ally list. A squad with no allies at
all (e.g. no raiders picked this tick) still falls back to the rear-zone
placement `_recompute_assignments` used before this module existed.
