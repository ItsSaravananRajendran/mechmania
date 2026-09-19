# `__init__.py` — orchestration

This file is glue, not an algorithm of its own: it wires a tick's
`GameState` through the modules documented alongside this file into a
`FleetAction`, and owns the one piece of cross-cutting logic none of them
own individually — *when* to recompute assignments versus replay the last
ones, and how a per-bot assignment dict turns into concrete move/turn/fire
orders.

## `plan1_strategy(state) -> FleetAction` — the recompute cadence

Every tick, three independent signals decide whether this tick recomputes
assignments (`_recompute_assignments`, the expensive path — targeting,
formation, mining spots) or replays the cached ones
(`_execute_assignments`, cheap):

- **`state.tick - last_assignment_tick >= 5`** — a floor cadence so
  assignments never go stale for more than 5 ticks even if nothing else
  changes.
- **`enemy_delta >= 2`** — 2+ enemies appeared or died since the last
  recompute (a single kill doesn't justify recomputing early; a real fight
  turning does).
- **`capture_changed`** — the *set* of our bots within payload capture range
  changed (someone entered or left the zone).
- **`bool(friendly_died)`** — any of our bots died since last tick.
- **`bool(friendly_born)`** — any of our bots were newly built since last
  tick. Without this, a fresh bot has no entry in the cached assignment dict
  until the next scheduled recompute (up to 5 ticks away) and falls back to
  "just head for the payload" in the meantime, ignoring
  formation/targeting — with fleets growing every few ticks from free builds
  and rush orders, that's a steady trickle of bots piling onto the payload
  point instead of holding formation.

Additionally, `budget.remaining <= 100_000` forces the cheap replay path
regardless of the above — the always-safe fallback the compute-budget
guidance calls for.

A **tick gap** (`state.tick - last_tick > 1`, meaning a tick was skipped —
most likely this bot sat out due to an exhausted budget) clears the cached
assignment dict outright, since anything cached from before the gap could be
arbitrarily stale.

`friendly_died`/`friendly_born` are also the *bots that changed*, computed
as a set difference between this tick's and last tick's fleet id sets —
cheap, and exact (not inferred from a count delta, which is why
`fabricator.py` takes `friendly_born` directly rather than re-deriving it).

## `_recompute_assignments(state, conf) -> FleetAction` — building the plan

Builds one `assignments: Dict[bot_id, dict]` for the whole fleet, in this
order (later steps can overwrite an earlier bot's entry, which is used
deliberately in two places noted below):

1. **Mining** — up to 3 extractors get `{"kind": "extractor", "mining_spot":
   ...}` from `formation._compute_mining_spots`.
2. **All-extractors payload fallback** — if the fleet has *no* Battle and
   *no* Healer left (every combat bot dead, only Extractors survive), the
   nearest extractor's assignment is overwritten to sit on the payload
   (`"mining_spot": payload, "at_payload": True`) instead of mining. Without
   this, every surviving bot just keeps mining and nothing ever holds the
   payload — which loses the match outright the instant the enemy commits a
   single bot to push it (capture only needs *a* bot in range, any class).
3. **Payload defenders** — one Healer and one Battle bot (if any exist) are
   pinned near the payload (`role: "payload"`), offset a little to either
   side of the exact point (`payload ± forward * payload_defense_offset`,
   capped at 1 unit or 40% of capture radius) rather than both sitting on
   the identical coordinate, which would let one splash hit take out both
   the objective's healer and its defender at once.
4. **Remaining healers** — each either gets a heal target (the lowest-health
   ally under 90% HP, stood near but not on top of, so the healer and its
   patient aren't stacked for a shared splash kill) or, if nobody needs
   healing, joins a rear formation zone behind the stand-off line
   (`formation._arrange_group` + `_resolve_slot` + `_dedupe_slots`) instead
   of defaulting to the literal payload point, which is exactly the kind of
   clustering formation exists to prevent.
5. **Remaining battles** — get a target from `targeting._assign_battle_targets`
   (computed once, up front, for the whole `battles` list) and a slot from
   `formation._compute_battle_formation`.
6. **Catch-all** — any bot not yet assigned (shouldn't normally happen; a
   safety net) defaults to heading for the payload.

## `_apply_assignments(action, state, conf, assignments, enemy_in_capture)` — turning a plan into orders

Runs every tick regardless of whether this tick recomputed, so it's what
actually has to stay correct against a *stale* cached assignment (a target
that died since the plan was made, say). Per bot, by `kind`:

- **Extractor** — move to its mining spot (or a default spot beside the
  deposit if none was assigned), face the deposit and mine — unless
  `at_payload` is set, in which case it faces the nearest enemy instead of
  turning its back to the deposit it's no longer near.
- **Healer** — face its heal target (or the payload/nearest-enemy per its
  role), heal if in range and in arc, move to its assigned point.
- **Battle** — the richest logic:
  - If its cached target died or it was never assigned one, fall back to the
    nearest *unclaimed* enemy (tracked in `claimed_targets`, seeded from
    every other bot's live cached target) rather than "nearest enemy"
    independently per bot, which would silently reintroduce the dogpiling
    `targeting.py` exists to prevent.
  - Fire only if in range, `_has_clear_shot`, *and* the target isn't
    currently invulnerable — firing at an invulnerable target burns the
    blaster's own cooldown for a shot that can't land.
  - If in range but *not* a clear shot, something's in the way: a wall calls
    for walking straight at the target (`navigate_to` already routes around
    walls on its own); a deposit or the payload isn't part of that wall
    topology, so `combat_los._flank_point` is used to step around it
    explicitly instead.
- **Anything without a recognized kind** (shouldn't happen — every bot is one
  of the three classes, and the default `kind` comes from the class name):
  heads for the payload, special action off.

## Edge cases this covers

- **A recompute lands with an already-dead target cached from a prior
  tick's assignment**: handled by `_apply_assignments`'s per-bot fallback,
  not by `_recompute_assignments` re-running early (that's what the
  `friendly_died` recompute trigger is for, but `_apply_assignments` doesn't
  rely on it having fired yet).
- **Every combat bot wiped**: the all-extractors payload fallback (step 2
  above) keeps at least one bot contesting the objective.
- **Budget exhausted mid-match**: `plan1_strategy` degrades to the cheap
  replay path, which still runs the full `_apply_assignments` logic (fire
  gating, fallback targeting, movement) against whatever was last
  computed — there's no cheaper "do nothing" tier below that.
