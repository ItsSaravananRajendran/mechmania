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
3. **Payload defender (Battle)** — one Battle bot (if any exist) is pinned
   near the payload (`role: "payload"`, `payload + forward * payload_defense_offset`).
4. **Remaining battles** (`mining_defense.py`) — split, in priority order:
   - **Miner escort** (`_pick_miner_escorts`) — a fixed `MINER_ESCORT_COUNT`
     (3) battle bots, lined up on the enemy-facing side of our deposit
     (`_compute_guard_positions`), `role: "guard_miners"`. Targets the
     nearest enemy within `blaster_range` of the deposit
     (`_nearest_threat_to_deposit`) over whatever the fleet-wide target pass
     handed it, so it never has to leave its post to chase something.
   - **Raid group** (`_pick_raiders`) — roughly a third of whoever's left,
     `role: "raid"`, headed for the nearest visible enemy `Extractor` or the
     enemy deposit (`_raid_target_pos`).
   - **Formation** — everyone still left gets a target from
     `targeting._assign_battle_targets` (computed once, up front, for the
     whole `battles` list) and a slot from `formation._compute_battle_formation`
     (passed `tick=state.tick`, so a bot an attack just landed on backs off
     out of the screen row for the rest of its invulnerability window and
     the next bot in line takes the front, per `formation.md`),
     `role: "combat"`.
5. **Healer roles** (`healer_roles.py`) — the healer roster is split the
   same way, by id, via `_pick_healer_groups`: `GOAL_HEALER_COUNT` (3) go to
   the goal/payload line, `MINER_HEALER_COUNT` (1) rides with the miner
   escort, `RAID_HEALER_COUNT` (1) rides with the raid group. Goal fills
   first — protecting the win condition outranks either economy job — so a
   roster smaller than 5 leaves the miner/raid healer slots empty rather
   than the goal line short-handed. Within the goal group, the first healer
   (nearest the payload, or already inside capture radius) is the dedicated
   payload defender (`role: "payload"`, pinned at
   `payload - forward * payload_defense_offset`, offset from the Battle
   defender above so one splash can't take out both); the rest of each
   group (goal's other 1-2, the miner healer, the raid healer) stands
   wherever covers the most of its own squad within heal range
   (`heal_coverage._heal_position`, `orchestration.md`'s squad, not the
   whole fleet) — at most `MAX_ATTACKERS_OUT_OF_HEAL_RANGE` (2) left
   outside it, when that's achievable with one healer — and heals whichever
   bot in the squad is hurt (lowest-health ally under 90% HP) from there.
   A squad with no allies at all (e.g. no raiders picked this tick) falls
   back to a rear zone behind that squad's own anchor instead
   (`formation._arrange_group` + `_resolve_slot` + `_dedupe_slots`) rather
   than defaulting to the literal payload point, which is exactly the kind
   of clustering formation exists to prevent.
6. **Catch-all** — any bot not yet assigned (shouldn't normally happen; a
   safety net) defaults to heading for the payload.

Every one of these role splits (`_pick_miner_escorts`, `_pick_raiders`,
`_pick_healer_groups`) is recomputed fresh, by id, on every call — not a
cached identity — so a bot built to replace one that died naturally
backfills whichever slot the id ordering says is short, without any
bookkeeping of "who used to hold this role."

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
  - **Hold instead of advance**: before anything target-specific, if *any*
    enemy is already within `blaster_range` of this bot and it isn't
    `formation._is_retreating`, its move target collapses to its own
    current position — pressing on toward a forward formation/guard/raid
    slot only walks past a fight that's already started, whether or not
    that nearby enemy happens to be this bot's own assigned target. A
    retreating bot is exempt: it's the one bot that's *supposed* to keep
    moving right now, falling back to get healed
    (`formation._is_retreating`), not holding a forward line. The
    target-specific logic below (aim, fire, or flank around an obstacle)
    still runs and can move the bot from there if it needs to maneuver for
    a shot — this only suppresses the *default* pull toward the assigned
    slot.
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
