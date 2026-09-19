# `mining_defense.py` — miner escorts and enemy-miner raiders

## Problem

`formation.py` gives every battle bot not tied down defending the payload a
spot on the payload/formation line -- but that leaves the mining line
(`_compute_mining_spots`, up to `MAX_ACTIVE_MINERS` extractors) completely
undefended, and gives us no way to apply the same pressure back on the
enemy's own miners. This module carves two small, fixed-size jobs off the
battle roster for that, before whatever's left goes to the formation line:

- **Escort** — a fixed crew (`MINER_ESCORT_COUNT`, matching
  `MAX_ACTIVE_MINERS`) camped on the enemy-facing side of our deposit,
  shooting whatever comes to harass the miners instead of the miners having
  to fend for themselves.
- **Raid** — roughly a third (`RAID_FRACTION`) of whoever's left, sent to
  hunt the enemy's own extractors (or their deposit, with none in view).

Both selections are picked fresh, by id, on every recompute -- not cached
identities -- so a bot built to replace one that died naturally slots into
whichever role the id ordering says is short, without any bookkeeping of
"who used to be an escort."

## `_pick_miner_escorts(battles, count=MINER_ESCORT_COUNT) -> List[BotState]`

The `count` lowest-id battle bots. Stable by construction: id order doesn't
reshuffle on its own, so the escort roster only changes membership when a
low-id escort actually dies (or a new build's id happens to be lower than an
existing escort's, which never happens -- ids only increase).

## `_pick_raiders(candidates, fraction=RAID_FRACTION) -> List[BotState]`

`round(len(candidates) * fraction)` of `candidates` (already filtered down
to "not an escort, not the payload defender" by the caller), lowest ids
first. Rounds to the nearest whole bot, but always sends at least one once
there are 2+ candidates to spare -- pulling the *only* remaining candidate
off would leave nothing for the payload/formation line to work with, so a
single leftover bot stays there instead.

## `_compute_guard_positions(escorts, state, forward, conf, wall_grid) -> Dict[bot_id, Vec2]`

Reuses `formation.py`'s `_arrange_group` / `_resolve_slot` / `_dedupe_slots`
primitives (same wall-safety fallbacks, same `_min_safe_spacing` floor) to
line the escorts up abreast on the enemy-facing side of the deposit
(`deposit_pos + forward * max(2·radius, 0.3·blaster_range)`) -- the side an
attacker has to cross to reach the miners, not somewhere behind them.

## `_nearest_threat_to_deposit(state, conf) -> Optional[BotState]`

The nearest enemy within `blaster_range` of our deposit, or `None`. This is
what an escort actually shoots at (see `orchestration.md`) instead of
whatever the fleet-wide `_assign_battle_targets` pass happened to hand it --
that assignment doesn't know an escort is meant to hold position, so it can
hand out a target the escort would have to leave its post to chase.

## `_raid_target_pos(state, raider) -> Vec2`

The raider's nearest visible enemy `Extractor`, or the enemy deposit itself
if none are currently in view (extractors spawn from and cluster around it,
so it's still the right place to go looking).

## How this plugs into `orchestration.md`

`_recompute_assignments` calls these in order, after the dedicated payload
defenders are picked and before `_compute_battle_formation` sees what's
left: escorts get `role: "guard_miners"`, raiders get `role: "raid"`, and
everyone else still goes through the normal formation/combat path
(`role: "combat"`). All three roles execute through the same generic battle
branch in `_apply_assignments` -- the role string is bookkeeping only, not
a fork in the movement/firing logic.
