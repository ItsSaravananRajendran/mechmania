# `targeting.py` — fleet-wide, cooldown-aware target assignment

## Problem

A naive "each bot shoots the nearest enemy" strategy causes two failures at
once: several bots dogpile the same juicy target while others go untouched,
and bots waste shots on a target that's still invulnerable from a hit this
same tick (`invulnerable_until_tick`), which only ever caps at one landed hit
per tick regardless of how many shots arrive. The goal is to spread fire
across the highest-priority *reachable* enemies, one attacker per target per
pass.

## `_target_priority(enemy, conf) -> float`

`class_weight * 1000 - health + low_hp_bonus`, where:

- `class_weight` is `_CLASS_PRIORITY = {Healer: 3, Extractor: 2, Battle: 1}`
  — healers keep the enemy fleet topped up, extractors fund it, battle bots
  are the least valuable kill of the three.
- `-health` — among equal-class targets, prefer the one closer to dying.
- `+50` if the target is at or below 30% of `conf.bot.health` — a push to
  finish a target that's nearly dead rather than spread damage thin.

## `_assign_battle_targets(battles, state, conf) -> Dict[bot_id, Optional[enemy_id]]`

A **greedy, not globally optimal**, assignment — optimal bipartite matching
isn't worth the compute for fleets this size; greedy already eliminates the
redundant-fire case that actually matters.

1. Build every `(bot, enemy)` pair within `blaster_range`, scored by
   `_target_priority`, with two penalties layered on:
   - `-5000` if `_has_clear_shot` says no (still a legal candidate — the
     caller can maneuver to fix that — but a clear shot elsewhere should win
     the pick).
   - `-10000` if the enemy is still invulnerable (`invulnerable_until_tick >
     tick`) — still assignable, so an idle bot has somewhere to aim while the
     window closes, but ranked behind anything hittable right now.
2. Sort all pairs by `(-priority, distance)` — highest priority first,
   nearest as the tiebreaker.
3. Walk the sorted list once: the first time a bot appears, claim its
   highest-ranked still-unclaimed enemy; skip pairs whose bot or enemy is
   already spoken for. This is what spreads fire — once an enemy is claimed
   this pass, every other bot's entry for that same enemy is skipped in
   favor of their next-best *different* target.
4. Any bot left unassigned (no candidate in range at all, or every candidate
   it could reach was claimed by someone else) gets the enemy nearest by
   **straight-line** distance — not `path_length`. This loop is
   bots-without-a-target × every enemy, which in the worst case (nobody in
   range yet, e.g. early game) is the full bot-pair count, and `path_length`
   in an all-pairs loop is one of the two most expensive things a strategy
   can do per tick (see the compute-budget guidance this repo follows). The
   wall-aware route there is `navigate_to`'s job once the bot is actually
   moving, not this pick's.

## Edge cases this handles

- **No battles**: returns `{}` immediately.
- **No enemies**: every battle bot maps to `None`.
- **Every enemy invulnerable**: still assigns (penalized) rather than
  leaving bots with nothing to aim at.
- **More bots than enemies**: excess bots fall through to the nearest-enemy
  fallback rather than idling.
- **A target neither in range nor clear-shot for anyone**: never assigned by
  the greedy pass, but still reachable via the nearest-by-distance fallback
  for whichever bot is closest to it.
