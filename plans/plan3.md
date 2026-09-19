# Plan 3 — Economic Tempo

**Strategic axis:** economy-first, late push.
**Posture:** don't fight the payload contest for the first ~80 ticks; stack extractors and let natural build cadence give a 2:1 fleet advantage by tick 100; commit the whole fleet onto the payload in one coordinated push.

## Thesis

Don't fight the payload contest for the first ~80 ticks. Stack extractors, hold every slot, and let the natural build cadence give you a 2:1 fleet advantage by tick 100. Then commit the whole fleet onto the payload in a single coordinated push, win via payload position + total HP tiebreaker if it goes long.

## Build order

- **T0–T60:** Extractor, Extractor, Extractor, Healer, Healer, Extractor, Battle, Battle, Healer — 9 bots, 4 extractors, 3 healers, 2 battles by tick 60. The battles are bodyguards for the healers on their way to the payload line.
- **T60–T150:** alternate Healer and Battle on the fabricator; do **not** build more extractors.
- **T120:** first `rush_order` — buy a Battle now that your token bank is fat. Second rush at T140 (Battle). Cap at ~16 bots.

## Payload tactic

For the first 60 ticks, **concede payload presence**. Let the enemy push toward your goal. Send your healer/battle bodyguard pair to a hold position ~5 tiles in front of your deposit, between your deposit and your goal. Their only job is to survive and screen your extractors. Only when fleet ≥ 10 bots does the full fleet rotate to the payload line, in one move.

## Economy tactic

This is the whole plan. **Fill every slot on your deposit.** Run a cheap tickly check: if `state.deposit_me.extractors` (your bitmask) doesn't cover the slot count, send the nearest idle bot to the deposit. *Don't touch the enemy deposit* — slot denial on the enemy side is wasted travel time at this stage; you want your extractors producing, not marching across the map.

## Combat tactic

Defensive only until T80. Healers face the closest enemy and heal whichever allied bot has lowest HP (the one in danger of dying first). Battles only fire if the target is *already* in `blaster_range` from where the battle is positioned — no advancing to shoot. **The moment a healer dies, fall back.** Recompute healer survival odds every 5 ticks. After T80, switch to the same focus-fire-targets-healers tactic as Plan 2.

## Endgame shift

This is where the tempo plan pays off: by endgame, you should have 14–16 bots where the enemy has 9–11. The payload tiebreaker + HP tiebreaker both favor you. In endgame, **concentrate the fleet at the payload's current position** rather than chasing kills — a wipe at endgame is instant loss, so all combat becomes "stay together, push payload, fire when in range." Self-destruct any extractor that can't reach the payload in time so the enemy doesn't get a free kill that doesn't matter.

## Compute strategy

Trivially cheap for the first 60 ticks (you're mostly holding positions). The expensive thing is the T80 transition — that's when you run the full plan in one tick. Gate it on `get_budget().remaining > 300_000`. If you don't have it, wait one more tick.

## Wins / loses

- **Wins when:** `fabricator.interval` is long enough that the enemy can't rush-spam their way back into the fleet-count game, and `endgame_ticks` is short enough that the late fleet advantage converts before time runs out.
- **Loses if:** the enemy is willing to push the payload 80% of the way in the first 60 ticks. Flag: "is the payload already past the center line at T60? if yes, abandon this plan and switch to Plan 1."

## Config dependencies

- `conf.fabricator.interval` long (≥ 50): stronger (more out-scale).
- `conf.fabricator.rush_cost` high (≥ 25% of natural interval tokens): stronger (enemy can't catch up).
- `conf.deposit` slot count ≥ 6: stronger (more cash flow).
- `conf.endgame_ticks` ≤ 500: stronger (less time for enemy to come back).
- `conf.bot.heal_per_tick` high: stronger (your healers sustain through the late push).

---

# Requirements — Plan 3

## Build order

- **R-P3.B1** Between ticks 0 and 60 inclusive, the strategy SHALL produce at least 4 Extractors and 3 Healers total. Battles in this window SHALL NOT exceed 2.
- **R-P3.B2** After tick 60 and before tick 120, the fabricator SHALL alternate `BotClass.Healer` and `BotClass.Battle` on natural builds, starting with Healer. No new Extractors SHALL be built in this window.
- **R-P3.B3** The first `rush_order` SHALL occur no earlier than tick 120 and SHALL produce `BotClass.Battle`. The second rush SHALL occur at least 20 ticks later and SHALL also produce `BotClass.Battle`.
- **R-P3.B4** After the second rush, the fabricator SHALL continue natural Healer/Battle alternation until fleet size reaches `min(16, conf.bot.squad_size)`.

## Payload

- **R-P3.P1** Before tick 60, the strategy SHALL maintain zero bots within payload capture range.
- **R-P3.P2** Before tick 60, the Healer/Battle bodyguard pair SHALL hold a position at the midpoint between `state.deposit_me.pos` and `state.payload_pos()`, on the friendly half of the map.
- **R-P3.P3** At tick 60, if fleet size ≥ 10, the entire fleet SHALL rotate toward payload capture range within 30 ticks. The rotation SHALL be a single coordinated move, not staggered.
- **R-P3.P4** If, at tick 60, the payload is already past the map center line (further toward friendly goal than enemy goal), R-P3.P3 SHALL be overridden and the strategy SHALL abandon Plan 3 in favor of Plan 1's payload hold.

## Economy

- **R-P3.E1** The strategy SHALL fill every available slot on `state.deposit_me` by tick 30. Slot count is determined from `conf.deposit` and verified against `state.deposit_me.extractors`.
- **R-P3.E2** No bot SHALL be sent to `state.deposit_other` at any point in the match.
- **R-P3.E3** Extractors SHALL be idle-parked (not moving) once they hold a slot. The mining ray SHALL be continuously active.

## Combat

- **R-P3.C1** Before tick 80, Battles SHALL fire only when the target is already within `conf.bot.blaster_range` from the bot's current position. Advancing to fire is PROHIBITED.
- **R-P3.C2** Before tick 80, Healer target selection SHALL be the lowest-HP allied bot within `heal_range` and arc. Allies at full HP SHALL NOT be targeted.
- **R-P3.C3** Before tick 80, if any friendly bot is destroyed, the surviving bots SHALL retreat to the bodyguard hold position from R-P3.P2.
- **R-P3.C4** At and after tick 80, target selection follows the same focus-Healers-first ranking as Plan 2 (R-P2.C1).

## Endgame

- **R-P3.END1** `action.rush_order` SHALL be False once `state.tick >= conf.max_ticks - conf.endgame_ticks`.
- **R-P3.END2** During endgame, every surviving bot SHALL be commanded toward payload capture range each tick. Bots that cannot reach capture range within 10 ticks SHALL self-destruct.
- **R-P3.END3** The strategy SHALL prefer a concentrated formation (≤ 2 tile spacing between adjacent bots) during endgame to maximize combined HP tiebreaker.

## Compute

- **R-P3.COMP1** Per-tick cost SHALL be <100 engine ticks for the first 60 ticks (position-hold only).
- **R-P3.COMP2** The T60 fleet rotation SHALL be the single most expensive tick. It SHALL execute only when `get_budget().remaining > 300_000`. If the budget is insufficient, the rotation SHALL be deferred by one tick and retried.
- **R-P3.COMP3** Fallback when in debt: hold position, heal lowest-HP ally, fire if in range.

## Cross-plan (applies here too)

- **R-X1** Call `get_budget()` exactly once per tick.
- **R-X2** Detect sitting-out gaps via `state.tick - last_tick` and downshift to fallback if gap > 1.
- **R-X3** No `path_length` inside bot-pair loops; cache results.
- **R-X4** No custom pathfinding; use `navigate_to`.
- **R-X5** All thresholds read from `get_config()`.
- **R-X6** Deterministic for identical state.
- **R-X7** `fabricator_next` always set.
- **R-X8** `action.bots[bot.id]` assigned for every alive bot every tick.
- **R-X9** Always return a `FleetAction`.
- **R-X10** No tick > 5.0s wall-clock.

## Verifiability

- `fabricator_next` per tick → R-P3.B*
- bot action per tick → R-P3.C*, R-P3.P*
- `rush_order` per tick → R-P3.B3, R-P3.END1
- position per bot per tick → R-P3.P*, R-P3.E*, R-P3.C3
- self-destruct events → R-P3.END2
- slot bitmask over time → R-P3.E1
- `get_budget().last_charge` distribution → R-P3.COMP*
