# Plan 5 — Skirmisher Kiter

**Strategic axis:** anti-splash, anti-rush.
**Posture:** build a fleet where no two bots share a tile **ever**; kite the enemy into shooting clumps of *their own* bots; commit at endgame.

## Thesis

Splash damage is the most snowbally mechanic in the game. Build a fleet where no two of your bots share a tile **ever**, and kite the enemy into shooting clumps of *their own* bots. Win by kiting the enemy deathball into your back line until its healers can't keep up.

## Build order

- **T0–T30:** Extractor, Battle, Battle, Battle, Healer, Battle. Four battles by tick 30, spaced apart.
- **T30–T90:** alternate Healer and Battle, never two of the same class back-to-back in adjacent fabricator cycles.
- No extractor beyond the first one; this plan concedes the economy.

## Payload tactic

Always have **one battle in capture range, never more**. That battle's job is to be in capture range, not to push. The other 3–4 battles form a screen 5 tiles behind the payload line. When an enemy approaches, the screen battles rotate *away* from the enemy (turning the engagement into a kite), never forward. The healer is 3 tiles behind the screen, facing whichever screen battle has the lowest HP. The payload advances slowly under this setup — but it advances, and the enemy can't catch your fleet without clumping.

## Economy tactic

One extractor at the deposit, idle-near if there's no slot available. Don't try to hold every slot. The token income is enough for the natural build cadence.

## Combat tactic

The key move: when an enemy battle is in `blaster_range` of your screen battle, your screen battle fires *one shot* then **backs up** using `navigate_to(retreat_point)` where the retreat point is `2 × blaster_range` further from the enemy. This forces the enemy to either advance (running into your healer's coverage as the range opens up) or hold (letting your battle reset its cooldown). Never chase a kill — every step forward puts two of your bots at risk of sharing a tile by accident.

### Targeting

Each screen battle's target is the enemy with the lowest HP that's currently in `line_of_sight`. Healers get focus fired only when isolated (no other enemy in `splash_radius` of them). Splash kills are opportunistic, never primary.

## Endgame shift

Stop the kite — endgame means a wipe is instant loss, so commit. All battles collapse to the payload line, healer rotates to the front-most battle, and you fight it out. The kiting advantage has already paid off in HP and kills; endgame is where you cash it in.

## Compute strategy

The kite is the expensive part. Retarget every 3 ticks; on the in-between ticks, keep the last assignment. Use `corridor_clear` to verify retreat paths don't push you through a wall — getting cornered with a fleet that won't clump is the worst-case failure mode.

## Wins / loses

- **Wins when:** splash radius is large (so enemy clumping is lethal to them), `blaster_cooldown` is long (so the kite advantage compounds), and the enemy runs Plan 2 (deathball).
- **Loses if:** the enemy runs Plan 4 with a healer dive — they won't clump, and your kite advantage disappears.

## Config dependencies

- `conf.bot.blaster_range` < 5: stronger (you can kite inside the healer's arc).
- `conf.bot.blaster_cooldown` > 5: stronger (the cooldown clock favors the kiter).
- Splash radius ≥ 2 tiles: stronger (enemy clumping is more lethal to themselves than to you).
- `conf.bot.heal_stack_cap` = 1: weaker (your single healer can't sustain as well).

---

# Requirements — Plan 5

## Build order

- **R-P5.B1** The fabricator SHALL produce `BotClass.Extractor` once and only once across the entire match, at the earliest opportunity.
- **R-P5.B2** The next 4 builds (in any combination of rush + natural) SHALL be `BotClass.Battle` until the fleet contains 4 Battles by tick 30.
- **R-P5.B3** After tick 30, the fabricator SHALL alternate `BotClass.Battle` and `BotClass.Healer` such that no two consecutive builds are the same class.
- **R-P5.B4** The fleet SHALL contain no more than 1 Extractor at any time.

## Payload

- **R-P5.P1** Exactly 1 friendly bot SHALL be in payload capture range at all times. If multiple bots enter capture range, the excess SHALL navigate out using `navigate_to(target_outside_capture_range)`.
- **R-P5.P2** The screen formation (Battles not in capture range) SHALL be positioned at a distance of `2 * conf.bot.blaster_range` from the payload on the friendly side, arranged in a line with ≥3 tile spacing between adjacent screen battles.
- **R-P5.P3** The Healer SHALL be positioned 3 tiles behind the screen centroid, facing the forward-most screen Battle.
- **R-P5.P4** The payload-capture Battle SHALL advance the payload by remaining in capture range; it SHALL NOT advance beyond the payload's current position.

## Economy

- **R-P5.E1** The single Extractor SHALL be parked at `state.deposit_me.pos` plus a stand-off offset of `2 * conf.deposit.radius` in the direction away from the enemy, mining continuously.
- **R-P5.E2** The strategy SHALL NOT attempt to hold denial slots. If the extractor cannot acquire a slot within 30 ticks, it SHALL hold position and mine by ray (`line_of_sight` to deposit) without entering the disc.

## Combat

- **R-P5.C1** When any enemy bot enters `blaster_range` of a screen Battle, that Battle SHALL fire at most once per enemy engagement, then navigate to a retreat point at distance `2 * conf.bot.blaster_range` away from the enemy centroid, on the line connecting the screen Battle to the friendly Healer.
- **R-P5.C2** The retreat path SHALL be verified with `corridor_clear(bot.pos, retreat_point, conf.bot.radius)` before execution. If the path is blocked, the Battle SHALL sidestep perpendicular to the enemy line by `1.5 * conf.bot.radius`.
- **R-P5.C3** Target selection: lowest-HP enemy in `line_of_sight`, with enemy Healers preferred when no other enemy is within the friendly bot's splash radius of the target.
- **R-P5.C4** The screen SHALL NOT chase a retreating enemy beyond `conf.bot.blaster_range` from its current position.
- **R-P5.C5** Friendly bots SHALL NEVER occupy the same tile. Distance check is enforced at the start of every tick; violations are corrected with `navigate_to(spread_target)`.

## Endgame

- **R-P5.END1** `action.rush_order` SHALL be False once `state.tick >= conf.max_ticks - conf.endgame_ticks`.
- **R-P5.END2** At endgame entry, the kiting posture SHALL be abandoned. All Battles SHALL collapse to within `2 * conf.bot.radius` of the payload capture position. The Healer SHALL be repositioned at the centroid of the Battle cluster, facing the enemy fleet centroid.
- **R-P5.END3** The single Extractor SHALL self-destruct at endgame entry unless `path_length(extractor.pos, payload) <= conf.bot.blaster_range * 4`.

## Compute

- **R-P5.COMP1** Retarget logic SHALL execute every 3 ticks; in between, the cached target assignment is used.
- **R-P5.COMP2** Retreat paths SHALL be computed once per enemy engagement, not per tick. An engagement is "the same" until all enemies leave `blaster_range` for ≥10 consecutive ticks.
- **R-P5.COMP3** Fallback when in debt: hold current screen position, fire cached target if in range, healer heals closest ally.

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

- `fabricator_next` per tick → R-P5.B*
- bot action per tick → R-P5.C*, R-P5.P*
- `rush_order` per tick → R-P5.END1
- position per bot per tick → R-P5.P*, R-P5.E*, R-P5.C5
- self-destruct events → R-P5.END3
- capture-range occupancy count → R-P5.P1
- inter-bot tile overlap count → R-P5.C5
- `get_budget().last_charge` distribution → R-P5.COMP*
