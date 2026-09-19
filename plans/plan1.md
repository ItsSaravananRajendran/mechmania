# Plan 1 — The Payload Squeeze

**Strategic axis:** payload presence first, economy steady, combat reactive.
**Posture:** symmetric fleet, hard commit to payload presence, deny enemy mining slots, win by tick 60–120.

## Thesis

The payload is the only objective that ends the match on its own. Hold ≥1 bot in capture range every tick, deny the enemy enough slots that they can't field a contesting force without starving, and the match ends in your favor.

## Build order (`fabricator_next` over time)

- **T0–T30:** Extractor, Extractor, Extractor, Healer, Battle. Five bots in roughly half a fabricator interval each. The point isn't income — it's that your first healer reaches the payload before the enemy's first battle does.
- **T30–T90:** alternate Battle and Healer; aim for a steady **3 Extractor : 2 Healer : 5 Battle** core by tick 90 (~10 bots).
- **T90 onward:** stop building Extractors unless you see the enemy gain a slot you don't hold. Fabricator flips to Battle unless a slot opens up.

## Payload tactic

Always park at least one Healer and one Battle inside capture range. The healer faces the payload and rotates to face the nearest enemy on the same tick it heals; the battle faces the closest enemy and fires only when `line_of_sight` is clear and within `blaster_range`. Don't push the payload fast — every step forward puts you deeper into enemy blaster range. **Sit on the payload at the line of symmetry.** The enemy either walks into your blaster/cooldown window or concedes the contest.

## Economy tactic

Two extractors on your deposit is enough for cash flow; the third slot is **denial**. Count `state.deposit_me.extractors` (your bitmask) and the enemy bitmask (`state.deposit_other.extractors`). If the enemy has any extractor on your deposit, the slot is occupied by them — you can't mine from your deposit there, but you've denied them one of the *enemy* deposit's slots too. Keep one extractor idle-near the deposit (facing it, not moving) so it holds a slot without pathing mistakes.

## Combat tactic

Stagger battles so they never share a tile — splash radius on your own fleet is the cleanest way to lose a fight. Keep 3+ tiles between battles. Use `line_of_sight` to gate every shot: a missed shot puts the blaster on cooldown for `conf.bot.blaster_cooldown` ticks. Assign targets by `path_length` to the enemy, not Euclidean distance — a battle that's behind a wall wastes a shot if you only check `dist`.

## Endgame shift

Stop `rush_order`ing. Anything you haven't built by `state.tick >= conf.max_ticks - conf.endgame_ticks` you won't build. Spend the saved tokens as a reserve. If you're behind on the payload tiebreaker, swap one healer off the payload and into a sacrificial rush on the enemy healer so the enemy can't accumulate that heal stack. If you're ahead, self-destruct your weakest extractor to deny the enemy a kill that doesn't matter — but only if you're at risk of being wiped (endgame wipe = instant loss).

## Compute strategy

Plan path/target assignments every 5 ticks; on the in-between ticks, repeat the last assignment. `get_budget().remaining > 100_000` gates the recompute; below that, fall back to "hold last assignment, fire if in range, heal if anyone is below half HP." Total cost should stay under ~300 engine ticks per tick (early game).

## Wins / loses

- **Wins when:** enemy over-extracts and leaves the payload unopposed, or enemy clumps battles and you multikill with splash.
- **Loses to:** a fast Battle rush that kills your healers before you have a second healer stacked.

## Config dependencies

- `conf.bot.blaster_range` < 6: stronger (you hold payload without being shot).
- `conf.bot.heal_stack_cap` = 1: weaker (you can't stack healer value).
- `conf.deposit` slot count ≥ 4: weaker (slot denial has less impact).
- `conf.endgame_ticks` ≥ 1000: stronger (more time for payload squeeze to convert).

---

# Requirements — Plan 1

## Build order

- **R-P1.B1** The fabricator SHALL set `fabricator_next` to `BotClass.Extractor` while the bot in slot 0 is missing (`not state.fleet_me.get(0)`).
- **R-P1.B2** Between ticks 0 and 30 inclusive, the fabricator SHALL produce at least 3 extractors total (natural builds + rush orders combined).
- **R-P1.B3** After tick 30, the fabricator SHALL alternate `BotClass.Battle` and `BotClass.Healer` on each natural build until fleet size reaches 10.
- **R-P1.B4** After fleet size reaches 10, the fabricator SHALL produce `BotClass.Battle` on every natural build UNLESS `state.deposit_me.extractors` shows fewer ally extractors mining than the enemy's extractor count on `state.deposit_me`, in which case the next build SHALL be `BotClass.Extractor`.

## Payload

- **R-P1.P1** At every tick, the strategy SHALL maintain ≥1 friendly bot within capture range of the payload.
- **R-P1.P2** When the payload has no enemy within capture range, the fleet SHALL NOT advance beyond the payload's current capture position by more than `2 * conf.bot.radius`.
- **R-P1.P3** At least one Healer SHALL be present in capture range whenever the fleet has ≥1 Healer alive.
- **R-P1.P4** The Healer in capture range SHALL face the payload and switch target to the nearest enemy bot the same tick an enemy enters capture range, then resume facing the payload when the enemy leaves.

## Economy

- **R-P1.E1** The strategy SHALL ensure `state.deposit_me.extractors` shows ≥2 ally extractors mining by tick 60.
- **R-P1.E2** An Extractor assigned to mine SHALL remain within `conf.bot.radius + conf.deposit.radius + 1.0` of `state.deposit_me.pos` and SHALL continuously issue `SpecialAction.Extractor(mine=True)` while it holds a slot.
- **R-P1.E3** When an enemy Extractor is occupying a slot on `state.deposit_me`, the strategy SHALL NOT divert combat bots to attack it; it SHALL wait for the slot to free.

## Combat

- **R-P1.C1** Every Battle bot SHALL have ≥3 tile Manhattan distance from every other friendly Battle bot. Distance is measured between bot centers; violations are corrected via `navigate_to` to a spreading target.
- **R-P1.C2** A Battle SHALL fire only when the target satisfies BOTH `bot.pos.dist(target.pos) <= conf.bot.blaster_range` AND `line_of_sight(bot.pos, target.pos)` is true.
- **R-P1.C3** Target selection SHALL rank enemy bots by `path_length(bot.pos, target.pos)`, ascending. The closest by walk-distance SHALL be selected.
- **R-P1.C4** A Battle SHALL NOT advance into enemy `blaster_range` if doing so would put it within `conf.bot.radius` of another friendly Battle.

## Endgame

- **R-P1.END1** The strategy SHALL set `action.rush_order = False` once `state.tick >= conf.max_ticks - conf.endgame_ticks`.
- **R-P1.END2** If, on entering endgame, total surviving fleet HP is below enemy fleet HP by ≥25%, the strategy SHALL self-destruct the weakest Extractor (lowest HP among extractors) using `BotAction.self_destruct = True`.
- **R-P1.END3** During endgame, at least one bot SHALL remain in payload capture range every tick until match end.

## Compute

- **R-P1.COMP1** The strategy SHALL call `get_budget()` exactly once per tick, at the top of the strategy function.
- **R-P1.COMP2** Target assignment and payload routing SHALL be recomputed at most every 5 ticks unless any of the following changes occur: enemy fleet size delta ≥ 2, any friendly bot enters or leaves capture range, or a friendly bot is destroyed.
- **R-P1.COMP3** When `get_budget().remaining <= 100_000`, the strategy SHALL execute the cheap fallback action defined in R-P1.COMP4 instead of recomputing targets.
- **R-P1.COMP4** Cheap fallback: each Battle fires on its current target if conditions in R-P1.C2 hold; each Healer heals its current target if the target is within `heal_range` and within the facing arc; each Extractor continues mining its current slot.

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

- `fabricator_next` per tick → R-P1.B*
- bot action per tick → R-P1.C*, R-P1.P*
- `rush_order` per tick → R-P1.END1
- position per bot per tick → R-P1.P*, R-P1.E*
- self-destruct events → R-P1.END2
- slot bitmask over time → R-P1.E1
- `get_budget().last_charge` distribution → R-P1.COMP*
