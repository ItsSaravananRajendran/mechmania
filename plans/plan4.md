# Plan 4 — Reactive Counter

**Strategic axis:** read-then-commit.
**Posture:** observe enemy composition in the first ~50 ticks, then go all-in on the counter-strategy with the saved rush tokens.

## Thesis

The enemy reveals a strategy in the first ~50 ticks through their bot count by class, their extractor placement, and how aggressively they push the payload. Don't commit to a plan until you've seen theirs, then go all-in on the counter-strategy with the build slots you saved.

## Detection (first 50 ticks, very cheap)

Count enemy classes by iterating `state.fleet_other`. The first ~5 enemy bots tell you their build order.

- If ≥2 enemy extractors on *your* deposit by tick 30 → they're running Plan 3 against you. Counter below.
- If enemy battles already at the payload by tick 20 → they're running Plan 1 or 2. Counter below.
- If enemy battles are hugging their own deposit → they're mining, not contesting yet. You have time.

## Counters

- **vs. Economy (Plan 3 enemy):** Save 2 rush tokens for a T40–T50 Battle rush. The moment they rotate their healers to the payload, your battles dive their extractors. Healers can't heal extractors across the map; the only thing protecting them is one battle. Send two battles, kill the protector, then the extractors. While you dive, send one healer + one battle to hold the payload so they can't push.
- **vs. Deathball (Plan 2 enemy):** Match their healer count, but never clump yours. Always keep 3+ tile spacing. Focus fire *one* of their healers at a time — if `heal_stack_cap` = 1, killing the first healer halves their effective sustain. Don't try to fight on the payload line; let them push the payload toward your goal while you kite backward. The payload moving toward your goal is fine *if* you keep your healer pair alive — at endgame, the payload tiebreaker is your friend if you've killed more of theirs.
- **vs. Squeeze (Plan 1 enemy):** You're already countering them by default — they're not rushing, so you have time to build extractors and a bigger fleet. Match their build order but push one more extractor early.

## Build order (default, before detection)

- **T0–T30:** Extractor, Healer, Battle, Extractor, Battle. Two extractors (one for cash, one for slot hold), one healer for sustain, two battles for defense.
- **T30–T50:** continue natural cadence, **do not rush**. You're saving the rush tokens for the counter.
- **T50+:** rush order triggered by detection event.

## Payload tactic

Default: 1 healer + 1 battle in capture range, never advancing. The moment you detect their composition, re-prioritize.

## Economy tactic

One extractor is mandatory, the second is flexible. Don't hold denial slots at first — your opponent's economy is the *thing you're going to attack*, so don't show that you're paying attention to it.

## Endgame shift

The reactive plan's endgame is the same as Plan 1's: stop rushing, consolidate on payload, prefer HP tiebreaker. The advantage you've built by countering is a fleet-quality edge, not a fleet-size edge, so don't throw it away on a reckless final fight.

## Compute strategy

Trivial for the first 50 ticks — you're just counting. The expensive bit is the T50 re-plan. Run it on a budget gate.

## Wins / loses

- **Wins when:** enemy composition is identifiable by tick 50 (almost always is), and your rush_order bank survives long enough to buy the counter.
- **Loses if:** the enemy commits so hard so early (≤20 ticks) that detection happens but you don't have the fleet yet to act on it. Mitigation: keep one battle near the payload from tick 0, as insurance.

## Config dependencies

- All config values matter less than enemy visibility: if the engine's state exposes enough to classify by tick 50, Plan 4 is dominant.
- `conf.bot.blaster_range` long: stronger (you can dive from far away).
- `conf.fabricator.rush_cost` low: stronger (you can afford 2+ rush counters).
- `conf.endgame_ticks` ≥ 800: weaker (enemy has more time to commit before your counter lands).

---

# Requirements — Plan 4

## Detection (ticks 0–50)

- **R-P4.D1** The strategy SHALL compute a snapshot of enemy composition at every multiple of 10 ticks between 0 and 50 inclusive. The snapshot records: enemy fleet size, count by class, count of enemy extractors on `state.deposit_me`, and the position of the enemy fleet centroid relative to the payload.
- **R-P4.D2** The strategy SHALL classify the enemy into one of `{ECONOMY, DEATHBALL, SQUEEZE, UNKNOWN}` based on the rules in R-P4.D3–R-P4.D6.
- **R-P4.D3** Classification `ECONOMY`: ≥2 enemy extractors on `state.deposit_me` at tick 30, OR enemy fleet centroid is within `2 * conf.deposit.radius` of `state.deposit_other.pos` at any detection snapshot.
- **R-P4.D4** Classification `DEATHBALL`: ≥3 enemy battles present, AND the enemy fleet centroid is within `blaster_range * 1.5` of the payload at any detection snapshot, AND enemy healer count ≥ 2.
- **R-P4.D5** Classification `SQUEEZE`: enemy battles present at payload capture range at tick 20 (any count ≥ 1).
- **R-P4.D6** Classification `UNKNOWN`: none of D3–D5 matched by tick 50.

## Build order

- **R-P4.B1** The strategy SHALL NOT issue any `rush_order` before tick 50, regardless of available tokens.
- **R-P4.B2** Between ticks 0 and 50, the fabricator SHALL produce: Extractor, Healer, Battle, Extractor, Battle (in cycle order, capped at fleet size 5 by tick 50).
- **R-P4.B3** After detection completes, the build order switches to the counter-strategy chosen per R-P4.S1.

## Counter selection

- **R-P4.S1** Once classification is non-`UNKNOWN`, the strategy SHALL switch to:
  - vs `ECONOMY` → Plan 2's denial deathball mode (R-P2.B2 onward), modified per R-P4.S2.
  - vs `DEATHBALL` → Plan 5's skirmisher kiter mode (R-P5.P1 onward), modified per R-P4.S3.
  - vs `SQUEEZE` → Plan 3's economic tempo (R-P3.B1 onward).
- **R-P4.S2** vs `ECONOMY` modification: two Battles SHALL dive the enemy Extractors on `state.deposit_me` at the counter-strategy switch tick. The dive target SHALL be the lowest-HP enemy extractor on `state.deposit_me`.
- **R-P4.S3** vs `DEATHBALL` modification: the Healer SHALL be positioned at the back of the friendly formation, 4+ tiles behind the front-most Battle. Kiting retreat is preferred over holding.

## Payload

- **R-P4.P1** At every tick from tick 0, at least one Battle SHALL be within payload capture range.
- **R-P4.P2** After counter-selection, payload posture is governed by the selected counter-plan.

## Combat

- **R-P4.C1** Detection is read-only. The strategy SHALL NOT issue attacks on enemy Extractors on `state.deposit_me` before classification.
- **R-P4.C2** Until tick 50, Battles fire on the closest enemy in `line_of_sight` and `blaster_range`, identical to R-P1.C2.

## Endgame

- **R-P4.END1** Same as R-P1.END1 — no rush orders in endgame.
- **R-P4.END2** Same as R-P2.END2 — extractors self-destruct if too far from payload.
- **R-P4.END3** During endgame, the strategy SHALL prefer HP tiebreaker: formation spacing per R-P3.END3.

## Compute

- **R-P4.COMP1** Detection snapshots (R-P4.D1) SHALL be O(enemy_fleet) per snapshot, computed at most 6 times per match.
- **R-P4.COMP2** The counter-selection at tick 50+ is the most expensive operation. It SHALL execute only when `get_budget().remaining > 200_000`.
- **R-P4.COMP3** Fallback when in debt: same as R-P1.COMP4.

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

- Detection snapshots per tick → R-P4.D*
- Classification transitions → R-P4.D2, R-P4.S1
- `fabricator_next` and `rush_order` per tick → R-P4.B*, R-P4.END1
- bot action per tick → R-P4.C*, R-P4.P*, R-P4.S2, R-P4.S3
- position per bot per tick → R-P4.P*, R-P4.S*
- self-destruct events → R-P4.END2
- `get_budget().last_charge` distribution → R-P4.COMP*
