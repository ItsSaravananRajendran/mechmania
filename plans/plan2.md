# Plan 2 — The Denial Deathball

**Strategic axis:** combat-first, slot-starved enemy.
**Posture:** rush a battle-heavy army, use healers as a stacked front, take payload by force while holding all the deposit slots.

## Thesis

The deposit is the real choke point. Fill every slot on your deposit with extractors, send the rest of the fleet as a stacked healer+battle front onto the payload. If you hold the payload with a battle front and your extractors are holding zero of *the enemy's* slots, the enemy is buying nothing and you're pushing for free.

## Build order

- **T0–T20:** rush-order two extractors immediately (only if `conf.fabricator.rush_cost` ≤ ~half your natural build interval's worth of tokens — flag this). You want 3 extractors on your deposit by tick 25.
- **T20–T60:** rush Healer, Healer, Battle, Battle, Healer, Battle. The two early healers are the heal-stack backbone.
- **T60–T120:** rush Battle as fast as tokens allow until fleet is at ~14–16 bots.

## Payload tactic

Walk the healer pair together — they must remain within ~2 tiles so the second healer can stack on the same Battle target the moment combat starts. The first healer is "tank" (front-most), the second healer is "anvil" (1–2 tiles behind, never on top of the first). The battle front advances in a staggered column: 2 tiles apart, never 1 tile apart. The point is to be inside enemy `blaster_range` only with one battle at a time when possible.

## Economy tactic

Hold **every slot on your own deposit** with extractors that are parked on `mining_spot = state.deposit_me.pos + Vec2(0, conf.deposit.radius + conf.bot.radius)` — hull-hugging the disc edge. Don't move them. Use `turn_towards(state.deposit_me.pos)` so the mining ray stays locked on the deposit. **Do not chase enemy extractors on your deposit** — the moment they leave, you fill the slot for free. Send one sacrificial battle to *the enemy's deposit* if `conf.bot.blaster_range` × 2 ≤ `path_length(enemy deposit, your healer)`, but only when your fleet has ≥10 bots. Most of the time, just mine.

## Combat tactic

Every battle's target is the same enemy healer (focus fire) — healers are the constraint on enemy burst, and killing one reshapes their cooldown curve. Don't shoot at battles unless no healers are in line-of-sight. Splash multi-kills only happen if the enemy clumps, so don't try to engineer them; just keep your own bot spacing at ≥2 tiles (splash radius will still let you score hits on grouped enemies without overcommitting yours).

## Endgame shift

By endgame, the deathball should be parked on the payload with 2 healers + 4 battles. If the enemy still has an extractor alive, the rush has already won you the slot game; consider `self_destruct` on your weakest extractor at endgame entry so the enemy can't get the kill. Save 1 token-bank reserve by **never rush-ordering in the last `endgame_ticks`** ticks — those tokens become nothing once the fabricator shuts off.

## Compute strategy

Target assignment is the same "nearest enemy healer" every tick — cache it once per 10 ticks and only recompute on visible enemy bot count change (cheap O(n²) scan over fleets). Use `path_length` not `dist` for healer targeting so you account for walls.

## Wins / loses

- **Wins when:** `heal_stack_cap` ≥ 2 (your healer stack works), enemy has fewer than 3 extractors on your deposit early (the slot game locks them out).
- **Loses if:** `blaster_range` is long enough that battles can kite your healer pair apart — the heal stack breaks.

## Config dependencies

- `conf.bot.heal_stack_cap` ≥ 2: required for heal-stack backbone.
- `conf.deposit` slot count < 4: stronger (slot-lockdown near-guaranteed).
- `conf.fabricator.rush_cost` low enough to fit 6+ rushes before endgame: required.
- `conf.bot.blaster_range` short enough that you reach healers before they kite: stronger.

---

# Requirements — Plan 2

## Build order

- **R-P2.B1** When `state.fabricator_me.tokens >= conf.fabricator.rush_cost` AND fleet size < `conf.bot.squad_size / 2`, the strategy SHALL set `action.rush_order = True` AND `fabricator_next = BotClass.Extractor` until the fleet contains ≥2 extractors.
- **R-P2.B2** The next two rush orders (after R-P2.B1) SHALL produce `BotClass.Healer` and `BotClass.Healer` in either order.
- **R-P2.B3** All subsequent rush orders SHALL produce `BotClass.Battle` until the fleet reaches 14 bots or `conf.bot.squad_size`, whichever is lower.
- **R-P2.B4** Natural builds SHALL default to `BotClass.Battle` while fleet size < 10, then alternate `Healer` / `Battle`.

## Payload

- **R-P2.P1** The two Healers SHALL be positioned such that the distance between them is in `[2 * conf.bot.radius, 3 * conf.bot.radius]` whenever both are alive. They SHALL move together.
- **R-P2.P2** The first Healer (lead) SHALL be the front-most bot in the fleet's advance order. The second Healer (anvil) SHALL be positioned 1–2 tiles behind the lead.
- **R-P2.P3** Battles in the front rank SHALL be spaced ≥2 tiles apart and SHALL advance in a staggered column (no two battles share the same Y axis when within `blaster_range` of the enemy fleet centroid).
- **R-P2.P4** When the enemy has zero bots within capture range, the front rank SHALL advance at `conf.bot.speed * 0.5` (half speed) toward the payload line.

## Economy

- **R-P2.E1** The strategy SHALL hold every available extraction slot on `state.deposit_me`. If `state.deposit_me.extractors` shows fewer ally slots filled than the total slot count, the nearest idle bot SHALL be redirected to the deposit within 10 ticks.
- **R-P2.E2** Each mining Extractor SHALL park at `state.deposit_me.pos + Vec2(0.0, conf.deposit.radius + conf.bot.radius)` rotated to the appropriate angle for its slot index. The Extractor SHALL issue `SpecialAction.Extractor(mine=True)` every tick it holds a slot.
- **R-P2.E3** The strategy SHALL NOT divert any bot to attack enemy Extractors on `state.deposit_me`. Slots freed by enemy retreat are reclaimed automatically.
- **R-P2.E4** An offensive extractor dive on `state.deposit_other` SHALL be initiated only when: (a) fleet size ≥ 10, AND (b) `path_length(state.deposit_other.pos, nearest_friendly_healer.pos) <= 2 * conf.bot.blaster_range`. Otherwise the strategy SHALL ignore the enemy deposit.

## Combat

- **R-P2.C1** Target selection SHALL rank enemy Healers first, then enemy Extractors, then enemy Battles. Within each rank, the closest by `path_length` is selected.
- **R-P2.C2** All friendly Battles within `blaster_range` of the same enemy target SHALL fire on the same tick only if their blaster is off cooldown (i.e., `bot.next_fire_tick <= state.tick`). Coordinated firing is preferred but NOT required.
- **R-P2.C3** The strategy SHALL avoid self-splash: no friendly bot SHALL be within the splash radius of another friendly bot's firing arc when that bot fires.

## Endgame

- **R-P2.END1** `action.rush_order` SHALL be False once `state.tick >= conf.max_ticks - conf.endgame_ticks`.
- **R-P2.END2** On endgame entry, the strategy SHALL self-destruct any Extractor that is more than `conf.bot.blaster_range * 3` tiles from the payload, using lowest-HP-first ordering.
- **R-P2.END3** During endgame, the Healer pair + every surviving Battle SHALL be within capture range of the payload every tick.

## Compute

- **R-P2.COMP1** Healer pairing target assignment SHALL be recomputed only on the following events: a friendly Healer dies, an enemy Healer dies, or 10 ticks have elapsed since the last recompute.
- **R-P2.COMP2** Target focus-fire ranking SHALL be cached per Battle for 5 ticks and invalidated when enemy fleet size changes.
- **R-P2.COMP3** Fallback action when bank is depleted: each Healer heals the closest ally in arc/range; each Battle fires on its cached target if conditions allow; each Extractor holds its slot.

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

- `fabricator_next` per tick → R-P2.B*
- bot action per tick → R-P2.C*, R-P2.P*
- `rush_order` per tick → R-P2.B*, R-P2.END1
- position per bot per tick → R-P2.P*, R-P2.E*
- self-destruct events → R-P2.END2
- slot bitmask over time → R-P2.E1
- `get_budget().last_charge` distribution → R-P2.COMP*
