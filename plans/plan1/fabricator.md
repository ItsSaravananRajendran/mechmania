# `fabricator.py` — build-order

## Problem

Decide `fabricator_next` every tick: which class the next natural build (or
rush) should be. The rule changes across three phases of the match.

## `_compute_fabricator_next(state, conf, cache, friendly_born=frozenset()) -> int`

In order:

1. **Bootstrap.** If bot slot 0 is missing (`not state.fleet_me.get(0)`),
   build an Extractor. Both fleets' very first build is always an Extractor
   landing in slot 0 (the engine's own default); slot 0 missing later means
   it died and should be replaced before anything else.
2. **Late-game slot denial** (fleet size ≥ 10). `state.deposit_me.extractors`
   is a `TeamPair` of **per-bot-id bitmasks** (`1 << id`), not counts —
   comparing them directly as integers compares the wrong thing entirely
   (bit 2 set alone, `0b100` = 4, beats bits 0+1 set, `0b011` = 3, even
   though the enemy holds *more* slots). This counts the held slots with
   `.bit_count()` before comparing: if we hold fewer deposit slots than the
   enemy does on our own deposit, build an Extractor to contest it;
   otherwise Battle.
3. **Early extractor ramp** (`tick <= 30` and fewer than 3 extractors alive):
   build Extractor.
4. **Steady-state alternation.** Alternate Battle/Healer, tracked by
   `cache["alt_count"]`. Incremented only by **combat** births this tick
   (`friendly_born` filtered to non-Extractor classes) — not by net fleet
   size change. Net size delta misses a birth landing the same tick as a
   death (size holds steady) and over-counts a later replacement for one
   that already left (size looks like it "grew" again); it also isn't aware
   of *which* bot was born, so an Extractor rebuild (slot 0 dying and
   respawning, say) would incorrectly flip the Battle/Healer alternation if
   size alone were the signal. Counting actual combat births avoids both.

## Edge cases this handles

- **Bitmask vs. count**: see point 2 above — this was a real, confirmed bug
  (verified against the engine's Rust source, where `extractors[team] &=
  !(1u32 << id)` shows the field is unambiguously a bitmask) and is the
  reason this module counts bits instead of comparing raw integers.
- **Simultaneous death + rebuild**: a bot dying and a new one landing the
  same tick nets to zero size change but is still one real combat birth (or
  zero, if the new bot is an Extractor) — `friendly_born` is checked
  directly rather than inferred from the size delta.
- **Multiple births same tick** (a natural build landing the same tick as a
  rush order): `combat_born` counts all of them, incrementing `alt_count` by
  more than 1 in that case, so the alternation parity stays correct.
