"""Build-order algorithm: what class the fabricator should queue next."""

from __future__ import annotations

from typing import Dict, FrozenSet

from .. import BotClass, GameState


def _compute_fabricator_next(state: GameState, conf, cache: Dict,
                              friendly_born: FrozenSet[int] = frozenset()) -> int:
    if not state.fleet_me.get(0):
        return int(BotClass.Extractor)

    if len(state.fleet_me) >= 10:
        # `extractors.me`/`.other` are per-bot-id bitmasks (`1 << id`), not
        # counts -- comparing them as plain integers compares the wrong
        # thing entirely (e.g. bit 2 set beats bits 0+1 set, 4 > 3, even
        # though the enemy holds more slots). Count the held slots instead.
        our_slots = state.deposit_me.extractors.me.bit_count()
        their_slots = state.deposit_me.extractors.other.bit_count()
        if our_slots < their_slots:
            return int(BotClass.Extractor)
        return int(BotClass.Battle)

    extractor_alive = sum(1 for b in state.fleet_me if b.class_ == BotClass.Extractor)
    if state.tick <= 30 and extractor_alive < 3:
        return int(BotClass.Extractor)

    # Net fleet-size growth misses a birth that lands the same tick as a
    # death (size holds steady) and over-counts a later replacement for one
    # that already left (size looks like it "grew" again) -- count actual
    # births instead of inferring them from the size delta. Only combat
    # (Battle/Healer) births should toggle the alternation: a rush or a
    # natural build landing the same tick as another counts as two, and an
    # Extractor rebuild (e.g. slot 0 dying and respawning) isn't part of the
    # Battle/Healer alternation at all and must not flip it.
    combat_born = sum(
        1 for bid in friendly_born
        if state.fleet_me.get(bid) and state.fleet_me.get(bid).class_ != BotClass.Extractor
    )
    if combat_born:
        cache["alt_count"] = cache.get("alt_count", 0) + combat_born

    alt_count = cache.get("alt_count", 0)
    return int(BotClass.Battle if alt_count % 2 == 0 else BotClass.Healer)
