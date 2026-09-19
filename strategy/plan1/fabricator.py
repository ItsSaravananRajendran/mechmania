"""Build-order algorithm: what class the fabricator should queue next."""

from __future__ import annotations

from typing import Dict, FrozenSet

from .. import BotClass, GameState
from .formation import MAX_ACTIVE_MINERS

# Minimum number of Healers to keep alive at all times once the opening is over.
# The spec (`plans/plan1.md` R-P1.B3, healer_roles.GOAL_HEALER_COUNT = 3) calls
# for 3 dedicated healers on the goal line plus support healers for the miner
# escort and raid group. Top tournament teams (JaniceKeepTalking, clankerbot,
# Gang) routinely run 4-9 healers alive throughout the match. Set the floor
# above the goal-line minimum so the alternation rule below can keep adding
# healers naturally past the floor instead of pinning us at exactly 3 -- the
# goal/escort/raid split in `healer_roles.py` only fills all three groups
# (3+1+1) when there are 5+ healers available.
MIN_HEALERS_ALIVE = 5


def _compute_fabricator_next(state: GameState, conf, cache: Dict,
                              friendly_born: FrozenSet[int] = frozenset()) -> int:
    if not state.fleet_me.get(0):
        return int(BotClass.Extractor)

    # Keep a usable mining crew topped up at all times, not just during the
    # opening ramp -- a miner lost to a raid at tick 3000 needs replacing
    # just as much as one that was never built, or mining (and the fleet's
    # whole economy) quietly stalls out for the rest of the match.
    extractor_alive = sum(1 for b in state.fleet_me if b.class_ == BotClass.Extractor)
    if extractor_alive < MAX_ACTIVE_MINERS:
        return int(BotClass.Extractor)

    # Healer replacement: top up to MIN_HEALERS_ALIVE whenever a healer has
    # died. This is checked before the alternation rule and the
    # slot-denial/extractors branch so a healer loss in the middle of the match
    # is replaced before any other build decision is considered. The alternation
    # rule below still controls the Battle vs Healer choice on normal ticks;
    # this branch only fires when a healer is genuinely missing.
    healer_alive = sum(1 for b in state.fleet_me if b.class_ == BotClass.Healer)
    if healer_alive < MIN_HEALERS_ALIVE:
        return int(BotClass.Healer)

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
