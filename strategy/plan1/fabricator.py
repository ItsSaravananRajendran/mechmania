"""Build-order algorithm: what class the fabricator should queue next."""

from __future__ import annotations

from typing import Dict

from .. import BotClass, GameState


def _compute_fabricator_next(state: GameState, conf, cache: Dict) -> int:
    if not state.fleet_me.get(0):
        return int(BotClass.Extractor)

    if len(state.fleet_me) >= 10:
        if state.deposit_me.extractors.me < state.deposit_me.extractors.other:
            return int(BotClass.Extractor)
        return int(BotClass.Battle)

    extractor_alive = sum(1 for b in state.fleet_me if b.class_ == BotClass.Extractor)
    if state.tick <= 30 and extractor_alive < 3:
        return int(BotClass.Extractor)

    last_size = cache.get("last_fleet_size", 0)
    size = len(state.fleet_me)
    if size > last_size:
        cache["alt_count"] = cache.get("alt_count", 0) + 1
    cache["last_fleet_size"] = size

    alt_count = cache.get("alt_count", 0)
    return int(BotClass.Battle if alt_count % 2 == 0 else BotClass.Healer)
