"""Endgame algorithm: once the fabricator goes dead and a wipe is instant
loss, decide whether to sacrifice a bot to shore up the health tiebreaker.
"""

from __future__ import annotations

from typing import Dict

from .. import BotClass, FleetAction, GameState


def _maybe_endgame_self_destruct(state: GameState, conf, action: FleetAction,
                                  cache: Dict) -> None:
    my_hp = sum(b.health for b in state.fleet_me)
    enemy_hp = sum(e.health for e in state.fleet_other)

    if enemy_hp > 0 and my_hp < enemy_hp * 0.75:
        extractors = [b for b in state.fleet_me if b.class_ == BotClass.Extractor]
        if extractors:
            weakest = min(extractors, key=lambda b: b.health)
            action.bots[weakest.id].self_destruct = True
            cache["endgame_sd_done"] = True
