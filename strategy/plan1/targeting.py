"""Fleet-wide, cooldown-aware target assignment for battle bots.

A hit bot is invulnerable for a few ticks (`invulnerable_until_tick`), and
only the first hit on a target in a given tick counts at all, so the goal
here is to spread fire across the highest-priority *reachable* enemies
instead of letting every bot independently pick "nearest enemy" and dogpile
one target while others go untouched.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from .. import BotClass, BotState, GameState
from .combat_los import _has_clear_shot

# Higher priority target classes get hit first: healers keep the fleet topped up,
# extractors fund it, battle bots are the least valuable kill of the three.
_CLASS_PRIORITY = {BotClass.Healer: 3.0, BotClass.Extractor: 2.0, BotClass.Battle: 1.0}


def _target_priority(enemy: BotState, conf) -> float:
    weight = _CLASS_PRIORITY.get(enemy.class_, 1.0)
    low_hp_bonus = 50.0 if enemy.health <= conf.bot.health * 0.3 else 0.0
    return weight * 1000.0 - enemy.health + low_hp_bonus


def _assign_battle_targets(
    battles: List[BotState], state: GameState, conf
) -> Dict[int, Optional[int]]:
    """Fleet-wide, cooldown-aware target assignment.

    A hit bot is invulnerable for a few ticks (`invulnerable_until_tick`), and
    only the first hit on a target in a given tick counts at all -- so once a
    target is claimed by one of our bots this pass, every other bot should
    look for the next-best *different* target instead of dogpiling. This is a
    greedy assignment (sort every bot/target pair that's actually reachable by
    priority, then walk the list claiming one bot and one target per pair), not
    a globally optimal matching -- optimal isn't worth the compute for fleets
    this size, greedy already eliminates the redundant-fire case that matters.
    """
    if not battles:
        return {}
    if not state.fleet_other:
        return {b.id: None for b in battles}

    tick = state.tick
    blaster_range = conf.bot.blaster_range

    # (-priority, distance, bot_id, enemy_id) so a plain sort gives us
    # highest-priority-first, nearest-first among ties.
    candidates: List[Tuple[float, float, int, int]] = []
    for bot in battles:
        for enemy in state.fleet_other:
            d = bot.pos.dist(enemy.pos)
            if d > blaster_range:
                continue
            priority = _target_priority(enemy, conf)
            if not _has_clear_shot(bot.pos, enemy.pos, state, conf):
                # Still worth claiming over leaving a bot with no target at
                # all (`_apply_assignments` will steer it to flank the
                # obstacle), but a clear shot elsewhere should win the pick.
                priority -= 5_000.0
            if enemy.invulnerable_until_tick > tick:
                # Still a legal candidate (so an idle bot has somewhere to aim
                # while the window closes) but ranked behind anything hittable
                # right now.
                priority -= 10_000.0
            candidates.append((-priority, d, bot.id, enemy.id))

    candidates.sort()

    assigned: Dict[int, int] = {}
    claimed_enemies: Set[int] = set()
    for _, _, bot_id, enemy_id in candidates:
        if bot_id in assigned or enemy_id in claimed_enemies:
            continue
        assigned[bot_id] = enemy_id
        claimed_enemies.add(enemy_id)

    # Bots with no in-range/LOS candidate at all (or whose only candidates
    # were already claimed by someone else) still need a target to close
    # distance towards. Nearest by straight-line distance, not `path_length`
    # -- this loop is bots-without-a-target x every enemy, which in the
    # worst case (nobody in range yet, e.g. early game) is the full
    # bot-pair count, and `path_length` in an all-pairs loop is one of the
    # two most expensive things a strategy can do per tick. The wall-aware
    # route there is `navigate_to`'s job once this bot is actually moving,
    # not this pick's.
    for bot in battles:
        if bot.id in assigned:
            continue
        best = min(state.fleet_other, key=lambda e: bot.pos.dist_sq(e.pos), default=None)
        assigned[bot.id] = best.id if best else None

    return {b.id: assigned.get(b.id) for b in battles}
