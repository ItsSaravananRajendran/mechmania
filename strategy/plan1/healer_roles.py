"""Healer role allocation: a fixed split of the healer roster across the
goal/payload line, the mining escort, and the raid group -- mirroring the
battle-bot split in `mining_defense.py` (`_pick_miner_escorts`/
`_pick_raiders`) so every squad gets its own dedicated support instead of
every healer independently drifting to whoever's nearest and hurt.
"""

from __future__ import annotations

from typing import List, Tuple

from .. import BotState

# 3 healers hold the goal/payload line, 1 rides with the mining escort, 1
# rides with the raid group -- protecting the win condition outranks either
# economy job, so goal is filled first.
GOAL_HEALER_COUNT = 3
MINER_HEALER_COUNT = 1
RAID_HEALER_COUNT = 1


def _pick_healer_groups(
    healers: List[BotState],
) -> Tuple[List[BotState], List[BotState], List[BotState]]:
    """Split `healers` by id into `(goal, miner, raid)` groups, goal filled
    first, then the miner escort, then the raid group. Stable by id like
    every other role split in this strategy (`_pick_miner_escorts`,
    `_pick_raiders`), so a newly built healer naturally backfills whichever
    group the id ordering says is short, rather than the whole roster
    reshuffling on every recompute.

    Degrades gracefully with a small roster: with only 2 healers, both go
    to goal and neither the miner nor raid group gets one at all -- holding
    the goal line outranks either support job, so a short roster shouldn't
    leave it short-handed to staff a lower-priority one."""
    ordered = sorted(healers, key=lambda h: h.id)
    goal = ordered[:GOAL_HEALER_COUNT]
    rest = ordered[GOAL_HEALER_COUNT:]
    miner = rest[:MINER_HEALER_COUNT]
    rest = rest[MINER_HEALER_COUNT:]
    raid = rest[:RAID_HEALER_COUNT]
    return goal, miner, raid
