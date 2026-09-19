"""Heal-coverage positioning: where a squad's healer should stand so it
keeps as much of that squad within heal range as possible, instead of
beelining for whichever single ally it happens to be actively healing this
tick and leaving the rest of the squad to fend for itself until they're
the lowest-health one.
"""

from __future__ import annotations

from typing import List, Tuple

from .. import BotState, Vec2
from .formation import _min_safe_spacing

# Per squad, this many attackers are allowed to sit outside heal range --
# not a hard guarantee (a big enough squad sharing one healer can't always
# be fully covered), but the target every healer's position is chosen
# against, so coverage degrades gracefully rather than by accident.
MAX_ATTACKERS_OUT_OF_HEAL_RANGE = 2


def _out_of_range_count(anchor: Vec2, allies: List[BotState], heal_range: float) -> int:
    """How many `allies` would be farther than `heal_range` from `anchor`."""
    return sum(1 for a in allies if anchor.dist(a.pos) > heal_range)


def _best_heal_anchor(allies: List[BotState], heal_range: float) -> Tuple[Vec2, int]:
    """The point -- among the squad's own current positions -- that leaves
    the fewest `allies` outside `heal_range`, and how many that still is.

    Every real formation slot is itself a legal candidate for "stand here",
    so trying each ally's own position and keeping whichever covers the
    most of the rest is a simple, exact solution for the squad sizes this
    strategy ever fields (a handful of bots) -- no need for a real minimum
    enclosing-circle solver. Ties keep whichever candidate was checked
    first, so callers that care about a particular tie-break (favoring a
    rearward bot, say) should pass `allies` pre-sorted that way."""
    best_anchor = allies[0].pos
    best_out = _out_of_range_count(best_anchor, allies, heal_range)
    for ally in allies[1:]:
        out = _out_of_range_count(ally.pos, allies, heal_range)
        if out < best_out:
            best_out = out
            best_anchor = ally.pos
    return best_anchor, best_out


def _heal_position(allies: List[BotState], forward: Vec2, conf) -> Vec2:
    """Where a squad's healer should stand: the `_best_heal_anchor` for
    `allies`, nudged a little further back (`-forward`) -- the same
    "healer stays out of the front line, past `_min_safe_spacing`"
    reasoning used everywhere else a healer is positioned -- as long as
    that step doesn't push the squad's out-of-range count past whichever
    is looser, its own best-achievable count or `MAX_ATTACKERS_OUT_OF_HEAL_RANGE`.
    A squad small enough (or clustered enough) to already fit entirely
    within heal range keeps fitting after the nudge; a squad too spread out
    for full coverage doesn't get made any worse by it."""
    heal_range = conf.bot.base_heal_range
    anchor, out_count = _best_heal_anchor(allies, heal_range)
    standoff = _min_safe_spacing(conf)
    nudged = anchor - forward * standoff
    budget = max(out_count, MAX_ATTACKERS_OUT_OF_HEAL_RANGE)
    if _out_of_range_count(nudged, allies, heal_range) <= budget:
        return nudged
    return anchor
