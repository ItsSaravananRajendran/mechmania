"""Miner protection and enemy-miner raiding.

Two jobs, both about the mining line rather than the payload line:

- A fixed-size **escort** of battle bots stays camped between our deposit and
  the enemy, so whichever enemy shows up to harass our extractors gets shot
  at instead of the extractors themselves.
- A fraction of whoever's left goes **raiding** -- hunting the enemy's own
  extractors (or their deposit, if none are in view) -- to put the same
  pressure on them.

Everyone else stays free for the payload/formation line (`formation.py`),
which is the point: only a fixed slice of the fleet is ever tied down here,
never "whatever's left after the payload".
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from .. import BotClass, BotState, GameState, Vec2
from .formation import _arrange_group, _dedupe_slots, _formation_scale, _min_safe_spacing, _resolve_slot
from .walls import WallGrid

# How many battle bots stay glued to the mining line, regardless of how big
# the rest of the fleet is -- matches `MAX_ACTIVE_MINERS` (formation.py):
# enough escorts to plausibly cover every active miner, not "the whole army".
MINER_ESCORT_COUNT = 3

# Roughly a third of whatever's left after the escort goes to raid the
# enemy's mining line -- see `_pick_raiders`.
RAID_FRACTION = 1.0 / 3.0


def _pick_miner_escorts(battles: List[BotState], count: int = MINER_ESCORT_COUNT) -> List[BotState]:
    """The first `count` battle bots, by id, assigned to guard the mining
    line. Stable across recomputes (id order never reshuffles on its own) so
    a bot built to replace a dead escort naturally inherits the same role
    instead of every escort's identity churning on every recompute."""
    return sorted(battles, key=lambda b: b.id)[:count]


def _pick_raiders(candidates: List[BotState], fraction: float = RAID_FRACTION) -> List[BotState]:
    """Roughly `fraction` of `candidates` (already excluding the escort and
    the dedicated payload defender), sent to hunt the enemy's miners. Stable
    by id for the same reason `_pick_miner_escorts` is. Rounds to the
    nearest whole bot, but always sends at least one once there are two or
    more candidates to spare -- with only one candidate, pulling it off
    leaves nothing to hold the payload/formation line, so it stays there."""
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda b: b.id)
    count = round(len(ordered) * fraction)
    if count == 0 and len(ordered) >= 2:
        count = 1
    return ordered[:count]


def _compute_guard_positions(
    escorts: List[BotState], state: GameState, forward: Vec2, conf, wall_grid: WallGrid
) -> Dict[int, Vec2]:
    """Line the escorts up on the enemy-facing side of our own deposit --
    the same side any attacker has to approach the miners from -- so an
    escort is the first thing an incoming enemy meets, not something
    standing behind the miners it's meant to protect."""
    if not escorts:
        return {}
    deposit_pos = state.deposit_me.pos
    perp = forward.rotate_deg(90.0)
    spacing, min_spacing, max_width, row_gap = _formation_scale(conf)
    guard_center = deposit_pos + forward * max(2.0 * conf.bot.radius, conf.bot.blaster_range * 0.3)
    slots, anchors = _arrange_group(
        wall_grid, guard_center, forward, perp, escorts, spacing, min_spacing, max_width, row_gap,
    )
    resolved = {bid: _resolve_slot(wall_grid, slot, anchors[bid]) for bid, slot in slots.items()}
    return _dedupe_slots(resolved, perp, spacing, min_distance=_min_safe_spacing(conf))


def _nearest_threat_to_deposit(state: GameState, conf) -> Optional[BotState]:
    """The nearest enemy within blaster range of our deposit -- what a
    miner escort should actually be shooting at, rather than whatever a
    fleet-wide target assignment (which doesn't know an escort is meant to
    stay put) happened to hand it."""
    deposit_pos = state.deposit_me.pos
    radius = conf.bot.blaster_range
    threats = [e for e in state.fleet_other if e.pos.dist(deposit_pos) <= radius]
    if not threats:
        return None
    return min(threats, key=lambda e: e.pos.dist(deposit_pos))


def _raid_target_pos(state: GameState, raider: BotState) -> Vec2:
    """Where a raider should head: its nearest visible enemy extractor, or
    the enemy deposit itself if none are currently in view -- extractors
    spawn from and cluster around it, so it's still the right place to go
    looking even with nothing visible there this exact tick."""
    extractors = [e for e in state.fleet_other if e.class_ == BotClass.Extractor]
    if extractors:
        return min(extractors, key=lambda e: raider.pos.dist(e.pos)).pos
    return state.deposit_other.pos
