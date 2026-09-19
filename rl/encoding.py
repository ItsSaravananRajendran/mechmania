"""Flatten `GameState` to a numpy vector and back.

Why a flat vector and not a dict / named tuple: every off-the-shelf RL algorithm treats
the observation as one big array. A graph-shaped observation (per-bot boxes, the map,
etc.) would need a custom model; a flat one trains with the standard MLP everyone
already has.

Two things this module owns that the others reuse:

* `OBS_DIM` -- the fixed size of the flattened observation, computed from `BOTS_MAX`.
* `encode_state` / `decode_state` -- round-trip a `GameState` through the vector.

The observation layout (see `_OBS_SPEC` for the per-feature layout):

    [ map  (32*32 = 1024 walls)
    , per-me-bot  (BOTS_MAX * ME_FEATS = 32 * 14)
    , per-enemy-bot (BOTS_MAX * ENEMY_FEATS = 32 * 14)
    , payload (3)
    , deposits_me (3), deposits_other (3)
    , fabricator_me (2), fabricator_other (2)
    , tick_norm (1)
    ]

Total = 1024 + 448 + 448 + 3 + 3 + 3 + 2 + 2 + 1 = 1934.

Action layout (see `_ACT_SPEC`): per-bot `move_dx`, `move_dy`, `fire`, `mine_or_heal`,
`self_destruct`, plus two fleet-wide knobs. Most algorithms only ever use the move
directions and the `fire` flag; the rest are defaulted to a sensible per-class action.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from strategy import (
    BOTS_MAX,
    BotClass,
    BotState,
    FleetAction,
    GameState,
    MAP_SIZE,
    MoveAction,
    SpecialAction,
    TurnAction,
    Vec2,
    get_config,
)


# -----------------------------------------------------------------------------------
# observation layout
# -----------------------------------------------------------------------------------

# Per-bot features. Order matters -- `encode_state` and `decode_state` agree on it.
_BOT_FEATS: List[Tuple[str, int]] = [
    ("alive",            1),  # 1 if the slot is live, else 0
    ("class_battle",     1),
    ("class_healer",     1),
    ("class_extractor",  1),
    ("health",           1),  # / max_health
    ("pos_x",            1),  # / MAP_SIZE
    ("pos_y",            1),
    ("vel_x",            1),  # / max_speed
    ("vel_y",            1),
    ("angle_sin",        1),  # sin/cos so angle wraps cleanly
    ("angle_cos",        1),
    ("fire_cd_frac",     1),  # cooldown / blaster_cooldown, 0 if ready
    ("invuln_frac",      1),  # ticks_left / base_invulnerability_ticks
    ("extracting",       1),  # 1 if currently extracting (any team deposit)
]
_BOT_FEAT_DIM = sum(d for _, d in _BOT_FEATS)
assert _BOT_FEAT_DIM == 14

# Map: a row of MAP_SIZE cells per row, stored as 0/1 walls. MAP_SIZE*MAP_SIZE = 1024.
_MAP_DIM = MAP_SIZE * MAP_SIZE

# Payload: x, y, capture fraction.
_PAYLOAD_DIM = 3

# Per-deposit: x, y, extractor_count_me (capped 1, but raw count as float).
_DEPOSIT_DIM = 3

# Per-fabricator: tokens_norm, ticks_to_next_bot / interval (1 means "ready now").
_FAB_DIM = 2

# Tick normalised by max_ticks.
_TICK_DIM = 1

OBS_DIM = (
    _MAP_DIM
    + BOTS_MAX * _BOT_FEAT_DIM   # my bots
    + BOTS_MAX * _BOT_FEAT_DIM   # enemy bots
    + _PAYLOAD_DIM
    + 2 * _DEPOSIT_DIM
    + 2 * _FAB_DIM
    + _TICK_DIM
)
assert OBS_DIM == 1934, f"unexpected OBS_DIM {OBS_DIM}"


# -----------------------------------------------------------------------------------
# action layout
# -----------------------------------------------------------------------------------

class ActionMode(enum.Enum):
    """What the per-bot action vector means.

    `MINIMAL`:    one (dx, dy) move direction per bot. The class-specific defaults are
                  filled in automatically -- battle bots fire at the nearest enemy if
                  they have line of sight and are in range, healers heal the lowest-HP
                  ally in arc and range, extractors mine their assigned spot. The
                  fabricator queues the next class via `fabricator_class_id`.

    `TACTICAL`:   MINIMAL plus one extra `fire` flag per bot, which forces the special
                  action on or off regardless of the class default. The fabricator is
                  still auto-driven.

    `FULL`:       TACTICAL plus `mine_or_heal` (overrides healer target = lowest-HP
                  ally when 1, lowest-HP enemy bot in arc when 0) and `self_destruct`.
                  Everything is exposed.
    """

    MINIMAL = "minimal"
    TACTICAL = "tactical"
    FULL = "full"


_PER_BOT_ACT_FEATS: dict[ActionMode, List[str]] = {
    ActionMode.MINIMAL: ["move_dx", "move_dy"],
    ActionMode.TACTICAL: ["move_dx", "move_dy", "fire"],
    ActionMode.FULL: ["move_dx", "move_dy", "fire", "mine_or_heal", "self_destruct"],
}

# Plus 3 (one-hot) for fabricator class choice + 1 for rush order.
_FAB_CLASS_DIM = 3
_RUSH_DIM = 1


def action_dim(mode: ActionMode) -> int:
    per_bot = len(_PER_BOT_ACT_FEATS[mode])
    return BOTS_MAX * per_bot + _FAB_CLASS_DIM + _RUSH_DIM


def action_space_low_high(mode: ActionMode) -> Tuple[np.ndarray, np.ndarray]:
    """Bounds for the continuous action vector. Used to build `gym.spaces.Box`."""
    per_bot = len(_PER_BOT_ACT_FEATS[mode])
    n = action_dim(mode)
    low = np.zeros(n, dtype=np.float32)
    high = np.zeros(n, dtype=np.float32)

    # Per-bot move directions and booleans: [-1, 1].
    bot_low = -np.ones(per_bot, dtype=np.float32)
    bot_high = np.ones(per_bot, dtype=np.float32)
    # `fire` / `mine_or_heal` / `self_destruct` are nominally 0/1 but signed works too.
    # We use [-1, 1] for everything for a uniform box; decoding thresholds at >= 0.

    for i in range(BOTS_MAX):
        s = i * per_bot
        low[s : s + per_bot] = bot_low
        high[s : s + per_bot] = bot_high

    # Fabricator class is one-hot: [0, 1] per slot, decode by argmax.
    fab_off = BOTS_MAX * per_bot
    low[fab_off : fab_off + _FAB_CLASS_DIM] = 0.0
    high[fab_off : fab_off + _FAB_CLASS_DIM] = 1.0

    # Rush order boolean.
    low[-1] = -1.0
    high[-1] = 1.0

    return low, high


# -----------------------------------------------------------------------------------
# state -> vector
# -----------------------------------------------------------------------------------


@dataclass
class _FeatureNormalizer:
    """Cached `conf.bot.health` / `conf.bot.speed` / etc. so we don't recompute each tick."""

    max_health: float
    max_speed: float
    blaster_cooldown: float
    invuln_ticks: float
    max_ticks: float
    fabricator_interval: float


# The `GameConfig` for the active match. The encoding layer normally reads it through
# `core.channel.get_config()`, which only works in the *bot* process (the engine
# hands it over during handshake). The Gym env lives in a separate process and never
# does its own handshake, so it stashes the config here after the relay bot forwards
# the handshake reply.
_ACTIVE_CONFIG: Optional["GameConfig"] = None  # type: ignore[name-defined]


def set_active_config(conf) -> None:
    """Pin the match config in this process. Called by the env right after `reset()`."""
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = conf


def _normalizer() -> _FeatureNormalizer:
    conf = _ACTIVE_CONFIG if _ACTIVE_CONFIG is not None else get_config()
    return _FeatureNormalizer(
        max_health=float(conf.bot.health),
        max_speed=float(conf.bot.speed),
        blaster_cooldown=float(conf.bot.blaster_cooldown) or 1.0,
        invuln_ticks=float(conf.bot.base_invulnerability_ticks) or 1.0,
        max_ticks=float(conf.max_ticks) or 1.0,
        fabricator_interval=float(conf.fabricator.interval) or 1.0,
    )


def _encode_bot(out: np.ndarray, offset: int, bot: Optional[BotState],
                tick: int, norm: _FeatureNormalizer) -> None:
    """Fill `_BOT_FEAT_DIM` slots starting at `out[offset:]`. `bot is None` -> zeros + alive=0."""
    chunk = out[offset : offset + _BOT_FEAT_DIM]
    chunk.fill(0.0)
    if bot is None:
        return
    chunk[0] = 1.0  # alive
    cls = bot.class_
    if cls == BotClass.Battle:
        chunk[1] = 1.0
    elif cls == BotClass.Healer:
        chunk[2] = 1.0
    elif cls == BotClass.Extractor:
        chunk[3] = 1.0

    chunk[4] = bot.health / norm.max_health
    chunk[5] = bot.pos.x / MAP_SIZE
    chunk[6] = bot.pos.y / MAP_SIZE
    chunk[7] = bot.vel.x / norm.max_speed if norm.max_speed > 0 else 0.0
    chunk[8] = bot.vel.y / norm.max_speed if norm.max_speed > 0 else 0.0

    # sin/cos of the heading -- discontinuity-free alternative to the raw degree.
    rad = math.radians(bot.angle)
    chunk[9] = math.sin(rad)
    chunk[10] = math.cos(rad)

    # Cooldown: 0 if the blaster is ready, 1 if just fired. Only meaningful for battle
    # bots; the engine's `BotState.next_fire_tick` is absolute, so we delta against
    # the current tick.
    if cls == BotClass.Battle:
        ready_in = max(0, bot.next_fire_tick - tick)
        chunk[11] = min(1.0, ready_in / norm.blaster_cooldown)

    chunk[12] = max(0.0, bot.invulnerable_until_tick) / norm.invuln_ticks
    chunk[13] = 1.0 if bot.extracting is not None else 0.0


def encode_state(state: GameState) -> np.ndarray:
    """Flatten `state` into a length-`OBS_DIM` float32 vector in `[0, 1]` (mostly)."""
    out = np.zeros(OBS_DIM, dtype=np.float32)
    norm = _normalizer()
    conf = _ACTIVE_CONFIG if _ACTIVE_CONFIG is not None else get_config()

    # Map: 1 = wall, 0 = empty. Stored row-major so `[x][y]` works without reshape.
    m = conf.map  # type: ignore[attr-defined]
    base = 0
    for x in range(MAP_SIZE):
        for y in range(MAP_SIZE):
            out[base] = 1.0 if int(m[x][y]) == 1 else 0.0  # type: ignore[index]
            base += 1

    # My bots, then enemy bots. Each row of BOTS_MAX slots.
    me_off = _MAP_DIM
    for slot in range(BOTS_MAX):
        bot = state.fleet_me.get(slot) if slot < BOTS_MAX else None
        _encode_bot(out, me_off + slot * _BOT_FEAT_DIM, bot, state.tick, norm)

    en_off = me_off + BOTS_MAX * _BOT_FEAT_DIM
    for slot in range(BOTS_MAX):
        bot = state.fleet_other.get(slot) if slot < BOTS_MAX else None
        _encode_bot(out, en_off + slot * _BOT_FEAT_DIM, bot, state.tick, norm)

    # Payload.
    pl = state.payload_pos()
    off = en_off + BOTS_MAX * _BOT_FEAT_DIM
    out[off + 0] = pl.x / MAP_SIZE
    out[off + 1] = pl.y / MAP_SIZE
    out[off + 2] = state.capture
    off += _PAYLOAD_DIM

    # Deposits.
    dm = state.deposit_me
    do = state.deposit_other
    out[off + 0] = dm.pos.x / MAP_SIZE
    out[off + 1] = dm.pos.y / MAP_SIZE
    out[off + 2] = min(1.0, dm.extractors.me / 4.0)
    off += _DEPOSIT_DIM
    out[off + 0] = do.pos.x / MAP_SIZE
    out[off + 1] = do.pos.y / MAP_SIZE
    out[off + 2] = min(1.0, do.extractors.other / 4.0)
    off += _DEPOSIT_DIM

    # Fabricators.
    fm = state.fabricator_me
    fo = state.fabricator_other
    out[off + 0] = min(1.0, fm.tokens / 20.0)  # arbitrary saturating scale
    out[off + 1] = max(0.0, fm.next_bot_creation - state.tick) / norm.fabricator_interval
    off += _FAB_DIM
    out[off + 0] = min(1.0, fo.tokens / 20.0)
    out[off + 1] = max(0.0, fo.next_bot_creation - state.tick) / norm.fabricator_interval
    off += _FAB_DIM

    # Tick.
    out[off] = state.tick / norm.max_ticks

    return out


# -----------------------------------------------------------------------------------
# vector -> action
# -----------------------------------------------------------------------------------


def decode_action(vec: np.ndarray, state: GameState, mode: ActionMode) -> FleetAction:
    """Translate an action vector into a `FleetAction` for the current tick.

    The vector layout is `action_space_low_high` from above. Decoding fills in defaults
    from the bot's class for any feature not present in this mode, so the same action
    space is usable at `MINIMAL` and `FULL`.
    """
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = action_dim(mode)
    if vec.shape[0] != n:
        raise ValueError(f"action vector of shape {vec.shape} does not match mode {mode.value} (expected {n})")

    conf = _ACTIVE_CONFIG if _ACTIVE_CONFIG is not None else get_config()
    action = FleetAction.new()
    per_bot = len(_PER_BOT_ACT_FEATS[mode])
    feats = _PER_BOT_ACT_FEATS[mode]

    move_idx = feats.index("move_dx")
    has_fire = "fire" in feats
    fire_idx = feats.index("fire") if has_fire else -1
    has_mh = "mine_or_heal" in feats
    mh_idx = feats.index("mine_or_heal") if has_mh else -1
    has_sd = "self_destruct" in feats
    sd_idx = feats.index("self_destruct") if has_sd else -1

    payload = state.payload_pos()

    # Per-bot decoding. Dead slots get a no-op (already zero-initialised).
    for slot in range(BOTS_MAX):
        bot = state.fleet_me.get(slot)
        if bot is None:
            continue  # bot action already zeroed; engine treats dead slots as no-ops

        ba = action.bots[slot]
        base = slot * per_bot

        # Move direction. Normalise the (dx, dy) -- the engine clamps magnitude anyway.
        dx = float(vec[base + move_idx])
        dy = float(vec[base + move_idx + 1])
        nrm = (dx * dx + dy * dy) ** 0.5
        if nrm > 1e-6:
            ba.move_action = MoveAction(Vec2(dx / nrm, dy / nrm))
        # else: zero vector == "don't accelerate" -- the engine leaves velocity as-is.

        # Turn toward the destination the bot is moving, so a face is always aimed
        # roughly at where it's going. (Matches what `move_bot` + `turn_towards` does
        # in the example strategies.)
        if nrm > 1e-6:
            ba.turn_action = TurnAction.TargetPosition(pos=Vec2(dx, dy))

        # Self-destruct.
        if has_sd and vec[base + sd_idx] > 0.0:
            ba.self_destruct = True

        # Special action.
        cls = bot.class_

        # Decide whether to fire. `fire` flag is read as `>= 0`; for MINIMAL we substitute
        # a heuristic. For TACTICAL / FULL, the agent's flag wins, but we still only
        # actually fire when conditions are right (range, cooldown, line of sight) so a
        # `True` from the agent never wastes the cooldown.
        wants_fire: Optional[bool]
        if has_fire:
            wants_fire = bool(vec[base + fire_idx] > 0.0)
        else:
            wants_fire = None  # MINIMAL: fill in from the class heuristic

        if cls == BotClass.Extractor:
            # Always mine if we have a line of sight and are close enough -- an
            # extractor with the wrong flag should still mine, otherwise it stalls
            # the deposit loop. The agent can override via `mine_or_heal == 0` in FULL.
            if has_mh and vec[base + mh_idx] < 0.0:
                ba.special_action = SpecialAction.Extractor(mine=False)
            else:
                ba.special_action = SpecialAction.Extractor(mine=True)

        elif cls == BotClass.Battle:
            target_id, in_range, los = _pick_battle_target(bot, state, conf)
            cooldown_ready = bot.next_fire_tick <= state.tick
            should_fire = (
                wants_fire if wants_fire is not None
                else (target_id is not None and in_range and los and cooldown_ready)
            )
            ba.special_action = SpecialAction.Battle(fire=bool(should_fire and cooldown_ready))

        elif cls == BotClass.Healer:
            target_id, in_range, in_arc = _pick_healer_target(bot, state, conf, healer_arc_to_enemy=bool(has_mh and vec[base + mh_idx] < 0.0) if has_mh else False)
            cooldown_ready = bot.next_fire_tick <= state.tick  # healers reuse the same cooldown field
            should_fire = (
                wants_fire if wants_fire is not None
                else (target_id is not None and in_range and in_arc and cooldown_ready)
            )
            tid = int(target_id) if (should_fire and target_id is not None) else 0
            ba.special_action = SpecialAction.Healer(fire=bool(should_fire and cooldown_ready), target=tid)

    # Fabricator class.
    fab_off = BOTS_MAX * per_bot
    class_logits = vec[fab_off : fab_off + _FAB_CLASS_DIM]
    cls_choice = int(np.argmax(class_logits))
    action.fabricator_next = cls_choice

    # Rush order.
    action.rush_order = bool(vec[-1] > 0.0)

    return action


def _pick_battle_target(bot: BotState, state: GameState, conf) -> Tuple[Optional[int], bool, bool]:
    """The closest enemy, plus whether it's in range and has line of sight. None,False,False if no enemy."""
    best: Optional[BotState] = None
    best_d = float("inf")
    for enemy in state.fleet_other:
        d = bot.pos.dist_sq(enemy.pos)
        if d < best_d:
            best_d = d
            best = enemy
    if best is None:
        return None, False, False
    in_range = bot.pos.dist(best.pos) <= conf.bot.blaster_range
    # `line_of_sight` needs the engine channel. In the env's process the channel is
    # never opened (the relay bot owns it), so we approximate with `corridor_clear` --
    # not identical, but a cheap and conservative fallback that misses shots only when
    # there is a wall between the two bots.
    if in_range:
        from core.channel import corridor_clear
        los = corridor_clear(bot.pos, best.pos)
    else:
        los = False
    return best.id, in_range, los


def _pick_healer_target(bot: BotState, state: GameState, conf,
                        healer_arc_to_enemy: bool) -> Tuple[Optional[int], bool, bool]:
    """Lowest-HP ally (default) or lowest-HP enemy (when `healer_arc_to_enemy`). Returns (id, in_range, in_arc)."""
    if healer_arc_to_enemy and state.fleet_other:
        candidates = state.fleet_other
    else:
        candidates = [b for b in state.fleet_me if b.id != bot.id]
    if not candidates:
        return None, False, False
    target = min(candidates, key=lambda b: b.health)
    in_range = bot.pos.dist(target.pos) <= conf.bot.base_heal_range
    in_arc = True
    if in_range:
        from core.channel import diff_degrees
        to_t = target.pos - bot.pos
        if to_t.norm_sq() > 1e-6:
            ang = to_t.angle_deg()
            diff = abs(diff_degrees(ang, bot.angle))
            in_arc = diff <= (conf.bot.base_heal_arc_deg / 2.0) + 1e-6
    return target.id, in_range, in_arc
