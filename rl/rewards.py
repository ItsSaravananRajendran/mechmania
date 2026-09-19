"""Per-tick reward shaping.

The match ends one of three ways: my team holds the payload (`capture == 1.0`), the
opponent does (`capture == 0.0`), or `tick >= max_ticks`. The terminal reward is
`+1` for a win, `-1` for a loss, `0` for a tie.

`reward_fn` is the single knob you can turn to change the agent's incentives. Two
ready-made variants:

* `default_reward` -- a small dense signal dominated by *who is winning right now*.
* `tactics_reward` -- adds dense per-action signals (extracting, blaster shots,
  captures, heal use) on top of `default_reward`. Use this when the agent cannot
  learn the basic mechanics from the slow payoff alone.

The signature is `(prev: GameState, curr: GameState, done: bool, winner: Optional[int]) -> float`.
"""

from __future__ import annotations

from typing import Callable, Optional

from strategy import BotClass, GameState, Vec2, get_config


RewardFn = Callable[[GameState, GameState, bool, Optional[int]], float]


def _conf():
    """The match config for the env-side process. Falls back to `get_config()` for
    in-process callers (e.g., bots that use these reward functions directly).

    `rl.encoding._ACTIVE_CONFIG` is looked up dynamically rather than imported --
    `from rl.encoding import _ACTIVE_CONFIG` would capture the value at import time,
    which is `None` before the first `set_active_config(...)`, and the binding would
    never refresh.
    """
    import rl.encoding as _enc
    return _enc._ACTIVE_CONFIG if _enc._ACTIVE_CONFIG is not None else get_config()


# -----------------------------------------------------------------------------------
# shared helpers
# -----------------------------------------------------------------------------------


def _terminal(winner: Optional[int]) -> float:
    if winner == 0:
        return 1.0
    if winner == 1:
        return -1.0
    return 0.0


def _bots_by_class(fleet) -> dict:
    out = {BotClass.Battle: [], BotClass.Healer: [], BotClass.Extractor: []}
    for b in fleet:
        out[b.class_].append(b)
    return out


# -----------------------------------------------------------------------------------
# default (coarse)
# -----------------------------------------------------------------------------------


def default_reward(prev: GameState, curr: GameState, done: bool, winner: Optional[int]) -> float:
    """A small dense reward for the agent's side.

    Components (per tick, additive):
        + 0.02 * curr.capture                  -- payload progress in `[-1, +1]`; 0 is neutral
        + 0.005 * (my_hp - enemy_hp) / max_hp
        + 0.002 * (my_extractors - their_extractors)

    Plus `+/- 1` at the terminal tick (or `0` for a tie at `max_ticks`).
    """
    if done:
        return _terminal(winner)

    conf = _conf()

    r = 0.02 * curr.capture

    my_hp = sum(b.health for b in curr.fleet_me)
    en_hp = sum(b.health for b in curr.fleet_other)
    max_hp = max(1.0, float(conf.bot.health))
    r += 0.005 * (my_hp - en_hp) / max_hp

    my_ext = curr.deposit_me.extractors.me
    en_ext = curr.deposit_me.extractors.other  # enemy's count on OUR deposit
    r += 0.002 * (my_ext - en_ext)

    return float(r)


def sparse_reward(prev: GameState, curr: GameState, done: bool, winner: Optional[int]) -> float:
    """`{+1, 0, -1}` at the terminal tick, `0` everywhere else."""
    if not done:
        return 0.0
    return _terminal(winner)


# -----------------------------------------------------------------------------------
# tactics (dense -- the one we recommend for first training runs)
# -----------------------------------------------------------------------------------


def tactics_reward(prev: GameState, curr: GameState, done: bool, winner: Optional[int]) -> float:
    """`default_reward` plus a layer of per-action signals the agent needs to learn.

    Components added on top of `default_reward` (all additive per tick):

        + 0.02 * (prev_mean_distance(my_bots, payload) - curr_mean_distance(...))
            -- *progress* toward the payload, not a static "how far are you" penalty.
               Rewards closing the gap each tick; telescopes to a constant total for
               closing a fixed distance regardless of how many ticks it takes (unlike
               a static per-tick distance penalty, which punishes slow travel on top
               of "not there yet"). At the real match length (`conf.max_ticks`, ~9000
               -- see `rl/IMPROVEMENT_PLAN.md`) a ~20-unit approach nets roughly
               0.02 * 20 = 0.4 total, spread densely across however many ticks it
               actually takes.
        + 0.001 * (my_in_capture_now - their_in_capture_now)
            -- reward every bot currently inside the payload capture radius
        + 0.0005 * (my_extracting_now - their_extracting_now)
            -- reward every tick an extractor is actively pulling from a deposit
        + 0.005 * (prev_mean_distance(my_extractors, my_deposit) - curr_mean_distance(...))
            -- progress pulling extractors toward our own deposit (same delta-based
               shape as the payload term above, scaled down since deposit distances
               are typically much smaller)
        + 0.0002 * (my_blasters_fired_now - their_blasters_fired_now)
            -- credit each blaster shot that landed (we detect via the `shot`
               field on `Battle` special state)
        + 0.0002 * (my_healers_healing_now - their_healers_healing_now)
            -- credit each tick a healer is actively landing a heal (via the
               `healing` field, which the engine only sets when there's a valid
               target in range/arc)
        + 0.1  * (curr.capture - prev.capture)
            -- the *delta* in payload capture, a much denser signal than the
               raw level

    The shaping is small enough that the terminal +/-1 still dominates, so the
    agent cannot game the per-tick signal at the expense of the win.
    """
    if done:
        return _terminal(winner)

    conf = _conf()
    r = default_reward(prev, curr, False, None)

    payload = curr.payload_pos()
    capture_r = conf.payload.capture_radius

    # Distance *progress* -- the dominant shaping signal for "go to the payload".
    # Delta-based (see docstring) rather than a static per-tick distance penalty.
    my_bots = list(curr.fleet_me)
    prev_bots = list(prev.fleet_me)
    if my_bots and prev_bots:
        prev_mean_dist = sum(b.pos.dist(prev.payload_pos()) for b in prev_bots) / len(prev_bots)
        curr_mean_dist = sum(b.pos.dist(payload) for b in my_bots) / len(my_bots)
        r += 0.02 * (prev_mean_dist - curr_mean_dist)

    # Capture-radius presence.
    my_in_capture = sum(1 for b in my_bots if b.pos.dist(payload) <= capture_r)
    en_in_capture = sum(1 for b in curr.fleet_other if b.pos.dist(payload) <= capture_r)
    r += 0.001 * (my_in_capture - en_in_capture)

    # Mining.
    my_extracting = sum(1 for b in my_bots if b.extracting is not None)
    en_extracting = sum(1 for b in curr.fleet_other if b.extracting is not None)
    r += 0.0005 * (my_extracting - en_extracting)

    # Progress pulling extractors toward our own deposit (delta-based, same shape
    # as the payload term above).
    my_deposit = curr.deposit_me.pos
    my_extractors = [b for b in my_bots if b.class_ == BotClass.Extractor]
    prev_extractors = [b for b in prev_bots if b.class_ == BotClass.Extractor]
    if my_extractors and prev_extractors:
        prev_deposit = prev.deposit_me.pos
        prev_mean_d = sum(b.pos.dist(prev_deposit) for b in prev_extractors) / len(prev_extractors)
        curr_mean_d = sum(b.pos.dist(my_deposit) for b in my_extractors) / len(my_extractors)
        r += 0.005 * (prev_mean_d - curr_mean_d)

    # Blaster fire count.
    my_shots = sum(1 for b in my_bots if b.shot is not None)
    en_shots = sum(1 for b in curr.fleet_other if b.shot is not None)
    r += 0.0002 * (my_shots - en_shots)

    # Heal usage -- credit each tick a healer is actively healing a *damaged* ally
    # (a healer parked on a full-health bot with `healing` set still shouldn't earn
    # this; the engine only sets `healing` when there's a valid target in range/arc,
    # so this only fires when a heal is actually landing).
    my_healing = sum(1 for b in my_bots if b.class_ == BotClass.Healer and b.healing is not None)
    en_healing = sum(1 for b in curr.fleet_other if b.class_ == BotClass.Healer and b.healing is not None)
    r += 0.0002 * (my_healing - en_healing)

    # Capture delta.
    r += 0.1 * (curr.capture - prev.capture)

    return float(r)


# -----------------------------------------------------------------------------------
# shaping decay -- anneal a dense reward toward sparse over a training run
# -----------------------------------------------------------------------------------


class ShapingSchedule:
    """Mutable, shared progress tracker for reward annealing.

    `value` ranges from `1.0` (full shaping) down to `0.0` (pure terminal ±1, i.e.
    `sparse_reward`). A trainer updates `.value` from a callback (e.g. once per SB3
    `_on_step`, based on `num_timesteps / total_timesteps`); `make_annealed_reward`
    reads it on every call. It's a plain mutable object rather than a plain float so
    the reward closure and the training callback can share the same instance without
    any extra plumbing through `MechManiaEnv` (whose `reward_fn` signature is fixed).
    """

    __slots__ = ("value",)

    def __init__(self, value: float = 1.0) -> None:
        self.value = value

    def set_linear(self, step: int, total_steps: int) -> None:
        """Linearly decay from 1.0 at `step == 0` to 0.0 at `step >= total_steps`."""
        if total_steps <= 0:
            self.value = 0.0
            return
        self.value = max(0.0, 1.0 - step / total_steps)


def make_annealed_reward(schedule: ShapingSchedule, base_fn: RewardFn = tactics_reward) -> RewardFn:
    """Wrap `base_fn` so its non-terminal shaping is scaled by `schedule.value`.

    The terminal ±1 (returned by `base_fn` when `done`) is never scaled -- only the
    dense per-tick shaping is annealed, so training starts with `base_fn`'s full
    signal and ends equivalent to `sparse_reward`, forcing the agent to optimize the
    actual win condition rather than the shaping proxy (e.g. farming heals/mining
    without ever pushing the payload).
    """

    def _reward(prev: GameState, curr: GameState, done: bool, winner: Optional[int]) -> float:
        if done:
            return _terminal(winner)
        return float(base_fn(prev, curr, False, None) * schedule.value)

    return _reward
