# `rl/` — Gymnasium integration for MechMania

This package wraps a single MechMania match in a Gymnasium environment so an off-the-shelf
RL algorithm can drive the game tick by tick. It is built on top of the existing `core/` and
`strategy/` code: the agent still talks to the engine over a shared memory mapping; the
Gym env just owns the loop that calls into it.

## How a step works

```
MechManiaEnv (parent)                relay_bot.py (subprocess)        mm-engine
        |                                   |                            |
        |-- spawns engine subprocess ------>|-- spawned as bot A -----> |
        |                                   |                            |
        |-- binds Unix socket               |                            |
        |<-- relay_bot connects -------------|                            |
        |                                   |-- handshake ---->          |
        |<-- ("hello", team, config) -------|                            |
        |                                   |                            |
        |-- step(action)                    |                            |
        |   encode -> send action  -------> |-- decode -> respond  ---> |
        |<-- GameState ---------------------|<-- next tick <-----------|
        |   reward = f(prev, curr)          |                            |
        |                                   |                            |
        |-- ... (loop) ...                  |                            |
        |                                   |                            |
        |<-- DONE (match ended) ------------|                            |
```

Why a relay bot: the engine is a Rust binary that owns the simulation. The only extension
point it offers is "spawn a bot subprocess that does its own handshake over shared memory".
So we spawn a tiny Python bot (`rl/relay_bot.py`) for one of the two slots, and that bot
forwards ticks over a Unix-domain socket to the parent Gym env.

## Quick start

```python
import sys; sys.path.insert(0, ".")
import gymnasium as gym
from rl.register import register
from rl.encoding import ActionMode
from rl.rewards import sparse_reward

register()

env = gym.make(
    "MechMania-v0",
    opponent="plan1",           # built-in opponent: plan1..plan5, basic, do_nothing
    action_mode=ActionMode.TACTICAL,
    reward_fn=sparse_reward,    # or default_reward (a small dense shaping)
    max_ticks=200,              # hard cap; matches end earlier on a capture
)

obs, info = env.reset(seed=0)
for _ in range(200):
    action = env.action_space.sample()
    obs, r, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

A worked PPO example lives at `rl/train_ppo.py`:

```bash
python -m rl.train_ppo --opponent plan1 --timesteps 10000 --max-ticks 200
```

## Layout

| File | What it does |
| ---- | ------------ |
| `rl/env.py` | The Gymnasium env class `MechManiaEnv`. Spawns the engine, owns the relay socket, runs the reset/step loop. |
| `rl/encoding.py` | Flattens `GameState` to a length-1934 float vector, and translates an action vector back to a `FleetAction`. Defines `OBS_DIM`, `ActionMode`, `action_space_low_high`. |
| `rl/rewards.py` | `default_reward` (small dense shaping: payload + HP + extractor edge) and `sparse_reward` (`{+1, 0, -1}`). |
| `rl/relay_bot.py` | The subprocess-side relay: opens the engine channel, sends the config and each tick over the socket, and decodes incoming actions. |
| `rl/run_relay.sh` | Shell wrapper so `mm-engine` can spawn the relay bot as a binary. |
| `rl/run_opponent.sh` | Auto-generated on first run; spawns the opponent bot the same way `.mm/run` does. |
| `rl/register.py` | Registers `MechMania-v0` with Gymnasium. Auto-runs on import. |
| `rl/train_ppo.py` | A PPO example using `stable_baselines3`. |

## Action space

Three modes, all continuous over `[-1, 1]` for the move direction and `[-1, 1]` for the
boolean-ish flags (decode threshold is `>= 0`):

| Mode      | Per-bot vector                            | Fleet-wide vec | Total dim |
| --------- | ----------------------------------------- | -------------- | --------- |
| `MINIMAL` | `[move_dx, move_dy]`                      | 3 + 1          | 68        |
| `TACTICAL`| `[move_dx, move_dy, fire]`                | 3 + 1          | 100       |
| `FULL`    | `[move_dx, move_dy, fire, mine_or_heal, self_destruct]` | 3 + 1 | 164       |

- `move_dx, move_dy` are normalised to unit length by the decoder (zero means "don't
  accelerate"). The bot also turns to face the direction it is moving.
- `fire` is the agent's vote to use the class's special action (blaster / heal). The
  decoder still gates by range, line of sight, and cooldown so a True vote never wastes
  the cooldown on a wall.
- `mine_or_heal` (only in `FULL`) for healers: `>= 0` means "heal the lowest-HP ally in
  arc and range"; `< 0` means "target the lowest-HP enemy in arc" (a more aggressive
  build). Extractors with `mine_or_heal < 0` disable mining.
- `self_destruct` (only in `FULL`) triggers the bot's `self_destruct` flag.
- The fleet-wide vector is a 3-way one-hot fabricator class (argmax -> Battle / Healer /
  Extractor) plus a `rush_order` boolean.

The action decoder fills in sensible defaults for any feature that is not present in the
mode, so the same observation always works under `MINIMAL` or `FULL`.

## Observation space

`OBS_DIM = 1934`, all in roughly `[0, 1]` (a few cells can go slightly negative, hence
the `Box(low=-1.5, high=1.5, ...)`):

| Section                  | Dim  |
| ------------------------ | ---- |
| Map (32x32 wall tiles)   | 1024 |
| My bots (32 * 14 feats)  | 448  |
| Enemy bots (32 * 14)     | 448  |
| Payload (x, y, capture)  | 3    |
| My deposit (3) + theirs (3) | 6 |
| My fabricator (2) + theirs (2) | 4 |
| Tick (normalised)        | 1    |

Per-bot features: alive, class one-hot, health, pos (x, y), vel (x, y), heading as
(sin, cos), blaster cooldown fraction, invulnerability fraction, extracting flag.

## Reward

`default_reward` (per tick, dense):

```
+ 0.02 * (capture - 0.5)            # payload progress; midpoint is neutral
+ 0.005 * (my_hp - enemy_hp) / max_hp
+ 0.002 * (my_extractors - their_extractors)
```

Plus `+/- 1` at the terminal tick (or `0` for a tie at `max_ticks`). `sparse_reward`
zeros out the per-tick signal.

## Opponents

The opponent is one of the bundled strategies. They are spawned as a separate bot
process and run unmodified:

| Name         | Where it lives |
| ------------ | -------------- |
| `do_nothing` | `strategy/main.py` |
| `basic`      | `strategy/main.py` |
| `plan1`..`plan5` | `strategy/plan*.py` (plan1 is the most aggressive; plan5 the most cautious) |

`plan1` is the default. Pick `do_nothing` if you want the env to be a toy for testing
the wrapper; pick `plan1` to actually train against something competent.

## Performance notes

- Each env spawns its own `mm-engine` subprocess and its own relay bot. On a laptop
  expect ~50-100 env steps per second for a single env with `max_ticks=200`.
- Vectorising with `--n-envs N` does *not* linearly speed up training, because the
  parent has to round-trip a match's worth of ticks sequentially through the socket.
  Two envs in parallel runs at roughly the same throughput as one. To scale up you
  want a per-process actor (each worker owns its own env), not a sync vector env.
- Use the engine's `--no-time-limit` flag locally if you want the agent to be allowed
  to be slow; the default per-tick budget is what a real match enforces.

## Caveats

- The env treats `capture == 1.0` as a win for "my team" (the agent's side after the
  engine's mirroring), `capture == 0.0` as a loss, and any other terminal (max_ticks)
  as a tie broken by total HP.
- Because the match is deterministic from the same engine seed but stochastic from the
  agent's perspective, two `reset(seed=42)` calls still produce different observations:
  the engine's seed comes from inside its binary. Don't rely on determinism across
  `reset()`s unless you also control the engine invocation.
