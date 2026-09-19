"""Gymnasium integration for MechMania.

The game itself is a Rust engine that runs a match and talks to each bot over a shared
memory mapping (`EngineChannel`). To make it look like a Gym env, this package:

  * spawns the engine as a subprocess and points one of its two bot slots at a small
    *relay bot* (`rl/relay_bot.py`) that opens the channel and forwards each
    `GameState` back to the parent over a Unix-domain socket;
  * encodes / decodes the wire types as numpy arrays, so an off-the-shelf RL algorithm
    can drive the match tick by tick;
  * computes a reward each tick from the state deltas (HP, captures, deposit ownership,
    final outcome).

`gym.register(...)` (in `rl/register.py`) hooks a single env id, `MechMania-v0`, with
defaults that pick `plan1_strategy` as the opponent. Pass kwargs to `gym.make(...)` to
swap the opponent, the action mode, or anything else.

Layout:

  rl/encoding.py    -- state <-> vector, action vector -> FleetAction
  rl/rewards.py     -- the per-tick reward shaping
  rl/relay_bot.py   -- the subprocess-side relay (also runnable as a bot binary)
  rl/env.py         -- the Gymnasium env (the only thing users normally touch)
  rl/register.py    -- `gym.register` plumbing
"""
