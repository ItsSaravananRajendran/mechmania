"""Minimal PPO training run on the MechMania Gym env.

Usage:

    python -m rl.train_ppo --opponent plan1 --timesteps 10000 --max-ticks 200

What this is for: confirming that off-the-shelf RL libraries can drive the env without
any extra glue. Expect very poor policy quality at low timestep counts -- the match is
a long-horizon game and PPO with the default MLP needs many millions of steps to learn
anything sensible. The script is here so you can plug in your own algorithm, not as a
demonstration that random init can win.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym

# Register `MechMania-v0` with gymnasium. Idempotent.
from rl.register import register
register()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--opponent", default="plan1", choices=["plan1", "plan2", "plan3", "plan4", "plan5", "basic", "do_nothing"])
    p.add_argument("--timesteps", type=int, default=2048)
    p.add_argument("--max-ticks", type=int, default=200)
    p.add_argument("--frame-skip", type=int, default=1,
                    help="repeat each action for N real engine ticks (see rl/env.py); "
                         "reduces decision count without changing real game time")
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--save-path", type=str, default="rl/saved_models/ppo_mechmania")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    args = p.parse_args()

    # Lazy import so the env module imports cleanly even without sb3 installed.
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    # We construct a single env first (needed for the spaces), then wrap in a
    # DummyVecEnv so SB3's `learn()` can batch. SB3's vector env wrapper resets each
    # sub-env in parallel; that's fine -- each sub-env spawns its own engine process.
    def _make():
        return gym.make(
            "MechMania-v0",
            opponent=args.opponent,
            max_ticks=args.max_ticks,
            frame_skip=args.frame_skip,
        )

    if args.n_envs > 1:
        env = DummyVecEnv([_make for _ in range(args.n_envs)])
    else:
        env = _make()

    print(f"action_space: {env.action_space}")
    print(f"observation_space: {env.observation_space}")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=64,
        batch_size=32,
        verbose=1,
        device="cpu",
    )

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps)
    elapsed = time.time() - t0

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"saved to {save_path}")
    print(f"trained for {elapsed:.1f}s on {args.timesteps} env steps ({args.timesteps / elapsed:.1f} steps/s)")

    env.close()


if __name__ == "__main__":
    main()
