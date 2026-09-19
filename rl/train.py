"""Train PPO on MechMania-v0.

Usage:

    python -m rl.train --opponent do_nothing --timesteps 20000 --max-ticks 100
    python -m rl.train --opponent plan1 --timesteps 20000 --max-ticks 200 --reward tactics

Saves checkpoints to `rl/runs/<run-name>/` along with a small JSON of hyperparameters
and the final mean reward. Resume with `--resume rl/runs/<run-name>/<step>.zip`.

Notes on the config:

* `--max-ticks` is the *training* cap; matches can end earlier if either side
  pushes the payload all the way. 100 keeps the early-training loop snappy.
* `--reward tactics` adds dense per-tick shaping (extracting, in-capture, blaster
  shots) on top of `default_reward`. Use `default` or `sparse` to ablate.
* The actor/critic MLP defaults match SB3's `MlpPolicy`; the input is 1934 floats
  so the default is fine for our action space of 100 floats.

This script is intentionally straightforward so it is easy to drop in alternative
algorithms (A2C, SAC, custom PPO) by swapping the model class.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np

# Idempotent.
from rl.register import register
from rl.rewards import default_reward, sparse_reward, tactics_reward
register()


_REWARD_FNS = {
    "default": default_reward,
    "sparse": sparse_reward,
    "tactics": tactics_reward,
}


def make_env(opponent: str, max_ticks: int, reward_name: str):
    reward_fn = _REWARD_FNS[reward_name]
    def _thunk():
        return gym.make(
            "MechMania-v0",
            opponent=opponent,
            max_ticks=max_ticks,
            reward_fn=reward_fn,
        )
    return _thunk


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--opponent", default="do_nothing", choices=["do_nothing", "basic", "plan1", "plan2", "plan3", "plan4", "plan5"])
    p.add_argument("--reward", default="tactics", choices=sorted(_REWARD_FNS))
    p.add_argument("--timesteps", type=int, default=20000)
    p.add_argument("--max-ticks", type=int, default=120)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", type=str, default="rl/runs")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--resume", type=str, default=None, help="path to a .zip saved by SB3")
    args = p.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback

    run_name = args.run_name or (
        f"{args.opponent}-{args.reward}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Persist the exact hyperparameters so a run can be reproduced / inspected later.
    config = {
        "opponent": args.opponent,
        "reward": args.reward,
        "max_ticks": args.max_ticks,
        "n_envs": args.n_envs,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "ent_coef": args.ent_coef,
        "seed": args.seed,
        "total_timesteps": args.timesteps,
    }
    (save_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"run dir: {save_dir}")
    print(f"config: {json.dumps(config, indent=2)}")

    if args.n_envs > 1:
        env_fns = [make_env(args.opponent, args.max_ticks, args.reward) for _ in range(args.n_envs)]
        env = DummyVecEnv(env_fns)
    else:
        env = gym.make(
            "MechMania-v0",
            opponent=args.opponent,
            max_ticks=args.max_ticks,
            reward_fn=_REWARD_FNS[args.reward],
        )

    if args.resume is not None:
        print(f"resuming from {args.resume}")
        model = PPO.load(args.resume, env=env, device="cpu")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            verbose=1,
            seed=args.seed,
            device="cpu",
        )

    ckpt_cb = CheckpointCallback(
        save_freq=max(args.n_steps * 4, 1000),
        save_path=str(save_dir),
        name_prefix="ckpt",
        save_replay_buffer=False,
    )

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=ckpt_cb, progress_bar=False)
    elapsed = time.time() - t0

    final_path = save_dir / "model.zip"
    model.save(str(final_path))
    env.close()

    summary = {
        "timesteps": args.timesteps,
        "elapsed_s": round(elapsed, 1),
        "throughput_steps_per_s": round(args.timesteps / elapsed, 2) if elapsed > 0 else None,
        "model_path": str(final_path),
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved final model to {final_path}")
    print(f"trained for {elapsed:.1f}s at {summary['throughput_steps_per_s']:.1f} steps/s")


if __name__ == "__main__":
    main()
