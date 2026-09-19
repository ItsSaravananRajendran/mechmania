"""Evaluate a trained PPO model against one or more opponents.

Usage:

    python -m rl.evaluate --model rl/runs/<run>/model.zip --episodes 20
    python -m rl.evaluate --model rl/runs/<run>/model.zip --opponents plan1 plan2 plan3

    # Round-robin every checkpoint in a directory against plan1..plan5, to see a real
    # skill curve over training instead of one checkpoint's snapshot stats:
    python -m rl.evaluate --checkpoint-dir rl/runs/<run>/snapshots/side_0 --episodes 10

Reports per-opponent win rate, average episode length, average final capture, and the
fraction of matches that ended on a payload capture (vs max_ticks tie). The `info`
dict from each episode is dumped to JSON so you can drill into individual matches.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np

from rl.register import register
register()


OPPONENTS = ["do_nothing", "basic", "plan1", "plan2", "plan3", "plan4", "plan5"]


def evaluate_one(model, env_id: str, opponent: str, max_ticks: int, episodes: int, deterministic: bool, seed: int, frame_skip: int = 1) -> dict:
    """Run `episodes` matches against `opponent`. Return aggregated stats."""
    env = gym.make(env_id, opponent=opponent, max_ticks=max_ticks, frame_skip=frame_skip)

    wins = 0
    losses = 0
    ties = 0
    capture_ends = 0  # matches that ended before max_ticks (one side captured)
    lengths = []
    final_captures = []
    final_my_hp = []
    final_en_hp = []

    rng = np.random.default_rng(seed)
    for i in range(episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        ep_len = 0
        ep_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, terminated, truncated, info = env.step(action)
            ep_reward += r
            ep_len += 1
            done = terminated or truncated
        final_capture = float(info.get("capture", env.unwrapped._state.capture if env.unwrapped._state else 0.5))
        winner = info.get("winner", None)
        # `ep_len` counts *decisions* (env.step() calls), not real engine ticks, once
        # `frame_skip > 1` -- compare the real tick count from `info` against
        # `max_ticks` (also in real ticks) instead, or every frame-skipped run would
        # look like it ended in an (incorrect) early capture.
        final_tick = int(info.get("tick", max_ticks))

        if winner == 0:
            wins += 1
        elif winner == 1:
            losses += 1
        else:
            ties += 1
        if final_tick < max_ticks - 1:
            capture_ends += 1
        lengths.append(ep_len)
        final_captures.append(final_capture)
        try:
            st = env.unwrapped._state
            if st is not None:
                final_my_hp.append(sum(b.health for b in st.fleet_me))
                final_en_hp.append(sum(b.health for b in st.fleet_other))
        except Exception:
            pass

    env.close()

    return {
        "opponent": opponent,
        "episodes": episodes,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / episodes if episodes else 0.0,
        "capture_end_rate": capture_ends / episodes if episodes else 0.0,
        "mean_episode_length": float(np.mean(lengths)) if lengths else 0.0,
        "mean_final_capture": float(np.mean(final_captures)) if final_captures else 0.5,
        "mean_my_hp_end": float(np.mean(final_my_hp)) if final_my_hp else 0.0,
        "mean_en_hp_end": float(np.mean(final_en_hp)) if final_en_hp else 0.0,
    }


def _checkpoint_step(path: Path) -> int:
    """`ckpt_00012000.zip` -> `12000`. Falls back to 0 for non-conforming names (e.g. the
    old-style `model_0.zip`) so they still sort/print, just at the front."""
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return 0


def round_robin(
    checkpoint_dir: Path,
    opponents: list,
    max_ticks: int,
    episodes: int,
    deterministic: bool,
    seed: int,
    frame_skip: int = 1,
) -> dict:
    """Evaluate every checkpoint in `checkpoint_dir` against every opponent.

    Returns `{"checkpoints": [...], "opponents": [...], "win_rate": [[...], ...]}`
    (rows = checkpoints in step order, cols = opponents) plus the full per-cell stats,
    so you can see a real skill curve over training instead of one checkpoint's stats.
    Watch for a *non-monotonic* win rate here even while training reward climbs -- that's
    the signature of self-play strategy cycling (see `rl/IMPROVEMENT_PLAN.md`).
    """
    from stable_baselines3 import PPO

    ckpts = sorted(checkpoint_dir.glob("*.zip"), key=_checkpoint_step)
    if not ckpts:
        raise FileNotFoundError(f"no .zip checkpoints found under {checkpoint_dir}")

    win_rate_grid = []
    cells = []
    for ckpt in ckpts:
        print(f"evaluating {ckpt.name} ...")
        model = PPO.load(str(ckpt), device="cpu")
        row = []
        for opponent in opponents:
            t0 = time.time()
            stats = evaluate_one(
                model=model,
                env_id="MechMania-v0",
                opponent=opponent,
                max_ticks=max_ticks,
                episodes=episodes,
                deterministic=deterministic,
                seed=seed,
                frame_skip=frame_skip,
            )
            stats["eval_seconds"] = round(time.time() - t0, 1)
            stats["checkpoint"] = ckpt.name
            stats["step"] = _checkpoint_step(ckpt)
            row.append(stats["win_rate"])
            cells.append(stats)
            print(
                f"  {ckpt.name:22s} vs {opponent:8s}: "
                f"win_rate={stats['win_rate']*100:5.1f}%  ({stats['eval_seconds']}s)"
            )
        win_rate_grid.append(row)

    return {
        "checkpoints": [c.name for c in ckpts],
        "steps": [_checkpoint_step(c) for c in ckpts],
        "opponents": list(opponents),
        "win_rate": win_rate_grid,
        "cells": cells,
    }


def _print_round_robin_table(result: dict) -> None:
    opponents = result["opponents"]
    header = "step".rjust(10) + "".join(o.rjust(10) for o in opponents) + "  mean".rjust(8)
    print(header)
    for ckpt, step, row in zip(result["checkpoints"], result["steps"], result["win_rate"]):
        mean = sum(row) / len(row) if row else 0.0
        line = str(step).rjust(10) + "".join(f"{w*100:9.1f}%" for w in row) + f"{mean*100:7.1f}%"
        print(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="path to a single .zip saved by stable_baselines3")
    p.add_argument("--checkpoint-dir", help="directory of checkpoints to round-robin "
                    "against --opponents instead of evaluating a single --model")
    p.add_argument("--opponents", nargs="+", default=["do_nothing"], choices=OPPONENTS)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-ticks", type=int, default=200)
    p.add_argument("--frame-skip", type=int, default=1,
                    help="repeat each action for N real engine ticks; must match how "
                         "the model was trained to be a fair comparison")
    p.add_argument("--deterministic", action="store_true", help="use deterministic policy (default: sample)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="optional path to write JSON results")
    args = p.parse_args()

    if bool(args.model) == bool(args.checkpoint_dir):
        raise SystemExit("pass exactly one of --model or --checkpoint-dir")

    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"checkpoint dir not found: {checkpoint_dir}")
        opponents = args.opponents if args.opponents != ["do_nothing"] else \
            ["plan1", "plan2", "plan3", "plan4", "plan5"]
        print(f"round-robin: {checkpoint_dir} vs opponents={opponents}, episodes={args.episodes}")
        result = round_robin(
            checkpoint_dir=checkpoint_dir,
            opponents=opponents,
            max_ticks=args.max_ticks,
            episodes=args.episodes,
            deterministic=args.deterministic,
            seed=args.seed,
            frame_skip=args.frame_skip,
        )
        _print_round_robin_table(result)
        out_path = args.out or str(checkpoint_dir / "round_robin.json")
        Path(out_path).write_text(json.dumps(result, indent=2))
        print(f"wrote {out_path}")
        return

    from stable_baselines3 import PPO

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    print(f"loading model from {model_path}")
    model = PPO.load(str(model_path), device="cpu")
    print(f"model policy: {model.policy}")

    print(f"evaluating on opponents={args.opponents}, episodes={args.episodes}, deterministic={args.deterministic}")

    results = {}
    for opponent in args.opponents:
        t0 = time.time()
        stats = evaluate_one(
            model=model,
            env_id="MechMania-v0",
            opponent=opponent,
            max_ticks=args.max_ticks,
            episodes=args.episodes,
            deterministic=args.deterministic,
            seed=args.seed,
            frame_skip=args.frame_skip,
        )
        stats["eval_seconds"] = round(time.time() - t0, 1)
        results[opponent] = stats
        print(
            f"  vs {opponent:11s}: wins={stats['wins']}/{stats['episodes']} "
            f"({stats['win_rate']*100:.0f}%)  captures={stats['capture_end_rate']*100:.0f}%  "
            f"len={stats['mean_episode_length']:.0f}  cap={stats['mean_final_capture']:.2f}  "
            f"({stats['eval_seconds']}s)"
        )

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
