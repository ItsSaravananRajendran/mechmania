"""Inspect a finished match and summarize what the agent actually did.

Reads the `.mmgl` log produced by `mm-engine` and replays it tick by tick against a
trained policy, so we can see per-tick decisions the agent is making. Outputs a JSON
of summary stats + a small text report.

Usage:

    python -m rl.analyze --model rl/runs/<run>/model.zip --log rl/logs/rl/match-<...>.mmgl

If `--log` is omitted, the script will run a fresh match under the model and analyze
that. The `.mmgl` files live in `rl/logs/rl/match-*.mmgl` by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np

from rl.register import register
register()


def _summarize_actions(actions: list[np.ndarray]) -> dict:
    """Reduce a sequence of action vectors to per-bot statistics."""
    actions = np.asarray(actions)
    n, total = actions.shape[0], actions.shape[1]
    per_bot = (total - 4) // 32  # last 4 = fabricator one-hot + rush bool
    if per_bot <= 0:
        return {}

    # Reshape to (n_ticks, 32, per_bot).
    bot_actions = actions[:, : per_bot * 32].reshape(n, 32, per_bot)

    out = {"per_bot": []}
    for slot in range(32):
        ba = bot_actions[:, slot, :]
        # Magnitude of the move direction (sqrt(dx^2 + dy^2)). Small == "stay still".
        if per_bot >= 2:
            mag = np.linalg.norm(ba[:, :2], axis=1)
        else:
            mag = np.zeros(n)
        out["per_bot"].append({
            "slot": slot,
            "mean_move_magnitude": float(mag.mean()),
            "fraction_moving": float((mag > 0.1).mean()),
            "fraction_acting": float((mag > 0.5).mean()),
        })

    # Fleet-wide: which fabricator class the agent picks most often.
    fab = actions[:, per_bot * 32 : per_bot * 32 + 3]
    fab_choices = fab.argmax(axis=1)
    fab_counts = Counter(int(x) for x in fab_choices)
    out["fabricator_class_picks"] = dict(fab_counts)
    out["rush_order_frac"] = float((actions[:, -1] > 0).mean())

    return out


def _summarize_state_trajectory(states: list) -> dict:
    """Per-tick state stats. `states` is a list of (tick, my_hp, en_hp, capture, extracting_me, extracting_them)."""
    if not states:
        return {}
    arr = np.asarray(states, dtype=float)
    return {
        "ticks": int(arr.shape[0]),
        "mean_my_hp": float(arr[:, 1].mean()),
        "mean_en_hp": float(arr[:, 2].mean()),
        "final_capture": float(arr[-1, 3]),
        "mean_my_extracting": float(arr[:, 4].mean()),
        "mean_en_extracting": float(arr[:, 5].mean()),
    }


def run_one_episode(model, opponent: str, max_ticks: int, deterministic: bool, seed: int, frame_skip: int = 1) -> dict:
    """Play one episode under the trained model and capture everything we need to analyze it."""
    env = gym.make("MechMania-v0", opponent=opponent, max_ticks=max_ticks, frame_skip=frame_skip)
    obs, info = env.reset(seed=seed)

    actions = []
    states = []
    info_history = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        actions.append(np.asarray(action, dtype=np.float32))

        # Snapshot the pre-step state for analysis.
        st = env.unwrapped._state
        if st is not None:
            states.append([
                int(st.tick),
                sum(b.health for b in st.fleet_me),
                sum(b.health for b in st.fleet_other),
                float(st.capture),
                sum(1 for b in st.fleet_me if b.extracting is not None),
                sum(1 for b in st.fleet_other if b.extracting is not None),
            ])

        obs, r, terminated, truncated, info = env.step(action)
        info_history.append({"tick": int(info.get("tick", -1)), "r": float(r)})
        done = terminated or truncated

    summary = {
        "opponent": opponent,
        "max_ticks": max_ticks,
        "winner": info.get("winner"),
        "final_capture": float(info.get("capture", 0.5)),
        "n_steps": len(actions),
        "total_reward": sum(h["r"] for h in info_history),
        "actions_summary": _summarize_actions(actions),
        "state_summary": _summarize_state_trajectory(states),
    }
    env.close()
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--opponent", default="do_nothing", choices=["do_nothing", "basic", "plan1", "plan2", "plan3", "plan4", "plan5"])
    p.add_argument("--max-ticks", type=int, default=200)
    p.add_argument("--frame-skip", type=int, default=1)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    from stable_baselines3 import PPO

    model = PPO.load(args.model, device="cpu")
    print(f"loaded {args.model}")

    all_results = []
    for i in range(args.episodes):
        seed = args.seed + i
        t0 = time.time()
        summary = run_one_episode(model, args.opponent, args.max_ticks, args.deterministic, seed, frame_skip=args.frame_skip)
        summary["wall_s"] = round(time.time() - t0, 1)
        all_results.append(summary)
        print(
            f"ep {i}: winner={summary['winner']} final_capture={summary['final_capture']:.2f} "
            f"steps={summary['n_steps']} reward={summary['total_reward']:.2f} ({summary['wall_s']}s)"
        )

    # Brief textual summary across episodes.
    wins = sum(1 for r in all_results if r["winner"] == 0)
    losses = sum(1 for r in all_results if r["winner"] == 1)
    print(f"\nover {len(all_results)} episodes vs {args.opponent}: {wins}W / {losses}L / {len(all_results) - wins - losses}T")
    fab = Counter()
    for r in all_results:
        for k, v in r["actions_summary"].get("fabricator_class_picks", {}).items():
            fab[k] += v
    print(f"fabricator class picks across episodes: {dict(fab)}  # 0=Battle, 1=Healer, 2=Extractor")

    if args.out:
        Path(args.out).write_text(json.dumps(all_results, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
