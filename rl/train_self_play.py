"""Self-play PPO: two agents train simultaneously against each other.

Each agent runs in its own process with its own PPO model and its own
`MechManiaEnv`. The opponent slot in each env is filled from the *other*
side's checkpoint pool, exchanged via files under `<save-dir>/snapshots/`:

    snapshots/side_0/ckpt_00000000.zip, ckpt_00002000.zip, ...  <-- worker 0's history
    snapshots/side_1/ckpt_00000000.zip, ckpt_00002000.zip, ...  <-- worker 1's history

Worker 0 samples its opponent from `side_1`'s checkpoints (and vice versa), not just
the latest one: ~`--recent-frac` of episodes use the most recent checkpoint, the rest
sample a uniform-random older one, and `--scripted-mix-rate` of episodes substitute a
`plan1..plan5` scripted bot instead. This is what turns "self-play" into training
against a small population rather than one mutable, ever-drifting snapshot -- a single
latest-snapshot opponent is prone to strategy cycling (A learns to beat B's current
style, B adapts, A forgets how it beat B's earlier style, repeat without net progress).
Sampling from history and mixing in fixed scripted opponents are both standard,
cheap mitigations (a lightweight version of AlphaStar-style league play). See
`rl/IMPROVEMENT_PLAN.md` for the full rationale.

Because the opponent is a freshly-spawned subprocess every match, the trainer picks up
a newly-sampled opponent on the next `env.reset()` -- staleness is bounded by one match
length.

Both workers run in parallel (multiprocessing), so both agents are training at the same
time. The main process initialises each side's first checkpoint before spawning the
workers so neither side starts with an empty opponent pool.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import gymnasium as gym

from rl.register import register
from rl.env import MechManiaEnv
from rl.env import _OPPONENTS as _OPPONENTS_LOOKUP
from rl.rewards import (
    ShapingSchedule,
    default_reward,
    make_annealed_reward,
    sparse_reward,
    tactics_reward,
)


_REWARD_FNS = {
    "default": default_reward,
    "sparse": sparse_reward,
    "tactics": tactics_reward,
}

_SCRIPTED_OPPONENTS = ["plan1", "plan2", "plan3", "plan4", "plan5"]


# -----------------------------------------------------------------------------------
# checkpoint pool
# -----------------------------------------------------------------------------------


def _side_dir(snapshot_dir: Path, worker_id: int) -> Path:
    return snapshot_dir / f"side_{worker_id}"


def _ckpt_path(side_dir: Path, trained: int) -> Path:
    return side_dir / f"ckpt_{trained:08d}.zip"


def _list_checkpoints(side_dir: Path) -> List[Path]:
    """All checkpoints for a side, oldest first. Re-globbed each call since the
    other worker's process is writing new ones concurrently."""
    return sorted(side_dir.glob("ckpt_*.zip"))


def _bootstrap_snapshot(path: Path, max_ticks: int, log_dir: Path) -> None:
    """Create an initial PPO snapshot on disk so the opponent pool isn't empty.

    Uses a throwaway env with `do_nothing` (the most inert opponent) just so
    PPO can build its policy/value nets against a valid observation/action space.
    The resulting policy is essentially random; that is fine -- self-play only
    needs *some* opponent at tick 0, and both workers improve together from there.
    """
    from stable_baselines3 import PPO
    env = MechManiaEnv(opponent="do_nothing", max_ticks=max_ticks, log_dir=log_dir)
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        gamma=0.995,
        ent_coef=0.01,
        verbose=0,
        device="cpu",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    env.close()


# -----------------------------------------------------------------------------------
# opponent sampling + shaping decay (wired in as an SB3 callback)
# -----------------------------------------------------------------------------------


def _sample_opponent(
    rng: np.random.Generator,
    opp_checkpoints: List[Path],
    scripted_mix_rate: float,
    recent_frac: float,
) -> Tuple[str, Optional[str]]:
    """Pick `(opponent_name, opponent_ppo_path)` for the next episode.

    ~`scripted_mix_rate` of the time: a random `plan1..plan5` scripted bot (a fixed,
    non-drifting anchor). Otherwise: `recent_frac` of the time the opponent's newest
    checkpoint, else a uniform-random older one -- this is what prevents the "latest
    snapshot only" cycling failure mode.
    """
    if scripted_mix_rate > 0.0 and rng.random() < scripted_mix_rate:
        return str(rng.choice(_SCRIPTED_OPPONENTS)), None

    if not opp_checkpoints:
        # Pool briefly empty (shouldn't happen after bootstrap, but don't crash a
        # training run over it) -- fall back to a scripted anchor.
        return "plan1", None

    if len(opp_checkpoints) == 1 or rng.random() < recent_frac:
        chosen = opp_checkpoints[-1]
    else:
        chosen = opp_checkpoints[rng.integers(0, len(opp_checkpoints) - 1)]
    return "ppo", str(chosen)


def _make_self_play_callback(
    opp_side_dir: Path,
    scripted_mix_rate: float,
    recent_frac: float,
    shaping_schedule: Optional[ShapingSchedule],
    total_steps: int,
    seed: int,
):
    from stable_baselines3.common.callbacks import BaseCallback

    class SelfPlayCallback(BaseCallback):
        """Resamples the opponent for the *next* episode whenever the current one
        ends, and (optionally) keeps a `ShapingSchedule` in sync with training
        progress. Runs inside the worker process, so `self.training_env` is this
        worker's own (single, non-vectorized) env."""

        def __init__(self) -> None:
            super().__init__()
            self._rng = np.random.default_rng(seed)
            self._checkpoints_cache: List[Path] = []

        def _on_training_start(self) -> None:
            self._checkpoints_cache = _list_checkpoints(opp_side_dir)

        def _on_step(self) -> bool:
            if shaping_schedule is not None:
                shaping_schedule.set_linear(self.num_timesteps, total_steps)

            dones = self.locals.get("dones")
            if dones is not None and bool(np.asarray(dones).any()):
                # Refresh cheaply -- the pool only grows by one file every
                # `snapshot_every` steps, so a re-glob per episode end is fine.
                self._checkpoints_cache = _list_checkpoints(opp_side_dir)
                name, path = _sample_opponent(
                    self._rng, self._checkpoints_cache, scripted_mix_rate, recent_frac
                )
                env = self.training_env.envs[0].unwrapped
                env.opponent_name = name
                env.opponent_callable_name = _OPPONENTS_LOOKUP[name]
                env.opponent_ppo_path = path
            return True

    return SelfPlayCallback()


# -----------------------------------------------------------------------------------
# worker
# -----------------------------------------------------------------------------------


def _worker(
    worker_id: int,
    total_steps: int,
    snapshot_dir: Path,
    max_ticks: int,
    frame_skip: int,
    reward_name: str,
    save_dir: Path,
    seed: int,
    learning_rate: float,
    n_steps: int,
    batch_size: int,
    gamma: float,
    ent_coef: float,
    snapshot_every: int,
    scripted_mix_rate: float,
    recent_frac: float,
    anneal_shaping: bool,
) -> None:
    """One side of the self-play: train a PPO against the *other* side's checkpoint
    pool, sampled per-episode by `SelfPlayCallback` (see `_make_self_play_callback`)."""
    from stable_baselines3 import PPO

    my_dir = _side_dir(snapshot_dir, worker_id)
    opp_dir = _side_dir(snapshot_dir, 1 - worker_id)
    my_dir.mkdir(parents=True, exist_ok=True)

    log_dir = save_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    shaping_schedule = ShapingSchedule(value=1.0) if anneal_shaping else None
    base_reward = _REWARD_FNS[reward_name]
    reward_fn = (
        make_annealed_reward(shaping_schedule, base_fn=base_reward)
        if shaping_schedule is not None
        else base_reward
    )

    # Initial opponent: newest checkpoint the other side has posted so far.
    initial_opp = _list_checkpoints(opp_dir)
    initial_opp_path = str(initial_opp[-1]) if initial_opp else None

    def _make_env():
        return MechManiaEnv(
            opponent="ppo" if initial_opp_path else "plan1",
            opponent_ppo_path=initial_opp_path,
            max_ticks=max_ticks,
            frame_skip=frame_skip,
            reward_fn=reward_fn,
            log_dir=log_dir,
        )

    env = _make_env()

    existing = _list_checkpoints(my_dir)
    if existing:
        my_latest = existing[-1]
        print(f"[worker {worker_id}] resuming from {my_latest}", flush=True)
        model = PPO.load(str(my_latest), env=env, device="cpu")
        # `_bootstrap_snapshot` saves with verbose=0 so bootstrapping itself stays
        # quiet; every worker loads from that (or an earlier checkpoint of its own)
        # and would otherwise silently inherit verbose=0 for the whole run, which
        # is why the SB3 rollout stats (ep_rew_mean, approx_kl, entropy_loss, ...)
        # weren't printing. Force it back on -- `configure_logger` reads `verbose`
        # fresh on each `learn()` call, so setting it post-load is enough.
        model.verbose = 1
    else:
        print(f"[worker {worker_id}] creating fresh policy", flush=True)
        model = PPO(
            "MlpPolicy", env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            ent_coef=ent_coef,
            verbose=1,
            seed=seed + worker_id,
            device="cpu",
        )

    callback = _make_self_play_callback(
        opp_side_dir=opp_dir,
        scripted_mix_rate=scripted_mix_rate,
        recent_frac=recent_frac,
        shaping_schedule=shaping_schedule,
        total_steps=total_steps,
        seed=seed + worker_id,
    )

    t0 = time.time()
    trained = int(existing[-1].stem.split("_")[-1]) if existing else 0
    start_trained = trained
    while trained - start_trained < total_steps:
        chunk = min(snapshot_every, total_steps - (trained - start_trained))
        model.learn(
            total_timesteps=chunk,
            reset_num_timesteps=False,
            progress_bar=False,
            callback=callback,
        )
        trained += chunk
        ckpt = _ckpt_path(my_dir, trained)
        model.save(str(ckpt))
        elapsed = time.time() - t0
        rate = (trained - start_trained) / elapsed if elapsed > 0 else 0.0
        print(
            f"[worker {worker_id}] {trained - start_trained}/{total_steps} steps this run "
            f"({trained} lifetime; {rate:.1f} steps/s); checkpoint -> {ckpt.name}",
            flush=True,
        )

    env.close()
    print(f"[worker {worker_id}] done; {len(_list_checkpoints(my_dir))} checkpoints under {my_dir}", flush=True)


# -----------------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=50000,
                   help="steps per worker (both workers train this many)")
    p.add_argument("--max-ticks", type=int, default=200)
    p.add_argument("--reward", default="tactics", choices=sorted(_REWARD_FNS))
    p.add_argument("--save-dir", default="rl/runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--snapshot-every", type=int, default=2000,
                   help="steps between checkpoints (per worker); also the pool's grain size")
    p.add_argument("--recent-frac", type=float, default=0.7,
                   help="fraction of episodes that sample the opponent's *newest* checkpoint "
                        "(vs. a uniform-random older one) -- see rl/IMPROVEMENT_PLAN.md")
    p.add_argument("--scripted-mix-rate", type=float, default=0.2,
                   help="fraction of episodes where the opponent is a random plan1-5 scripted "
                        "bot instead of a self-play checkpoint")
    p.add_argument("--no-anneal-shaping", dest="anneal_shaping", action="store_false",
                   help="disable annealing the reward's dense shaping to 0 over training "
                        "(shaping stays constant for the whole run)")
    p.set_defaults(anneal_shaping=True)
    p.add_argument("--frame-skip", type=int, default=1,
                   help="repeat each action for N real engine ticks (see rl/env.py); "
                        "reduces decision count without changing real game time")
    args = p.parse_args()

    register()

    run_name = args.run_name or (
        f"selfplay-{args.reward}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    snapshot_dir = save_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "mode": "self_play",
        "total_timesteps_per_worker": args.total_timesteps,
        "max_ticks": args.max_ticks,
        "frame_skip": args.frame_skip,
        "reward": args.reward,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "ent_coef": args.ent_coef,
        "snapshot_every": args.snapshot_every,
        "recent_frac": args.recent_frac,
        "scripted_mix_rate": args.scripted_mix_rate,
        "anneal_shaping": args.anneal_shaping,
    }
    (save_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"run dir: {save_dir}")
    print(f"config: {json.dumps(config, indent=2)}")

    bootstrap_log_dir = save_dir / "bootstrap_logs"
    bootstrap_log_dir.mkdir(parents=True, exist_ok=True)

    # Pre-create each side's first checkpoint (`ckpt_00000000.zip`) so neither side's
    # opponent pool is empty when their first env.reset() spawns the opponent subprocess.
    print("bootstrapping initial checkpoints ...")
    _bootstrap_snapshot(_ckpt_path(_side_dir(snapshot_dir, 0), 0), args.max_ticks, bootstrap_log_dir)
    _bootstrap_snapshot(_ckpt_path(_side_dir(snapshot_dir, 1), 0), args.max_ticks, bootstrap_log_dir)

    print("spawning parallel workers ...")
    common_args = (
        args.total_timesteps, snapshot_dir, args.max_ticks, args.frame_skip,
        args.reward, save_dir, args.seed,
        args.learning_rate, args.n_steps, args.batch_size,
        args.gamma, args.ent_coef, args.snapshot_every,
        args.scripted_mix_rate, args.recent_frac, args.anneal_shaping,
    )
    procs = [
        mp.Process(target=_worker, args=(0, *common_args)),
        mp.Process(target=_worker, args=(1, *common_args)),
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()

    summary = {
        "timesteps_per_worker": args.total_timesteps,
        "side_0_checkpoints": [str(p) for p in _list_checkpoints(_side_dir(snapshot_dir, 0))],
        "side_1_checkpoints": [str(p) for p in _list_checkpoints(_side_dir(snapshot_dir, 1))],
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"both workers finished; checkpoints under {snapshot_dir}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()