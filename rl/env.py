"""The Gymnasium environment wrapping one MechMania match.

How a step actually works under the hood:

  1. `reset()` spawns the `mm-engine` binary as a subprocess with two bot slots:
     - bot A: `rl/run_relay.sh` -- the relay bot in this package.
     - bot B: a fixed opponent, by default a Python bot that runs one of the
       `planN_strategy` functions in `strategy/`. We spawn that with the existing
       `.mm/run` wrapper so it inherits the right PYTHONPATH and engine layout.
  2. The env binds a Unix-domain socket and waits for the relay bot to connect.
  3. The engine kicks off the match; the relay bot opens the shared memory mapping
     and starts forwarding `GameState`s.
  4. `step(action)` pickles the action and sends it down the socket; the relay bot
     decodes it, calls `respond()`, and forwards the next `GameState` back. That
     round trip is what we call "one tick".

Why a relay and not in-process: the engine is a Rust binary that owns the simulation,
so we can't put the gym loop inside the same process. A separate bot subprocess is
the engine's only extension point.

Why a socket and not stdin/stdout: the bot's stdout is the gamelog, so we'd race the
match output. A Unix-domain socket is also a clean Python <-> Python boundary.

Why `select`-style non-blocking IO: each tick must end with exactly one frame in
each direction. We use blocking IO with a small timeout because it is simpler and
the engine is always going to produce a state within ~ms.
"""

from __future__ import annotations

import atexit
import os
import pickle
import signal
import socket
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core._generated.bindings import FleetAction, GameState
from rl.encoding import (
    ActionMode,
    OBS_DIM,
    action_space_low_high,
    decode_action,
    encode_state,
    set_active_config,
)
from rl.rewards import RewardFn, default_reward
from rl.relay_bot import _SENTINEL_DONE


# -----------------------------------------------------------------------------------
# opponent discovery
# -----------------------------------------------------------------------------------

# Each entry is a (display-name, callable-from-strategy-module) the env can spin up as
# the opponent bot. Lives here so callers don't have to know the strategy names.
# `ppo` is special: the opponent subprocess loads a saved PPO model whose path is
# passed via the `MM_PPO_MODEL_PATH` env var. See `strategy/ppo_strategy.py`.
_OPPONENTS = {
    "plan1": "plan1_strategy",
    "plan2": "plan2_strategy",
    "plan3": "plan3_strategy",
    "plan4": "plan4_strategy",
    "plan5": "plan5_strategy",
    "basic": "basic_strategy",
    "do_nothing": "do_nothing",
    "ppo": "ppo_strategy",
}


@dataclass
class _MatchProcess:
    """The running `mm-engine` subprocess + the socket the relay bot is connected to."""

    proc: subprocess.Popen
    conn: socket.socket
    socket_path: str
    team: int  # which side the relay bot ended up on (0 = bottom-left, 1 = top-right)

    def close(self) -> None:
        for s in (self.conn,):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                s.close()
            except OSError:
                pass
        if self.proc.poll() is None:
            self._kill_match()
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass

    def _kill_match(self) -> None:
        """Terminate the engine *and* any bot subprocesses it spawned.

        The engine forks each bot into its own session, so they are NOT in the same
        process group as the engine even with `start_new_session=True`. We kill the
        engine first, then sweep `pgrep -P` to find any reparented bots and kill them
        by their shared-memory path argument -- that's the only stable handle on them.
        """
        # Collect opponent bot paths (one per pid) while the engine is still alive so
        # we can match them later by the `/tmp/.tmpXXXXXX` shmem argument.
        bot_pids: List[int] = []
        try:
            ps_out = subprocess.check_output(
                ["ps", "-eo", "pid,cmd"], text=True, timeout=2.0
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            ps_out = ""
        for line in ps_out.splitlines():
            line = line.strip()
            if "__main__.py /tmp/" in line or "relay_bot.py /tmp/" in line:
                try:
                    pid = int(line.split(None, 1)[0])
                except ValueError:
                    continue
                if pid != self.proc.pid:
                    bot_pids.append(pid)

        # Kill the engine (its process group, in case it has any children of its own).
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass

        # Now any bots that were children of the engine are reparented to init. The
        # path-based match from above may have caught them already; if not, look up
        # children of the engine pid again -- if the engine is gone, pgrep on its pid
        # returns nothing, so fall back to scanning all processes for the same shmem
        # path argument we saw earlier.
        try:
            survivors = subprocess.check_output(
                ["pgrep", "-f", "__main__.py /tmp/"], text=True, timeout=2.0
            ).split()
        except (subprocess.SubprocessError, FileNotFoundError):
            survivors = []

        for pid in survivors:
            try:
                pid_int = int(pid)
            except ValueError:
                continue
            try:
                os.kill(pid_int, signal.SIGTERM)
            except ProcessLookupError:
                continue
        if survivors:
            time.sleep(0.3)
            for pid in survivors:
                try:
                    pid_int = int(pid)
                    os.kill(pid_int, signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    continue

        # Belt-and-braces: also kill whatever we caught in the first scan.
        for pid in bot_pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
        if bot_pids:
            time.sleep(0.3)
            for pid in bot_pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    continue


# -----------------------------------------------------------------------------------
# the env
# -----------------------------------------------------------------------------------


class MechManiaEnv(gym.Env):
    """A Gymnasium env wrapping a single MechMania match.

    Args:
        opponent: Name of a built-in opponent strategy. One of `plan1`..`plan5`, `basic`,
            `do_nothing`. Defaults to `plan1` (the most aggressive of the bundled bots).
        action_mode: `MINIMAL`, `TACTICAL`, or `FULL`. See `rl.encoding.ActionMode`.
        reward_fn: Per-tick shaping. Default `default_reward`; pass `sparse_reward` (or
            your own callable) to override.
        max_ticks: Hard cap on match length in *real engine ticks* (not decisions --
            see `frame_skip`). The engine already enforces `conf.max_ticks`; this is a
            *second* cap that returns `truncated=True`.
        frame_skip: Repeat each decoded action for this many consecutive real engine
            ticks before asking the policy for a new one (classic action-repeat).
            Reward is summed over all `frame_skip` ticks into one `env.step()` return.
            This does *not* change real game time -- the engine still simulates every
            tick -- it only reduces how many *decisions* an episode takes, which is
            what actually helps long-horizon credit assignment (this game's real
            `conf.max_ticks` is 9000; see `rl/IMPROVEMENT_PLAN.md`). Default `1` (off).
        engine_path: Path to the `mm-engine` binary. Defaults to the
            `<repo>/.mm/bin/mm-engine` produced by `mm-cli run`.
        bot_root: Path to the directory containing `__main__.py` and `strategy/`.
            Defaults to this repo. The env needs a working directory it can run the
            opponent bot from, with `core/_generated/` already populated.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent: str = "plan1",
        action_mode: ActionMode = ActionMode.TACTICAL,
        reward_fn: RewardFn = default_reward,
        max_ticks: Optional[int] = None,
        frame_skip: int = 1,
        engine_path: Optional[str] = None,
        bot_root: Optional[str] = None,
        log_dir: Optional[str] = None,
        opponent_ppo_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        if opponent not in _OPPONENTS:
            raise ValueError(
                f"unknown opponent {opponent!r}; pick one of {sorted(_OPPONENTS)}"
            )

        if opponent == "ppo" and not opponent_ppo_path:
            raise ValueError("opponent='ppo' requires opponent_ppo_path=<path to .zip>")

        self.opponent_name = opponent
        self.opponent_callable_name = _OPPONENTS[opponent]
        self.opponent_ppo_path = str(opponent_ppo_path) if opponent_ppo_path else None
        self.action_mode = action_mode
        self.reward_fn = reward_fn
        self.max_ticks = max_ticks
        if frame_skip < 1:
            raise ValueError(f"frame_skip must be >= 1, got {frame_skip}")
        self.frame_skip = frame_skip
        self.engine_path = Path(engine_path) if engine_path else self._default_engine_path()
        self.bot_root = Path(bot_root) if bot_root else Path(__file__).resolve().parent.parent
        self.log_dir = Path(log_dir) if log_dir else self.bot_root / "logs" / "rl"

        if not self.engine_path.exists():
            raise FileNotFoundError(
                f"mm-engine binary not found at {self.engine_path}; run `mm-cli run` first"
            )

        # Spaces.
        low, high = action_space_low_high(action_mode)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.5, high=1.5, shape=(OBS_DIM,), dtype=np.float32
        )

        # Lazily populated on `reset()`.
        self._match: Optional[_MatchProcess] = None
        self._prev_state: Optional[GameState] = None
        self._state: Optional[GameState] = None
        self._done = False
        self._winner: Optional[int] = None
        self._episode_steps = 0

        # Best-effort cleanup if the process is GC'd without `close()`.
        self._closed = False
        atexit.register(self._atexit_cleanup)

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _default_engine_path() -> Path:
        # Walk up from this file until we find `.mm/bin/mm-engine`.
        here = Path(__file__).resolve().parent
        for _ in range(6):
            candidate = here / ".mm" / "bin" / "mm-engine"
            if candidate.exists():
                return candidate
            here = here.parent
        raise FileNotFoundError(
            "could not locate mm-engine; pass engine_path=... to MechManiaEnv(...)"
        )

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        # Hand the seed to `gym.Env`'s RNG so `check_env` is happy. We do not use
        # this RNG to seed the engine (the engine has its own seed inside its binary);
        # the only consumer is whatever calls `env.action_space.sample()` between
        # `reset` and `step`.
        super().reset(seed=seed)

        if self._match is not None:
            self._close_match()
        self._done = False
        self._winner = None
        self._episode_steps = 0
        self._state = None
        self._prev_state = None

        self._match = self._launch_match()
        first_state = self._recv_state_or_done()
        if first_state is None:
            raise RuntimeError("engine closed the channel before any tick was delivered")

        self._state = first_state
        return encode_state(first_state), {
            "team": self._match.team,
            "opponent": self.opponent_name,
            "tick": int(first_state.tick),
        }

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self._match is None or self._state is None:
            raise RuntimeError("env.step() called before reset(); call env.reset() first")
        if self._done:
            raise RuntimeError("env.step() called on a finished episode; call reset() first")

        # Decode once -- with `frame_skip > 1` the same decoded action is resent for
        # up to `frame_skip` consecutive *real* engine ticks (classic action-repeat).
        # The engine still simulates every real tick and every tick's reward is summed
        # into this one `step()` call, so this doesn't change real game time or total
        # shaping magnitude -- it only reduces how often the policy has to make a new
        # decision, which is what actually helps PPO's credit assignment over a
        # long-horizon match (see rl/IMPROVEMENT_PLAN.md).
        try:
            fa = decode_action(np.asarray(action, dtype=np.float32), self._state, self.action_mode)
        except Exception as exc:
            raise ValueError(f"failed to decode action: {exc}") from exc

        total_reward = 0.0
        for _ in range(self.frame_skip):
            self._prev_state = self._state
            self._send_action(fa)
            self._episode_steps += 1

            # Receive the next tick. `_recv_state_or_done` returns `None` on match end.
            next_state = self._recv_state_or_done()
            if next_state is None:
                # The engine closed the channel: episode is over. Decide who won from
                # the last state we had -- `capture == 1` means my team held the
                # payload, `capture == 0` means the opponent did, and max_ticks is a
                # tie.
                self._done = True
                self._winner = self._decide_winner(self._prev_state)
                total_reward += self.reward_fn(self._prev_state, self._prev_state, True, self._winner)
                obs = encode_state(self._prev_state)
                info = {
                    "winner": self._winner,
                    "tick": int(self._prev_state.tick),
                    "reason": "engine closed channel",
                }
                return obs, total_reward, True, False, info

            self._state = next_state
            if self.max_ticks is not None and self._episode_steps >= self.max_ticks:
                self._done = True
                self._winner = self._decide_winner(next_state)
                total_reward += self.reward_fn(self._prev_state, next_state, True, self._winner)
                info = {
                    "tick": int(next_state.tick),
                    "capture": float(next_state.capture),
                    "winner": self._winner,
                }
                return encode_state(next_state), total_reward, True, True, info

            total_reward += self.reward_fn(self._prev_state, next_state, False, None)

        # Ran all `frame_skip` ticks without hitting a terminal condition.
        info = {
            "tick": int(self._state.tick),
            "capture": float(self._state.capture),
        }
        return encode_state(self._state), total_reward, False, False, info

    def close(self) -> None:
        self._close_match()
        self._closed = True

    def render(self) -> None:
        # No rendering: the engine writes a `.mmgl` log under `logs/`. Tweak `log_dir`
        # if you want them somewhere else.
        return None

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _launch_match(self) -> _MatchProcess:
        """Spawn the engine, then build the relay socket the relay bot will dial."""
        socket_path = self._new_socket_path()
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(socket_path))
        server.listen(1)
        server.settimeout(30.0)  # generous: opponent boot + handshake

        env = os.environ.copy()
        env["MM_RL_SOCKET"] = str(socket_path)
        # Make sure the opponent bot finds its own modules.
        env["PYTHONPATH"] = (
            str(self.bot_root) + os.pathsep + env.get("PYTHONPATH", "")
        )
        env.setdefault("MM_NATIVE_LIB", str(self.bot_root / "native" / "target" / "release" / "libmm_python_native.so"))

        # When the opponent is a PPO snapshot, point the subprocess at it and have
        # `strategy.main.get_strategy` swap in the PPO-backed strategy. The subprocess
        # is respawned per match, so an updated snapshot is picked up on the next match.
        if self.opponent_name == "ppo" and self.opponent_ppo_path is not None:
            env["MM_OPPONENT_STRATEGY"] = "ppo"
            env["MM_PPO_MODEL_PATH"] = self.opponent_ppo_path

        # Bot A is our relay bot. Bot B is the built-in opponent -- we point the engine
        # at a tiny shell wrapper that execs `python3 __main__.py`, the same path
        # `.mm/run` uses. This keeps the opponent on the same module / conf layout.
        opponent_runner = self.bot_root / "rl" / "run_opponent.sh"
        if not opponent_runner.exists():
            self._write_opponent_runner(opponent_runner)
        relay_runner = self.bot_root / "rl" / "run_relay.sh"

        log_dir = self.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"match-{int(time.time())}-{uuid.uuid4().hex[:6]}.mmgl"

        cmd = [
            str(self.engine_path),
            "-o", f"g:{log_path}",
            str(relay_runner),
            str(opponent_runner),
        ]
        # Run from `bot_root` so the opponent's `__main__.py` resolves its own
        # `from strategy.main import ...` imports.
        proc = subprocess.Popen(
            cmd,
            cwd=str(self.bot_root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,  # detach so the env can kill the whole group
        )

        # The relay bot dials the socket as soon as it has done its handshake. Block
        # until either it connects or the engine crashes first.
        try:
            conn, _ = server.accept()
        except socket.timeout:
            proc.kill()
            stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            raise RuntimeError(
                f"relay bot did not connect to {socket_path} within 30s\nengine stderr:\n{stderr}"
            )
        finally:
            server.close()

        conn.settimeout(30.0)
        # First frame from the relay bot is `("hello", team, config)`. The relay bot
        # forwards the `GameConfig` it received at handshake so the env can build
        # observations and rewards without its own handshake.
        header = _recv_exact(conn, 4)
        n = int.from_bytes(header, "big")
        hello = pickle.loads(_recv_exact(conn, n))
        if not (isinstance(hello, tuple) and len(hello) == 3 and hello[0] == "hello"):
            raise RuntimeError(f"unexpected hello frame: {hello!r}")
        team = int(hello[1])
        config = hello[2]
        # Stash the config in this process so `encode_state` / `decode_action` /
        # reward shaping can find it without re-handshaking.
        set_active_config(config)

        # If we drew team 1, the engine already mirrored -- nothing for the algorithm
        # to do, and `fleet_me` / `fleet_other` still read as "my side / their side"
        # from the agent's view.
        if team == 1:
            pass

        return _MatchProcess(
            proc=proc,
            conn=conn,
            socket_path=str(socket_path),
            team=team,
        )

    def _write_opponent_runner(self, path: Path) -> None:
        """A `sh` wrapper that exec's `python3 __main__.py`. Same shape as `.mm/run`."""
        body = (
            "#!/bin/sh\n"
            f'exec python3 "{self.bot_root}/__main__.py" "$@"\n'
        )
        path.write_text(body, encoding="utf-8")
        os.chmod(path, 0o755)

    def _new_socket_path(self) -> str:
        # `/tmp` is the only path that is reliably short enough for `sun_path`.
        d = Path(tempfile.gettempdir()) / f"mm-rl-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock"
        return str(d)

    def _recv_state_or_done(self) -> Optional[GameState]:
        """Block for the next `GameState` from the relay bot, or `None` if it sent DONE."""
        if self._match is None:
            raise RuntimeError("no live match")
        payload = _recv_frame(self._match.conn)
        if payload is None:
            return None
        frame = pickle.loads(payload)
        if frame == _SENTINEL_DONE:
            return None
        return frame  # type: ignore[return-value]

    def _send_action(self, action: FleetAction) -> None:
        if self._match is None:
            raise RuntimeError("no live match")
        try:
            _send_frame(self._match.conn, pickle.dumps(action))
        except (BrokenPipeError, ConnectionError, OSError):
            # The engine/opponent subprocess died between our last recv and this
            # send (crash, OOM, etc.). Swallow it rather than let it propagate and
            # kill the whole training process -- the next `_recv_state_or_done()`
            # call in `step()` will see the closed socket and return `None`, which
            # already routes through the "engine closed channel" episode-end path
            # below. Mirrors how `_recv_frame` already handles a closed connection.
            pass

    @staticmethod
    def _decide_winner(state: GameState) -> Optional[int]:
        """Read the outcome from the last state we saw.

        `capture` ranges in `[-1.0, +1.0]`: 0 is the neutral start state, +1 means my
        team pushed the payload all the way to their end, -1 means the opponent did.
        The engine ends the match the moment capture hits either bound; reaching
        `max_ticks` with capture still around 0 is a tie.
        """
        if state.capture >= 1.0:
            return 0
        if state.capture <= -1.0:
            return 1
        # Inconclusive at the cap. Pick by total HP as a tiebreak.
        my_hp = sum(b.health for b in state.fleet_me)
        en_hp = sum(b.health for b in state.fleet_other)
        if my_hp > en_hp:
            return 0
        if en_hp > my_hp:
            return 1
        return None

    def _close_match(self) -> None:
        if self._match is not None:
            self._match.close()
            self._match = None

    def _atexit_cleanup(self) -> None:
        if not self._closed:
            try:
                self._close_match()
            except Exception:
                pass


# -----------------------------------------------------------------------------------
# socket helpers (kept here too so env.py is self-contained)
# -----------------------------------------------------------------------------------


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed mid-frame")
        buf.extend(chunk)
    return bytes(buf)


def _send_frame(conn: socket.socket, payload: bytes) -> None:
    conn.sendall(len(payload).to_bytes(4, "big") + payload)


def _recv_frame(conn: socket.socket) -> Optional[bytes]:
    try:
        header = _recv_exact(conn, 4)
    except ConnectionError:
        return None
    n = int.from_bytes(header, "big")
    if n == 0:
        return b""
    return _recv_exact(conn, n)
