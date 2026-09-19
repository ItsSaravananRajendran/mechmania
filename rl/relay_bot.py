"""The bot subprocess that bridges the engine's shared memory to a parent Gym env.

The engine launches us with one argument -- the path to the shared memory mapping it
created -- and expects us to handshake then loop `await_tick`/`respond` until the
channel closes. The Gym env wants to drive those ticks itself with RL actions, so
this script opens a Unix-domain socket the parent has already bound, sends the
serialised `GameState` over it, and waits for a serialised `FleetAction` to send back.

Protocol (each frame is one length-prefixed pickle):

    parent <- child:  GameState
    parent -> child:  FleetAction

The channel closes when `await_tick()` returns `None`; we forward that to the parent
as a `None` frame and exit 0.
"""

from __future__ import annotations

import os
import pickle
import socket
import sys
import traceback
from pathlib import Path
from typing import Optional

from core.channel import EngineChannel
from core._generated import bindings as _b  # noqa: F401  -- imported for its side effects


_SENTINEL_DONE = b"DONE"


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    """Read exactly `n` bytes or raise."""
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed mid-frame")
        buf.extend(chunk)
    return bytes(buf)


def _send_frame(conn: socket.socket, payload: bytes) -> None:
    """Length-prefixed frame: 4-byte big-endian length + payload."""
    conn.sendall(len(payload).to_bytes(4, "big") + payload)


def _recv_frame(conn: socket.socket) -> Optional[bytes]:
    """Return the next payload, or `None` on EOF."""
    try:
        header = _recv_exact(conn, 4)
    except ConnectionError:
        return None
    n = int.from_bytes(header, "big")
    if n == 0:
        return b""
    return _recv_exact(conn, n)


def _serve(conn: socket.socket, shmem_path: str) -> None:
    """The bot-side loop. Opens the engine channel, relays ticks until the match ends."""
    with EngineChannel.from_path(shmem_path) as chan:
        team = chan.handshake()  # 0 = bottom-left, 1 = top-right
        # Tell the parent which team we are and hand over the `GameConfig` so the
        # parent can build observation / reward without its own handshake. The relay
        # bot is the only process that ever talks to the engine for this match, and
        # the config is immutable for the match, so a single copy travels over.
        from core.channel import get_config as _get_config
        config = _get_config()
        _send_frame(conn, pickle.dumps(("hello", int(team), config)))

        while True:
            state = chan.await_tick()
            if state is None:
                # Match over. Tell the parent, then exit cleanly.
                _send_frame(conn, pickle.dumps(_SENTINEL_DONE))
                return

            _send_frame(conn, pickle.dumps(state))

            action_bytes = _recv_frame(conn)
            if action_bytes is None:
                # Parent died -- exit so the engine can shut us down.
                return

            action = pickle.loads(action_bytes)
            chan.respond(action)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: relay_bot.py <shmem_path>", file=sys.stderr)
        sys.exit(2)

    socket_path = os.environ.get("MM_RL_SOCKET")
    if not socket_path:
        print("MM_RL_SOCKET is not set", file=sys.stderr)
        sys.exit(2)

    sock_path = Path(socket_path)
    # The parent binds a Unix-domain socket at this path before launching us.
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        conn.connect(str(sock_path))
    except (FileNotFoundError, ConnectionRefusedError) as exc:
        print(f"could not connect to {sock_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        _serve(conn, sys.argv[1])
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    finally:
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        conn.close()


if __name__ == "__main__":
    main()
