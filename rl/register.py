"""`gym.register` plumbing for MechMania.

`register()` is idempotent: importing the package or calling it twice is a no-op the
second time. The env id is `MechMania-v0`; `gym.make("MechMania-v0", ...)` returns a
fresh env that overrides the defaults with the kwargs you passed.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym

from rl.encoding import ActionMode
from rl.env import MechManiaEnv
from rl.rewards import RewardFn, default_reward


_REGISTERED = False
_ENTRY_POINT = "rl.env:MechManiaEnv"


def register() -> None:
    """Idempotent registration. Safe to call multiple times."""
    global _REGISTERED
    if _REGISTERED:
        return
    gym.register(
        id="MechMania-v0",
        entry_point=_ENTRY_POINT,
        kwargs={
            "opponent": "plan1",
            "action_mode": ActionMode.TACTICAL,
            "reward_fn": default_reward,
        },
    )
    _REGISTERED = True


# Auto-register on import so `import gymnasium as gym; gym.make("MechMania-v0")`
# works without an extra call from the user.
register()
