"""The one piece of state plan1 keeps between ticks.

A plain module-level dict, shared by every submodule that needs to remember
something across ticks (the wall grid, the deposit-to-deposit "forward"
direction, cached assignments, tick-over-tick bookkeeping) -- Python modules
are singletons, so `from .cache import _cache` gives every file the same
dict `plan1_strategy` itself reads and writes.
"""

from typing import Dict

_cache: Dict[str, object] = {}
