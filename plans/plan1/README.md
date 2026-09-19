# Plan 1 algorithms

`../plan1.md` is the strategic spec (build order, tactics, requirements
`R-P1.*`/`R-X*`). This directory documents how `strategy/plan1/` actually
implements it — one file per module, each covering: what problem the module
solves, the data it works with, the algorithm itself, and the edge cases it
was written to survive. Tests for each live in `tests/plan1/`, one file per
module, mirroring this layout.

| Module | Doc | What it does |
|---|---|---|
| `cache.py` | (below, no separate doc — it's one line) | The shared cross-tick state dict every other module reads/writes. |
| `walls.py` | [walls.md](walls.md) | Static wall grid built once from `conf.map`. |
| `combat_los.py` | [combat_los.md](combat_los.md) | Clear-shot checks against walls, deposits, and the payload; side-stepping a blocked shot. |
| `targeting.py` | [targeting.md](targeting.md) | Fleet-wide, cooldown-aware target assignment. |
| `formation.py` | [formation.md](formation.md) | Zone/positioning so bots don't cluster into one splash-vulnerable blob. |
| `fabricator.py` | [fabricator.md](fabricator.md) | Build-order: what class to queue next. |
| `endgame.py` | [endgame.md](endgame.md) | The endgame self-destruct call. |
| `__init__.py` | [orchestration.md](orchestration.md) | The glue: assignment cache, recompute cadence, and per-tick action wiring. |

## `cache.py`

```python
_cache: Dict[str, object] = {}
```

That's the whole module. Python modules are singletons, so every other
`strategy/plan1/*.py` file does `from .cache import _cache` and gets the
*same* dict `plan1_strategy` itself reads and writes — one shared store for
everything that needs to persist across ticks (the wall grid, the
deposit-to-deposit "forward" direction, the cached assignment dict,
tick-over-tick bookkeeping like `last_enemy_count`). It's deliberately a
plain dict with no schema: each module owns its own key(s) and nobody else
reaches into them.
