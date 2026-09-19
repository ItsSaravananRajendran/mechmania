# Tests for `strategy/plan1/`

One test file per module, mirroring `strategy/plan1/*.py` and documented
alongside `plans/plan1/*.md`. Plain `unittest` (stdlib only, no pytest
dependency) so `python3 -m unittest discover -s tests` works anywhere.

## Why these don't need a live match or engine channel

Most of `strategy/plan1`'s logic is pure Python math over `Vec2`/`BotClass`
(both plain ctypes/enum types that work standalone) plus a handful of
engine-channel calls (`point_free`, `line_of_sight`, `point_seg_dist`,
`navigate_to`, `path_length`) that need a live handshake with the engine.
`fakes.py` provides duck-typed stand-ins for `GameState`, `BotState`,
`GameConfig`, and friends -- objects exposing only the attributes the
algorithms actually read -- and each test file patches the engine-channel
calls at the name the module under test imported them as (e.g.
`strategy.plan1.combat_los.line_of_sight`, not `core.channel.line_of_sight`
-- `from .. import X` binds a *local* name in each module, so that's the one
that needs patching).

This is the same approach used to track down the corner-avoidance
oscillation bug earlier: a small synthetic scenario run directly against the
algorithm, instead of a full multi-thousand-tick match, to get a fast,
deterministic answer to "does this specific piece of logic do the right
thing."

## Running

```sh
python3 -m unittest discover -s tests -v
```

or a single module's tests:

```sh
python3 -m unittest tests.plan1.test_formation -v
```

## Layout

- `fakes.py` -- shared fakes, not a test file itself.
- `test_walls.py`, `test_combat_los.py`, `test_targeting.py`,
  `test_formation.py`, `test_fabricator.py`, `test_endgame.py` -- one module,
  one file.
- `test_orchestration.py` -- `strategy/plan1/__init__.py`'s glue:
  `_recompute_assignments`, `_apply_assignments`, `_is_in_heal_arc`, and
  `plan1_strategy`'s recompute-cadence decision.
