"""Tests for strategy/plan1/corners.py -- corner-hugging avoidance.

No live engine/simulation is used: `navigate_to`/`move_bot` are monkeypatched
at the name `corners.py` bound them under (`from .. import ...`), same
convention as the rest of tests/plan1/. Multi-tick "does it ever reach the
target" checks drive `_widen_path` directly with a hand-rolled step, standing
in for what the real engine's movement integration would do.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import strategy.plan1.corners as corners_mod  # noqa: E402
import strategy.plan1.walls as walls_mod  # noqa: E402
from tests.plan1.fakes import FakeBotConfig, FakeMapConf, Vec2  # noqa: E402


def _grid(wall_cells=()):
    return walls_mod.WallGrid.build(FakeMapConf(wall_cells=set(wall_cells)))


def _step_towards(pos: Vec2, aim: Vec2, step_len: float) -> Vec2:
    d = aim - pos
    dist = (d.x ** 2 + d.y ** 2) ** 0.5
    if dist < 1e-9:
        return pos
    scale = min(step_len, dist) / dist
    return Vec2(pos.x + d.x * scale, pos.y + d.y * scale)


class FindConvexCornersTests(unittest.TestCase):
    def test_open_map_has_no_corners(self):
        grid = _grid()
        self.assertEqual(corners_mod._find_convex_corners(grid), [])

    def test_single_wall_cell_yields_its_four_corners(self):
        grid = _grid({(5, 5)})
        corners = corners_mod._find_convex_corners(grid)
        positions = {(c.x, c.y) for c, _ in corners}
        self.assertEqual(positions, {(5.0, 5.0), (5.0, 6.0), (6.0, 5.0), (6.0, 6.0)})

    def test_away_direction_points_into_open_space(self):
        grid = _grid({(5, 5)})
        corners = {(c.x, c.y): away for c, away in corners_mod._find_convex_corners(grid)}
        # The wall occupies quadrant (+x, +y) of corner (5, 5) -- away must
        # point into the opposite, open quadrant.
        away = corners[(5.0, 5.0)]
        self.assertLess(away.x, 0.0)
        self.assertLess(away.y, 0.0)

    def test_map_boundary_alone_produces_no_corners(self):
        # Every lattice point on the map edge has >= 2 "solid" (out-of-bounds)
        # quadrants, so the border itself must never register as a corner --
        # only an actual wall poking into the open map should.
        grid = _grid()
        corners = corners_mod._find_convex_corners(grid)
        self.assertEqual(corners, [])


class WidenPathTests(unittest.TestCase):
    def setUp(self):
        corners_mod._cache.pop("wall_corners", None)

    def tearDown(self):
        corners_mod._cache.pop("wall_corners", None)

    def test_no_wall_returns_original_target(self):
        grid = _grid()
        target = Vec2(11.0, 11.0)
        routed = corners_mod._widen_path(Vec2(0.0, 0.0), target, grid, buffer=0.75, detect_radius=3.0)
        self.assertEqual((routed.x, routed.y), (target.x, target.y))

    def test_far_from_any_corner_returns_original_target(self):
        grid = _grid({(5, 5)})
        target = Vec2(2.0, 0.0)
        routed = corners_mod._widen_path(Vec2(0.0, 0.0), target, grid, buffer=0.75, detect_radius=3.0)
        self.assertEqual((routed.x, routed.y), (target.x, target.y))

    def test_path_grazing_a_corner_gets_routed_around_it(self):
        grid = _grid({(5, 5)})
        # Within `detect_radius` (3.0) of the corner already, unlike the
        # multi-tick test below which starts far away and walks in.
        bot = Vec2(3.0, 3.0)
        target = Vec2(11.0, 11.0)
        routed = corners_mod._widen_path(bot, target, grid, buffer=0.75, detect_radius=3.0)
        self.assertNotEqual((routed.x, routed.y), (target.x, target.y))
        # Routed point should sit further from the wall corner than the
        # bot's bare radius would -- i.e. roughly `buffer` away from (5, 5).
        self.assertAlmostEqual(Vec2(5.0, 5.0).dist(routed), 0.75, places=2)

    def test_bot_already_past_the_corner_is_not_pinned_there(self):
        # Regression: once the bot has arrived at the widened waypoint, the
        # straight line onward to the real target can still graze the same
        # corner within `buffer`, since the corner sits almost exactly on
        # that line. A naive `dist(bot, corner) <= buffer` check flip-flops
        # right at that boundary (the bot lands within float noise of
        # `buffer`, some ticks just inside, some just outside) and the bot
        # freezes there forever instead of continuing on.
        grid = _grid({(5, 5)})
        target = Vec2(11.0, 11.0)
        buffer = 0.75
        # Land exactly on the widened waypoint, with the float noise this
        # bug actually hit in practice (fractionally *past* `buffer`).
        bot = Vec2(4.469669818878174, 4.469669818878174)
        routed = corners_mod._widen_path(bot, target, grid, buffer=buffer, detect_radius=3.0)
        self.assertNotAlmostEqual(routed.x, bot.x, places=3)
        self.assertNotAlmostEqual(routed.y, bot.y, places=3)

    def test_multi_tick_walk_reaches_the_real_target(self):
        # Full regression for the freeze: repeatedly aim at whatever
        # `_widen_path` returns and take a small step towards it, exactly
        # like the engine moving a bot towards its `MoveAction` each tick.
        # Before the fix this converged to a fixed point at the corner
        # detour and never got closer to `target`.
        grid = _grid({(5, 5)})
        target = Vec2(11.0, 11.0)
        pos = Vec2(0.0, 0.0)
        buffer = 0.75
        detect_radius = buffer * 4.0
        for _ in range(500):
            aim = corners_mod._widen_path(pos, target, grid, buffer, detect_radius)
            pos = _step_towards(pos, aim, step_len=0.15)
        self.assertLess(pos.dist(target), 0.2)


class SteerTests(unittest.TestCase):
    def setUp(self):
        corners_mod._cache.pop("wall_corners", None)

    def tearDown(self):
        corners_mod._cache.pop("wall_corners", None)

    def test_delegates_to_navigate_to_and_move_bot_with_routed_target(self):
        grid = _grid()  # no walls -> routed target == target
        conf = type("Conf", (), {"bot": FakeBotConfig()})()
        bot_pos = Vec2(0.0, 0.0)
        target = Vec2(4.0, 0.0)
        with patch.object(corners_mod, "navigate_to", return_value=Vec2(1.0, 0.0)) as nav, \
             patch.object(corners_mod, "move_bot", return_value="MOVE") as mv:
            result = corners_mod._steer(bot_pos, target, grid, conf)
        nav.assert_called_once()
        called_bot_pos, called_target = nav.call_args[0]
        self.assertEqual((called_bot_pos.x, called_bot_pos.y), (bot_pos.x, bot_pos.y))
        self.assertEqual((called_target.x, called_target.y), (target.x, target.y))
        mv.assert_called_once_with(Vec2(1.0, 0.0))
        self.assertEqual(result, "MOVE")

    def test_never_returns_a_zero_length_aim_when_target_is_reachable(self):
        # A `_steer` call that ends up aiming a moving bot back at itself is
        # exactly the "not moving" symptom -- guard the wrapper end-to-end.
        grid = _grid({(5, 5)})
        conf = type("Conf", (), {"bot": FakeBotConfig()})()
        bot_pos = Vec2(0.0, 0.0)
        target = Vec2(11.0, 11.0)
        with patch.object(corners_mod, "navigate_to", side_effect=lambda a, b: b - a), \
             patch.object(corners_mod, "move_bot", side_effect=lambda v: v):
            aim = corners_mod._steer(bot_pos, target, grid, conf)
        self.assertGreater((aim.x ** 2 + aim.y ** 2) ** 0.5, 1e-6)


if __name__ == "__main__":
    unittest.main()
