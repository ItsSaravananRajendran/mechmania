"""Tests for strategy/plan1/walls.py -- see plans/plan1/walls.md."""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import strategy.plan1.walls as walls_mod  # noqa: E402
from tests.plan1.fakes import FakeMapConf, Vec2  # noqa: E402


class WallGridBuildTests(unittest.TestCase):
    def test_marks_wall_tiles(self):
        conf = FakeMapConf(wall_cells={(5, 5), (5, 6)})
        grid = walls_mod.WallGrid.build(conf)
        self.assertTrue(grid.is_wall_rc(5, 5))
        self.assertTrue(grid.is_wall_rc(5, 6))

    def test_leaves_other_tiles_empty(self):
        conf = FakeMapConf(wall_cells={(5, 5)})
        grid = walls_mod.WallGrid.build(conf)
        self.assertFalse(grid.is_wall_rc(0, 0))
        self.assertFalse(grid.is_wall_rc(5, 4))
        self.assertFalse(grid.is_wall_rc(4, 5))

    def test_out_of_bounds_counts_as_wall(self):
        conf = FakeMapConf(wall_cells=set())
        grid = walls_mod.WallGrid.build(conf)
        self.assertTrue(grid.is_wall_rc(-1, 0))
        self.assertTrue(grid.is_wall_rc(0, -1))
        self.assertTrue(grid.is_wall_rc(grid.size, 0))
        self.assertTrue(grid.is_wall_rc(0, grid.size))

    def test_is_wall_at_uses_x_then_y(self):
        # A cell that's a wall at (x=3, y=7) but not at the transposed
        # (x=7, y=3) -- if `is_wall_at` ever indexed (y, x) instead of
        # (x, y), this would come back inverted.
        conf = FakeMapConf(wall_cells={(3, 7)})
        grid = walls_mod.WallGrid.build(conf)
        self.assertTrue(grid.is_wall_at(Vec2(3.4, 7.9)))
        self.assertFalse(grid.is_wall_at(Vec2(7.4, 3.9)))

    def test_is_wall_at_truncates_not_rounds(self):
        conf = FakeMapConf(wall_cells={(5, 5)})
        grid = walls_mod.WallGrid.build(conf)
        # 5.99 truncates to tile 5, still inside the wall tile's [5, 6) span.
        self.assertTrue(grid.is_wall_at(Vec2(5.99, 5.01)))
        # 6.0 truncates to tile 6, outside the wall tile.
        self.assertFalse(grid.is_wall_at(Vec2(6.0, 5.5)))


class GetWallGridCacheTests(unittest.TestCase):
    def setUp(self):
        walls_mod._cache.clear()

    def tearDown(self):
        walls_mod._cache.clear()

    def test_caches_across_calls(self):
        conf = FakeMapConf(wall_cells={(1, 1)})
        first = walls_mod._get_wall_grid(conf)
        second = walls_mod._get_wall_grid(conf)
        self.assertIs(first, second)

    def test_builds_only_once(self):
        conf = FakeMapConf(wall_cells={(1, 1)})
        with patch.object(walls_mod.WallGrid, "build", wraps=walls_mod.WallGrid.build) as build:
            walls_mod._get_wall_grid(conf)
            walls_mod._get_wall_grid(conf)
            walls_mod._get_wall_grid(conf)
        build.assert_called_once()


class IsSlotFreeTests(unittest.TestCase):
    def test_false_when_on_wall_grid_tile(self):
        conf = FakeMapConf(wall_cells={(5, 5)})
        grid = walls_mod.WallGrid.build(conf)
        with patch.object(walls_mod, "point_free", return_value=True):
            self.assertFalse(walls_mod._is_slot_free(grid, Vec2(5.5, 5.5)))

    def test_false_when_engine_point_free_says_no(self):
        # Free on the coarse grid (no wall tile), but the engine's own
        # point_free -- which accounts for the bot's hull -- says no (e.g.
        # right at a wall's edge).
        conf = FakeMapConf(wall_cells=set())
        grid = walls_mod.WallGrid.build(conf)
        with patch.object(walls_mod, "point_free", return_value=False):
            self.assertFalse(walls_mod._is_slot_free(grid, Vec2(1.0, 1.0)))

    def test_true_when_both_checks_pass(self):
        conf = FakeMapConf(wall_cells=set())
        grid = walls_mod.WallGrid.build(conf)
        with patch.object(walls_mod, "point_free", return_value=True):
            self.assertTrue(walls_mod._is_slot_free(grid, Vec2(1.0, 1.0)))

    def test_wall_grid_short_circuits_before_calling_point_free(self):
        conf = FakeMapConf(wall_cells={(5, 5)})
        grid = walls_mod.WallGrid.build(conf)
        with patch.object(walls_mod, "point_free") as point_free:
            walls_mod._is_slot_free(grid, Vec2(5.5, 5.5))
        point_free.assert_not_called()


if __name__ == "__main__":
    unittest.main()
