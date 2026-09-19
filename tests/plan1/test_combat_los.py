"""Tests for strategy/plan1/combat_los.py -- see plans/plan1/combat_los.md."""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import strategy.plan1.combat_los as los_mod  # noqa: E402
from tests.plan1.fakes import FakeConf, FakeState, Vec2  # noqa: E402


def _patch_los(value):
    return patch.object(los_mod, "line_of_sight", return_value=value)


def _patch_seg_dist(fn):
    """fn(pos, a, b) -> float, so different obstacles can return different distances."""
    return patch.object(los_mod, "point_seg_dist", side_effect=fn)


class HasClearShotTests(unittest.TestCase):
    def test_false_when_wall_blocks(self):
        state = FakeState()
        conf = FakeConf()
        with _patch_los(False), _patch_seg_dist(lambda *_: 999.0):
            self.assertFalse(los_mod._has_clear_shot(Vec2(0, 0), Vec2(5, 5), state, conf))

    def test_false_when_own_deposit_blocks(self):
        state = FakeState()
        state.deposit_me.pos = Vec2(2.0, 2.0)
        conf = FakeConf()
        conf.deposit.radius = 0.5

        def seg_dist(pos, a, b):
            return 0.1 if pos is state.deposit_me.pos else 999.0

        with _patch_los(True), _patch_seg_dist(seg_dist):
            self.assertFalse(los_mod._has_clear_shot(Vec2(0, 0), Vec2(5, 5), state, conf))

    def test_false_when_enemy_deposit_blocks(self):
        state = FakeState()
        conf = FakeConf()
        conf.deposit.radius = 0.5

        def seg_dist(pos, a, b):
            return 0.1 if pos is state.deposit_other.pos else 999.0

        with _patch_los(True), _patch_seg_dist(seg_dist):
            self.assertFalse(los_mod._has_clear_shot(Vec2(0, 0), Vec2(5, 5), state, conf))

    def test_false_when_payload_blocks(self):
        state = FakeState(payload=Vec2(3.0, 3.0))
        conf = FakeConf()
        conf.payload.radius = 0.75

        def seg_dist(pos, a, b):
            return 0.1 if pos is state._payload else 999.0

        with _patch_los(True), _patch_seg_dist(seg_dist):
            self.assertFalse(los_mod._has_clear_shot(Vec2(0, 0), Vec2(5, 5), state, conf))

    def test_true_when_nothing_blocks(self):
        state = FakeState()
        conf = FakeConf()
        with _patch_los(True), _patch_seg_dist(lambda *_: 999.0):
            self.assertTrue(los_mod._has_clear_shot(Vec2(0, 0), Vec2(5, 5), state, conf))


class BlockingObstacleTests(unittest.TestCase):
    def test_none_when_nothing_blocks(self):
        state = FakeState()
        conf = FakeConf()
        with _patch_seg_dist(lambda *_: 999.0):
            self.assertIsNone(los_mod._blocking_obstacle(Vec2(0, 0), Vec2(5, 5), state, conf))

    def test_returns_the_single_blocker(self):
        state = FakeState()
        conf = FakeConf()
        conf.deposit.radius = 0.5

        def seg_dist(pos, a, b):
            return 0.1 if pos is state.deposit_me.pos else 999.0

        with _patch_seg_dist(seg_dist):
            result = los_mod._blocking_obstacle(Vec2(0, 0), Vec2(5, 5), state, conf)
        self.assertIsNotNone(result)
        pos, radius = result
        self.assertIs(pos, state.deposit_me.pos)
        self.assertEqual(radius, 0.5)

    def test_picks_the_one_closest_to_the_shooter_when_two_qualify(self):
        state = FakeState()
        state.deposit_me.pos = Vec2(1.0, 0.0)
        state.deposit_other.pos = Vec2(9.0, 0.0)
        conf = FakeConf()
        conf.deposit.radius = 0.5
        shooter = Vec2(0.0, 0.0)

        def seg_dist(pos, a, b):
            return 0.1  # both "qualify" as blocking

        with _patch_seg_dist(seg_dist):
            pos, _ = los_mod._blocking_obstacle(shooter, Vec2(10.0, 0.0), state, conf)
        self.assertIs(pos, state.deposit_me.pos)  # nearer to the shooter at (0,0)


class FlankPointTests(unittest.TestCase):
    def test_prefers_the_bots_current_side(self):
        bot_pos = Vec2(0.0, 1.0)  # "above" the bot->target line
        target_pos = Vec2(10.0, 0.0)
        obstacle_pos = Vec2(5.0, 0.0)
        with patch.object(los_mod, "point_free", return_value=True):
            p = los_mod._flank_point(bot_pos, target_pos, obstacle_pos, clearance=2.0)
        # Should land on the same side of the travel line as the bot (y > 0-ish side).
        direction = (target_pos - bot_pos).normalize_or_zero()
        perp = direction.rotate_deg(90.0)
        side = (bot_pos - obstacle_pos).dot(perp)
        chosen_side = (p - obstacle_pos).dot(perp)
        self.assertGreater(side * chosen_side, 0.0)

    def test_falls_back_to_other_side_if_preferred_blocked(self):
        bot_pos = Vec2(0.0, 1.0)
        target_pos = Vec2(10.0, 0.0)
        obstacle_pos = Vec2(5.0, 0.0)
        direction = (target_pos - bot_pos).normalize_or_zero()
        perp = direction.rotate_deg(90.0)
        preferred_side = 1.0 if (bot_pos - obstacle_pos).dot(perp) >= 0.0 else -1.0

        def fake_point_free(p):
            side = (p - obstacle_pos).dot(perp)
            # Block whichever point is on the preferred side, leave the other free.
            return (side >= 0) != (preferred_side >= 0)

        with patch.object(los_mod, "point_free", side_effect=fake_point_free):
            p = los_mod._flank_point(bot_pos, target_pos, obstacle_pos, clearance=2.0)
        chosen_side = (p - obstacle_pos).dot(perp)
        self.assertLess(preferred_side * chosen_side, 0.0)

    def test_falls_back_to_target_if_both_sides_blocked(self):
        bot_pos = Vec2(0.0, 1.0)
        target_pos = Vec2(10.0, 0.0)
        obstacle_pos = Vec2(5.0, 0.0)
        with patch.object(los_mod, "point_free", return_value=False):
            p = los_mod._flank_point(bot_pos, target_pos, obstacle_pos, clearance=2.0)
        self.assertEqual((p.x, p.y), (target_pos.x, target_pos.y))

    def test_degenerate_same_position_does_not_crash(self):
        with patch.object(los_mod, "point_free", return_value=True):
            p = los_mod._flank_point(Vec2(5.0, 5.0), Vec2(5.0, 5.0), Vec2(5.0, 0.0), clearance=1.0)
        self.assertIsInstance(p, Vec2)


if __name__ == "__main__":
    unittest.main()
