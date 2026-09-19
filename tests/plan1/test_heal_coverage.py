"""Tests for strategy/plan1/heal_coverage.py -- squad healer positioning."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import strategy.plan1.heal_coverage as heal_coverage_mod  # noqa: E402
from tests.plan1.fakes import FakeBot, FakeConf, Vec2  # noqa: E402


class OutOfRangeCountTests(unittest.TestCase):
    def test_all_within_range(self):
        allies = [FakeBot(0, Vec2(0, 0)), FakeBot(1, Vec2(1, 0))]
        self.assertEqual(heal_coverage_mod._out_of_range_count(Vec2(0, 0), allies, heal_range=3.0), 0)

    def test_some_out_of_range(self):
        allies = [FakeBot(0, Vec2(0, 0)), FakeBot(1, Vec2(10, 0))]
        self.assertEqual(heal_coverage_mod._out_of_range_count(Vec2(0, 0), allies, heal_range=3.0), 1)

    def test_exactly_at_range_counts_as_in_range(self):
        allies = [FakeBot(0, Vec2(3, 0))]
        self.assertEqual(heal_coverage_mod._out_of_range_count(Vec2(0, 0), allies, heal_range=3.0), 0)


class BestHealAnchorTests(unittest.TestCase):
    def test_single_ally_is_its_own_anchor(self):
        allies = [FakeBot(0, Vec2(5, 5))]
        anchor, out = heal_coverage_mod._best_heal_anchor(allies, heal_range=3.0)
        self.assertEqual((anchor.x, anchor.y), (5.0, 5.0))
        self.assertEqual(out, 0)

    def test_tight_cluster_fully_covered_from_any_member(self):
        allies = [FakeBot(i, Vec2(i * 0.5, 0.0)) for i in range(4)]  # span 1.5, well under range
        anchor, out = heal_coverage_mod._best_heal_anchor(allies, heal_range=3.0)
        self.assertEqual(out, 0)

    def test_picks_the_position_that_covers_the_most(self):
        # Bots 0/1/2 clustered near the origin, bot 3 far away -- standing
        # at 0/1/2's position covers 3 of the 4 (missing only the far one);
        # standing at bot 3's position only covers itself.
        allies = [
            FakeBot(0, Vec2(0.0, 0.0)),
            FakeBot(1, Vec2(1.0, 0.0)),
            FakeBot(2, Vec2(-1.0, 0.0)),
            FakeBot(3, Vec2(100.0, 0.0)),
        ]
        anchor, out = heal_coverage_mod._best_heal_anchor(allies, heal_range=3.0)
        self.assertEqual(out, 1)
        self.assertNotEqual((round(anchor.x), round(anchor.y)), (100, 0))

    def test_ties_keep_the_first_candidate_checked(self):
        allies = [FakeBot(0, Vec2(0.0, 0.0)), FakeBot(1, Vec2(100.0, 0.0))]
        # Each only covers itself (both equally far apart) -- first wins.
        anchor, out = heal_coverage_mod._best_heal_anchor(allies, heal_range=3.0)
        self.assertEqual((anchor.x, anchor.y), (0.0, 0.0))
        self.assertEqual(out, 1)


class HealPositionTests(unittest.TestCase):
    def test_nudges_behind_when_coverage_is_unaffected(self):
        allies = [FakeBot(0, Vec2(5.0, 5.0))]
        conf = FakeConf()
        forward = Vec2(1.0, 0.0)
        pos = heal_coverage_mod._heal_position(allies, forward, conf)
        self.assertLess(pos.x, 5.0)  # stepped back along -forward
        self.assertEqual(pos.y, 5.0)
        self.assertEqual(heal_coverage_mod._out_of_range_count(pos, allies, conf.bot.base_heal_range), 0)

    def test_small_nudge_that_stays_within_the_cap_is_still_applied(self):
        # Coverage goes from perfect (0 out) to imperfect (1 out) once
        # nudged, but 1 is still within `MAX_ATTACKERS_OUT_OF_HEAL_RANGE` --
        # the safety nudge (matching every other healer placement in this
        # strategy) wins as long as it doesn't cross that ceiling.
        conf = FakeConf()
        heal_range = conf.bot.base_heal_range
        forward = Vec2(1.0, 0.0)
        anchor_bot = FakeBot(0, Vec2(0.0, 0.0))
        edge_bot = FakeBot(1, Vec2(heal_range, 0.0))
        allies = [anchor_bot, edge_bot]
        pos = heal_coverage_mod._heal_position(allies, forward, conf)
        self.assertLess(pos.x, 0.0)

    def test_does_not_nudge_past_the_out_of_range_cap(self):
        # Already at the 2-straggler cap from two permanently-out-of-range
        # allies -- nudging back and pushing a *third* (the edge bot, sitting
        # exactly at heal_range) out too would exceed the cap, so the nudge
        # must be rejected and the un-nudged anchor used instead.
        conf = FakeConf()
        heal_range = conf.bot.base_heal_range
        forward = Vec2(1.0, 0.0)
        anchor_bot = FakeBot(0, Vec2(0.0, 0.0))
        edge_bot = FakeBot(1, Vec2(heal_range, 0.0))
        far1 = FakeBot(2, Vec2(0.0, 1000.0))
        far2 = FakeBot(3, Vec2(0.0, -1000.0))
        allies = [anchor_bot, edge_bot, far1, far2]
        pos = heal_coverage_mod._heal_position(allies, forward, conf)
        self.assertEqual((pos.x, pos.y), (0.0, 0.0))
        self.assertEqual(heal_coverage_mod._out_of_range_count(pos, allies, heal_range), 2)

    def test_large_spread_squad_caps_out_of_range_at_the_configured_max(self):
        conf = FakeConf()
        forward = Vec2(1.0, 0.0)
        # 6 bots spread far apart (pairwise distances all >> heal_range) --
        # no single point can cover more than one of them.
        allies = [FakeBot(i, Vec2(i * 50.0, 0.0)) for i in range(6)]
        pos = heal_coverage_mod._heal_position(allies, forward, conf)
        out = heal_coverage_mod._out_of_range_count(pos, allies, conf.bot.base_heal_range)
        # Best achievable here is 5 out of 6 (only the anchor's own bot is
        # ever in range) -- more than MAX_ATTACKERS_OUT_OF_HEAL_RANGE, but
        # the function must still return its best effort, not crash or
        # silently do something worse.
        self.assertEqual(out, 5)


if __name__ == "__main__":
    unittest.main()
