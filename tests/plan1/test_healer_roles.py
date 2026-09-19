"""Tests for strategy/plan1/healer_roles.py -- the goal/miner/raid healer split."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import strategy.plan1.healer_roles as healer_roles_mod  # noqa: E402
from tests.plan1.fakes import FakeBot, Vec2  # noqa: E402


class PickHealerGroupsTests(unittest.TestCase):
    def test_five_healers_split_three_one_one(self):
        healers = [FakeBot(i, Vec2(0, 0)) for i in range(5)]
        goal, miner, raid = healer_roles_mod._pick_healer_groups(healers)
        self.assertEqual([h.id for h in goal], [0, 1, 2])
        self.assertEqual([h.id for h in miner], [3])
        self.assertEqual([h.id for h in raid], [4])

    def test_ordering_is_by_id_not_input_order(self):
        healers = [FakeBot(i, Vec2(0, 0)) for i in (4, 0, 3, 1, 2)]
        goal, miner, raid = healer_roles_mod._pick_healer_groups(healers)
        self.assertEqual([h.id for h in goal], [0, 1, 2])
        self.assertEqual([h.id for h in miner], [3])
        self.assertEqual([h.id for h in raid], [4])

    def test_more_than_five_healers_extras_go_nowhere_special(self):
        # Only the first 5 (by id) get one of the three fixed roles; extras
        # beyond that aren't part of this split at all -- the caller decides
        # what to do with whoever's left (in practice: nobody's left, since
        # `_recompute_assignments` only calls this with the full roster and
        # everyone gets *some* assignment either way).
        healers = [FakeBot(i, Vec2(0, 0)) for i in range(7)]
        goal, miner, raid = healer_roles_mod._pick_healer_groups(healers)
        self.assertEqual(len(goal) + len(miner) + len(raid), 5)

    def test_empty_roster(self):
        goal, miner, raid = healer_roles_mod._pick_healer_groups([])
        self.assertEqual((goal, miner, raid), ([], [], []))

    def test_fewer_than_three_healers_all_go_to_goal(self):
        healers = [FakeBot(i, Vec2(0, 0)) for i in (5, 1)]
        goal, miner, raid = healer_roles_mod._pick_healer_groups(healers)
        self.assertEqual([h.id for h in goal], [1, 5])
        self.assertEqual(miner, [])
        self.assertEqual(raid, [])

    def test_four_healers_goal_then_miner_before_raid(self):
        # Goal (3) fills completely before miner gets its one slot; raid
        # only gets a healer once both higher-priority groups are full.
        healers = [FakeBot(i, Vec2(0, 0)) for i in range(4)]
        goal, miner, raid = healer_roles_mod._pick_healer_groups(healers)
        self.assertEqual([h.id for h in goal], [0, 1, 2])
        self.assertEqual([h.id for h in miner], [3])
        self.assertEqual(raid, [])

    def test_replacement_healer_backfills_by_id_like_battle_roles(self):
        # Healer 1 (a goal healer) dies and is replaced by a freshly built
        # healer with a higher id -- healer 3 (previously the miner escort's
        # healer) shifts up into goal, and the new healer lands wherever the
        # id ordering now puts it, mirroring `_pick_miner_escorts`.
        before = [FakeBot(i, Vec2(0, 0)) for i in (0, 1, 2, 3, 4)]
        after = [FakeBot(i, Vec2(0, 0)) for i in (0, 2, 3, 4, 9)]
        goal_before, miner_before, _ = healer_roles_mod._pick_healer_groups(before)
        goal_after, miner_after, raid_after = healer_roles_mod._pick_healer_groups(after)
        self.assertEqual({h.id for h in goal_before}, {0, 1, 2})
        self.assertEqual({h.id for h in miner_before}, {3})
        self.assertEqual({h.id for h in goal_after}, {0, 2, 3})
        self.assertEqual({h.id for h in miner_after}, {4})
        self.assertEqual({h.id for h in raid_after}, {9})


if __name__ == "__main__":
    unittest.main()
