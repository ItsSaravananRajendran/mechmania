"""Tests for strategy/plan1/mining_defense.py -- miner escorts and raiders."""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import strategy.plan1.mining_defense as mining_defense_mod  # noqa: E402
import strategy.plan1.walls as walls_mod  # noqa: E402
from tests.plan1.fakes import (  # noqa: E402
    BotClass, FakeBot, FakeConf, FakeDeposit, FakeMapConf, FakeState, Vec2,
)


def _all_free():
    return patch.object(walls_mod, "point_free", return_value=True)


def _empty_grid():
    return walls_mod.WallGrid.build(FakeMapConf(wall_cells=set()))


class PickMinerEscortsTests(unittest.TestCase):
    def test_takes_the_first_n_by_id(self):
        battles = [FakeBot(i, Vec2(0, 0)) for i in (5, 1, 3, 2, 4)]
        result = mining_defense_mod._pick_miner_escorts(battles, count=3)
        self.assertEqual(sorted(b.id for b in result), [1, 2, 3])

    def test_fewer_battles_than_escort_count_takes_them_all(self):
        battles = [FakeBot(i, Vec2(0, 0)) for i in (7, 2)]
        result = mining_defense_mod._pick_miner_escorts(battles, count=3)
        self.assertEqual(sorted(b.id for b in result), [2, 7])

    def test_empty_battles(self):
        self.assertEqual(mining_defense_mod._pick_miner_escorts([], count=3), [])

    def test_replacement_bot_inherits_the_escort_slot(self):
        # A dead escort (id 1) is replaced by a newly built bot (id 8) --
        # the *set* of escort ids should be stable in composition (still 3
        # of the lowest surviving ids), even though the specific bot that
        # fills the third slot changes.
        before = [FakeBot(i, Vec2(0, 0)) for i in (1, 2, 3, 4)]
        after = [FakeBot(i, Vec2(0, 0)) for i in (2, 3, 4, 8)]
        escorts_before = {b.id for b in mining_defense_mod._pick_miner_escorts(before, count=3)}
        escorts_after = {b.id for b in mining_defense_mod._pick_miner_escorts(after, count=3)}
        self.assertEqual(escorts_before, {1, 2, 3})
        self.assertEqual(escorts_after, {2, 3, 4})


class PickRaidersTests(unittest.TestCase):
    def test_roughly_a_third_by_id(self):
        battles = [FakeBot(i, Vec2(0, 0)) for i in range(9)]
        result = mining_defense_mod._pick_raiders(battles)
        self.assertEqual([b.id for b in result], [0, 1, 2])

    def test_empty_candidates(self):
        self.assertEqual(mining_defense_mod._pick_raiders([]), [])

    def test_single_candidate_stays_put(self):
        # Nobody to spare -- pulling the only remaining bot off would leave
        # the payload/formation line with nothing.
        battles = [FakeBot(0, Vec2(0, 0))]
        self.assertEqual(mining_defense_mod._pick_raiders(battles), [])

    def test_two_candidates_still_sends_one(self):
        battles = [FakeBot(i, Vec2(0, 0)) for i in (5, 1)]
        result = mining_defense_mod._pick_raiders(battles)
        self.assertEqual([b.id for b in result], [1])


class NearestThreatToDepositTests(unittest.TestCase):
    def test_returns_nearest_enemy_within_range(self):
        conf = FakeConf()
        deposit = FakeDeposit(Vec2(0.0, 0.0))
        enemies = [FakeBot(0, Vec2(5.0, 0.0)), FakeBot(1, Vec2(2.0, 0.0))]
        state = FakeState(deposit_me=deposit, fleet_other=enemies)
        result = mining_defense_mod._nearest_threat_to_deposit(state, conf)
        self.assertEqual(result.id, 1)

    def test_ignores_enemies_out_of_range(self):
        conf = FakeConf()  # default blaster_range=10.0
        deposit = FakeDeposit(Vec2(0.0, 0.0))
        enemies = [FakeBot(0, Vec2(50.0, 0.0))]
        state = FakeState(deposit_me=deposit, fleet_other=enemies)
        self.assertIsNone(mining_defense_mod._nearest_threat_to_deposit(state, conf))

    def test_no_enemies_at_all(self):
        conf = FakeConf()
        state = FakeState(deposit_me=FakeDeposit(Vec2(0.0, 0.0)), fleet_other=[])
        self.assertIsNone(mining_defense_mod._nearest_threat_to_deposit(state, conf))


class RaidTargetPosTests(unittest.TestCase):
    def test_heads_for_the_nearest_visible_enemy_extractor(self):
        state = FakeState(fleet_other=[
            FakeBot(0, Vec2(10.0, 0.0), class_=BotClass.Extractor),
            FakeBot(1, Vec2(1.0, 0.0), class_=BotClass.Extractor),
            FakeBot(2, Vec2(0.5, 0.0), class_=BotClass.Battle),
        ])
        raider = FakeBot(9, Vec2(0.0, 0.0))
        result = mining_defense_mod._raid_target_pos(state, raider)
        self.assertEqual((result.x, result.y), (1.0, 0.0))

    def test_falls_back_to_the_enemy_deposit_with_no_extractors_in_view(self):
        state = FakeState(
            fleet_other=[FakeBot(2, Vec2(0.5, 0.0), class_=BotClass.Battle)],
            deposit_other=FakeDeposit(Vec2(30.0, 30.0)),
        )
        raider = FakeBot(9, Vec2(0.0, 0.0))
        result = mining_defense_mod._raid_target_pos(state, raider)
        self.assertEqual((result.x, result.y), (30.0, 30.0))

    def test_no_enemies_at_all_falls_back_to_deposit(self):
        state = FakeState(fleet_other=[], deposit_other=FakeDeposit(Vec2(30.0, 30.0)))
        raider = FakeBot(9, Vec2(0.0, 0.0))
        result = mining_defense_mod._raid_target_pos(state, raider)
        self.assertEqual((result.x, result.y), (30.0, 30.0))


class ComputeGuardPositionsTests(unittest.TestCase):
    def test_empty_escorts(self):
        grid = _empty_grid()
        state = FakeState()
        conf = FakeConf()
        result = mining_defense_mod._compute_guard_positions([], state, Vec2(1, 0), conf, grid)
        self.assertEqual(result, {})

    def test_guards_sit_between_deposit_and_enemy(self):
        grid = _empty_grid()
        conf = FakeConf()
        deposit = FakeDeposit(Vec2(10.0, 10.0))
        state = FakeState(deposit_me=deposit)
        forward = Vec2(1.0, 0.0)  # towards the enemy
        escorts = [FakeBot(i, Vec2(0, 0)) for i in range(3)]
        with _all_free():
            result = mining_defense_mod._compute_guard_positions(escorts, state, forward, conf, grid)
        self.assertEqual(len(result), 3)
        for pos in result.values():
            # On the enemy-facing side of the deposit, not behind it.
            self.assertGreater((pos - deposit.pos).dot(forward), 0.0)

    def test_guard_slots_are_distinct(self):
        grid = _empty_grid()
        conf = FakeConf()
        state = FakeState(deposit_me=FakeDeposit(Vec2(10.0, 10.0)))
        escorts = [FakeBot(i, Vec2(0, 0)) for i in range(3)]
        with _all_free():
            result = mining_defense_mod._compute_guard_positions(escorts, state, Vec2(1, 0), conf, grid)
        coords = {(round(v.x, 4), round(v.y, 4)) for v in result.values()}
        self.assertEqual(len(coords), 3)


if __name__ == "__main__":
    unittest.main()
