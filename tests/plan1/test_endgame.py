"""Tests for strategy/plan1/endgame.py -- see plans/plan1/endgame.md."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core._generated.bindings import FleetAction  # noqa: E402
import strategy.plan1.endgame as endgame_mod  # noqa: E402
from tests.plan1.fakes import BotClass, FakeBot, FakeConf, FakeState, Vec2  # noqa: E402


class MaybeEndgameSelfDestructTests(unittest.TestCase):
    def test_self_destructs_weakest_extractor_when_far_behind(self):
        fleet = [
            FakeBot(0, Vec2(0, 0), class_=BotClass.Extractor, health=2.0),
            FakeBot(1, Vec2(0, 0), class_=BotClass.Extractor, health=1.0),
            FakeBot(2, Vec2(0, 0), class_=BotClass.Battle, health=1.0),
        ]
        # total 4.0 < 10 * 0.75 = 7.5 -- comfortably below the self-destruct threshold
        enemy = [FakeBot(10, Vec2(1, 1), class_=BotClass.Battle, health=10.0)]
        state = FakeState(fleet_me=fleet, fleet_other=enemy)
        conf = FakeConf()
        action = FleetAction.new()
        cache = {}
        endgame_mod._maybe_endgame_self_destruct(state, conf, action, cache)
        self.assertTrue(action.bots[1].self_destruct)   # the weakest extractor (id=1)
        self.assertFalse(action.bots[0].self_destruct)
        self.assertTrue(cache.get("endgame_sd_done"))

    def test_does_nothing_when_health_ratio_is_fine(self):
        fleet = [FakeBot(0, Vec2(0, 0), class_=BotClass.Extractor, health=10.0)]
        enemy = [FakeBot(10, Vec2(1, 1), class_=BotClass.Battle, health=10.0)]
        state = FakeState(fleet_me=fleet, fleet_other=enemy)
        conf = FakeConf()
        action = FleetAction.new()
        cache = {}
        endgame_mod._maybe_endgame_self_destruct(state, conf, action, cache)
        self.assertFalse(action.bots[0].self_destruct)
        self.assertNotIn("endgame_sd_done", cache)

    def test_does_nothing_when_enemy_already_wiped(self):
        fleet = [FakeBot(0, Vec2(0, 0), class_=BotClass.Extractor, health=1.0)]
        state = FakeState(fleet_me=fleet, fleet_other=[])
        conf = FakeConf()
        action = FleetAction.new()
        cache = {}
        endgame_mod._maybe_endgame_self_destruct(state, conf, action, cache)
        self.assertFalse(action.bots[0].self_destruct)
        self.assertNotIn("endgame_sd_done", cache)

    def test_does_nothing_when_no_extractors_left(self):
        fleet = [FakeBot(0, Vec2(0, 0), class_=BotClass.Battle, health=1.0)]
        enemy = [FakeBot(10, Vec2(1, 1), class_=BotClass.Battle, health=10.0)]
        state = FakeState(fleet_me=fleet, fleet_other=enemy)
        conf = FakeConf()
        action = FleetAction.new()
        cache = {}
        endgame_mod._maybe_endgame_self_destruct(state, conf, action, cache)
        self.assertFalse(action.bots[0].self_destruct)
        self.assertNotIn("endgame_sd_done", cache)


if __name__ == "__main__":
    unittest.main()
