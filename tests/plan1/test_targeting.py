"""Tests for strategy/plan1/targeting.py -- see plans/plan1/targeting.md."""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import strategy.plan1.targeting as targeting_mod  # noqa: E402
from tests.plan1.fakes import BotClass, FakeBot, FakeConf, FakeState, Vec2  # noqa: E402


def _always_clear(*_a, **_k):
    return True


class TargetPriorityTests(unittest.TestCase):
    def test_healer_outranks_extractor_outranks_battle(self):
        conf = FakeConf()
        healer = FakeBot(0, Vec2(0, 0), class_=BotClass.Healer, health=10.0)
        extractor = FakeBot(1, Vec2(0, 0), class_=BotClass.Extractor, health=10.0)
        battle = FakeBot(2, Vec2(0, 0), class_=BotClass.Battle, health=10.0)
        ph = targeting_mod._target_priority(healer, conf)
        pe = targeting_mod._target_priority(extractor, conf)
        pb = targeting_mod._target_priority(battle, conf)
        self.assertGreater(ph, pe)
        self.assertGreater(pe, pb)

    def test_lower_health_ranks_higher_within_a_class(self):
        conf = FakeConf()
        healthy = FakeBot(0, Vec2(0, 0), class_=BotClass.Battle, health=9.0)
        hurt = FakeBot(1, Vec2(0, 0), class_=BotClass.Battle, health=2.0)
        self.assertGreater(
            targeting_mod._target_priority(hurt, conf),
            targeting_mod._target_priority(healthy, conf),
        )

    def test_low_hp_bonus_applies_at_or_under_30_percent(self):
        conf = FakeConf()
        conf.bot.health = 10.0
        just_above = FakeBot(0, Vec2(0, 0), class_=BotClass.Battle, health=3.01)
        just_at = FakeBot(1, Vec2(0, 0), class_=BotClass.Battle, health=3.0)
        p_above = targeting_mod._target_priority(just_above, conf)
        p_at = targeting_mod._target_priority(just_at, conf)
        # The bonus (+50) should more than make up for the tiny health gap.
        self.assertGreater(p_at, p_above)


class AssignBattleTargetsTests(unittest.TestCase):
    def test_empty_battles_returns_empty_dict(self):
        state = FakeState(fleet_other=[FakeBot(0, Vec2(1, 1), class_=BotClass.Battle)])
        conf = FakeConf()
        with patch.object(targeting_mod, "_has_clear_shot", side_effect=_always_clear):
            result = targeting_mod._assign_battle_targets([], state, conf)
        self.assertEqual(result, {})

    def test_no_enemies_maps_every_bot_to_none(self):
        battles = [FakeBot(0, Vec2(0, 0), class_=BotClass.Battle)]
        state = FakeState(fleet_other=[])
        conf = FakeConf()
        result = targeting_mod._assign_battle_targets(battles, state, conf)
        self.assertEqual(result, {0: None})

    def test_two_bots_two_reachable_enemies_get_different_targets(self):
        conf = FakeConf()
        b0 = FakeBot(0, Vec2(0.0, 0.0), class_=BotClass.Battle)
        b1 = FakeBot(1, Vec2(0.1, 0.1), class_=BotClass.Battle)
        e0 = FakeBot(10, Vec2(1.0, 0.0), class_=BotClass.Battle, health=10.0)
        e1 = FakeBot(11, Vec2(1.0, 0.2), class_=BotClass.Healer, health=10.0)  # higher priority
        state = FakeState(fleet_other=[e0, e1])
        with patch.object(targeting_mod, "_has_clear_shot", side_effect=_always_clear):
            result = targeting_mod._assign_battle_targets([b0, b1], state, conf)
        self.assertEqual(set(result.values()), {10, 11})  # not both on the healer

    def test_invulnerable_enemy_still_assigned_when_only_option(self):
        conf = FakeConf()
        bot = FakeBot(0, Vec2(0.0, 0.0), class_=BotClass.Battle)
        enemy = FakeBot(10, Vec2(1.0, 0.0), class_=BotClass.Battle, invulnerable_until_tick=999)
        state = FakeState(tick=5, fleet_other=[enemy])
        with patch.object(targeting_mod, "_has_clear_shot", side_effect=_always_clear):
            result = targeting_mod._assign_battle_targets([bot], state, conf)
        self.assertEqual(result, {0: 10})

    def test_prefers_hittable_enemy_over_invulnerable_one(self):
        conf = FakeConf()
        bot = FakeBot(0, Vec2(0.0, 0.0), class_=BotClass.Battle)
        hittable = FakeBot(10, Vec2(1.0, 0.0), class_=BotClass.Battle, invulnerable_until_tick=0)
        invuln = FakeBot(11, Vec2(1.0, 0.5), class_=BotClass.Healer, invulnerable_until_tick=999)
        state = FakeState(tick=5, fleet_other=[hittable, invuln])
        with patch.object(targeting_mod, "_has_clear_shot", side_effect=_always_clear):
            result = targeting_mod._assign_battle_targets([bot], state, conf)
        # Even though the healer is higher class-priority, the -10000 invulnerability
        # penalty should push the hittable battle bot ahead of it.
        self.assertEqual(result, {0: 10})

    def test_out_of_range_enemy_never_claimed_by_greedy_pass(self):
        conf = FakeConf()
        conf.bot.blaster_range = 5.0
        bot = FakeBot(0, Vec2(0.0, 0.0), class_=BotClass.Battle)
        far_enemy = FakeBot(10, Vec2(100.0, 0.0), class_=BotClass.Battle)
        state = FakeState(fleet_other=[far_enemy])
        # Fallback path still assigns it (nearest by straight-line, the only option).
        result = targeting_mod._assign_battle_targets([bot], state, conf)
        self.assertEqual(result, {0: 10})

    def test_no_target_left_with_none_when_enemies_exist(self):
        # Every bot should get *some* target when enemies exist, even if none
        # are in range or have a clear shot -- the nearest-by-distance fallback.
        conf = FakeConf()
        conf.bot.blaster_range = 1.0
        bots = [FakeBot(0, Vec2(0.0, 0.0), class_=BotClass.Battle),
                FakeBot(1, Vec2(50.0, 50.0), class_=BotClass.Battle)]
        enemies = [FakeBot(10, Vec2(5.0, 5.0), class_=BotClass.Battle),
                   FakeBot(11, Vec2(60.0, 60.0), class_=BotClass.Battle)]
        state = FakeState(fleet_other=enemies)
        result = targeting_mod._assign_battle_targets(bots, state, conf)
        self.assertNotIn(None, result.values())
        self.assertEqual(result[0], 10)
        self.assertEqual(result[1], 11)

    def test_does_not_call_path_length(self):
        # Regression guard: the fallback loop must use straight-line distance,
        # not `path_length` (all-pairs `path_length` is one of the two most
        # expensive things a strategy can do per tick).
        conf = FakeConf()
        conf.bot.blaster_range = 0.0  # nobody in range -> everyone hits the fallback
        bots = [FakeBot(i, Vec2(float(i), 0.0), class_=BotClass.Battle) for i in range(4)]
        enemies = [FakeBot(10 + i, Vec2(float(i), 5.0), class_=BotClass.Battle) for i in range(4)]
        state = FakeState(fleet_other=enemies)
        self.assertFalse(hasattr(targeting_mod, "path_length"))
        targeting_mod._assign_battle_targets(bots, state, conf)  # should not raise / not call path_length


if __name__ == "__main__":
    unittest.main()
