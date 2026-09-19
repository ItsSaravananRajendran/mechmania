"""Tests for strategy/plan1/fabricator.py -- see plans/plan1/fabricator.md."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import strategy.plan1.fabricator as fabricator_mod  # noqa: E402
from tests.plan1.fakes import BotClass, FakeBot, FakeConf, FakeDeposit, FakeState, Vec2  # noqa: E402


def _fleet_of(n, cls=BotClass.Battle, start_id=0):
    return [FakeBot(start_id + i, Vec2(0, 0), class_=cls) for i in range(n)]


class BootstrapTests(unittest.TestCase):
    def test_builds_extractor_when_slot_0_missing(self):
        state = FakeState(fleet_me=[FakeBot(1, Vec2(0, 0), class_=BotClass.Battle)])
        conf = FakeConf()
        result = fabricator_mod._compute_fabricator_next(state, conf, {})
        self.assertEqual(result, int(BotClass.Extractor))


class SlotDenialTests(unittest.TestCase):
    """Regression coverage for the bitmask-vs-count bug: `extractors.me`/
    `.other` are per-bot-id bitmasks (`1 << id`), not counts."""

    def _fleet_with_full_miner_crew(self):
        # 3 extractors (a full `MAX_ACTIVE_MINERS` crew, so the "keep miners
        # topped up" rule below doesn't shadow the bitmask logic these tests
        # are actually about) + 7 Battle bots, 10 total.
        fleet = _fleet_of(10, start_id=0)
        fleet[0] = FakeBot(0, Vec2(0, 0), class_=BotClass.Extractor)
        fleet[1] = FakeBot(1, Vec2(0, 0), class_=BotClass.Extractor)
        fleet[2] = FakeBot(2, Vec2(0, 0), class_=BotClass.Extractor)
        return fleet

    def test_fewer_bits_set_builds_extractor_even_if_numerically_larger(self):
        # We hold bit 2 (value 4, one slot); enemy holds bits 0+1 (value 3,
        # two slots). A raw integer compare (4 < 3 -> False) would wrongly
        # conclude we're ahead and build a Battle instead.
        fleet = self._fleet_with_full_miner_crew()
        deposit = FakeDeposit(Vec2(2, 2), extractors_me=0b100, extractors_other=0b011)
        state = FakeState(fleet_me=fleet, deposit_me=deposit)
        conf = FakeConf()
        result = fabricator_mod._compute_fabricator_next(state, conf, {})
        self.assertEqual(result, int(BotClass.Extractor))

    def test_more_bits_set_builds_battle(self):
        fleet = self._fleet_with_full_miner_crew()
        deposit = FakeDeposit(Vec2(2, 2), extractors_me=0b011, extractors_other=0b001)
        state = FakeState(fleet_me=fleet, deposit_me=deposit)
        conf = FakeConf()
        result = fabricator_mod._compute_fabricator_next(state, conf, {})
        self.assertEqual(result, int(BotClass.Battle))

    def test_equal_slots_builds_battle(self):
        fleet = self._fleet_with_full_miner_crew()
        deposit = FakeDeposit(Vec2(2, 2), extractors_me=0b101, extractors_other=0b011)
        state = FakeState(fleet_me=fleet, deposit_me=deposit)
        conf = FakeConf()
        # Both hold 2 slots -- not behind, so Battle.
        result = fabricator_mod._compute_fabricator_next(state, conf, {})
        self.assertEqual(result, int(BotClass.Battle))

    def test_large_fleet_still_replaces_a_lost_miner_first(self):
        # A large (>=10) fleet down to 2 live extractors should rebuild the
        # mining crew before it even looks at the slot-contest bitmasks.
        fleet = _fleet_of(10, start_id=0)
        fleet[0] = FakeBot(0, Vec2(0, 0), class_=BotClass.Extractor)
        fleet[1] = FakeBot(1, Vec2(0, 0), class_=BotClass.Extractor)
        deposit = FakeDeposit(Vec2(2, 2), extractors_me=0b011, extractors_other=0b001)
        state = FakeState(fleet_me=fleet, deposit_me=deposit)
        conf = FakeConf()
        result = fabricator_mod._compute_fabricator_next(state, conf, {})
        self.assertEqual(result, int(BotClass.Extractor))


class EarlyRampTests(unittest.TestCase):
    def test_builds_extractor_early_with_few_extractors(self):
        fleet = [FakeBot(0, Vec2(0, 0), class_=BotClass.Extractor)]
        state = FakeState(tick=10, fleet_me=fleet)
        conf = FakeConf()
        result = fabricator_mod._compute_fabricator_next(state, conf, {})
        self.assertEqual(result, int(BotClass.Extractor))

    def test_keeps_replenishing_a_short_mining_crew_past_tick_30(self):
        # A miner lost well into the match (well past the old tick-30 ramp
        # window) still needs replacing -- "usable amount of miners" is a
        # standing invariant, not a one-time opening ramp.
        fleet = [FakeBot(0, Vec2(0, 0), class_=BotClass.Extractor)]
        state = FakeState(tick=3000, fleet_me=fleet)
        conf = FakeConf()
        result = fabricator_mod._compute_fabricator_next(state, conf, {})
        self.assertEqual(result, int(BotClass.Extractor))

    def test_stops_ramp_once_three_extractors_alive_even_late(self):
        fleet = [FakeBot(i, Vec2(0, 0), class_=BotClass.Extractor) for i in range(3)]
        state = FakeState(tick=3000, fleet_me=fleet)
        conf = FakeConf()
        result = fabricator_mod._compute_fabricator_next(state, conf, {})
        self.assertNotEqual(result, int(BotClass.Extractor))

    def test_stops_ramp_once_three_extractors_alive(self):
        fleet = [FakeBot(i, Vec2(0, 0), class_=BotClass.Extractor) for i in range(3)]
        state = FakeState(tick=10, fleet_me=fleet)
        conf = FakeConf()
        result = fabricator_mod._compute_fabricator_next(state, conf, {})
        self.assertNotEqual(result, int(BotClass.Extractor))


class AlternationTests(unittest.TestCase):
    def _state(self):
        fleet = [FakeBot(i, Vec2(0, 0), class_=BotClass.Extractor) for i in range(3)]
        return FakeState(tick=100, fleet_me=fleet)

    def test_no_births_keeps_alternation_stable(self):
        conf = FakeConf()
        cache = {}
        first = fabricator_mod._compute_fabricator_next(self._state(), conf, cache, frozenset())
        second = fabricator_mod._compute_fabricator_next(self._state(), conf, cache, frozenset())
        self.assertEqual(first, second)

    def test_combat_birth_flips_alternation(self):
        conf = FakeConf()
        cache = {}
        state = self._state()
        state.fleet_me = type(state.fleet_me)(
            list(state.fleet_me) + [FakeBot(99, Vec2(0, 0), class_=BotClass.Battle)]
        )
        before = fabricator_mod._compute_fabricator_next(state, conf, dict(cache), frozenset())
        after = fabricator_mod._compute_fabricator_next(state, conf, cache, frozenset({99}))
        self.assertNotEqual(before, after)

    def test_extractor_birth_does_not_flip_alternation(self):
        conf = FakeConf()
        cache = {}
        state = self._state()
        state.fleet_me = type(state.fleet_me)(
            list(state.fleet_me) + [FakeBot(99, Vec2(0, 0), class_=BotClass.Extractor)]
        )
        before = fabricator_mod._compute_fabricator_next(state, conf, dict(cache), frozenset())
        after = fabricator_mod._compute_fabricator_next(state, conf, cache, frozenset({99}))
        self.assertEqual(before, after)

    def test_simultaneous_death_and_birth_still_counts_as_a_birth(self):
        # Net fleet size unchanged (one died, one born same tick), but the
        # birth must still flip the alternation -- this is the case a
        # size-delta-based check would silently miss.
        conf = FakeConf()
        cache = {"alt_count": 0}
        fleet = [FakeBot(0, Vec2(0, 0), class_=BotClass.Extractor),
                 FakeBot(1, Vec2(0, 0), class_=BotClass.Extractor),
                 FakeBot(2, Vec2(0, 0), class_=BotClass.Extractor),
                 FakeBot(50, Vec2(0, 0), class_=BotClass.Healer)]  # replaced a dead Battle
        state = FakeState(tick=100, fleet_me=fleet)
        result = fabricator_mod._compute_fabricator_next(state, conf, cache, frozenset({50}))
        self.assertEqual(cache["alt_count"], 1)
        self.assertEqual(result, int(BotClass.Healer))


if __name__ == "__main__":
    unittest.main()
