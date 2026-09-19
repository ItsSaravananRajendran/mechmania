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
        # are actually about) + 7 Battle bots, 10 total. Healers are at or
        # above `MIN_HEALERS_ALIVE` so the healer-replacement rule (which
        # has higher priority than slot denial) doesn't shadow this test.
        fleet = _fleet_of(10, start_id=0)
        fleet[0] = FakeBot(0, Vec2(0, 0), class_=BotClass.Extractor)
        fleet[1] = FakeBot(1, Vec2(0, 0), class_=BotClass.Extractor)
        fleet[2] = FakeBot(2, Vec2(0, 0), class_=BotClass.Extractor)
        fleet[3] = FakeBot(3, Vec2(0, 0), class_=BotClass.Healer)
        fleet[4] = FakeBot(4, Vec2(0, 0), class_=BotClass.Healer)
        fleet[5] = FakeBot(5, Vec2(0, 0), class_=BotClass.Healer)
        fleet[6] = FakeBot(6, Vec2(0, 0), class_=BotClass.Healer)
        fleet[7] = FakeBot(7, Vec2(0, 0), class_=BotClass.Healer)
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
        # Healers at `MIN_HEALERS_ALIVE` so healer-replacement doesn't shadow.
        fleet = _fleet_of(15, start_id=0)
        fleet[0] = FakeBot(0, Vec2(0, 0), class_=BotClass.Extractor)
        fleet[1] = FakeBot(1, Vec2(0, 0), class_=BotClass.Extractor)
        fleet[2] = FakeBot(2, Vec2(0, 0), class_=BotClass.Healer)
        fleet[3] = FakeBot(3, Vec2(0, 0), class_=BotClass.Healer)
        fleet[4] = FakeBot(4, Vec2(0, 0), class_=BotClass.Healer)
        fleet[5] = FakeBot(5, Vec2(0, 0), class_=BotClass.Healer)
        fleet[6] = FakeBot(6, Vec2(0, 0), class_=BotClass.Healer)
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
        # Healers at MIN_HEALERS_ALIVE so the healer-replacement rule doesn't
        # shadow the alternation logic these tests are actually about.
        fleet = [FakeBot(i, Vec2(0, 0), class_=BotClass.Extractor) for i in range(3)]
        fleet += [FakeBot(3 + i, Vec2(0, 0), class_=BotClass.Healer) for i in range(5)]
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
        # size-delta-based check would silently miss. The fleet has
        # MIN_HEALERS_ALIVE healers so the healer-replacement rule does not
        # shadow the alternation logic this test is actually about.
        conf = FakeConf()
        cache = {"alt_count": 0}
        fleet = [FakeBot(0, Vec2(0, 0), class_=BotClass.Extractor),
                 FakeBot(1, Vec2(0, 0), class_=BotClass.Extractor),
                 FakeBot(2, Vec2(0, 0), class_=BotClass.Extractor),
                 FakeBot(3, Vec2(0, 0), class_=BotClass.Healer),
                 FakeBot(4, Vec2(0, 0), class_=BotClass.Healer),
                 FakeBot(5, Vec2(0, 0), class_=BotClass.Healer),
                 FakeBot(6, Vec2(0, 0), class_=BotClass.Healer),
                 FakeBot(7, Vec2(0, 0), class_=BotClass.Healer),
                 FakeBot(50, Vec2(0, 0), class_=BotClass.Battle)]  # a Battle born, replacing a dead Battle
        state = FakeState(tick=100, fleet_me=fleet)
        result = fabricator_mod._compute_fabricator_next(state, conf, cache, frozenset({50}))
        self.assertEqual(cache["alt_count"], 1)
        # alt_count=1 -> Healer (alt_count%2==1). Verifies alternation fired,
        # not the healer-replacement rule.
        self.assertEqual(result, int(BotClass.Healer))


class HealerReplacementTests(unittest.TestCase):
    """Healer top-up: missing healers must be rebuilt before any other
    build decision. Without this, the fleet runs the rest of the match
    with zero healers after the initial 3 die. Floor is `MIN_HEALERS_ALIVE`
    (=5) so the alternation can keep adding healers naturally past the
    goal-line minimum (3) up through the miner-escort (4) and raid (5)
    allocations in `healer_roles.py`."""

    def _state_with(self, healers, extractors=3, battles=7):
        fleet = [FakeBot(i, Vec2(0, 0), class_=BotClass.Extractor) for i in range(extractors)]
        for i, h in enumerate(healers):
            fleet.append(FakeBot(extractors + i, Vec2(0, 0), class_=BotClass.Healer))
        for i in range(battles):
            fleet.append(FakeBot(extractors + len(healers) + i, Vec2(0, 0), class_=BotClass.Battle))
        return FakeState(tick=2000, fleet_me=fleet)

    def test_replaces_missing_healer_even_with_full_miner_crew(self):
        # Fleet: 3 extractors, 4 healers (one below floor), 8 battles. The
        # healer-replacement rule has higher priority than slot denial or
        # alternation, so the next build must be Healer.
        state = self._state_with(healers=[BotClass.Healer]*4)
        conf = FakeConf()
        result = fabricator_mod._compute_fabricator_next(state, conf, {})
        self.assertEqual(result, int(BotClass.Healer))

    def test_at_five_healers_falls_through_to_normal_logic(self):
        # Exactly MIN_HEALERS_ALIVE healers: the top-up rule does NOT fire,
        # so the alternation/slot-denial branches decide.
        state = self._state_with(healers=[BotClass.Healer]*5)
        conf = FakeConf()
        # No death/birth so alt_count stays 0 -> Battle.
        result = fabricator_mod._compute_fabricator_next(state, conf, {})
        self.assertEqual(result, int(BotClass.Battle))

    def test_zero_healers_replaces_before_alternation_and_slot_denial(self):
        # All healers died. Fleet still has 3 extractors (the miner-replace
        # branch does not fire), and we are ahead on slots (so the slot-
        # denial branch would say Battle). Healer-replacement wins over
        # both lower-priority branches.
        state = self._state_with(healers=[], extractors=3, battles=10)
        conf = FakeConf()
        result = fabricator_mod._compute_fabricator_next(state, conf, {})
        self.assertEqual(result, int(BotClass.Healer))


if __name__ == "__main__":
    unittest.main()
