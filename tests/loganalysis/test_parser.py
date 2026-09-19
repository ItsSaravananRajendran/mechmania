"""Tests for tools/loganalysis/parser.py."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.loganalysis.fixtures import _bot, write_tempfile  # noqa: E402
from tools.loganalysis.parser import (  # noqa: E402
    ROLE_BATTLE,
    ROLE_EXTRACTOR,
    ROLE_HEALER,
    iter_ticks,
    load_config,
    load_match_result,
)


class TestConfig(unittest.TestCase):
    def test_load_config(self):
        path = write_tempfile([])
        self.addCleanup(os.remove, path)
        cfg = load_config(path)
        self.assertEqual(cfg.max_ticks, 9000)
        self.assertEqual(cfg.bot["blaster_range"], 10.0)


class TestIterTicks(unittest.TestCase):
    def test_first_tick_is_a_full_fleet(self):
        records = [
            {
                "tick": 1,
                "capture": 0.0,
                "fleet_a": [_bot(0, 1.0, 1.0)],
                "fleet_b": [_bot(0, 30.0, 30.0)],
                "deposit_a": {"pos": {"x": 23.0, "y": 30.0}, "extractors": {"a": 0, "b": 0}},
                "deposit_b": {"pos": {"x": 9.0, "y": 2.0}, "extractors": {"a": 0, "b": 0}},
                "fabricator_a": {"tokens": 750.0, "next_bot_creation": 1},
                "fabricator_b": {"tokens": 750.0, "next_bot_creation": 1},
            }
        ]
        path = write_tempfile(records)
        self.addCleanup(os.remove, path)
        ticks = list(iter_ticks(path))
        self.assertEqual(len(ticks), 1)
        snap = ticks[0]
        self.assertEqual(len(snap.bots("a")), 1)
        self.assertEqual(snap.bots("a")[0].pos, (1.0, 1.0))
        self.assertEqual(snap.deposits["a"].x, 23.0)
        self.assertEqual(snap.fabricators["a"].tokens, 750.0)

    def test_delta_added_changed_removed(self):
        records = [
            {"tick": 1, "fleet_a": [_bot(0, 1.0, 1.0)], "fleet_b": [_bot(0, 30.0, 30.0)]},
            # tick 2: add a second bot to fleet_a, move bot 0
            {
                "tick": 2,
                "fleet_a": {
                    "added": {"1": _bot(1, 2.0, 2.0)},
                    "changed": {"0": {"pos": {"x": 1.5, "y": 1.0}}},
                },
            },
            # tick 3: bot 0 removed (dies), bot 1 untouched
            {"tick": 3, "fleet_a": {"removed": [0]}},
        ]
        path = write_tempfile(records)
        self.addCleanup(os.remove, path)
        ticks = list(iter_ticks(path))
        self.assertEqual(len(ticks), 3)

        t2 = ticks[1]
        self.assertEqual(len(t2.bots("a")), 2)
        bot0 = t2.fleets["a"][0]
        self.assertEqual(bot0.pos, (1.5, 1.0))
        self.assertEqual(bot0.health, 10.0)  # unset fields carry forward

        t3 = ticks[2]
        self.assertEqual(len(t3.bots("a")), 1)
        self.assertNotIn(0, t3.fleets["a"])
        self.assertEqual(t3.removed["a"], [0])

    def test_deposit_and_fabricator_carry_forward_unset_fields(self):
        records = [
            {
                "tick": 1,
                "fleet_a": [_bot(0, 1.0, 1.0)],
                "fleet_b": [_bot(0, 30.0, 30.0)],
                "deposit_a": {"pos": {"x": 23.0, "y": 30.0}, "extractors": {"a": 0, "b": 0}},
                "fabricator_a": {"tokens": 750.0, "next_bot_creation": 1},
            },
            # only extractors/tokens change -- pos and next_bot_creation
            # are omitted, matching the real engine's encoding.
            {
                "tick": 2,
                "deposit_a": {"extractors": {"a": 1, "b": 0}},
                "fabricator_a": {"tokens": 700.0},
            },
        ]
        path = write_tempfile(records)
        self.addCleanup(os.remove, path)
        ticks = list(iter_ticks(path))
        dep = ticks[1].deposits["a"]
        self.assertEqual(dep.x, 23.0)  # carried forward
        self.assertEqual(dep.extractors_a, 1)
        fab = ticks[1].fabricators["a"]
        self.assertEqual(fab.tokens, 700.0)
        self.assertEqual(fab.next_bot_creation, 1)  # carried forward

    def test_role_property_reads_special_key(self):
        records = [
            {
                "tick": 1,
                "fleet_a": [
                    _bot(0, 1.0, 1.0, special={"Extractor": {"extracting": "None"}}),
                    _bot(1, 1.0, 1.0, special={"Healer": {"healing": "None"}}),
                    _bot(2, 1.0, 1.0, special={"Battle": {"next_fire_tick": 0, "shot": "None"}}),
                ],
            }
        ]
        path = write_tempfile(records)
        self.addCleanup(os.remove, path)
        snap = next(iter_ticks(path))
        roles = {b.id: b.role for b in snap.bots("a")}
        self.assertEqual(roles, {0: ROLE_EXTRACTOR, 1: ROLE_HEALER, 2: ROLE_BATTLE})


class TestMatchResult(unittest.TestCase):
    def test_parses_trailing_comment_lines(self):
        path = write_tempfile(
            [{"tick": 1, "fleet_a": [_bot(0, 1.0, 1.0)]}],
            footer_lines=[
                '# result: {"reason":"elimination","tick":6736,"winner":"A"}',
                "# time elapsed: 20.965644236s",
            ],
        )
        self.addCleanup(os.remove, path)
        result = load_match_result(path)
        self.assertEqual(result.reason, "elimination")
        self.assertEqual(result.winner, "A")
        self.assertEqual(result.tick, 6736)
        self.assertAlmostEqual(result.elapsed_seconds, 20.965644236)

    def test_missing_result_returns_none(self):
        path = write_tempfile([{"tick": 1, "fleet_a": [_bot(0, 1.0, 1.0)]}])
        self.addCleanup(os.remove, path)
        self.assertIsNone(load_match_result(path))


if __name__ == "__main__":
    unittest.main()
