"""Tests for tools/loganalysis/analyze.py."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.loganalysis.fixtures import _bot, write_tempfile  # noqa: E402
from tools.loganalysis.analyze import (  # noqa: E402
    analyze_economy,
    analyze_engagements,
    analyze_formation_shape,
    analyze_healer_movement,
    analyze_hit_reactions,
    analyze_mining_deployment,
    analyze_payload_contest,
)
from tools.loganalysis.parser import ROLE_BATTLE, ROLE_EXTRACTOR, ROLE_HEALER, iter_ticks  # noqa: E402


def _battle(id_, x, y, health=10.0, vx=0.0, vy=0.0):
    return _bot(id_, x, y, health=health, vx=vx, vy=vy, special={"Battle": {"next_fire_tick": 0, "shot": "None"}})


class TestHitReactions(unittest.TestCase):
    def test_hit_then_flee_is_recorded(self):
        records = [
            {"tick": 1, "fleet_b": [_battle(0, 5.0, 5.0, health=10.0, vx=0.0, vy=0.0)]},
            # takes 3 damage, then moves fast next tick -- a flee.
            {"tick": 2, "fleet_b": {"changed": {"0": {"health": 7.0}}}},
            {
                "tick": 3,
                "fleet_b": {"changed": {"0": {"vel": {"x": 0.5, "y": 0.0}, "pos": {"x": 5.5, "y": 5.0}}}},
            },
        ]
        path = write_tempfile(records)
        self.addCleanup(os.remove, path)
        ticks = list(iter_ticks(path))
        summary = analyze_hit_reactions(ticks, "b", window=10)
        self.assertEqual(summary.total_hits, 1)
        self.assertEqual(summary.deaths, 0)
        self.assertTrue(summary.events[0].fled)

    def test_id_reuse_does_not_splice_lives_and_death_is_detected(self):
        # Bot 0 takes a fatal-looking hit (down to 1hp, never logged at 0),
        # then gets removed (died). A tick later a *new* bot 0 is added --
        # its fresh 10hp must not be read as "healed" by the dead one.
        records = [
            {"tick": 1, "fleet_b": [_battle(0, 5.0, 5.0, health=10.0)]},
            {"tick": 2, "fleet_b": {"changed": {"0": {"health": 1.0}}}},
            {"tick": 3, "fleet_b": {"removed": [0]}},
            {"tick": 4, "fleet_b": {"added": {"0": _battle(0, 1.0, 1.0, health=10.0)}}},
        ]
        path = write_tempfile(records)
        self.addCleanup(os.remove, path)
        ticks = list(iter_ticks(path))
        summary = analyze_hit_reactions(ticks, "b", window=10)
        self.assertEqual(summary.total_hits, 1)
        self.assertEqual(summary.deaths, 1)
        self.assertTrue(summary.events[0].died)


class TestHealerMovement(unittest.TestCase):
    def test_tracks_target_and_wander(self):
        records = [
            {
                "tick": 1,
                "fleet_a": [
                    _bot(0, 10.0, 10.0, special={"Healer": {"healing": "None"}}),
                ],
            },
            {
                "tick": 2,
                "fleet_a": {
                    "changed": {
                        "0": {"pos": {"x": 10.5, "y": 10.0}, "special": {"Healer": {"healing": {"Some": 7}}}}
                    }
                },
            },
        ]
        path = write_tempfile(records)
        self.addCleanup(os.remove, path)
        ticks = list(iter_ticks(path))
        summary = analyze_healer_movement(ticks, "a")
        self.assertEqual(summary.healer_count, 1)
        p = summary.profiles[0]
        self.assertEqual(p.ticks_observed, 2)
        self.assertEqual(p.pct_ticks_with_target, 50.0)
        self.assertEqual(p.unique_targets_healed, 1)


class TestMiningDeployment(unittest.TestCase):
    def test_counts_miners_and_guards_near_deposit(self):
        records = [
            {
                "tick": 20,
                "fleet_b": [
                    _bot(0, 23.0, 30.0, special={"Extractor": {"extracting": "None"}}),
                    _bot(1, 50.0, 50.0, special={"Extractor": {"extracting": "None"}}),
                    _battle(2, 23.5, 30.0),
                ],
                "deposit_a": {"pos": {"x": 23.0, "y": 30.0}, "extractors": {"a": 0, "b": 1}},
                "deposit_b": {"pos": {"x": 9.0, "y": 2.0}, "extractors": {"a": 0, "b": 0}},
            }
        ]
        path = write_tempfile(records)
        self.addCleanup(os.remove, path)
        ticks = list(iter_ticks(path))
        summary = analyze_mining_deployment(ticks, "b", extract_range=5.0, guard_range=10.0, sample_every=20)
        self.assertEqual(summary.max_miners["a"], 1)  # only the close one counts
        self.assertEqual(summary.max_guards["a"], 1)


class TestFormationShape(unittest.TestCase):
    def test_colinear_bots_read_as_a_line(self):
        records = [
            {
                "tick": 20,
                "fleet_b": [_battle(i, float(i), 0.0) for i in range(5)],
            }
        ]
        path = write_tempfile(records)
        self.addCleanup(os.remove, path)
        ticks = list(iter_ticks(path))
        summary = analyze_formation_shape(ticks, "b", ROLE_BATTLE, sample_every=20)
        self.assertEqual(summary.dominant_shape, "line")

    def test_tight_cluster_reads_as_cluster(self):
        records = [
            {
                "tick": 20,
                "fleet_b": [_battle(i, 10.0 + 0.01 * i, 10.0) for i in range(5)],
            }
        ]
        path = write_tempfile(records)
        self.addCleanup(os.remove, path)
        ticks = list(iter_ticks(path))
        summary = analyze_formation_shape(ticks, "b", ROLE_BATTLE, sample_every=20)
        self.assertEqual(summary.dominant_shape, "cluster")


class TestPayloadContest(unittest.TestCase):
    def test_direction_and_stall_counted(self):
        records = [
            {"tick": 1, "capture": 0.0},
            {"tick": 2, "capture": 0.1},  # pushed toward A
            {"tick": 3, "capture": 0.1},  # stalled
            {"tick": 4, "capture": 0.05},  # pushed toward B
        ]
        path = write_tempfile(records)
        self.addCleanup(os.remove, path)
        ticks = list(iter_ticks(path))
        summary = analyze_payload_contest(ticks)
        self.assertEqual(summary.ticks_pushed_toward_a_win, 1)
        self.assertEqual(summary.ticks_pushed_toward_b_win, 1)
        self.assertEqual(summary.ticks_stalled, 1)
        self.assertEqual(summary.leader_at_end, "a")  # net change (0.05) is still positive


class TestEconomy(unittest.TestCase):
    def test_tracks_tokens_builds_and_slot_occupancy(self):
        records = [
            {
                "tick": 1,
                "fleet_b": [_battle(0, 5.0, 5.0)],
                "fabricator_b": {"tokens": 100.0, "next_bot_creation": 200},
                "deposit_a": {"pos": {"x": 23.0, "y": 30.0}, "extractors": {"a": 0, "b": 4}},
            },
            {
                "tick": 20,
                "fleet_b": {"added": {"1": _bot(1, 6.0, 6.0, special={"Extractor": {"extracting": "None"}})}},
                "fabricator_b": {"tokens": 50.0},
            },
        ]
        path = write_tempfile(records)
        self.addCleanup(os.remove, path)
        ticks = list(iter_ticks(path))
        summary = analyze_economy(ticks, "b", extractor_cap=8, sample_every=20)
        self.assertEqual(summary.token_end, 50.0)
        # tick 1's starting bot counts as a "build" too -- the parser
        # treats the log's initial full-fleet line the same as any other
        # `added` event (see test_parser.test_first_tick_is_a_full_fleet).
        self.assertEqual(summary.build_counts_by_role, {"Battle": 1, "Extractor": 1})
        self.assertAlmostEqual(summary.deposit_occupancy_max["a"], 4 / 8)


class TestEngagements(unittest.TestCase):
    def test_clusters_hits_separated_by_a_quiet_gap(self):
        records = [
            {"tick": 1, "fleet_b": [_battle(0, 5.0, 5.0, health=10.0)]},
            {"tick": 2, "fleet_b": {"changed": {"0": {"health": 7.0}}}},
            # a second hit right after -- same fight
            {"tick": 3, "fleet_b": {"changed": {"0": {"health": 4.0}}}},
            # a long quiet stretch, then a new hit -- a separate fight
            {"tick": 200, "fleet_b": {"changed": {"0": {"health": 1.0}}}},
        ]
        path = write_tempfile(records)
        self.addCleanup(os.remove, path)
        ticks = list(iter_ticks(path))
        hits = analyze_hit_reactions(ticks, "b", window=10)
        summary = analyze_engagements(hits.events, gap_ticks=60)
        self.assertEqual(summary.count, 2)
        self.assertEqual(summary.engagements[0].hits, 2)
        self.assertEqual(summary.engagements[1].hits, 1)


if __name__ == "__main__":
    unittest.main()
