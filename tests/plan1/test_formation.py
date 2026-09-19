"""Tests for strategy/plan1/formation.py -- see plans/plan1/formation.md.

`_is_slot_free` (imported from `walls.py`) calls the engine's `point_free`;
tests patch `strategy.plan1.walls.point_free` -- the name binding
`_is_slot_free` actually resolves at call time -- rather than anything in
`formation.py` itself, which doesn't import `point_free` directly.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import strategy.plan1.formation as formation_mod  # noqa: E402
import strategy.plan1.walls as walls_mod  # noqa: E402
from tests.plan1.fakes import (  # noqa: E402
    FakeBot, FakeBotConfig, FakeConf, FakeMapConf, FakeState, Vec2,
)


def _all_free():
    return patch.object(walls_mod, "point_free", return_value=True)


def _none_free():
    return patch.object(walls_mod, "point_free", return_value=False)


def _empty_grid():
    return walls_mod.WallGrid.build(FakeMapConf(wall_cells=set()))


class LateralSlotTests(unittest.TestCase):
    def test_centers_symmetrically_around_center(self):
        center = Vec2(10.0, 10.0)
        perp = Vec2(1.0, 0.0)
        slots = [formation_mod._lateral_slot(center, perp, i, 4, 2.0) for i in range(4)]
        xs = sorted(s.x for s in slots)
        # 4 slots at spacing 2 centered on x=10: -3, -1, 1, 3 relative -> 7, 9, 11, 13
        self.assertEqual(xs, [7.0, 9.0, 11.0, 13.0])

    def test_single_slot_lands_on_center(self):
        center = Vec2(5.0, 5.0)
        perp = Vec2(0.0, 1.0)
        slot = formation_mod._lateral_slot(center, perp, 0, 1, 2.0)
        self.assertEqual((slot.x, slot.y), (5.0, 5.0))

    def test_evenly_spaced(self):
        center = Vec2(0.0, 0.0)
        perp = Vec2(1.0, 0.0)
        slots = [formation_mod._lateral_slot(center, perp, i, 5, 3.0) for i in range(5)]
        xs = [s.x for s in slots]
        gaps = [round(xs[i + 1] - xs[i], 9) for i in range(len(xs) - 1)]
        self.assertTrue(all(g == 3.0 for g in gaps))


class MinSafeSpacingTests(unittest.TestCase):
    def test_just_greater_than_splash_radius(self):
        conf = FakeConf(bot=FakeBotConfig(radius=0.1, base_blaster_splash_radius=0.3))
        result = formation_mod._min_safe_spacing(conf)
        self.assertGreater(result, conf.bot.base_blaster_splash_radius)
        self.assertAlmostEqual(result, 0.3 * formation_mod.SPLASH_SAFETY_MARGIN)

    def test_floored_by_twice_bot_radius(self):
        # A splash radius near zero shouldn't collapse spacing to near zero
        # too -- bots still need room not to stand inside one another.
        conf = FakeConf(bot=FakeBotConfig(radius=1.0, base_blaster_splash_radius=0.01))
        result = formation_mod._min_safe_spacing(conf)
        self.assertEqual(result, 2.0 * conf.bot.radius)


class ResolveSlotTests(unittest.TestCase):
    def test_returns_candidate_when_free(self):
        grid = _empty_grid()
        candidate = Vec2(5.0, 5.0)
        anchor = Vec2(0.0, 0.0)
        with _all_free():
            result = formation_mod._resolve_slot(grid, candidate, anchor)
        self.assertEqual((result.x, result.y), (5.0, 5.0))

    def test_interpolates_toward_anchor_when_blocked(self):
        grid = _empty_grid()
        candidate = Vec2(10.0, 0.0)
        anchor = Vec2(0.0, 0.0)
        # Free only at exactly 50% of the way from anchor to candidate.
        with patch.object(walls_mod, "point_free", side_effect=lambda p: abs(p.x - 5.0) < 1e-6):
            result = formation_mod._resolve_slot(grid, candidate, anchor)
        self.assertAlmostEqual(result.x, 5.0)

    def test_last_resort_is_not_identical_to_anchor(self):
        grid = _empty_grid()
        candidate = Vec2(10.0, 0.0)
        anchor = Vec2(0.0, 0.0)
        with _none_free():
            result = formation_mod._resolve_slot(grid, candidate, anchor)
        self.assertNotEqual((result.x, result.y), (anchor.x, anchor.y))
        # Should still be a small step from the anchor towards the candidate.
        self.assertAlmostEqual(result.x, 0.2, places=5)


class FreeAnchorTests(unittest.TestCase):
    def test_returns_anchor_if_already_free(self):
        grid = _empty_grid()
        anchor = Vec2(1.0, 1.0)
        with _all_free():
            result = formation_mod._free_anchor(grid, anchor, Vec2(1.0, 0.0), step=1.0)
        self.assertEqual((result.x, result.y), (1.0, 1.0))

    def test_nudges_sideways_until_free(self):
        grid = _empty_grid()
        anchor = Vec2(0.0, 0.0)
        perp = Vec2(1.0, 0.0)
        # Free only 2 steps to the +perp side.
        with patch.object(walls_mod, "point_free", side_effect=lambda p: abs(p.x - 2.0) < 1e-6):
            result = formation_mod._free_anchor(grid, anchor, perp, step=1.0)
        self.assertAlmostEqual(result.x, 2.0)

    def test_gives_up_and_returns_anchor_if_nothing_found(self):
        grid = _empty_grid()
        anchor = Vec2(0.0, 0.0)
        with _none_free():
            result = formation_mod._free_anchor(grid, anchor, Vec2(1.0, 0.0), step=1.0, max_tries=3)
        self.assertEqual((result.x, result.y), (0.0, 0.0))


class DedupeSlotsTests(unittest.TestCase):
    def test_no_duplicates_left_unchanged(self):
        slots = {0: Vec2(0.0, 0.0), 1: Vec2(5.0, 5.0)}
        result = formation_mod._dedupe_slots(slots, Vec2(1.0, 0.0), spacing=2.0)
        self.assertEqual((result[0].x, result[0].y), (0.0, 0.0))
        self.assertEqual((result[1].x, result[1].y), (5.0, 5.0))

    def test_exact_duplicate_gets_nudged_apart(self):
        slots = {0: Vec2(3.0, 3.0), 1: Vec2(3.0, 3.0)}
        result = formation_mod._dedupe_slots(slots, Vec2(1.0, 0.0), spacing=2.0)
        self.assertNotEqual((result[0].x, result[0].y), (result[1].x, result[1].y))

    def test_three_way_duplicate_all_distinct(self):
        slots = {0: Vec2(1.0, 1.0), 1: Vec2(1.0, 1.0), 2: Vec2(1.0, 1.0)}
        result = formation_mod._dedupe_slots(slots, Vec2(1.0, 0.0), spacing=2.0)
        coords = {(round(v.x, 2), round(v.y, 2)) for v in result.values()}
        self.assertEqual(len(coords), 3)

    def test_min_distance_pushes_apart_near_but_distinct_slots(self):
        # Not exact duplicates (so the old exact-match check alone would
        # leave them as-is), but well inside a 2.0 minimum distance.
        slots = {0: Vec2(0.0, 0.0), 1: Vec2(0.1, 0.0)}
        result = formation_mod._dedupe_slots(slots, Vec2(0.0, 1.0), spacing=2.0, min_distance=2.0)
        self.assertGreaterEqual(result[0].dist(result[1]), 2.0)

    def test_min_distance_zero_keeps_old_behaviour(self):
        slots = {0: Vec2(0.0, 0.0), 1: Vec2(0.1, 0.0)}
        result = formation_mod._dedupe_slots(slots, Vec2(0.0, 1.0), spacing=2.0)
        self.assertAlmostEqual(result[1].x, 0.1, places=5)
        self.assertAlmostEqual(result[1].y, 0.0, places=5)

    def test_min_distance_satisfied_for_a_larger_group(self):
        slots = {i: Vec2(0.0, 0.0) for i in range(8)}
        result = formation_mod._dedupe_slots(slots, Vec2(1.0, 0.0), spacing=1.0, min_distance=1.5)
        positions = list(result.values())
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                self.assertGreaterEqual(positions[i].dist(positions[j]), 1.5)


class StackBehindTests(unittest.TestCase):
    def test_empty_front_positions(self):
        grid = _empty_grid()
        slots, anchors = formation_mod._stack_behind(grid, [], Vec2(1, 0), [FakeBot(0, Vec2(0, 0))], 1.0)
        self.assertEqual(slots, {})
        self.assertEqual(anchors, {})

    def test_single_trailing_bot_lands_directly_behind_the_front_bot(self):
        grid = _empty_grid()
        front = [Vec2(10.0, 10.0)]
        trailing = [FakeBot(0, Vec2(0, 0))]
        with _all_free():
            slots, anchors = formation_mod._stack_behind(grid, front, Vec2(1.0, 0.0), trailing, 2.0)
        # "Directly behind" -- same lateral (y) coordinate, stepped back
        # along -forward (x) by depth_gap.
        self.assertAlmostEqual(slots[0].y, 10.0)
        self.assertAlmostEqual(slots[0].x, 8.0)
        self.assertEqual((anchors[0].x, anchors[0].y), (10.0, 10.0))

    def test_extra_trailing_bots_wrap_to_deeper_rows_in_the_same_column(self):
        # Front position kept well clear of the map edge (`MAP_SIZE`=32) --
        # stacking back along -forward from something near x=0 would read as
        # out-of-bounds ("wall"), tripping `_resolve_slot`'s fallback instead
        # of exercising the stacking math this test is actually about.
        grid = _empty_grid()
        front = [Vec2(20.0, 20.0)]
        trailing = [FakeBot(i, Vec2(0, 0)) for i in range(3)]
        with _all_free():
            slots, _ = formation_mod._stack_behind(grid, front, Vec2(1.0, 0.0), trailing, 2.0)
        # All in the single column (same lateral coordinate)...
        for s in slots.values():
            self.assertAlmostEqual(s.y, 20.0)
        # ...but at increasing depth behind the front bot.
        xs = sorted(s.x for s in slots.values())
        self.assertEqual(xs, [14.0, 16.0, 18.0])

    def test_trailing_bots_round_robin_across_multiple_columns(self):
        grid = _empty_grid()
        front = [Vec2(20.0, 10.0), Vec2(20.0, 15.0)]
        trailing = [FakeBot(i, Vec2(0, 0)) for i in range(2)]
        with _all_free():
            slots, anchors = formation_mod._stack_behind(grid, front, Vec2(1.0, 0.0), trailing, 2.0)
        # Bot 0 -> column 0 (y=10), bot 1 -> column 1 (y=15), both one deep.
        self.assertAlmostEqual(slots[0].y, 10.0)
        self.assertAlmostEqual(slots[1].y, 15.0)
        self.assertAlmostEqual(slots[0].x, 18.0)
        self.assertAlmostEqual(slots[1].x, 18.0)


class ArrangeGroupTests(unittest.TestCase):
    def test_empty_group(self):
        grid = _empty_grid()
        slots, anchors = formation_mod._arrange_group(
            grid, Vec2(0, 0), Vec2(1, 0), Vec2(0, 1), [], 1.0, 1.0, 10.0, 1.0
        )
        self.assertEqual(slots, {})
        self.assertEqual(anchors, {})

    def test_wraps_into_a_second_row_past_row_capacity(self):
        grid = _empty_grid()
        bots = [FakeBot(i, Vec2(0, 0)) for i in range(6)]
        # max_width=4, min_spacing=2 -> row_capacity = 4//2 + 1 = 3
        with _all_free():
            slots, anchors = formation_mod._arrange_group(
                grid, Vec2(0, 0), Vec2(1, 0), Vec2(0, 1), bots,
                spacing=2.0, min_spacing=2.0, max_width=4.0, row_gap=3.0,
            )
        # Row centers step along `forward` (here Vec2(1, 0)) by `row_gap`
        # per row, so a bot in the second row should differ from the first
        # row's anchor along x, not y (y is the lateral/`perp` axis).
        row0_anchor_x = anchors[bots[0].id].x
        row1_anchor_x = anchors[bots[5].id].x
        self.assertNotAlmostEqual(row0_anchor_x, row1_anchor_x)

    def test_all_bots_get_distinct_slots_when_free(self):
        grid = _empty_grid()
        bots = [FakeBot(i, Vec2(0, 0)) for i in range(5)]
        with _all_free():
            slots, _ = formation_mod._arrange_group(
                grid, Vec2(0, 0), Vec2(1, 0), Vec2(0, 1), bots,
                spacing=2.0, min_spacing=2.0, max_width=100.0, row_gap=3.0,
            )
        coords = {(round(v.x, 6), round(v.y, 6)) for v in slots.values()}
        self.assertEqual(len(coords), len(bots))


class IsRetreatingTests(unittest.TestCase):
    def test_still_invulnerable_after_a_hit_is_retreating(self):
        bot = FakeBot(0, Vec2(0, 0), invulnerable_until_tick=20)
        self.assertTrue(formation_mod._is_retreating(bot, tick=10))

    def test_invulnerability_expired_is_not_retreating(self):
        bot = FakeBot(0, Vec2(0, 0), invulnerable_until_tick=5)
        self.assertFalse(formation_mod._is_retreating(bot, tick=10))

    def test_never_hit_is_not_retreating(self):
        bot = FakeBot(0, Vec2(0, 0), invulnerable_until_tick=0)
        self.assertFalse(formation_mod._is_retreating(bot, tick=10))

    def test_no_tick_given_disables_retreat(self):
        bot = FakeBot(0, Vec2(0, 0), invulnerable_until_tick=999)
        self.assertFalse(formation_mod._is_retreating(bot, tick=None))


class ComputeBattleFormationTests(unittest.TestCase):
    def test_empty_battles(self):
        grid = _empty_grid()
        conf = FakeConf()
        result = formation_mod._compute_battle_formation([], Vec2(16, 16), Vec2(1, 0), conf, grid)
        self.assertEqual(result, {})

    def test_no_exact_duplicate_slots_for_a_large_group(self):
        grid = _empty_grid()
        conf = FakeConf()
        battles = [FakeBot(i, Vec2(16, 16)) for i in range(20)]
        with _all_free():
            result = formation_mod._compute_battle_formation(battles, Vec2(16, 16), Vec2(1, 0), conf, grid)
        coords = [(round(v.x, 2), round(v.y, 2)) for v in result.values()]
        self.assertEqual(len(coords), len(set(coords)))
        self.assertEqual(len(result), 20)

    def test_all_slots_satisfy_the_minimum_safe_distance(self):
        # The "minimum distance between each bot after it moves away from
        # the origin" invariant: no two computed formation slots should ever
        # end up closer than `_min_safe_spacing(conf)` -- just over one
        # splash radius -- from each other.
        grid = _empty_grid()
        conf = FakeConf()
        battles = [FakeBot(i, Vec2(0, 0)) for i in range(10)]
        with _all_free():
            result = formation_mod._compute_battle_formation(battles, Vec2(16, 16), Vec2(1, 0), conf, grid)
        min_safe = formation_mod._min_safe_spacing(conf)
        positions = list(result.values())
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                self.assertGreaterEqual(positions[i].dist(positions[j]), min_safe)

    def test_grid_has_equal_columns_and_rows_for_a_perfect_square_count(self):
        # 9 battles -> ceil(sqrt(9)) = 3 columns, and 9 bots split 3-per-
        # column over those 3 columns is exactly 3 rows deep too.
        grid = _empty_grid()
        conf = FakeConf()
        forward = Vec2(1.0, 0.0)
        perp = forward.rotate_deg(90.0)
        battles = [FakeBot(i, Vec2(0, 0)) for i in range(9)]
        with _all_free():
            result = formation_mod._compute_battle_formation(battles, Vec2(16, 16), forward, conf, grid)
        columns = {round(v.dot(perp), 4) for v in result.values()}
        self.assertEqual(len(columns), 3)
        for col in columns:
            depths = [round(v.dot(forward), 4) for v in result.values() if round(v.dot(perp), 4) == col]
            self.assertEqual(len(depths), 3)

    def test_grid_columns_and_rows_stay_close_for_a_non_square_count(self):
        # 7 battles -> ceil(sqrt(7)) = 3 columns, ceil(7/3) = 3 rows -- as
        # square as an integer count allows, never one wide row of 7.
        grid = _empty_grid()
        conf = FakeConf()
        forward = Vec2(1.0, 0.0)
        perp = forward.rotate_deg(90.0)
        battles = [FakeBot(i, Vec2(0, 0)) for i in range(7)]
        with _all_free():
            result = formation_mod._compute_battle_formation(battles, Vec2(16, 16), forward, conf, grid)
        columns = {round(v.dot(perp), 4) for v in result.values()}
        self.assertEqual(len(columns), 3)
        max_depth = max(
            sum(1 for v in result.values() if round(v.dot(perp), 4) == col)
            for col in columns
        )
        self.assertEqual(max_depth, 3)

    def test_recently_hit_bot_drops_out_of_the_screen_row(self):
        # 9 battles -> screen (front row) would normally be ids 0,1,2. Mark
        # 0 as still in its post-hit invulnerability window (tick=10 <
        # invulnerable_until_tick=100) -- it must back off out of the screen
        # row, and whoever was next in line (id 3) takes the vacated slot.
        grid = _empty_grid()
        conf = FakeConf()
        forward = Vec2(1.0, 0.0)
        battles = [FakeBot(i, Vec2(0, 0), invulnerable_until_tick=(100 if i == 0 else 0))
                   for i in range(9)]
        with _all_free():
            result = formation_mod._compute_battle_formation(
                battles, Vec2(16, 16), forward, conf, grid, tick=10,
            )
        screen_depth = round(result[1].dot(forward), 4)  # a screen bot untouched by the swap
        self.assertEqual(round(result[3].dot(forward), 4), screen_depth,
                          "bot 3 should be promoted into the vacated screen slot")
        self.assertLess(round(result[0].dot(forward), 4), screen_depth,
                         "the recently-hit bot must not be in the screen row")

    def test_retreating_bot_ends_up_deepest_in_its_column(self):
        grid = _empty_grid()
        conf = FakeConf()
        forward = Vec2(1.0, 0.0)
        battles = [FakeBot(i, Vec2(0, 0), invulnerable_until_tick=(100 if i == 0 else 0))
                   for i in range(9)]
        with _all_free():
            result = formation_mod._compute_battle_formation(
                battles, Vec2(16, 16), forward, conf, grid, tick=10,
            )
        depths = [round(v.dot(forward), 4) for v in result.values()]
        self.assertEqual(round(result[0].dot(forward), 4), min(depths))

    def test_no_tick_keeps_original_screen_selection(self):
        # Without a tick, a bot's invulnerability window is irrelevant --
        # existing callers that don't pass `tick` must see no change at all.
        grid = _empty_grid()
        conf = FakeConf()
        forward = Vec2(1.0, 0.0)
        battles_no_retreat = [FakeBot(i, Vec2(0, 0)) for i in range(9)]
        battles_would_retreat = [FakeBot(i, Vec2(0, 0), invulnerable_until_tick=(100 if i == 0 else 0))
                                  for i in range(9)]
        with _all_free():
            baseline = formation_mod._compute_battle_formation(
                battles_no_retreat, Vec2(16, 16), forward, conf, grid,
            )
            result = formation_mod._compute_battle_formation(
                battles_would_retreat, Vec2(16, 16), forward, conf, grid,
            )
        for bid in range(9):
            self.assertAlmostEqual(result[bid].x, baseline[bid].x)
            self.assertAlmostEqual(result[bid].y, baseline[bid].y)

    def test_small_group_is_unaffected_by_retreat(self):
        # n<=3 collapses to a single spread line regardless -- not enough
        # bots for a screen/column split to retreat out of.
        grid = _empty_grid()
        conf = FakeConf()
        forward = Vec2(1.0, 0.0)
        battles = [FakeBot(i, Vec2(0, 0), invulnerable_until_tick=(100 if i == 0 else 0))
                   for i in range(3)]
        with _all_free():
            retreating = formation_mod._compute_battle_formation(
                battles, Vec2(16, 16), forward, conf, grid, tick=10,
            )
            baseline_battles = [FakeBot(i, Vec2(0, 0)) for i in range(3)]
            baseline = formation_mod._compute_battle_formation(
                baseline_battles, Vec2(16, 16), forward, conf, grid,
            )
        for bid in range(3):
            self.assertAlmostEqual(retreating[bid].x, baseline[bid].x)
            self.assertAlmostEqual(retreating[bid].y, baseline[bid].y)

    def test_rest_of_fleet_stacks_directly_behind_a_screen_bot(self):
        # Battles beyond the screen line should land in the column "shadow"
        # of a screen bot -- same lateral offset, stepped straight back
        # along -forward -- not off in their own separate lateral line.
        grid = _empty_grid()
        conf = FakeConf()
        forward = Vec2(1.0, 0.0)
        perp = forward.rotate_deg(90.0)
        battles = [FakeBot(i, Vec2(0, 0)) for i in range(6)]  # n_screen = 3
        with _all_free():
            result = formation_mod._compute_battle_formation(battles, Vec2(16, 16), forward, conf, grid)
        screen_ids = sorted(b.id for b in battles)[:3]
        trailing_ids = sorted(b.id for b in battles)[3:]
        screen_perp_coords = {round(result[bid].dot(perp), 4) for bid in screen_ids}
        for bid in trailing_ids:
            self.assertIn(round(result[bid].dot(perp), 4), screen_perp_coords)

    def test_small_group_does_not_sit_exactly_on_payload(self):
        # Guards the "middle slot lands on the anchor" collision with the
        # dedicated payload defender, which sits at literal `payload`.
        grid = _empty_grid()
        conf = FakeConf()
        payload = Vec2(16.0, 16.0)
        battles = [FakeBot(0, payload)]  # a single battle bot: odd count, n<=3 branch
        with _all_free():
            result = formation_mod._compute_battle_formation(battles, payload, Vec2(1, 0), conf, grid)
        slot = result[0]
        self.assertNotEqual((round(slot.x, 6), round(slot.y, 6)), (payload.x, payload.y))


class ComputeMiningSpotsTests(unittest.TestCase):
    def test_empty_extractors(self):
        grid = _empty_grid()
        state = FakeState()
        conf = FakeConf()
        self.assertEqual(formation_mod._compute_mining_spots([], state, conf, grid), {})

    def test_spots_are_on_the_far_side_from_the_enemy(self):
        # forward points deposit_me -> deposit_other; mining spots should be
        # offset the *other* way (using the deposit's own bulk as cover).
        grid = _empty_grid()
        state = FakeState(
            deposit_me=None,
            deposit_other=None,
        )
        state.deposit_me.pos = Vec2(2.0, 2.0)
        state.deposit_other.pos = Vec2(30.0, 30.0)
        conf = FakeConf()
        extractors = [FakeBot(0, Vec2(2, 2))]
        formation_mod._cache.pop("forward_dir", None)
        with _all_free():
            result = formation_mod._compute_mining_spots(extractors, state, conf, grid)
        spot = result[0]
        forward = formation_mod._forward_direction(state)
        to_spot = spot - state.deposit_me.pos
        # Dot product with `forward` should be negative -- the spot is behind
        # the deposit relative to the enemy direction, not in front of it.
        self.assertLess(to_spot.dot(forward), 0.0)
        formation_mod._cache.pop("forward_dir", None)

    def test_caps_at_three_spots(self):
        grid = _empty_grid()
        state = FakeState()
        conf = FakeConf()
        extractors = [FakeBot(i, Vec2(2, 2)) for i in range(5)]
        with _all_free():
            result = formation_mod._compute_mining_spots(extractors, state, conf, grid)
        self.assertEqual(len(result), 3)


class ForwardDirectionTests(unittest.TestCase):
    def test_points_from_our_deposit_to_theirs(self):
        formation_mod._cache.pop("forward_dir", None)
        state = FakeState()
        state.deposit_me.pos = Vec2(0.0, 0.0)
        state.deposit_other.pos = Vec2(10.0, 0.0)
        forward = formation_mod._forward_direction(state)
        self.assertAlmostEqual(forward.x, 1.0, places=5)
        self.assertAlmostEqual(forward.y, 0.0, places=5)
        formation_mod._cache.pop("forward_dir", None)

    def test_is_cached(self):
        formation_mod._cache.pop("forward_dir", None)
        state = FakeState()
        state.deposit_me.pos = Vec2(0.0, 0.0)
        state.deposit_other.pos = Vec2(10.0, 0.0)
        first = formation_mod._forward_direction(state)
        state.deposit_other.pos = Vec2(0.0, 10.0)  # changed, but should be ignored now
        second = formation_mod._forward_direction(state)
        self.assertIs(first, second)
        formation_mod._cache.pop("forward_dir", None)

    def test_degenerate_same_position_defaults_to_a_unit_vector(self):
        formation_mod._cache.pop("forward_dir", None)
        state = FakeState()
        state.deposit_me.pos = Vec2(5.0, 5.0)
        state.deposit_other.pos = Vec2(5.0, 5.0)
        forward = formation_mod._forward_direction(state)
        self.assertAlmostEqual(forward.norm(), 1.0, places=5)
        formation_mod._cache.pop("forward_dir", None)


if __name__ == "__main__":
    unittest.main()
