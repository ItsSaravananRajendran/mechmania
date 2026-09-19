"""Tests for strategy/plan1/__init__.py -- see plans/plan1/orchestration.md.

`navigate_to`, `line_of_sight`, `point_seg_dist`, and `point_free` all need a
live engine channel; `move_bot` and `turn_towards` don't (they're pure
Python-side ctypes constructors), so those run for real. Every test patches
the engine-channel functions at the name each module actually bound them
under (`from .. import X` creates a *local* name in that module).
"""

import os
import sys
import unittest
from contextlib import ExitStack
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core._generated.bindings import FleetAction  # noqa: E402
import strategy.plan1 as plan1_pkg  # noqa: E402
import strategy.plan1.combat_los as los_mod  # noqa: E402
import strategy.plan1.corners as corners_mod  # noqa: E402
import strategy.plan1.walls as walls_mod  # noqa: E402
from tests.plan1.fakes import (  # noqa: E402
    BotClass, FakeBot, FakeBudget, FakeConf, FakeDeposit, FakeState, Vec2,
)


class _EnginePatch:
    """Patches every engine-channel call reachable from plan1 so tests never
    need a live handshake: `navigate_to` returns a straight-line direction,
    `line_of_sight`/`point_free` always say "clear", `point_seg_dist` always
    says "far away" (nothing blocks)."""

    def __enter__(self):
        self._stack = ExitStack()
        # `_apply_assignments` moves bots through `_steer` (in `corners.py`),
        # not `navigate_to`/`move_bot` directly -- those names live there now.
        self._stack.enter_context(patch.object(corners_mod, "navigate_to", side_effect=lambda a, b: b - a))
        self._stack.enter_context(patch.object(plan1_pkg, "line_of_sight", return_value=True))
        self._stack.enter_context(patch.object(walls_mod, "point_free", return_value=True))
        self._stack.enter_context(patch.object(los_mod, "line_of_sight", return_value=True))
        self._stack.enter_context(patch.object(los_mod, "point_free", return_value=True))
        self._stack.enter_context(patch.object(los_mod, "point_seg_dist", return_value=999.0))
        return self

    def __exit__(self, *exc):
        self._stack.close()
        return False


def _clear_cache():
    plan1_pkg._cache.clear()


class RecomputeAssignmentsTests(unittest.TestCase):
    def setUp(self):
        _clear_cache()

    def tearDown(self):
        _clear_cache()

    def test_all_extractors_fleet_sends_one_to_hold_payload(self):
        fleet = [FakeBot(i, Vec2(2, 2), class_=BotClass.Extractor) for i in range(3)]
        state = FakeState(fleet_me=fleet, payload=Vec2(16.0, 16.0))
        conf = FakeConf()
        with _EnginePatch():
            action = plan1_pkg._recompute_assignments(state, conf)
        assignments = plan1_pkg._cache["assignments"]
        holders = [a for a in assignments.values() if a.get("at_payload")]
        self.assertEqual(len(holders), 1)
        self.assertIsInstance(action, FleetAction)

    def test_mixed_fleet_does_not_send_extractor_to_payload(self):
        fleet = [
            FakeBot(0, Vec2(2, 2), class_=BotClass.Extractor),
            FakeBot(1, Vec2(15, 15), class_=BotClass.Battle),
            FakeBot(2, Vec2(15, 16), class_=BotClass.Healer),
        ]
        state = FakeState(fleet_me=fleet, payload=Vec2(16.0, 16.0))
        conf = FakeConf()
        with _EnginePatch():
            plan1_pkg._recompute_assignments(state, conf)
        assignments = plan1_pkg._cache["assignments"]
        self.assertFalse(any(a.get("at_payload") for a in assignments.values()))

    def test_payload_healer_and_battle_are_not_on_the_identical_point(self):
        fleet = [
            FakeBot(0, Vec2(15, 15), class_=BotClass.Battle),
            FakeBot(1, Vec2(15, 16), class_=BotClass.Healer),
        ]
        state = FakeState(fleet_me=fleet, payload=Vec2(16.0, 16.0))
        conf = FakeConf()
        with _EnginePatch():
            plan1_pkg._recompute_assignments(state, conf)
        assignments = plan1_pkg._cache["assignments"]
        healer_spot = assignments[1]["move_target"]
        battle_spot = assignments[0]["move_target"]
        self.assertNotEqual((round(healer_spot.x, 6), round(healer_spot.y, 6)),
                             (round(battle_spot.x, 6), round(battle_spot.y, 6)))

    def test_combat_healer_move_target_is_behind_its_patient(self):
        # "Directly behind the attacker": a goal-line healer not tied down
        # defending the payload itself should land on its patient's own
        # -forward side, at least `_min_safe_spacing` away so a hit on the
        # patient can't also down the healer standing behind it.
        #
        # Battle 1 sits in the payload's capture radius and becomes the
        # dedicated payload defender; battles 2-4 are the miner escort (3
        # lowest remaining ids); battle 10 is the only one left, so it lands
        # in the formation/combat group. With exactly 3 healers, all 3 go to
        # the goal group (see `healer_roles.py`): healer 20 (in capture
        # radius) becomes the dedicated payload healer, leaving 21/22 free
        # to heal battle 10 as its combat healer.
        fleet = [
            FakeBot(1, Vec2(16.5, 16), class_=BotClass.Battle),
            FakeBot(2, Vec2(5, 5), class_=BotClass.Battle),
            FakeBot(3, Vec2(5, 5), class_=BotClass.Battle),
            FakeBot(4, Vec2(5, 5), class_=BotClass.Battle),
            FakeBot(10, Vec2(5, 5), class_=BotClass.Battle, health=3.0),
            FakeBot(20, Vec2(16, 16.5), class_=BotClass.Healer),
            FakeBot(21, Vec2(5, 6), class_=BotClass.Healer),
            FakeBot(22, Vec2(5, 7), class_=BotClass.Healer),
        ]
        state = FakeState(fleet_me=fleet, payload=Vec2(16.0, 16.0))
        state.deposit_me.pos = Vec2(0.0, 0.0)
        state.deposit_other.pos = Vec2(32.0, 0.0)
        conf = FakeConf()
        with _EnginePatch():
            plan1_pkg._recompute_assignments(state, conf)
        assignments = plan1_pkg._cache["assignments"]
        self.assertEqual(assignments[20]["role"], "payload")
        healer_assignment = assignments[21]
        self.assertEqual(healer_assignment["role"], "combat")
        self.assertEqual(healer_assignment["target_id"], 10)
        move_target = healer_assignment["move_target"]
        patient = next(b for b in fleet if b.id == 10)
        forward = plan1_pkg._forward_direction(state)
        to_healer = move_target - patient.pos
        # Points opposite `forward` (behind the patient relative to the enemy).
        self.assertLess(to_healer.dot(forward), 0.0)
        min_safe = plan1_pkg._min_safe_spacing(conf)
        self.assertGreaterEqual(move_target.dist(patient.pos), min_safe)

    def test_battle_fleet_splits_into_payload_escort_raid_and_formation(self):
        # 7 battle bots, none in the payload's capture radius -> 1 dedicated
        # payload defender (the fallback pick), 3 miner escorts, 1 raider
        # (round(3 * 1/3) of the remaining 3), 2 left for the formation
        # line.
        battles = [FakeBot(i, Vec2(5.0, 5.0), class_=BotClass.Battle) for i in range(7)]
        state = FakeState(fleet_me=battles, payload=Vec2(16.0, 16.0))
        conf = FakeConf()
        with _EnginePatch():
            plan1_pkg._recompute_assignments(state, conf)
        assignments = plan1_pkg._cache["assignments"]
        roles = {bid: a["role"] for bid, a in assignments.items()}
        counts = {}
        for role in roles.values():
            counts[role] = counts.get(role, 0) + 1
        self.assertEqual(counts.get("payload"), 1)
        self.assertEqual(counts.get("guard_miners"), 3)
        self.assertEqual(counts.get("raid"), 1)
        self.assertEqual(counts.get("combat"), 2)

    def test_miner_escort_targets_a_threat_near_the_deposit_over_the_global_pick(self):
        battles = [FakeBot(i, Vec2(5.0, 5.0), class_=BotClass.Battle) for i in range(4)]
        # A lone enemy sitting right on top of our deposit -- exactly the
        # "enemy came to hit the miners" case an escort must engage.
        enemy_at_deposit = FakeBot(50, Vec2(2.0, 2.0), class_=BotClass.Battle)
        state = FakeState(fleet_me=battles, fleet_other=[enemy_at_deposit], payload=Vec2(16.0, 16.0))
        state.deposit_me.pos = Vec2(2.0, 2.0)
        conf = FakeConf()
        with _EnginePatch():
            plan1_pkg._recompute_assignments(state, conf)
        assignments = plan1_pkg._cache["assignments"]
        escort_ids = [bid for bid, a in assignments.items() if a.get("role") == "guard_miners"]
        self.assertEqual(len(escort_ids), 3)
        for bid in escort_ids:
            self.assertEqual(assignments[bid]["target_id"], 50)

    def test_raider_heads_towards_the_enemy_deposit_area(self):
        battles = [FakeBot(i, Vec2(5.0, 5.0), class_=BotClass.Battle) for i in range(7)]
        state = FakeState(fleet_me=battles, payload=Vec2(16.0, 16.0))
        state.deposit_me.pos = Vec2(0.0, 0.0)
        state.deposit_other.pos = Vec2(30.0, 30.0)
        conf = FakeConf()
        with _EnginePatch():
            plan1_pkg._recompute_assignments(state, conf)
        assignments = plan1_pkg._cache["assignments"]
        raider_ids = [bid for bid, a in assignments.items() if a.get("role") == "raid"]
        self.assertEqual(len(raider_ids), 1)
        move_target = assignments[raider_ids[0]]["move_target"]
        self.assertEqual((move_target.x, move_target.y), (30.0, 30.0))

    def test_escort_and_raider_roles_survive_a_replacement_bot(self):
        # 7 battle bots -> escorts {0,1,2}, raider {3}, formation {4,5}
        # (bot 6 is the payload defender, at the payload). Kill escort 1 --
        # composition is picked fresh by id each recompute (see
        # `_pick_miner_escorts`), so bot 3 (next-lowest id after the gap)
        # shifts up to take the vacated escort slot, and whichever bot is
        # newly built backfills further down the chain (raid or formation)
        # rather than needing to specifically "know" it should guard.
        # Either way, the escort/raid *counts* must stay exactly the same
        # size after every recompute, dead bot or not.
        battles = [FakeBot(i, Vec2(5.0, 5.0), class_=BotClass.Battle) for i in range(7)]
        battles[6].pos = Vec2(16.5, 16.0)  # near the payload -> becomes the defender
        state = FakeState(fleet_me=battles, payload=Vec2(16.0, 16.0))
        conf = FakeConf()
        with _EnginePatch():
            plan1_pkg._recompute_assignments(state, conf)
        assignments = plan1_pkg._cache["assignments"]
        self.assertEqual(assignments[0]["role"], "guard_miners")
        self.assertEqual(assignments[1]["role"], "guard_miners")
        self.assertEqual(assignments[3]["role"], "raid")

        replacement = [b for b in battles if b.id != 1] + [FakeBot(20, Vec2(0.0, 0.0), class_=BotClass.Battle)]
        state2 = FakeState(fleet_me=replacement, payload=Vec2(16.0, 16.0))
        with _EnginePatch():
            plan1_pkg._recompute_assignments(state2, conf)
        assignments2 = plan1_pkg._cache["assignments"]
        roles2 = {bid: a["role"] for bid, a in assignments2.items()}
        self.assertEqual(sum(1 for r in roles2.values() if r == "guard_miners"), 3)
        self.assertEqual(sum(1 for r in roles2.values() if r == "raid"), 1)
        # Bot 3 (previously the raider) shifts up into the escort slot 1
        # left vacant; the newly-built bot (highest id) lands further down
        # the chain, still fully assigned rather than dropped.
        self.assertEqual(roles2.get(3), "guard_miners")
        self.assertIn(roles2.get(20), {"raid", "combat"})

    def test_five_healers_split_three_goal_one_miner_one_raid(self):
        # 7 battles -> payload(1) + escort(3) + raid(1) + formation(2), as
        # in `test_battle_fleet_splits_into_payload_escort_raid_and_formation`.
        # 5 healers -> goal(3) + miner(1) + raid(1), per `healer_roles.py`.
        battles = [FakeBot(i, Vec2(5.0, 5.0), class_=BotClass.Battle) for i in range(7)]
        healers = [FakeBot(20 + i, Vec2(5.0, 5.0), class_=BotClass.Healer) for i in range(5)]
        state = FakeState(fleet_me=battles + healers, payload=Vec2(16.0, 16.0))
        conf = FakeConf()
        with _EnginePatch():
            plan1_pkg._recompute_assignments(state, conf)
        assignments = plan1_pkg._cache["assignments"]
        healer_roles = [assignments[h.id]["role"] for h in healers]
        # payload(1, dedicated) + combat(2, goal's other pair, idle since
        # nobody's hurt) + miner_support(1) + raid_support(1).
        self.assertEqual(sorted(healer_roles),
                          ["combat", "combat", "miner_support", "payload", "raid_support"])

    def test_miner_support_healer_heals_a_hurt_escort(self):
        battles = [FakeBot(i, Vec2(5.0, 5.0), class_=BotClass.Battle) for i in range(7)]
        healers = [FakeBot(20 + i, Vec2(5.0, 5.0), class_=BotClass.Healer) for i in range(5)]
        state = FakeState(fleet_me=battles + healers, payload=Vec2(16.0, 16.0))
        conf = FakeConf()
        with _EnginePatch():
            plan1_pkg._recompute_assignments(state, conf)
        assignments = plan1_pkg._cache["assignments"]
        # Escorts are battles {1, 2, 3} (remaining battles sorted by id,
        # first 3, after battle 0 takes the payload-defender fallback slot).
        escort_ids = {bid for bid, a in assignments.items() if a.get("role") == "guard_miners"}
        self.assertEqual(escort_ids, {1, 2, 3})
        hurt_escort = min(escort_ids)
        # Re-run with the escort hurt, and confirm the miner-support healer
        # picks it up as its patient.
        for b in battles:
            if b.id == hurt_escort:
                b.health = 2.0
        with _EnginePatch():
            plan1_pkg._recompute_assignments(state, conf)
        assignments = plan1_pkg._cache["assignments"]
        miner_healer_ids = [bid for bid, a in assignments.items() if a.get("role") == "miner_support"]
        self.assertEqual(len(miner_healer_ids), 1)
        self.assertEqual(assignments[miner_healer_ids[0]]["target_id"], hurt_escort)

    def test_raid_support_healer_heals_the_raider(self):
        battles = [FakeBot(i, Vec2(5.0, 5.0), class_=BotClass.Battle) for i in range(7)]
        healers = [FakeBot(20 + i, Vec2(5.0, 5.0), class_=BotClass.Healer) for i in range(5)]
        state = FakeState(fleet_me=battles + healers, payload=Vec2(16.0, 16.0))
        conf = FakeConf()
        with _EnginePatch():
            plan1_pkg._recompute_assignments(state, conf)
        assignments = plan1_pkg._cache["assignments"]
        raider_ids = [bid for bid, a in assignments.items() if a.get("role") == "raid"]
        self.assertEqual(len(raider_ids), 1)
        for b in battles:
            if b.id == raider_ids[0]:
                b.health = 1.0
        with _EnginePatch():
            plan1_pkg._recompute_assignments(state, conf)
        assignments = plan1_pkg._cache["assignments"]
        raid_healer_ids = [bid for bid, a in assignments.items() if a.get("role") == "raid_support"]
        self.assertEqual(len(raid_healer_ids), 1)
        self.assertEqual(assignments[raid_healer_ids[0]]["target_id"], raider_ids[0])

    def test_two_healers_both_default_to_goal(self):
        # Fewer than 3 healers -> all of them go to goal; miner/raid healer
        # groups stay empty rather than starving the goal line.
        battles = [FakeBot(i, Vec2(5.0, 5.0), class_=BotClass.Battle) for i in range(7)]
        healers = [FakeBot(20, Vec2(16.5, 16.0), class_=BotClass.Healer),
                   FakeBot(21, Vec2(5.0, 5.0), class_=BotClass.Healer)]
        state = FakeState(fleet_me=battles + healers, payload=Vec2(16.0, 16.0))
        conf = FakeConf()
        with _EnginePatch():
            plan1_pkg._recompute_assignments(state, conf)
        assignments = plan1_pkg._cache["assignments"]
        self.assertNotIn("miner_support", {assignments[h.id]["role"] for h in healers})
        self.assertNotIn("raid_support", {assignments[h.id]["role"] for h in healers})
        roles = sorted(assignments[h.id]["role"] for h in healers)
        self.assertEqual(roles, ["combat", "payload"])

    def test_recently_hit_formation_bot_backs_off_and_is_replaced_at_the_front(self):
        # 10 battles -> payload(1) + escort(3) + raid(2) + formation(4), the
        # formation group big enough (n=4 > 3) to have its own screen/column
        # split. Mark the formation group's front bot as freshly hit and
        # confirm it backs off while some previously-second-row bot is
        # promoted into the now-vacant front slot.
        battles = [FakeBot(i, Vec2(5.0, 5.0), class_=BotClass.Battle) for i in range(10)]
        state = FakeState(fleet_me=battles, payload=Vec2(16.0, 16.0), tick=50)
        conf = FakeConf()
        with _EnginePatch():
            plan1_pkg._recompute_assignments(state, conf)
        assignments = plan1_pkg._cache["assignments"]
        formation_ids = sorted(bid for bid, a in assignments.items() if a.get("role") == "combat")
        self.assertGreaterEqual(len(formation_ids), 3)
        forward = plan1_pkg._forward_direction(state)

        depths_before = {bid: round(assignments[bid]["move_target"].dot(forward), 4) for bid in formation_ids}
        front_depth = max(depths_before.values())
        front_ids_before = {bid for bid, d in depths_before.items() if d == front_depth}
        front_id = min(front_ids_before)

        # Front bot was just hit -- still inside its invulnerability window.
        for b in battles:
            if b.id == front_id:
                b.invulnerable_until_tick = state.tick + 10
        with _EnginePatch():
            plan1_pkg._recompute_assignments(state, conf)
        assignments = plan1_pkg._cache["assignments"]

        new_depth = round(assignments[front_id]["move_target"].dot(forward), 4)
        self.assertLess(new_depth, front_depth, "the hit bot must back off out of the front row")

        # Someone who was *not* already at the front before now is.
        depths_after = {bid: round(assignments[bid]["move_target"].dot(forward), 4) for bid in formation_ids}
        promoted = [bid for bid, d in depths_after.items()
                    if d == front_depth and bid not in front_ids_before]
        self.assertTrue(promoted, "a previously-non-front bot must take the vacated front slot")

    def test_support_healer_ends_up_behind_the_retreating_bot(self):
        # "Safely in the middle": once the goal healer's patient backs off
        # the front row, the healer -- which stands behind its patient --
        # ends up further back too, not still up at the old front-line spot.
        # Two healers so the goal group has one beyond the dedicated payload
        # defender (id 20) free to act as a combat healer (id 21).
        battles = [FakeBot(i, Vec2(5.0, 5.0), class_=BotClass.Battle) for i in range(10)]
        healers = [FakeBot(20, Vec2(5.0, 5.0), class_=BotClass.Healer),
                   FakeBot(21, Vec2(5.0, 5.0), class_=BotClass.Healer)]
        state = FakeState(fleet_me=battles + healers, payload=Vec2(16.0, 16.0), tick=50)
        conf = FakeConf()
        with _EnginePatch():
            plan1_pkg._recompute_assignments(state, conf)
        assignments = plan1_pkg._cache["assignments"]
        self.assertEqual(assignments[20]["role"], "payload")
        formation_ids = sorted(bid for bid, a in assignments.items() if a.get("role") == "combat" and bid < 20)
        front_id = formation_ids[0]
        forward = plan1_pkg._forward_direction(state)

        for b in battles:
            if b.id == front_id:
                b.health = 2.0
                b.invulnerable_until_tick = state.tick + 10
        with _EnginePatch():
            plan1_pkg._recompute_assignments(state, conf)
        assignments = plan1_pkg._cache["assignments"]

        healer_move_target = assignments[21]["move_target"]
        patient_depth = round(assignments[front_id]["move_target"].dot(forward), 4)
        healer_depth = round(healer_move_target.dot(forward), 4)
        self.assertEqual(assignments[21]["target_id"], front_id)
        self.assertLess(healer_depth, patient_depth)

    def test_goal_healer_leaves_at_most_two_formation_attackers_out_of_heal_range(self):
        # 10 battles -> formation group is 4 bots (ids 6-9, per the split
        # established in `test_recently_hit_formation_bot_backs_off_...`),
        # clustered close enough together for full coverage to be possible.
        # A second goal healer (beyond the dedicated payload one) should
        # position itself so at most 2 of {formation group + payload
        # defender} are left outside heal range.
        battles = [FakeBot(i, Vec2(5.0, 5.0), class_=BotClass.Battle) for i in range(10)]
        healers = [FakeBot(20, Vec2(16.5, 16.0), class_=BotClass.Healer),
                   FakeBot(21, Vec2(5.0, 5.0), class_=BotClass.Healer)]
        state = FakeState(fleet_me=battles + healers, payload=Vec2(16.0, 16.0))
        conf = FakeConf()
        with _EnginePatch():
            plan1_pkg._recompute_assignments(state, conf)
        assignments = plan1_pkg._cache["assignments"]
        self.assertEqual(assignments[20]["role"], "payload")
        self.assertEqual(assignments[21]["role"], "combat")

        goal_ally_ids = [bid for bid, a in assignments.items()
                         if bid < 20 and a.get("role") in ("payload", "combat")]
        move_target = assignments[21]["move_target"]
        out_of_range = sum(
            1 for bid in goal_ally_ids
            if move_target.dist(next(b for b in battles if b.id == bid).pos) > conf.bot.base_heal_range
        )
        self.assertLessEqual(out_of_range, 2)

    def test_miner_escort_healer_leaves_at_most_two_escorts_out_of_range(self):
        battles = [FakeBot(i, Vec2(5.0, 5.0), class_=BotClass.Battle) for i in range(7)]
        healers = [FakeBot(20 + i, Vec2(5.0, 5.0), class_=BotClass.Healer) for i in range(5)]
        state = FakeState(fleet_me=battles + healers, payload=Vec2(16.0, 16.0))
        conf = FakeConf()
        with _EnginePatch():
            plan1_pkg._recompute_assignments(state, conf)
        assignments = plan1_pkg._cache["assignments"]
        escort_ids = [bid for bid, a in assignments.items() if a.get("role") == "guard_miners"]
        miner_healer_ids = [bid for bid, a in assignments.items() if a.get("role") == "miner_support"]
        self.assertEqual(len(miner_healer_ids), 1)
        move_target = assignments[miner_healer_ids[0]]["move_target"]
        out_of_range = sum(
            1 for bid in escort_ids
            if move_target.dist(next(b for b in battles if b.id == bid).pos) > conf.bot.base_heal_range
        )
        self.assertLessEqual(out_of_range, 2)

    def test_every_battle_group_that_has_a_healer_is_covered_by_it(self):
        # "No attackers without healers": every squad that was allotted a
        # healer (see `healer_roles.py`) must actually get one positioned
        # for it -- never an empty move_target/target_id, and never left to
        # the generic payload-heading catch-all.
        battles = [FakeBot(i, Vec2(5.0, 5.0), class_=BotClass.Battle) for i in range(7)]
        healers = [FakeBot(20 + i, Vec2(5.0, 5.0), class_=BotClass.Healer) for i in range(5)]
        state = FakeState(fleet_me=battles + healers, payload=Vec2(16.0, 16.0))
        conf = FakeConf()
        with _EnginePatch():
            plan1_pkg._recompute_assignments(state, conf)
        assignments = plan1_pkg._cache["assignments"]
        healer_roles = {bid: assignments[bid]["role"] for bid in range(20, 25)}
        self.assertEqual(sorted(healer_roles.values()),
                          sorted(["payload", "combat", "combat", "miner_support", "raid_support"]))
        for bid in range(20, 25):
            self.assertIsNotNone(assignments[bid].get("move_target"))

    def test_every_bot_gets_an_assignment(self):
        fleet = [
            FakeBot(0, Vec2(2, 2), class_=BotClass.Extractor),
            FakeBot(1, Vec2(15, 15), class_=BotClass.Battle),
            FakeBot(2, Vec2(15, 16), class_=BotClass.Battle),
            FakeBot(3, Vec2(15, 17), class_=BotClass.Healer),
        ]
        state = FakeState(fleet_me=fleet, payload=Vec2(16.0, 16.0))
        conf = FakeConf()
        with _EnginePatch():
            plan1_pkg._recompute_assignments(state, conf)
        assignments = plan1_pkg._cache["assignments"]
        for bot in fleet:
            self.assertIn(bot.id, assignments)


class ApplyAssignmentsTests(unittest.TestCase):
    def test_produces_a_valid_action_for_every_bot(self):
        fleet = [
            FakeBot(0, Vec2(2, 2), class_=BotClass.Extractor),
            FakeBot(1, Vec2(15, 15), class_=BotClass.Battle),
            FakeBot(2, Vec2(15, 16), class_=BotClass.Healer),
        ]
        enemy = [FakeBot(10, Vec2(17, 17), class_=BotClass.Battle)]
        state = FakeState(fleet_me=fleet, fleet_other=enemy, payload=Vec2(16.0, 16.0))
        conf = FakeConf()
        action = FleetAction.new()
        with _EnginePatch():
            plan1_pkg._apply_assignments(action, state, conf, {}, enemy_in_capture=False)
        for bot in fleet:
            self.assertIsNotNone(action.bots[bot.id])

    def test_battle_does_not_fire_at_out_of_range_target(self):
        bot = FakeBot(0, Vec2(0, 0), class_=BotClass.Battle)
        enemy = FakeBot(10, Vec2(100.0, 100.0), class_=BotClass.Battle)
        state = FakeState(fleet_me=[bot], fleet_other=[enemy], payload=Vec2(16, 16))
        conf = FakeConf()
        assignments = {0: {"kind": "battle", "target_id": 10, "move_target": Vec2(16, 16)}}
        action = FleetAction.new()
        with _EnginePatch():
            plan1_pkg._apply_assignments(action, state, conf, assignments, enemy_in_capture=False)
        self.assertFalse(action.bots[0].special_action.as_battle.fire)

    def test_battle_holds_fire_on_invulnerable_target_in_range(self):
        bot = FakeBot(0, Vec2(0, 0), class_=BotClass.Battle)
        enemy = FakeBot(10, Vec2(1.0, 0.0), class_=BotClass.Battle, invulnerable_until_tick=999)
        state = FakeState(tick=5, fleet_me=[bot], fleet_other=[enemy], payload=Vec2(16, 16))
        conf = FakeConf()
        assignments = {0: {"kind": "battle", "target_id": 10, "move_target": Vec2(16, 16)}}
        action = FleetAction.new()
        with _EnginePatch():
            plan1_pkg._apply_assignments(action, state, conf, assignments, enemy_in_capture=False)
        self.assertFalse(action.bots[0].special_action.as_battle.fire)

    def test_battle_fires_on_hittable_in_range_clear_target(self):
        bot = FakeBot(0, Vec2(0, 0), class_=BotClass.Battle)
        enemy = FakeBot(10, Vec2(1.0, 0.0), class_=BotClass.Battle, invulnerable_until_tick=0)
        state = FakeState(tick=5, fleet_me=[bot], fleet_other=[enemy], payload=Vec2(16, 16))
        conf = FakeConf()
        assignments = {0: {"kind": "battle", "target_id": 10, "move_target": Vec2(16, 16)}}
        action = FleetAction.new()
        with _EnginePatch():
            plan1_pkg._apply_assignments(action, state, conf, assignments, enemy_in_capture=False)
        self.assertTrue(action.bots[0].special_action.as_battle.fire)

    def test_battle_holds_position_to_fire_instead_of_walking_off_the_los(self):
        """The engine moves every bot before any blaster fires that same
        tick, so a bot that keeps walking toward its formation slot while
        it fires shoots from wherever that step lands it, not from the spot
        `_has_clear_shot` just verified -- fine in the open, but a single
        step is enough to swing a ray from clear to clipping a wall's
        corner. A firing bot must hold position for the tick it shoots."""
        bot = FakeBot(0, Vec2(0, 0), class_=BotClass.Battle)
        enemy = FakeBot(10, Vec2(1.0, 0.0), class_=BotClass.Battle, invulnerable_until_tick=0)
        state = FakeState(tick=5, fleet_me=[bot], fleet_other=[enemy], payload=Vec2(16, 16))
        conf = FakeConf()
        # A formation slot far from the bot's current position -- if the bot
        # still steers toward it this tick, the shot would leave from a
        # different point than the one just checked for a clear line.
        assignments = {0: {"kind": "battle", "target_id": 10, "move_target": Vec2(16, 16)}}
        action = FleetAction.new()
        with _EnginePatch():
            plan1_pkg._apply_assignments(action, state, conf, assignments, enemy_in_capture=False)
        self.assertTrue(action.bots[0].special_action.as_battle.fire)
        move = action.bots[0].move_action.direction
        self.assertAlmostEqual(move.x, 0.0, places=6)
        self.assertAlmostEqual(move.y, 0.0, places=6)

    def test_raider_holds_position_when_an_enemy_closes_to_blaster_range(self):
        """A raider en route to the enemy deposit shouldn't keep walking
        deeper into enemy territory once a fight has already started at
        blaster range -- it should hold and clear that threat first."""
        bot = FakeBot(0, Vec2(0, 0), class_=BotClass.Battle)
        enemy = FakeBot(10, Vec2(1.0, 0.0), class_=BotClass.Battle, invulnerable_until_tick=0)
        state = FakeState(tick=5, fleet_me=[bot], fleet_other=[enemy], payload=Vec2(16, 16))
        conf = FakeConf()
        # A raid move_target far past the enemy that's already engaging it.
        assignments = {0: {"kind": "battle", "role": "raid", "target_id": 10,
                            "move_target": Vec2(30, 30)}}
        action = FleetAction.new()
        with _EnginePatch():
            plan1_pkg._apply_assignments(action, state, conf, assignments, enemy_in_capture=False)
        move = action.bots[0].move_action.direction
        self.assertAlmostEqual(move.x, 0.0, places=6)
        self.assertAlmostEqual(move.y, 0.0, places=6)

    def test_retreating_bot_still_falls_back_despite_a_nearby_enemy(self):
        """A bot still inside its post-hit invulnerability window is the one
        bot that should keep moving (back into formation to get healed),
        even though some enemy is within blaster range -- the hold-in-place
        rule is for healthy bots, not one already falling back. The nearby
        enemy here isn't this bot's own assigned target, so it can't also be
        held by the (separate, pre-existing) "stop to take a clear shot"
        behavior -- only the new retreat-exempt hold logic is in play."""
        bot = FakeBot(0, Vec2(0, 0), class_=BotClass.Battle, invulnerable_until_tick=10)
        near_enemy = FakeBot(11, Vec2(1.0, 0.0), class_=BotClass.Battle)
        far_target = FakeBot(10, Vec2(100.0, 100.0), class_=BotClass.Battle)
        state = FakeState(tick=5, fleet_me=[bot], fleet_other=[near_enemy, far_target],
                           payload=Vec2(16, 16))
        conf = FakeConf()
        assignments = {0: {"kind": "battle", "role": "combat", "target_id": 10,
                            "move_target": Vec2(-5, 0)}}
        action = FleetAction.new()
        with _EnginePatch():
            plan1_pkg._apply_assignments(action, state, conf, assignments, enemy_in_capture=False)
        move = action.bots[0].move_action.direction
        self.assertAlmostEqual(move.x, -5.0, places=6)
        self.assertAlmostEqual(move.y, 0.0, places=6)

    def test_formation_bot_still_advances_when_no_enemy_is_near(self):
        """With nothing within blaster range yet, a bot should still close
        in on its assigned formation slot rather than holding early."""
        bot = FakeBot(0, Vec2(0, 0), class_=BotClass.Battle)
        enemy = FakeBot(10, Vec2(100.0, 100.0), class_=BotClass.Battle)
        state = FakeState(tick=5, fleet_me=[bot], fleet_other=[enemy], payload=Vec2(16, 16))
        conf = FakeConf()
        assignments = {0: {"kind": "battle", "role": "combat", "target_id": 10,
                            "move_target": Vec2(5, 0)}}
        action = FleetAction.new()
        with _EnginePatch():
            plan1_pkg._apply_assignments(action, state, conf, assignments, enemy_in_capture=False)
        move = action.bots[0].move_action.direction
        self.assertAlmostEqual(move.x, 5.0, places=6)
        self.assertAlmostEqual(move.y, 0.0, places=6)

    def test_stale_dead_target_falls_back_without_crashing(self):
        bot = FakeBot(0, Vec2(0, 0), class_=BotClass.Battle)
        live_enemy = FakeBot(11, Vec2(1.0, 0.0), class_=BotClass.Battle)
        state = FakeState(fleet_me=[bot], fleet_other=[live_enemy], payload=Vec2(16, 16))
        conf = FakeConf()
        # target_id=999 no longer exists in fleet_other -- must fall back, not KeyError.
        assignments = {0: {"kind": "battle", "target_id": 999, "move_target": Vec2(16, 16)}}
        action = FleetAction.new()
        with _EnginePatch():
            plan1_pkg._apply_assignments(action, state, conf, assignments, enemy_in_capture=False)
        # Should have picked up the live enemy instead of leaving no target at all --
        # it's in range and clear, so the fallback should let it fire.
        self.assertTrue(action.bots[0].special_action.as_battle.fire)


class IsInHealArcTests(unittest.TestCase):
    def test_target_dead_ahead_is_in_arc(self):
        healer = FakeBot(0, Vec2(0, 0), angle=0.0)
        target = FakeBot(1, Vec2(5.0, 0.0))
        conf = FakeConf()
        self.assertTrue(plan1_pkg._is_in_heal_arc(healer, target, conf))

    def test_target_directly_behind_is_out_of_arc(self):
        healer = FakeBot(0, Vec2(0, 0), angle=0.0)
        target = FakeBot(1, Vec2(-5.0, 0.0))
        conf = FakeConf()
        conf.bot.base_heal_arc_deg = 90.0
        self.assertFalse(plan1_pkg._is_in_heal_arc(healer, target, conf))

    def test_same_position_counts_as_in_arc(self):
        healer = FakeBot(0, Vec2(5.0, 5.0), angle=123.0)
        target = FakeBot(1, Vec2(5.0, 5.0))
        conf = FakeConf()
        self.assertTrue(plan1_pkg._is_in_heal_arc(healer, target, conf))


class PlanStrategyRecomputeCadenceTests(unittest.TestCase):
    """`plan1_strategy` itself, with `get_budget`/`get_config` patched --
    the only two engine calls it makes directly beyond what `_recompute_/
    _apply_assignments` already need `_EnginePatch` for."""

    def setUp(self):
        _clear_cache()

    def tearDown(self):
        _clear_cache()

    def _run(self, state, conf, budget_remaining=500_000):
        with _EnginePatch(), \
             patch.object(plan1_pkg, "get_budget", return_value=FakeBudget(budget_remaining)), \
             patch.object(plan1_pkg, "get_config", return_value=conf):
            return plan1_pkg.plan1_strategy(state)

    def test_first_tick_recomputes(self):
        fleet = [FakeBot(0, Vec2(2, 2), class_=BotClass.Extractor)]
        state = FakeState(tick=1, fleet_me=fleet, payload=Vec2(16, 16))
        conf = FakeConf()
        self._run(state, conf)
        self.assertEqual(plan1_pkg._cache["last_assignment_tick"], 1)

    def test_new_bot_triggers_recompute_before_five_ticks(self):
        fleet = [FakeBot(0, Vec2(2, 2), class_=BotClass.Extractor)]
        state = FakeState(tick=1, fleet_me=fleet, payload=Vec2(16, 16))
        conf = FakeConf()
        self._run(state, conf)
        # Tick 2: a new bot appears. Even though it's only 1 tick later
        # (well under the 5-tick floor), this must still trigger a recompute.
        fleet2 = fleet + [FakeBot(1, Vec2(2, 2), class_=BotClass.Battle)]
        state2 = FakeState(tick=2, fleet_me=fleet2, payload=Vec2(16, 16))
        self._run(state2, conf)
        self.assertEqual(plan1_pkg._cache["last_assignment_tick"], 2)
        self.assertIn(1, plan1_pkg._cache["assignments"])

    def test_low_budget_skips_recompute(self):
        fleet = [FakeBot(0, Vec2(2, 2), class_=BotClass.Extractor)]
        state = FakeState(tick=1, fleet_me=fleet, payload=Vec2(16, 16))
        conf = FakeConf()
        self._run(state, conf, budget_remaining=1)
        # Never recomputed -- last_assignment_tick stays at its default (-100).
        self.assertEqual(plan1_pkg._cache.get("last_assignment_tick", -100), -100)


if __name__ == "__main__":
    unittest.main()
