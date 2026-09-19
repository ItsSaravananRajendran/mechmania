"""Plan 1 -- The Payload Squeeze.

See plans/plan1.md for the full spec (requirements R-P1.* and R-X*).

Each algorithm lives in its own module so a change to one (say, formation
spacing) never has to touch the others:

- `cache`: the one piece of state kept between ticks, shared by the rest.
- `walls`: the static wall grid built once from `conf.map`.
- `combat_los`: clear-shot checks against walls, deposits, and the payload.
- `targeting`: fleet-wide, cooldown-aware target assignment.
- `formation`: zone/positioning so bots don't cluster.
- `fabricator`: build-order.
- `endgame`: the endgame self-destruct call.

This file is the glue: it wires a tick's `GameState` through those pieces
into a `FleetAction`, and owns the assignment cache/recompute cadence that
ties them together.
"""

from __future__ import annotations

from typing import Dict, Optional, Set

from .. import (
    BotClass,
    BotState,
    FleetAction,
    GameState,
    SpecialAction,
    Vec2,
    diff_degrees,
    get_budget,
    get_config,
    line_of_sight,
    turn_towards,
)
from .cache import _cache
from .combat_los import _blocking_obstacle, _flank_point, _has_clear_shot
from .corners import _steer
from .endgame import _maybe_endgame_self_destruct
from .fabricator import _compute_fabricator_next
from .formation import (
    _arrange_group,
    _compute_battle_formation,
    _compute_mining_spots,
    _dedupe_slots,
    _forward_direction,
    _formation_scale,
    _resolve_slot,
)
from .targeting import _assign_battle_targets
from .walls import _get_wall_grid


def plan1_strategy(state: GameState) -> FleetAction:
    budget = get_budget()
    conf = get_config()
    in_endgame = state.tick >= conf.max_ticks - conf.endgame_ticks

    last_tick = _cache.get("last_tick", -1)
    if last_tick >= 0 and state.tick - last_tick > 1:
        _cache["assignments"] = {}

    payload = state.payload_pos()
    capture_radius = conf.payload.capture_radius

    enemy_count = len(state.fleet_other)
    last_enemy_count = _cache.get("last_enemy_count", enemy_count)

    friendly_ids = frozenset(state.fleet_me.ids())
    last_friendly_ids = _cache.get("last_friendly_ids", frozenset())
    friendly_died = last_friendly_ids - friendly_ids
    # A newly built/rushed bot has no entry in the cached assignment dict
    # until the next recompute -- until then it falls back to "just head for
    # the payload", ignoring formation/targeting entirely. Fleets can grow
    # every few ticks (free build + rush orders), so waiting out the regular
    # 5-tick cadence left a steady trickle of unassigned bots piling onto the
    # payload point. Recompute as soon as one appears instead.
    friendly_born = friendly_ids - last_friendly_ids

    friendly_in_capture = frozenset(
        b.id for b in state.fleet_me if b.pos.dist(payload) <= capture_radius
    )
    last_friendly_in_capture = _cache.get("last_friendly_in_capture", frozenset())

    last_assignment_tick = _cache.get("last_assignment_tick", -100)

    enemy_delta = abs(enemy_count - last_enemy_count)
    capture_changed = friendly_in_capture != last_friendly_in_capture

    needs_recompute = (
        last_assignment_tick < 0
        or state.tick - last_assignment_tick >= 5
        or enemy_delta >= 2
        or capture_changed
        or bool(friendly_died)
        or bool(friendly_born)
    )

    if budget.remaining <= 100_000:
        action = _execute_assignments(state, conf)
    elif needs_recompute:
        action = _recompute_assignments(state, conf)
        _cache["last_assignment_tick"] = state.tick
    else:
        action = _execute_assignments(state, conf)

    _cache["last_tick"] = state.tick
    _cache["last_enemy_count"] = enemy_count
    _cache["last_friendly_ids"] = friendly_ids
    _cache["last_friendly_in_capture"] = friendly_in_capture

    action.fabricator_next = _compute_fabricator_next(state, conf, _cache)

    action.rush_order = (
        not in_endgame
        and state.fabricator_me.tokens >= conf.fabricator.rush_cost
    )

    if in_endgame and not _cache.get("endgame_sd_done", False):
        _maybe_endgame_self_destruct(state, conf, action, _cache)

    return action


def _recompute_assignments(state: GameState, conf) -> FleetAction:
    action = FleetAction.new()
    payload = state.payload_pos()
    capture_radius = conf.payload.capture_radius
    wall_grid = _get_wall_grid(conf)
    forward = _forward_direction(state)

    healers = [b for b in state.fleet_me if b.class_ == BotClass.Healer]
    # Sorted by id so the screen/stand-off split is stable across recomputes
    # -- otherwise fleet-iteration order reshuffling which bot lands in which
    # zone every few ticks reads as bots drifting rather than holding a slot.
    battles = sorted((b for b in state.fleet_me if b.class_ == BotClass.Battle), key=lambda b: b.id)
    extractors = [b for b in state.fleet_me if b.class_ == BotClass.Extractor]

    assignments: Dict[int, dict] = {}

    enemy_in_capture = any(
        e.pos.dist(payload) <= capture_radius for e in state.fleet_other
    )

    battle_targets = _assign_battle_targets(battles, state, conf)
    mining_spots = _compute_mining_spots(extractors, state, conf, wall_grid)

    for ext in extractors[:3]:
        assignments[ext.id] = {
            "kind": "extractor",
            "mining_spot": mining_spots.get(ext.id),
        }

    # The dedicated payload defenders (one healer, one battle bot) both need
    # to stay within `capture_radius` to contest the payload, but that's a
    # 2.5+ unit-wide zone, not a single point -- pin them both to the exact
    # same coordinate and one splash hit on the payload takes out both at
    # once. Split them a little to either side of it instead.
    payload_defense_offset = min(1.0, capture_radius * 0.4)

    payload_healer_id: Optional[int] = None
    if healers:
        in_cap = [h for h in healers if h.pos.dist(payload) <= capture_radius]
        if in_cap:
            payload_healer_id = in_cap[0].id
        else:
            payload_healer_id = min(healers, key=lambda h: h.pos.dist(payload)).id

        candidates = [b for b in state.fleet_me
                      if b.id != payload_healer_id
                      and b.pos.dist(payload) <= capture_radius]
        target_id = min(candidates, key=lambda b: b.health).id if candidates else None

        assignments[payload_healer_id] = {
            "kind": "healer",
            "role": "payload",
            "target_id": target_id,
            "move_target": payload - forward * payload_defense_offset,
        }

    payload_battle_id: Optional[int] = None
    if battles:
        in_cap = [b for b in battles if b.pos.dist(payload) <= capture_radius]
        battle_candidates = [b for b in (in_cap if in_cap else battles)
                           if b.id != (payload_healer_id if payload_healer_id else -1)]
        if battle_candidates:
            payload_battle_id = battle_candidates[0].id
        else:
            payload_battle_id = min(battles, key=lambda b: b.pos.dist(payload)).id

        assignments[payload_battle_id] = {
            "kind": "battle",
            "role": "payload",
            "target_id": battle_targets.get(payload_battle_id),
            "move_target": payload + forward * payload_defense_offset,
        }

    remaining_healers = [h for h in healers if h.id != payload_healer_id]
    idle_healers = []
    for h in remaining_healers:
        candidates = [b for b in state.fleet_me
                      if b.id != h.id and b.health < conf.bot.health * 0.9]
        if candidates:
            target = min(candidates, key=lambda b: b.health)
            # Stand near the patient, not literally on it -- heal range
            # (`base_heal_range`) is well past melee distance, and a healer
            # stacked on the ally it's healing hands the enemy a single
            # splash hit that can down both of them at once.
            standoff = min(2.0, conf.bot.base_heal_range * 0.5)
            assignments[h.id] = {
                "kind": "healer",
                "role": "combat",
                "target_id": target.id,
                "move_target": target.pos - forward * standoff,
            }
        else:
            idle_healers.append(h)

    if idle_healers:
        # Nothing to heal right now -- previously these all defaulted to the
        # literal payload point and stacked there, which is exactly the kind
        # of splash-bait clustering formation is meant to avoid. Give them a
        # rear zone instead, behind the battle stand-off line.
        perp = forward.rotate_deg(90.0)
        spacing, min_spacing, max_width, row_gap = _formation_scale(conf)
        rear_center = payload - forward * max(6.0 * conf.bot.radius, conf.bot.blaster_range * 0.8)
        rear_slots, rear_anchors = _arrange_group(
            wall_grid, rear_center, -forward, perp, idle_healers, spacing, min_spacing, max_width, row_gap
        )
        rear_resolved = {
            h.id: _resolve_slot(wall_grid, rear_slots.get(h.id, rear_anchors.get(h.id, rear_center)),
                                 rear_anchors.get(h.id, rear_center))
            for h in idle_healers
        }
        rear_resolved = _dedupe_slots(rear_resolved, perp, spacing)
        for h in idle_healers:
            slot = rear_resolved[h.id]
            assignments[h.id] = {
                "kind": "healer",
                "role": "combat",
                "target_id": None,
                "move_target": slot,
            }

    remaining_battles = [b for b in battles if b.id != payload_battle_id]
    formation_slots = _compute_battle_formation(remaining_battles, payload, forward, conf, wall_grid)
    for battle in remaining_battles:
        assignments[battle.id] = {
            "kind": "battle",
            "role": "combat",
            "target_id": battle_targets.get(battle.id),
            "move_target": formation_slots.get(battle.id, payload),
        }

    for bot in state.fleet_me:
        if bot.id in assignments:
            continue
        assignments[bot.id] = {
            "kind": bot.class_.name.lower(),
            "role": "combat",
            "target_id": battle_targets.get(bot.id) if bot.class_ == BotClass.Battle else None,
            "move_target": payload,
        }

    _cache["assignments"] = assignments
    _apply_assignments(action, state, conf, assignments, enemy_in_capture)
    return action


def _execute_assignments(state: GameState, conf) -> FleetAction:
    action = FleetAction.new()
    payload = state.payload_pos()
    capture_radius = conf.payload.capture_radius
    assignments = _cache.get("assignments", {})

    enemy_in_capture = any(
        e.pos.dist(payload) <= capture_radius for e in state.fleet_other
    )

    _apply_assignments(action, state, conf, assignments, enemy_in_capture)
    return action


def _is_in_heal_arc(healer: BotState, target: BotState, conf) -> bool:
    to_target = target.pos - healer.pos
    if to_target.norm_sq() < 0.0001:
        return True
    angle_to_target = to_target.angle_deg()
    diff = abs(diff_degrees(angle_to_target, healer.angle))
    half_arc = conf.bot.base_heal_arc_deg / 2.0
    return diff <= half_arc + 1e-6


def _apply_assignments(action: FleetAction, state: GameState, conf,
                       assignments: Dict, enemy_in_capture: bool) -> None:
    payload = state.payload_pos()
    enemies_by_id = {e.id: e for e in state.fleet_other}
    wall_grid = _get_wall_grid(conf)

    # Bots whose cached target has since died (or that never got an
    # assignment at all, e.g. built between recomputes) need a fallback --
    # but picking "nearest enemy" independently per bot is exactly the
    # dogpile behaviour targeting was built to avoid. Track which enemies are
    # already spoken for (by a live cached assignment, or by an earlier bot's
    # fallback pick this same tick) so fallbacks spread out too.
    claimed_targets: Set[int] = {
        a["target_id"]
        for a in assignments.values()
        if a.get("kind") == "battle" and a.get("target_id") in enemies_by_id
    }

    for bot in state.fleet_me:
        bot_action = action.bots[bot.id]
        assignment = assignments.get(bot.id, {})

        kind = assignment.get("kind", bot.class_.name.lower())
        role = assignment.get("role", "combat")

        if kind == "extractor" or bot.class_ == BotClass.Extractor:
            mining_spot = assignment.get("mining_spot")
            if mining_spot is None:
                mining_spot = state.deposit_me.pos + Vec2(
                    0.0, conf.deposit.radius + conf.bot.radius
                )
            bot_action.move_action = _steer(bot.pos, mining_spot, wall_grid, conf)
            bot_action.turn_action = turn_towards(state.deposit_me.pos)
            bot_action.special_action = SpecialAction.Extractor(mine=True)

        elif kind == "healer" or bot.class_ == BotClass.Healer:
            target_id = assignment.get("target_id")
            target = state.fleet_me.get(target_id) if target_id is not None else None

            if role == "payload":
                if enemy_in_capture:
                    nearest = min(state.fleet_other, key=lambda e: e.pos.dist(bot.pos), default=None)
                    if nearest:
                        bot_action.turn_action = turn_towards(nearest.pos)
                else:
                    bot_action.turn_action = turn_towards(payload)
            else:
                if target is not None:
                    bot_action.turn_action = turn_towards(target.pos)

            if target is not None and bot.class_ == BotClass.Healer:
                in_range = bot.pos.dist(target.pos) <= conf.bot.base_heal_range
                in_arc = _is_in_heal_arc(bot, target, conf)
                bot_action.special_action = SpecialAction.Healer(
                    fire=in_range and in_arc,
                    target=target.id,
                )
            else:
                bot_action.special_action = SpecialAction.Healer(fire=False, target=0)

            move_target = assignment.get("move_target", payload)
            bot_action.move_action = _steer(bot.pos, move_target, wall_grid, conf)

        elif kind == "battle" or bot.class_ == BotClass.Battle:
            target_id = assignment.get("target_id")
            target = enemies_by_id.get(target_id) if target_id is not None else None

            # The cached assignment can go stale between recomputes (target
            # died, or the state gap-detection above cleared the cache) --
            # fall back to nearest-by-straight-line rather than stand idle,
            # preferring an enemy nothing else has claimed this tick yet.
            if target is None and state.fleet_other:
                unclaimed = [e for e in state.fleet_other if e.id not in claimed_targets]
                pool = unclaimed if unclaimed else state.fleet_other
                target = min(pool, key=lambda e: bot.pos.dist_sq(e.pos))
                claimed_targets.add(target.id)

            move_target = assignment.get("move_target", payload)

            if target is not None:
                in_range = bot.pos.dist(target.pos) <= conf.bot.blaster_range
                clear_shot = in_range and _has_clear_shot(bot.pos, target.pos, state, conf)
                # A hit target is invulnerable for a few ticks -- firing at it
                # anyway would only burn our own cooldown on a shot that can't
                # land, at the cost of a bot that could be shooting something
                # else. Hold fire until the window closes.
                hittable = target.invulnerable_until_tick <= state.tick
                bot_action.special_action = SpecialAction.Battle(fire=clear_shot and hittable)
                bot_action.turn_action = turn_towards(target.pos)

                # In range but no clear shot: something is in the way, so
                # break formation just enough to get one. A wall calls for
                # routing around it (`navigate_to` already does that towards
                # any point); a deposit or the payload isn't part of that
                # wall topology, so step around it explicitly instead.
                if in_range and not clear_shot:
                    if not line_of_sight(bot.pos, target.pos):
                        move_target = target.pos
                    else:
                        obstacle = _blocking_obstacle(bot.pos, target.pos, state, conf)
                        if obstacle is not None:
                            obstacle_pos, obstacle_radius = obstacle
                            clearance = obstacle_radius + conf.bot.radius * 3.0
                            move_target = _flank_point(bot.pos, target.pos, obstacle_pos, clearance)
            else:
                bot_action.special_action = SpecialAction.Battle(fire=False)
                bot_action.turn_action = turn_towards(payload)

            bot_action.move_action = _steer(bot.pos, move_target, wall_grid, conf)

        else:
            bot_action.move_action = _steer(bot.pos, payload, wall_grid, conf)
            if bot.class_ == BotClass.Extractor:
                bot_action.special_action = SpecialAction.Extractor(mine=True)
            elif bot.class_ == BotClass.Healer:
                bot_action.special_action = SpecialAction.Healer(fire=False, target=0)
            else:
                bot_action.special_action = SpecialAction.Battle(fire=False)
