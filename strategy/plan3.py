"""Plan 3 — Economic Tempo.

See plans/plan3.md for the full spec (requirements R-P3.* and R-X*).
"""

from __future__ import annotations

import math
from typing import Dict, FrozenSet, List, Optional, Tuple

from . import (
    BotAction,
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
    move_bot,
    navigate_to,
    path_length,
    turn_towards,
)

_cache: Dict[str, object] = {}


def plan3_strategy(state: GameState) -> FleetAction:
    budget = get_budget()
    conf = get_config()
    in_endgame = state.tick >= conf.max_ticks - conf.endgame_ticks

    last_tick = _cache.get("last_tick", -1)
    if last_tick >= 0 and state.tick - last_tick > 1:
        _cache["assignments"] = {}

    payload = state.payload_pos()
    capture_radius = conf.payload.capture_radius
    bot_radius = conf.bot.radius

    enemy_count = len(state.fleet_other)
    last_enemy_count = _cache.get("last_enemy_count", enemy_count)

    friendly_ids = frozenset(state.fleet_me.ids())
    last_friendly_ids = _cache.get("last_friendly_ids", frozenset())
    friendly_died = last_friendly_ids - friendly_ids

    friendly_in_capture = frozenset(
        b.id for b in state.fleet_me if b.pos.dist(payload) <= capture_radius
    )
    last_friendly_in_capture = _cache.get("last_friendly_in_capture", frozenset())

    last_assignment_tick = _cache.get("last_assignment_tick", -100)

    enemy_delta = abs(enemy_count - last_enemy_count)
    capture_changed = friendly_in_capture != last_friendly_in_capture

    payload_past_center = (
        payload.dist(state.deposit_me.pos) < payload.dist(state.deposit_other.pos)
    )

    needs_recompute = (
        last_assignment_tick < 0
        or state.tick - last_assignment_tick >= 5
        or enemy_delta >= 2
        or capture_changed
        or bool(friendly_died)
        or (state.tick == 60 and payload_past_center)
    )

    pre_tick60 = state.tick < 60
    fleet_size = len(state.fleet_me)
    bodyguard_pair_assigned = _cache.get("bodyguard_pair_assigned", False)

    should_rotate = (
        state.tick == 60
        and fleet_size >= 10
        and not pre_tick60
        and not _cache.get("rotation_done", False)
        and not payload_past_center
    )
    if should_rotate and budget.remaining <= 300_000:
        should_rotate = False

    if budget.remaining <= 100_000:
        action = _execute_fallback(state, conf, in_endgame)
    elif needs_recompute or should_rotate:
        action = _recompute_assignments(state, conf, in_endgame, should_rotate)
        _cache["last_assignment_tick"] = state.tick
        if should_rotate:
            _cache["rotation_done"] = True
    else:
        action = _execute_assignments(state, conf, in_endgame)

    _cache["last_tick"] = state.tick
    _cache["last_enemy_count"] = enemy_count
    _cache["last_friendly_ids"] = friendly_ids
    _cache["last_friendly_in_capture"] = friendly_in_capture

    action.fabricator_next = _compute_fabricator_next(state, conf, _cache, in_endgame)

    rush_order = (
        not in_endgame
        and state.fabricator_me.tokens >= conf.fabricator.rush_cost
    )
    first_rush_tick = _cache.get("first_rush_tick", -1)
    second_rush_tick = _cache.get("second_rush_tick", -1)

    if rush_order:
        if first_rush_tick < 0 and state.tick >= 120:
            _cache["first_rush_tick"] = state.tick
        elif first_rush_tick >= 0 and second_rush_tick < 0 and state.tick >= first_rush_tick + 20:
            _cache["second_rush_tick"] = state.tick

    action.rush_order = rush_order

    if in_endgame:
        _maybe_endgame_self_destruct(state, conf, action, _cache)

    return action


def _recompute_assignments(state: GameState, conf, in_endgame: bool,
                           should_rotate: bool) -> FleetAction:
    action = FleetAction.new()
    payload = state.payload_pos()
    capture_radius = conf.payload.capture_radius
    bot_radius = conf.bot.radius

    healers = [b for b in state.fleet_me if b.class_ == BotClass.Healer]
    battles = [b for b in state.fleet_me if b.class_ == BotClass.Battle]
    extractors = [b for b in state.fleet_me if b.class_ == BotClass.Extractor]

    enemy_in_capture = any(
        e.pos.dist(payload) <= capture_radius for e in state.fleet_other
    )

    path_cache: Dict[Tuple[int, int], Optional[float]] = {}
    for battle in battles:
        for enemy in state.fleet_other:
            d = path_length(battle.pos, enemy.pos)
            path_cache[(battle.id, enemy.id)] = d

    mining_spot_base = state.deposit_me.pos + Vec2(0.0, conf.deposit.radius + bot_radius)

    assignments: Dict[int, dict] = {}

    pre_tick60 = state.tick < 60
    use_ranked_targets = state.tick >= 80

    if pre_tick60 and not should_rotate:
        bodyguard_pos = _compute_bodyguard_hold(state, conf)
        _cache["bodyguard_pair_assigned"] = True

        extractor_cap_per_team = conf.deposit.extractor_cap // 2
        for idx, ext in enumerate(extractors):
            if idx >= extractor_cap_per_team:
                break
            offset_x = idx * 2.0 * bot_radius
            spot = Vec2(mining_spot_base.x + offset_x, mining_spot_base.y)
            assignments[ext.id] = {
                "kind": "extractor",
                "mining_spot": spot,
            }

        if healers:
            lead_healer = healers[0]
            assignments[lead_healer.id] = {
                "kind": "healer",
                "role": "bodyguard",
                "move_target": bodyguard_pos,
                "target_id": None,
            }
            if len(healers) >= 2:
                bodyguard_healer = healers[1]
                assignments[bodyguard_healer.id] = {
                    "kind": "healer",
                    "role": "bodyguard",
                    "move_target": bodyguard_pos,
                    "target_id": None,
                }

        if battles and len(battles) >= 2:
            for b in battles[:2]:
                assignments[b.id] = {
                    "kind": "battle",
                    "role": "bodyguard",
                    "move_target": bodyguard_pos,
                    "target_id": None,
                }
        elif battles:
            assignments[battles[0].id] = {
                "kind": "battle",
                "role": "bodyguard",
                "move_target": bodyguard_pos,
                "target_id": None,
            }

        remaining_healers = [h for h in healers if h.id not in assignments]
        for h in remaining_healers:
            candidates = [b for b in state.fleet_me
                          if b.id != h.id and b.health < conf.bot.health * 0.9]
            target = min(candidates, key=lambda b: b.health) if candidates else None
            assignments[h.id] = {
                "kind": "healer",
                "role": "combat",
                "target_id": target.id if target else None,
                "move_target": bodyguard_pos,
            }

        remaining_battles = [b for b in battles if b.id not in assignments]
        for b in remaining_battles:
            target = _pick_target_enemy(b, state, path_cache) if use_ranked_targets else None
            assignments[b.id] = {
                "kind": "battle",
                "role": "combat",
                "target_id": target.id if target else None,
                "move_target": bodyguard_pos,
            }

    elif should_rotate:
        _cache["bodyguard_pair_assigned"] = False
        target_positions = _compute_payload_formation(state, conf)
        for i, bot in enumerate(state.fleet_me):
            if i < len(target_positions):
                target_pos = target_positions[i]
            else:
                target_pos = payload
            assignments[bot.id] = {
                "kind": bot.class_.name.lower(),
                "role": "payload",
                "move_target": target_pos,
                "target_id": None,
            }

    else:
        _cache["bodyguard_pair_assigned"] = False
        extractor_cap_per_team = conf.deposit.extractor_cap // 2
        for idx, ext in enumerate(extractors):
            if idx >= extractor_cap_per_team:
                break
            offset_x = idx * 2.0 * bot_radius
            spot = Vec2(mining_spot_base.x + offset_x, mining_spot_base.y)
            assignments[ext.id] = {
                "kind": "extractor",
                "mining_spot": spot,
            }

        if healers and not in_endgame:
            in_cap_h = [h for h in healers if h.pos.dist(payload) <= capture_radius]
            if in_cap_h:
                payload_healer_id = in_cap_h[0].id
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
                "move_target": payload,
            }

            remaining_healers = [h for h in healers if h.id != payload_healer_id]
            for h in remaining_healers:
                low_hp = [b for b in state.fleet_me
                          if b.id != h.id and b.health < conf.bot.health * 0.9]
                t = min(low_hp, key=lambda b: b.health) if low_hp else None
                assignments[h.id] = {
                    "kind": "healer",
                    "role": "combat",
                    "target_id": t.id if t else None,
                    "move_target": payload,
                }
        elif healers:
            for h in healers:
                low_hp = [b for b in state.fleet_me
                          if b.id != h.id and b.health < conf.bot.health * 0.9]
                t = min(low_hp, key=lambda b: b.health) if low_hp else None
                assignments[h.id] = {
                    "kind": "healer",
                    "role": "combat",
                    "target_id": t.id if t else None,
                    "move_target": payload,
                }

        if battles:
            in_cap_b = [b for b in battles if b.pos.dist(payload) <= capture_radius]
            payload_battle_id = None
            battle_candidates = in_cap_b if in_cap_b else battles
            battle_candidates = [b for b in battle_candidates
                                 if b.id not in assignments]
            if battle_candidates:
                payload_battle_id = battle_candidates[0].id
                target_enemy = _pick_target_enemy(
                    next(b for b in battles if b.id == payload_battle_id),
                    state, path_cache, use_ranked_targets
                )
                assignments[payload_battle_id] = {
                    "kind": "battle",
                    "role": "payload",
                    "target_id": target_enemy.id if target_enemy else None,
                    "move_target": payload,
                }

            remaining_battles = [b for b in battles if b.id not in assignments]
            for b in remaining_battles:
                target_enemy = _pick_target_enemy(b, state, path_cache, use_ranked_targets)
                assignments[b.id] = {
                    "kind": "battle",
                    "role": "combat",
                    "target_id": target_enemy.id if target_enemy else None,
                    "move_target": payload,
                }

    _cache["assignments"] = assignments
    _apply_assignments(action, state, conf, assignments, enemy_in_capture, use_ranked_targets)
    return action


def _execute_assignments(state: GameState, conf, in_endgame: bool) -> FleetAction:
    action = FleetAction.new()
    payload = state.payload_pos()
    capture_radius = conf.payload.capture_radius
    assignments = _cache.get("assignments", {})
    enemies_by_id = {e.id: e for e in state.fleet_other}

    enemy_in_capture = any(
        e.pos.dist(payload) <= capture_radius for e in state.fleet_other
    )

    use_ranked_targets = state.tick >= 80

    for bot in state.fleet_me:
        bot_action = action.bots[bot.id]
        assignment = assignments.get(bot.id, {})

        kind = assignment.get("kind", bot.class_.name.lower())

        if kind == "extractor" or bot.class_ == BotClass.Extractor:
            mining_spot = assignment.get("mining_spot")
            if mining_spot is None:
                mining_spot = state.deposit_me.pos + Vec2(
                    0.0, conf.deposit.radius + conf.bot.radius
                )
            bot_action.move_action = move_bot(navigate_to(bot.pos, mining_spot))
            bot_action.turn_action = turn_towards(state.deposit_me.pos)
            bot_action.special_action = SpecialAction.Extractor(mine=True)

        elif kind == "healer" or bot.class_ == BotClass.Healer:
            target_id = assignment.get("target_id")
            target = state.fleet_me.get(target_id) if target_id is not None else None

            if target is not None:
                bot_action.turn_action = turn_towards(target.pos)
            else:
                nearest_enemy = min(state.fleet_other, key=lambda e: e.pos.dist(bot.pos), default=None)
                if nearest_enemy:
                    bot_action.turn_action = turn_towards(nearest_enemy.pos)
                else:
                    bot_action.turn_action = turn_towards(payload)

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
            bot_action.move_action = move_bot(navigate_to(bot.pos, move_target))

        elif kind == "battle" or bot.class_ == BotClass.Battle:
            target_id = assignment.get("target_id")
            target = enemies_by_id.get(target_id) if target_id is not None else None

            if target is not None:
                in_range = bot.pos.dist(target.pos) <= conf.bot.blaster_range
                los = line_of_sight(bot.pos, target.pos)
                bot_action.special_action = SpecialAction.Battle(fire=in_range and los)
                bot_action.turn_action = turn_towards(target.pos)
            else:
                bot_action.special_action = SpecialAction.Battle(fire=False)
                bot_action.turn_action = turn_towards(payload)

            move_target = assignment.get("move_target", payload)
            bot_action.move_action = move_bot(navigate_to(bot.pos, move_target))

        else:
            bot_action.move_action = move_bot(navigate_to(bot.pos, payload))
            if bot.class_ == BotClass.Extractor:
                bot_action.special_action = SpecialAction.Extractor(mine=True)
            elif bot.class_ == BotClass.Healer:
                bot_action.special_action = SpecialAction.Healer(fire=False, target=0)
            else:
                bot_action.special_action = SpecialAction.Battle(fire=False)

    return action


def _execute_fallback(state: GameState, conf, in_endgame: bool) -> FleetAction:
    action = FleetAction.new()
    payload = state.payload_pos()

    for bot in state.fleet_me:
        bot_action = action.bots[bot.id]
        bot_action.move_action = move_bot(navigate_to(bot.pos, payload))
        bot_action.turn_action = turn_towards(payload)

        if bot.class_ == BotClass.Healer:
            low_hp = [b for b in state.fleet_me
                      if b.id != bot.id and b.health < conf.bot.health]
            target = min(low_hp, key=lambda b: b.health) if low_hp else None
            if target is not None:
                in_range = bot.pos.dist(target.pos) <= conf.bot.base_heal_range
                in_arc = _is_in_heal_arc(bot, target, conf)
                bot_action.special_action = SpecialAction.Healer(
                    fire=in_range and in_arc,
                    target=target.id,
                )
            else:
                bot_action.special_action = SpecialAction.Healer(fire=False, target=0)

        elif bot.class_ == BotClass.Battle:
            nearest = min(state.fleet_other, key=lambda e: e.pos.dist(bot.pos), default=None)
            if nearest:
                in_range = bot.pos.dist(nearest.pos) <= conf.bot.blaster_range
                los = line_of_sight(bot.pos, nearest.pos)
                bot_action.special_action = SpecialAction.Battle(fire=in_range and los)
            else:
                bot_action.special_action = SpecialAction.Battle(fire=False)

        else:
            bot_action.special_action = SpecialAction.Extractor(mine=True)

    return action


def _compute_fabricator_next(state: GameState, conf, cache: Dict, in_endgame: bool) -> int:
    if not state.fleet_me.get(0):
        return int(BotClass.Extractor)

    extractor_count = sum(1 for b in state.fleet_me if b.class_ == BotClass.Extractor)
    healer_count = sum(1 for b in state.fleet_me if b.class_ == BotClass.Healer)
    battle_count = sum(1 for b in state.fleet_me if b.class_ == BotClass.Battle)

    if state.tick <= 60:
        if extractor_count < 4:
            return int(BotClass.Extractor)
        if healer_count < 3:
            return int(BotClass.Healer)
        if battle_count < 2:
            return int(BotClass.Battle)
        return int(BotClass.Healer)

    last_alt = cache.get("last_alt", 0)
    alt_count = cache.get("alt_count", 0)
    fleet_size = len(state.fleet_me)
    if fleet_size > last_alt:
        alt_count += 1
        cache["alt_count"] = alt_count
    cache["last_alt"] = fleet_size

    return int(BotClass.Battle if alt_count % 2 == 0 else BotClass.Healer)


def _pick_target_enemy(bot: BotState, state: GameState,
                       path_cache: Dict[Tuple[int, int], Optional[float]],
                       use_ranked: bool = False) -> Optional[BotState]:
    if not state.fleet_other:
        return None

    if use_ranked:
        ranked = _rank_targets(state, path_cache)
        return ranked[0] if ranked else None

    best = None
    best_dist = float("inf")
    for enemy in state.fleet_other:
        d = path_cache.get((bot.id, enemy.id))
        if d is None:
            d = bot.pos.dist(enemy.pos)
        if d < best_dist:
            best_dist = d
            best = enemy
    return best


def _rank_targets(state: GameState,
                  path_cache: Dict[Tuple[int, int], Optional[float]]) -> List[BotState]:
    healers = [e for e in state.fleet_other if e.class_ == BotClass.Healer]
    extractors = [e for e in state.fleet_other if e.class_ == BotClass.Extractor]
    battles = [e for e in state.fleet_other if e.class_ == BotClass.Battle]

    result: List[BotState] = []

    for rank_list in [healers, extractors, battles]:
        if not rank_list:
            continue
        best_dist: Dict[int, float] = {}
        for enemy in rank_list:
            min_d = float("inf")
            for battle in state.fleet_me:
                if battle.class_ != BotClass.Battle:
                    continue
                d = path_cache.get((battle.id, enemy.id))
                if d is None:
                    d = battle.pos.dist(enemy.pos)
                if d < min_d:
                    min_d = d
            best_dist[enemy.id] = min_d

        sorted_enemies = sorted(rank_list, key=lambda e: best_dist[e.id])
        result.extend(sorted_enemies)

    return result


def _compute_bodyguard_hold(state: GameState, conf) -> Vec2:
    deposit_pos = state.deposit_me.pos
    payload_pos = state.payload_pos()
    return deposit_pos * 0.5 + payload_pos * 0.5


def _compute_payload_formation(state: GameState, conf) -> List[Vec2]:
    payload = state.payload_pos()
    capture_radius = conf.payload.capture_radius
    bot_radius = conf.bot.radius
    n = len(state.fleet_me)

    positions = []
    if n == 0:
        return positions

    if n <= 3:
        for i in range(n):
            angle = (i / max(n - 1, 1)) * 2.0 * math.pi
            offset = Vec2(math.cos(angle), math.sin(angle)) * capture_radius
            positions.append(payload + offset)
    else:
        rings = _compute_formation_rings(n, capture_radius, bot_radius)
        for ring in rings:
            for pos in ring:
                positions.append(payload + pos)

    while len(positions) < n:
        positions.append(payload)

    return positions[:n]


def _compute_formation_rings(n: int, capture_radius: float,
                             bot_radius: float) -> List[List[Vec2]]:
    rings = []
    remaining = n
    ring_index = 0

    while remaining > 0:
        ring_size = max(6, 6 * ring_index) if ring_index > 0 else 1
        if ring_index == 0:
            ring_size = 1
        elif ring_index == 1:
            ring_size = 6

        radius = capture_radius + ring_index * 2.5 * bot_radius
        ring = []
        actual_size = min(ring_size, remaining)
        for i in range(actual_size):
            angle = (i / actual_size) * 2.0 * math.pi
            offset = Vec2(math.cos(angle), math.sin(angle)) * radius
            ring.append(offset)
        rings.append(ring)
        remaining -= actual_size
        ring_index += 1

    return rings


def _is_in_heal_arc(healer: BotState, target: BotState, conf) -> bool:
    to_target = target.pos - healer.pos
    if to_target.norm_sq() < 0.0001:
        return True
    angle_to_target = to_target.angle_deg()
    diff = abs(diff_degrees(angle_to_target, healer.angle))
    half_arc = conf.bot.base_heal_arc_deg / 2.0
    return diff <= half_arc + 1e-6


def _apply_assignments(action: FleetAction, state: GameState, conf,
                       assignments: Dict, enemy_in_capture: bool,
                       use_ranked: bool) -> None:
    payload = state.payload_pos()
    enemies_by_id = {e.id: e for e in state.fleet_other}

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
            bot_action.move_action = move_bot(navigate_to(bot.pos, mining_spot))
            bot_action.turn_action = turn_towards(state.deposit_me.pos)
            bot_action.special_action = SpecialAction.Extractor(mine=True)

        elif kind == "healer" or bot.class_ == BotClass.Healer:
            target_id = assignment.get("target_id")
            target = state.fleet_me.get(target_id) if target_id is not None else None

            if target is not None:
                bot_action.turn_action = turn_towards(target.pos)
            else:
                nearest_enemy = min(state.fleet_other, key=lambda e: e.pos.dist(bot.pos), default=None)
                if nearest_enemy:
                    bot_action.turn_action = turn_towards(nearest_enemy.pos)
                else:
                    bot_action.turn_action = turn_towards(payload)

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
            bot_action.move_action = move_bot(navigate_to(bot.pos, move_target))

        elif kind == "battle" or bot.class_ == BotClass.Battle:
            target_id = assignment.get("target_id")
            target = enemies_by_id.get(target_id) if target_id is not None else None

            if target is not None:
                in_range = bot.pos.dist(target.pos) <= conf.bot.blaster_range
                los = line_of_sight(bot.pos, target.pos)
                bot_action.special_action = SpecialAction.Battle(fire=in_range and los)
                bot_action.turn_action = turn_towards(target.pos)
            else:
                bot_action.special_action = SpecialAction.Battle(fire=False)
                bot_action.turn_action = turn_towards(payload)

            move_target = assignment.get("move_target", payload)
            bot_action.move_action = move_bot(navigate_to(bot.pos, move_target))

        else:
            bot_action.move_action = move_bot(navigate_to(bot.pos, payload))
            if bot.class_ == BotClass.Extractor:
                bot_action.special_action = SpecialAction.Extractor(mine=True)
            elif bot.class_ == BotClass.Healer:
                bot_action.special_action = SpecialAction.Healer(fire=False, target=0)
            else:
                bot_action.special_action = SpecialAction.Battle(fire=False)


def _maybe_endgame_self_destruct(state: GameState, conf, action: FleetAction,
                                 cache: Dict) -> None:
    payload = state.payload_pos()
    capture_radius = conf.payload.capture_radius
    ticks_remaining = conf.max_ticks - state.tick

    for bot in state.fleet_me:
        dist_to_capture_edge = max(0.0, bot.pos.dist(payload) - capture_radius)
        travel_per_tick = conf.bot.speed
        ticks_to_reach = dist_to_capture_edge / travel_per_tick if travel_per_tick > 0 else float("inf")
        if travel_per_tick > 0 and ticks_to_reach > ticks_remaining:
            action.bots[bot.id].self_destruct = True
