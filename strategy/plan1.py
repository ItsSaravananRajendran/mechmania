"""Plan 1 — The Payload Squeeze.

See plans/plan1.md for the full spec (requirements R-P1.* and R-X*).
"""

from __future__ import annotations

import math
from typing import Dict, FrozenSet, Optional, Tuple

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


def plan1_strategy(state: GameState) -> FleetAction:
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

    needs_recompute = (
        last_assignment_tick < 0
        or state.tick - last_assignment_tick >= 5
        or enemy_delta >= 2
        or capture_changed
        or bool(friendly_died)
    )

    if budget.remaining <= 100_000:
        action = _execute_assignments(state, conf, in_endgame)
    elif needs_recompute:
        action = _recompute_assignments(state, conf, in_endgame)
        _cache["last_assignment_tick"] = state.tick
    else:
        action = _execute_assignments(state, conf, in_endgame)

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


def _recompute_assignments(state: GameState, conf, in_endgame: bool) -> FleetAction:
    action = FleetAction.new()
    payload = state.payload_pos()
    capture_radius = conf.payload.capture_radius
    bot_radius = conf.bot.radius

    healers = [b for b in state.fleet_me if b.class_ == BotClass.Healer]
    battles = [b for b in state.fleet_me if b.class_ == BotClass.Battle]
    extractors = [b for b in state.fleet_me if b.class_ == BotClass.Extractor]

    assignments: Dict[int, dict] = {}

    enemy_in_capture = any(
        e.pos.dist(payload) <= capture_radius for e in state.fleet_other
    )

    path_cache: Dict[Tuple[int, int], Optional[float]] = {}
    for battle in battles:
        for enemy in state.fleet_other:
            d = path_length(battle.pos, enemy.pos)
            path_cache[(battle.id, enemy.id)] = d

    mining_spot_base = state.deposit_me.pos + Vec2(0.0, conf.deposit.radius + conf.bot.radius)

    assigned_extractor_ids = []
    for idx, ext in enumerate(extractors[:3]):
        offset_x = idx * 2.0 * bot_radius
        spot = Vec2(mining_spot_base.x + offset_x, mining_spot_base.y)
        assignments[ext.id] = {
            "kind": "extractor",
            "mining_spot": spot,
        }
        assigned_extractor_ids.append(ext.id)

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
            "move_target": payload,
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

        target_enemy = _pick_target_enemy(
            next(b for b in battles if b.id == payload_battle_id),
            state, path_cache
        )

        assignments[payload_battle_id] = {
            "kind": "battle",
            "role": "payload",
            "target_id": target_enemy.id if target_enemy else None,
            "move_target": payload,
        }

    remaining_healers = [h for h in healers if h.id != payload_healer_id]
    for h in remaining_healers:
        candidates = [b for b in state.fleet_me
                      if b.id != h.id and b.health < conf.bot.health * 0.9]
        if candidates:
            target = min(candidates, key=lambda b: b.health)
            assignments[h.id] = {
                "kind": "healer",
                "role": "combat",
                "target_id": target.id,
                "move_target": target.pos,
            }
        else:
            assignments[h.id] = {
                "kind": "healer",
                "role": "combat",
                "target_id": None,
                "move_target": payload,
            }

    remaining_battles = [b for b in battles if b.id != payload_battle_id]
    battle_positions = []
    for i, battle in enumerate(remaining_battles):
        target_enemy = _pick_target_enemy(battle, state, path_cache)
        slot_index = i
        n_slots = len(remaining_battles)
        move_target = _compute_battle_move_target(
            slot_index, n_slots, payload, conf
        )
        assignments[battle.id] = {
            "kind": "battle",
            "role": "combat",
            "target_id": target_enemy.id if target_enemy else None,
            "move_target": move_target,
        }
        battle_positions.append((battle.id, move_target))

    for bot in state.fleet_me:
        if bot.id in assignments:
            continue
        target_enemy = _pick_target_enemy(bot, state, path_cache) if bot.class_ == BotClass.Battle else None
        assignments[bot.id] = {
            "kind": bot.class_.name.lower(),
            "role": "combat",
            "target_id": target_enemy.id if target_enemy else None,
            "move_target": payload,
        }

    _cache["assignments"] = assignments
    _apply_assignments(action, state, conf, assignments, enemy_in_capture, path_cache)
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

            is_payload = assignment.get("role") == "payload"
            if is_payload:
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


def _compute_fabricator_next(state: GameState, conf, cache: Dict) -> int:
    if not state.fleet_me.get(0):
        return int(BotClass.Extractor)

    if len(state.fleet_me) >= 10:
        if state.deposit_me.extractors.me < state.deposit_me.extractors.other:
            return int(BotClass.Extractor)
        return int(BotClass.Battle)

    extractor_alive = sum(1 for b in state.fleet_me if b.class_ == BotClass.Extractor)
    if state.tick <= 30 and extractor_alive < 3:
        return int(BotClass.Extractor)

    last_size = cache.get("last_fleet_size", 0)
    size = len(state.fleet_me)
    if size > last_size:
        cache["alt_count"] = cache.get("alt_count", 0) + 1
    cache["last_fleet_size"] = size

    alt_count = cache.get("alt_count", 0)
    return int(BotClass.Battle if alt_count % 2 == 0 else BotClass.Healer)


def _pick_target_enemy(bot: BotState, state: GameState,
                        path_cache: Dict[Tuple[int, int], Optional[float]]) -> Optional[BotState]:
    if not state.fleet_other:
        return None
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


def _compute_battle_move_target(slot_index: int, n_battles: int,
                                payload: Vec2, conf) -> Vec2:
    if n_battles <= 0:
        return payload
    angle = (slot_index / n_battles) * 2.0 * math.pi
    radius = 3.0 * conf.bot.radius
    offset = Vec2(math.cos(angle) * radius, math.sin(angle) * radius)
    return payload + offset


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
                       path_cache: Dict[Tuple[int, int], Optional[float]]) -> None:
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
    my_hp = sum(b.health for b in state.fleet_me)
    enemy_hp = sum(e.health for e in state.fleet_other)

    if enemy_hp > 0 and my_hp < enemy_hp * 0.75:
        extractors = [b for b in state.fleet_me if b.class_ == BotClass.Extractor]
        if extractors:
            weakest = min(extractors, key=lambda b: b.health)
            action.bots[weakest.id].self_destruct = True
            cache["endgame_sd_done"] = True