"""Plan 4 — Reactive Counter.

See plans/plan4.md for the full spec (requirements R-P4.* and R-X*).
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

CLASSIFICATION_UNKNOWN = "UNKNOWN"
CLASSIFICATION_ECONOMY = "ECONOMY"
CLASSIFICATION_DEATHBALL = "DEATHBALL"
CLASSIFICATION_SQUEEZE = "SQUEEZE"


def plan4_strategy(state: GameState) -> FleetAction:
    budget = get_budget()
    conf = get_config()
    in_endgame = state.tick >= conf.max_ticks - conf.endgame_ticks

    last_tick = _cache.get("last_tick", -1)
    if last_tick >= 0 and state.tick - last_tick > 1:
        _cache["assignments"] = {}

    classification = _cache.get("classification", CLASSIFICATION_UNKNOWN)
    counter_switch_tick = _cache.get("counter_switch_tick", -1)

    if classification == CLASSIFICATION_UNKNOWN and state.tick <= 50:
        if state.tick % 10 == 0:
            _update_detection_snapshot(state, conf)

        classification = _compute_classification(state, conf, _cache)
        if classification != CLASSIFICATION_UNKNOWN:
            _cache["classification"] = classification
            counter_switch_tick = state.tick
            _cache["counter_switch_tick"] = counter_switch_tick

    elif classification != CLASSIFICATION_UNKNOWN:
        pass

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

    if classification == CLASSIFICATION_UNKNOWN or state.tick <= 50:
        if budget.remaining <= 100_000:
            action = _execute_assignments_default(state, conf, in_endgame)
        elif needs_recompute:
            action = _recompute_assignments_default(state, conf, in_endgame)
            _cache["last_assignment_tick"] = state.tick
        else:
            action = _execute_assignments_default(state, conf, in_endgame)
    else:
        if budget.remaining <= 100_000:
            action = _execute_counter_assignments(state, conf, in_endgame, classification)
        elif needs_recompute:
            action = _recompute_counter_assignments(state, conf, in_endgame, classification)
            _cache["last_assignment_tick"] = state.tick
        else:
            action = _execute_counter_assignments(state, conf, in_endgame, classification)

    _cache["last_tick"] = state.tick
    _cache["last_enemy_count"] = enemy_count
    _cache["last_friendly_ids"] = friendly_ids
    _cache["last_friendly_in_capture"] = friendly_in_capture

    action.fabricator_next = _compute_fabricator_next(state, conf, _cache, classification, counter_switch_tick)

    if in_endgame or state.tick < 50:
        action.rush_order = False
    else:
        action.rush_order = (
            not in_endgame
            and state.fabricator_me.tokens >= conf.fabricator.rush_cost
        )

    if in_endgame and not _cache.get("endgame_sd_done", False):
        _maybe_endgame_self_destruct(state, conf, action, _cache, classification)

    return action


def _update_detection_snapshot(state: GameState, conf) -> None:
    snapshots = _cache.get("detection_snapshots", [])
    enemy_battles = [e for e in state.fleet_other if e.class_ == BotClass.Battle]
    enemy_healers = [e for e in state.fleet_other if e.class_ == BotClass.Healer]
    enemy_extractors = [e for e in state.fleet_other if e.class_ == BotClass.Extractor]

    enemy_extractors_on_my_deposit = [
        e for e in enemy_extractors
        if e.pos.dist(state.deposit_me.pos) <= conf.deposit.radius * 2.0
    ]

    centroid = Vec2(0.0, 0.0)
    if state.fleet_other:
        for e in state.fleet_other:
            centroid = centroid + e.pos
        centroid = centroid * (1.0 / len(state.fleet_other))

    payload = state.payload_pos()
    centroid_to_payload = centroid.dist(payload) if state.fleet_other else float("inf")

    snapshot = {
        "tick": state.tick,
        "fleet_size": len(state.fleet_other),
        "battles": len(enemy_battles),
        "healers": len(enemy_healers),
        "extractors": len(enemy_extractors),
        "extractors_on_my_deposit": len(enemy_extractors_on_my_deposit),
        "centroid": centroid,
        "centroid_to_payload": centroid_to_payload,
        "centroid_to_my_deposit_other": centroid.dist(state.deposit_other.pos) if state.fleet_other else float("inf"),
    }
    snapshots.append(snapshot)
    _cache["detection_snapshots"] = snapshots


def _compute_classification(state: GameState, conf, cache: Dict) -> str:
    snapshots = cache.get("detection_snapshots", [])
    if not snapshots:
        return CLASSIFICATION_UNKNOWN

    for snap in snapshots:
        if snap["tick"] == 30 and snap["extractors_on_my_deposit"] >= 2:
            return CLASSIFICATION_ECONOMY

    for snap in snapshots:
        if snap["centroid_to_my_deposit_other"] <= 2.0 * conf.deposit.radius:
            return CLASSIFICATION_ECONOMY

    for snap in snapshots:
        if snap["battles"] >= 3 and snap["centroid_to_payload"] <= conf.bot.blaster_range * 1.5 and snap["healers"] >= 2:
            return CLASSIFICATION_DEATHBALL

    for snap in snapshots:
        if snap["tick"] == 20 and snap["fleet_size"] > 0:
            centroid = snap["centroid"]
            payload = state.payload_pos()
            if centroid.dist(payload) <= conf.payload.capture_radius:
                return CLASSIFICATION_SQUEEZE

    if state.tick >= 50:
        return CLASSIFICATION_UNKNOWN

    return CLASSIFICATION_UNKNOWN


def _compute_fabricator_next(state: GameState, conf, cache: Dict, classification: str, counter_switch_tick: int) -> int:
    if counter_switch_tick > 0 and classification != CLASSIFICATION_UNKNOWN:
        return _compute_counter_fabricator_next(state, conf, cache, classification)

    fleet_size = len(state.fleet_me)
    if fleet_size == 0:
        return int(BotClass.Extractor)

    extractor_count = sum(1 for b in state.fleet_me if b.class_ == BotClass.Extractor)
    healer_count = sum(1 for b in state.fleet_me if b.class_ == BotClass.Healer)
    battle_count = sum(1 for b in state.fleet_me if b.class_ == BotClass.Battle)

    if fleet_size < 5:
        cycle = [BotClass.Extractor, BotClass.Healer, BotClass.Battle, BotClass.Extractor, BotClass.Battle]
        idx = fleet_size
        if idx < len(cycle):
            return int(cycle[idx])

    last_alt = cache.get("last_alt", 0)
    alt_count = cache.get("alt_count", 0)
    if fleet_size > last_alt:
        alt_count += 1
        cache["alt_count"] = alt_count
    cache["last_alt"] = fleet_size

    return int(BotClass.Healer if alt_count % 2 == 1 else BotClass.Battle)


def _compute_counter_fabricator_next(state: GameState, conf, cache: Dict, classification: str) -> int:
    fleet_size = len(state.fleet_me)
    if fleet_size == 0:
        return int(BotClass.Extractor)

    if classification == CLASSIFICATION_ECONOMY:
        extractor_count = sum(1 for b in state.fleet_me if b.class_ == BotClass.Extractor)
        healer_count = sum(1 for b in state.fleet_me if b.class_ == BotClass.Healer)
        battle_count = sum(1 for b in state.fleet_me if b.class_ == BotClass.Battle)

        if extractor_count < 2:
            return int(BotClass.Extractor)
        if healer_count < 2:
            return int(BotClass.Healer)
        if battle_count < 4:
            return int(BotClass.Battle)

        last_alt = cache.get("counter_alt", 0)
        alt_count = cache.get("counter_alt_count", 0)
        if fleet_size > last_alt:
            alt_count += 1
            cache["counter_alt_count"] = alt_count
        cache["counter_alt"] = fleet_size

        return int(BotClass.Healer if alt_count % 2 == 1 else BotClass.Battle)

    elif classification == CLASSIFICATION_DEATHBALL:
        battle_count = sum(1 for b in state.fleet_me if b.class_ == BotClass.Battle)
        healer_count = sum(1 for b in state.fleet_me if b.class_ == BotClass.Healer)

        if battle_count < 4:
            return int(BotClass.Battle)
        if healer_count < 2:
            return int(BotClass.Healer)

        last_alt = cache.get("counter_alt", 0)
        alt_count = cache.get("counter_alt_count", 0)
        if fleet_size > last_alt:
            alt_count += 1
            cache["counter_alt_count"] = alt_count
        cache["counter_alt"] = fleet_size

        return int(BotClass.Healer if alt_count % 2 == 1 else BotClass.Battle)

    elif classification == CLASSIFICATION_SQUEEZE:
        extractor_count = sum(1 for b in state.fleet_me if b.class_ == BotClass.Extractor)
        healer_count = sum(1 for b in state.fleet_me if b.class_ == BotClass.Healer)

        if extractor_count < 4:
            return int(BotClass.Extractor)
        if healer_count < 3:
            return int(BotClass.Healer)

        last_alt = cache.get("counter_alt", 0)
        alt_count = cache.get("counter_alt_count", 0)
        if fleet_size > last_alt:
            alt_count += 1
            cache["counter_alt_count"] = alt_count
        cache["counter_alt"] = fleet_size

        return int(BotClass.Healer if alt_count % 2 == 1 else BotClass.Battle)

    return int(BotClass.Battle)


def _recompute_assignments_default(state: GameState, conf, in_endgame: bool) -> FleetAction:
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

    for idx, ext in enumerate(extractors[:3]):
        offset_x = idx * 2.0 * bot_radius
        spot = Vec2(mining_spot_base.x + offset_x, mining_spot_base.y)
        assignments[ext.id] = {
            "kind": "extractor",
            "mining_spot": spot,
        }

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
    for i, battle in enumerate(remaining_battles):
        target_enemy = _pick_target_enemy(battle, state, path_cache)
        slot_index = i
        n_slots = len(remaining_battles)
        move_target = _compute_battle_move_target(slot_index, n_slots, payload, conf)
        assignments[battle.id] = {
            "kind": "battle",
            "role": "combat",
            "target_id": target_enemy.id if target_enemy else None,
            "move_target": move_target,
        }

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
    _apply_assignments(action, state, conf, assignments, enemy_in_capture)
    return action


def _execute_assignments_default(state: GameState, conf, in_endgame: bool) -> FleetAction:
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


def _recompute_counter_assignments(state: GameState, conf, in_endgame: bool, classification: str) -> FleetAction:
    if classification == CLASSIFICATION_ECONOMY:
        return _recompute_economy_counter(state, conf, in_endgame)
    elif classification == CLASSIFICATION_DEATHBALL:
        return _recompute_deathball_counter(state, conf, in_endgame)
    elif classification == CLASSIFICATION_SQUEEZE:
        return _recompute_squeeze_counter(state, conf, in_endgame)
    return _recompute_assignments_default(state, conf, in_endgame)


def _recompute_economy_counter(state: GameState, conf, in_endgame: bool) -> FleetAction:
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

    enemy_extractors_on_my_deposit = [
        e for e in state.fleet_other
        if e.class_ == BotClass.Extractor and e.pos.dist(state.deposit_me.pos) <= conf.deposit.radius * 2.0
    ]

    dive_target = None
    if enemy_extractors_on_my_deposit:
        dive_target = min(enemy_extractors_on_my_deposit, key=lambda e: e.health)

    path_cache: Dict[Tuple[int, int], Optional[float]] = {}
    for battle in battles:
        for enemy in state.fleet_other:
            d = path_length(battle.pos, enemy.pos)
            path_cache[(battle.id, enemy.id)] = d

    mining_spot_base = state.deposit_me.pos + Vec2(0.0, conf.deposit.radius + conf.bot.radius)

    for idx, ext in enumerate(extractors[:3]):
        offset_x = idx * 2.0 * bot_radius
        spot = Vec2(mining_spot_base.x + offset_x, mining_spot_base.y)
        assignments[ext.id] = {
            "kind": "extractor",
            "mining_spot": spot,
        }

    if len(battles) >= 2 and dive_target:
        dive_battles = battles[:2]
        for b in dive_battles:
            assignments[b.id] = {
                "kind": "battle",
                "role": "dive",
                "target_id": dive_target.id,
                "move_target": dive_target.pos,
            }

    payload_battle_id: Optional[int] = None
    remaining_battles = [b for b in battles if b.id not in assignments]

    if remaining_battles:
        in_cap = [b for b in remaining_battles if b.pos.dist(payload) <= capture_radius]
        if in_cap:
            payload_battle_id = in_cap[0].id
        else:
            payload_battle_id = min(remaining_battles, key=lambda b: b.pos.dist(payload)).id

        target_enemy = _pick_target_enemy(
            next(b for b in remaining_battles if b.id == payload_battle_id),
            state, path_cache
        )
        assignments[payload_battle_id] = {
            "kind": "battle",
            "role": "payload",
            "target_id": target_enemy.id if target_enemy else None,
            "move_target": payload,
        }

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

    remaining_healers = [h for h in healers if h.id != payload_healer_id]
    remaining_battles = [b for b in battles if b.id not in assignments]

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

    for b in remaining_battles:
        target_enemy = _pick_target_enemy(b, state, path_cache)
        slot_index = 0
        n_slots = len(remaining_battles)
        move_target = _compute_battle_move_target(slot_index, max(n_slots, 1), payload, conf)
        assignments[b.id] = {
            "kind": "battle",
            "role": "combat",
            "target_id": target_enemy.id if target_enemy else None,
            "move_target": move_target,
        }

    for bot in state.fleet_me:
        if bot.id in assignments:
            continue
        assignments[bot.id] = {
            "kind": bot.class_.name.lower(),
            "role": "combat",
            "target_id": None,
            "move_target": payload,
        }

    _cache["assignments"] = assignments
    _apply_assignments(action, state, conf, assignments, enemy_in_capture)
    return action


def _recompute_deathball_counter(state: GameState, conf, in_endgame: bool) -> FleetAction:
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

    for idx, ext in enumerate(extractors[:3]):
        offset_x = idx * 2.0 * bot_radius
        spot = Vec2(mining_spot_base.x + offset_x, mining_spot_base.y)
        assignments[ext.id] = {
            "kind": "extractor",
            "mining_spot": spot,
        }

    healers_sorted = sorted(healers, key=lambda h: h.id)
    if len(healers_sorted) >= 2:
        lead_healer = healers_sorted[0]
        back_healer = healers_sorted[1]

        lead_spot = payload + Vec2(0.0, -4.0 * bot_radius)
        back_spot = payload + Vec2(0.0, -8.0 * bot_radius)

        assignments[lead_healer.id] = {
            "kind": "healer",
            "role": "lead",
            "target_id": None,
            "move_target": lead_spot,
            "turn_target": payload,
        }
        assignments[back_healer.id] = {
            "kind": "healer",
            "role": "back",
            "target_id": lead_healer.id,
            "move_target": back_spot,
            "turn_target": payload,
        }
    elif len(healers_sorted) == 1:
        h = healers_sorted[0]
        spot = payload + Vec2(0.0, -6.0 * bot_radius)
        assignments[h.id] = {
            "kind": "healer",
            "role": "solo",
            "target_id": None,
            "move_target": spot,
            "turn_target": payload,
        }

    battle_slots = []
    for i, b in enumerate(battles):
        angle = (i / max(len(battles), 1)) * 2.0 * math.pi
        radius = 3.0 * bot_radius
        offset = Vec2(math.cos(angle) * radius, math.sin(angle) * radius)
        spot = payload + offset
        battle_slots.append((b.id, spot))

    for bid, spot in battle_slots:
        target_enemy = None
        for e in state.fleet_other:
            if e.class_ == BotClass.Healer:
                target_enemy = e
                break
        if not target_enemy:
            target_enemy = _pick_target_enemy(next(b for b in battles if b.id == bid), state, path_cache)

        assignments[bid] = {
            "kind": "battle",
            "role": "screen",
            "target_id": target_enemy.id if target_enemy else None,
            "move_target": spot,
            "kite": True,
        }

    for bot in state.fleet_me:
        if bot.id in assignments:
            continue
        assignments[bot.id] = {
            "kind": bot.class_.name.lower(),
            "role": "combat",
            "target_id": None,
            "move_target": payload,
        }

    _cache["assignments"] = assignments
    _apply_deathball_assignments(action, state, conf, assignments, enemy_in_capture)
    return action


def _recompute_squeeze_counter(state: GameState, conf, in_endgame: bool) -> FleetAction:
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

    for idx, ext in enumerate(extractors[:4]):
        offset_x = idx * 2.0 * bot_radius
        spot = Vec2(mining_spot_base.x + offset_x, mining_spot_base.y)
        assignments[ext.id] = {
            "kind": "extractor",
            "mining_spot": spot,
        }

    if battles:
        payload_battle_id = battles[0].id
        target_enemy = _pick_target_enemy(battles[0], state, path_cache)
        assignments[payload_battle_id] = {
            "kind": "battle",
            "role": "payload",
            "target_id": target_enemy.id if target_enemy else None,
            "move_target": payload,
        }

    if healers:
        payload_healer_id = healers[0].id
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

    remaining_healers = [h for h in healers if h.id not in assignments]
    remaining_battles = [b for b in battles if b.id not in assignments]

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

    for b in remaining_battles:
        target_enemy = _pick_target_enemy(b, state, path_cache)
        slot_index = 0
        n_slots = len(remaining_battles)
        move_target = _compute_battle_move_target(slot_index, max(n_slots, 1), payload, conf)
        assignments[b.id] = {
            "kind": "battle",
            "role": "combat",
            "target_id": target_enemy.id if target_enemy else None,
            "move_target": move_target,
        }

    for bot in state.fleet_me:
        if bot.id in assignments:
            continue
        assignments[bot.id] = {
            "kind": bot.class_.name.lower(),
            "role": "combat",
            "target_id": None,
            "move_target": payload,
        }

    _cache["assignments"] = assignments
    _apply_assignments(action, state, conf, assignments, enemy_in_capture)
    return action


def _execute_counter_assignments(state: GameState, conf, in_endgame: bool, classification: str) -> FleetAction:
    if classification == CLASSIFICATION_ECONOMY:
        return _execute_economy_assignments(state, conf, in_endgame)
    elif classification == CLASSIFICATION_DEATHBALL:
        return _execute_deathball_assignments(state, conf, in_endgame)
    elif classification == CLASSIFICATION_SQUEEZE:
        return _execute_squeeze_assignments(state, conf, in_endgame)
    return _execute_assignments_default(state, conf, in_endgame)


def _execute_economy_assignments(state: GameState, conf, in_endgame: bool) -> FleetAction:
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
        role = assignment.get("role", "combat")

        if kind == "extractor" or bot.class_ == BotClass.Extractor:
            mining_spot = assignment.get("mining_spot")
            if mining_spot is None:
                mining_spot = state.deposit_me.pos + Vec2(0.0, conf.deposit.radius + conf.bot.radius)
            bot_action.move_action = move_bot(navigate_to(bot.pos, mining_spot))
            bot_action.turn_action = turn_towards(state.deposit_me.pos)
            bot_action.special_action = SpecialAction.Extractor(mine=True)

        elif kind == "healer" or bot.class_ == BotClass.Healer:
            target_id = assignment.get("target_id")
            target = state.fleet_me.get(target_id) if target_id is not None else None

            is_payload = role == "payload"
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


def _execute_deathball_assignments(state: GameState, conf, in_endgame: bool) -> FleetAction:
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
        role = assignment.get("role", "combat")
        kite = assignment.get("kite", False)

        if kind == "extractor" or bot.class_ == BotClass.Extractor:
            mining_spot = assignment.get("mining_spot")
            if mining_spot is None:
                mining_spot = state.deposit_me.pos + Vec2(0.0, conf.deposit.radius + conf.bot.radius)
            bot_action.move_action = move_bot(navigate_to(bot.pos, mining_spot))
            bot_action.turn_action = turn_towards(state.deposit_me.pos)
            bot_action.special_action = SpecialAction.Extractor(mine=True)

        elif kind == "healer" or bot.class_ == BotClass.Healer:
            target_id = assignment.get("target_id")
            target = state.fleet_me.get(target_id) if target_id is not None else None

            turn_target = assignment.get("turn_target", payload)
            bot_action.turn_action = turn_towards(turn_target)

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

                if kite and in_range and los:
                    retreat_dir = bot.pos - target.pos
                    if retreat_dir.norm_sq() > 0.0001:
                        retreat_dir = retreat_dir.normalize_or_zero()
                    retreat_point = bot.pos + retreat_dir * (2.0 * conf.bot.blaster_range)
                    bot_action.move_action = move_bot(navigate_to(bot.pos, retreat_point))
                else:
                    move_target = assignment.get("move_target", payload)
                    bot_action.move_action = move_bot(navigate_to(bot.pos, move_target))
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


def _execute_squeeze_assignments(state: GameState, conf, in_endgame: bool) -> FleetAction:
    return _execute_economy_assignments(state, conf, in_endgame)


def _apply_deathball_assignments(action: FleetAction, state: GameState, conf,
                                  assignments: Dict, enemy_in_capture: bool) -> None:
    payload = state.payload_pos()
    enemies_by_id = {e.id: e for e in state.fleet_other}

    for bot in state.fleet_me:
        bot_action = action.bots[bot.id]
        assignment = assignments.get(bot.id, {})

        kind = assignment.get("kind", bot.class_.name.lower())
        role = assignment.get("role", "combat")
        kite = assignment.get("kite", False)

        if kind == "extractor" or bot.class_ == BotClass.Extractor:
            mining_spot = assignment.get("mining_spot")
            if mining_spot is None:
                mining_spot = state.deposit_me.pos + Vec2(0.0, conf.deposit.radius + conf.bot.radius)
            bot_action.move_action = move_bot(navigate_to(bot.pos, mining_spot))
            bot_action.turn_action = turn_towards(state.deposit_me.pos)
            bot_action.special_action = SpecialAction.Extractor(mine=True)

        elif kind == "healer" or bot.class_ == BotClass.Healer:
            target_id = assignment.get("target_id")
            target = state.fleet_me.get(target_id) if target_id is not None else None

            turn_target = assignment.get("turn_target", payload)
            bot_action.turn_action = turn_towards(turn_target)

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

                if kite and in_range and los:
                    retreat_dir = bot.pos - target.pos
                    if retreat_dir.norm_sq() > 0.0001:
                        retreat_dir = retreat_dir.normalize_or_zero()
                    retreat_point = bot.pos + retreat_dir * (2.0 * conf.bot.blaster_range)
                    bot_action.move_action = move_bot(navigate_to(bot.pos, retreat_point))
                else:
                    move_target = assignment.get("move_target", payload)
                    bot_action.move_action = move_bot(navigate_to(bot.pos, move_target))
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


def _apply_assignments(action: FleetAction, state: GameState, conf,
                       assignments: Dict, enemy_in_capture: bool) -> None:
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


def _maybe_endgame_self_destruct(state: GameState, conf, action: FleetAction,
                                 cache: Dict, classification: str) -> None:
    my_hp = sum(b.health for b in state.fleet_me)
    enemy_hp = sum(e.health for e in state.fleet_other)

    if enemy_hp > 0 and my_hp < enemy_hp * 0.75:
        extractors = [b for b in state.fleet_me if b.class_ == BotClass.Extractor]
        if extractors:
            weakest = min(extractors, key=lambda b: b.health)
            action.bots[weakest.id].self_destruct = True
            cache["endgame_sd_done"] = True
