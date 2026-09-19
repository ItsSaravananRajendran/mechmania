"""Plan 2 — The Denial Deathball.

See plans/plan2.md for the full spec (requirements R-P2.* and R-X*).
"""

from __future__ import annotations

import math
from typing import Dict, FrozenSet, List, Optional, Tuple

from . import (
    BOTS_MAX,
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


def plan2_strategy(state: GameState) -> FleetAction:
    budget = get_budget()
    conf = get_config()
    in_endgame = state.tick >= conf.max_ticks - conf.endgame_ticks

    last_tick = _cache.get("last_tick", -1)
    if last_tick >= 0 and state.tick - last_tick > 1:
        _cache["assignments"] = {}

    enemy_count = len(state.fleet_other)
    last_enemy_count = _cache.get("last_enemy_count", enemy_count)

    friendly_ids = frozenset(state.fleet_me.ids())
    last_friendly_ids = _cache.get("last_friendly_ids", frozenset())
    friendly_died = last_friendly_ids - friendly_ids

    healers = [b for b in state.fleet_me if b.class_ == BotClass.Healer]
    enemy_healers = [e for e in state.fleet_other if e.class_ == BotClass.Healer]
    last_enemy_healer_count = _cache.get("last_enemy_healer_count", len(enemy_healers))
    last_friendly_healer_count = _cache.get("last_friendly_healer_count", len(healers))
    healer_died = (
        len(healers) < last_friendly_healer_count or
        len(enemy_healers) < last_enemy_healer_count
    )

    last_assignment_tick = _cache.get("last_assignment_tick", -100)
    last_healer_pair_tick = _cache.get("last_healer_pair_tick", -100)

    enemy_delta = abs(enemy_count - last_enemy_count)

    needs_recompute = (
        last_assignment_tick < 0
        or state.tick - last_assignment_tick >= 5
        or enemy_delta >= 2
        or bool(friendly_died)
    )

    healer_pair_needs_recompute = (
        last_healer_pair_tick < 0
        or state.tick - last_healer_pair_tick >= 10
        or healer_died
    )

    if budget.remaining <= 100_000:
        action = _execute_assignments(state, conf, in_endgame)
    elif needs_recompute or healer_pair_needs_recompute:
        action = _recompute_assignments(state, conf, in_endgame)
        _cache["last_assignment_tick"] = state.tick
        if healer_pair_needs_recompute:
            _cache["last_healer_pair_tick"] = state.tick
    else:
        action = _execute_assignments(state, conf, in_endgame)

    _cache["last_tick"] = state.tick
    _cache["last_enemy_count"] = enemy_count
    _cache["last_friendly_ids"] = friendly_ids
    _cache["last_friendly_healer_count"] = len(healers)
    _cache["last_enemy_healer_count"] = len(enemy_healers)

    action.fabricator_next = _compute_fabricator_next(state, conf, _cache, in_endgame)
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
    bot_radius = conf.bot.radius

    healers = [b for b in state.fleet_me if b.class_ == BotClass.Healer]
    battles = [b for b in state.fleet_me if b.class_ == BotClass.Battle]
    extractors = [b for b in state.fleet_me if b.class_ == BotClass.Extractor]

    path_cache: Dict[Tuple[int, int], Optional[float]] = {}
    for battle in battles:
        for enemy in state.fleet_other:
            d = path_length(battle.pos, enemy.pos)
            path_cache[(battle.id, enemy.id)] = d

    enemy_centroid = Vec2(0.0, 0.0)
    if state.fleet_other:
        for e in state.fleet_other:
            enemy_centroid = enemy_centroid + e.pos
        enemy_centroid = enemy_centroid * (1.0 / len(state.fleet_other))

    enemy_in_capture = any(
        e.pos.dist(payload) <= conf.payload.capture_radius for e in state.fleet_other
    )

    target_by_battle: Dict[int, Optional[BotState]] = {}
    if battles:
        ranked = _rank_targets(state, path_cache)
        for battle in battles:
            target_by_battle[battle.id] = ranked[0].id if ranked else None

    assignments: Dict[int, dict] = {}

    if len(healers) >= 2:
        lead = healers[0]
        anvil = healers[1]
        anvil_spot = lead.pos + Vec2(0.0, -(2.5 * bot_radius))

        if enemy_in_capture:
            lead_move_target = payload
            lead_turn_target = enemy_centroid
        else:
            lead_move_target = payload
            lead_turn_target = payload

        assignments[lead.id] = {
            "kind": "healer",
            "role": "lead",
            "target_id": None,
            "move_target": lead_move_target,
            "turn_target": lead_turn_target,
            "half_speed": True,
        }
        assignments[anvil.id] = {
            "kind": "healer",
            "role": "anvil",
            "target_id": lead.id,
            "move_target": anvil_spot,
            "turn_target": payload,
            "half_speed": False,
        }
    elif len(healers) == 1:
        h = healers[0]
        assignments[h.id] = {
            "kind": "healer",
            "role": "lead",
            "target_id": None,
            "move_target": payload,
            "turn_target": payload,
            "half_speed": True,
        }

    battle_front: List[BotState] = []
    battle_back: List[BotState] = []
    if battles:
        lead_healer_pos = assignments.get(healers[0].id, {}).get("move_target", payload) if healers else payload
        for b in battles:
            if b.pos.dist(lead_healer_pos) <= 2 * bot_radius:
                battle_front.append(b)
            else:
                battle_back.append(b)

        slot_index = 0
        for b in battle_front:
            angle = (slot_index / max(len(battle_front), 1)) * 2.0 * math.pi
            offset = Vec2(math.cos(angle) * 2.0 * bot_radius, math.sin(angle) * 2.0 * bot_radius)
            pos_behind = lead_healer_pos + Vec2(0.0, -3.0 * bot_radius) + offset
            assignments[b.id] = {
                "kind": "battle",
                "role": "front",
                "target_id": target_by_battle.get(b.id, None),
                "move_target": pos_behind,
                "half_speed": True,
            }
            slot_index += 1

        for i, b in enumerate(battle_back):
            offset_x = (i % 2) * 2.0 * bot_radius
            pos_back = lead_healer_pos + Vec2(offset_x, -(5.0 + i * 2.0) * bot_radius)
            assignments[b.id] = {
                "kind": "battle",
                "role": "back",
                "target_id": target_by_battle.get(b.id, None),
                "move_target": pos_back,
                "half_speed": False,
            }

    extractor_cap = conf.deposit.extractor_cap
    my_extractors = state.deposit_me.extractors.me
    mining_spot_base = state.deposit_me.pos + Vec2(0.0, conf.deposit.radius + conf.bot.radius)

    for idx in range(my_extractors, min(extractor_cap, my_extractors + 3)):
        if len(extractors) <= idx:
            break
        ext = extractors[idx]
        offset_x = idx * 2.0 * bot_radius
        spot = Vec2(mining_spot_base.x + offset_x, mining_spot_base.y)
        assignments[ext.id] = {
            "kind": "extractor",
            "mining_spot": spot,
        }

    for bot in state.fleet_me:
        if bot.id in assignments:
            continue
        if bot.class_ == BotClass.Healer:
            candidates = [b for b in state.fleet_me if b.id != bot.id and b.health < conf.bot.health * 0.9]
            target = min(candidates, key=lambda b: b.health).id if candidates else None
            assignments[bot.id] = {
                "kind": "healer",
                "role": "combat",
                "target_id": target,
                "move_target": payload,
                "turn_target": payload,
            }
        elif bot.class_ == BotClass.Battle:
            assignments[bot.id] = {
                "kind": "battle",
                "role": "combat",
                "target_id": target_by_battle.get(bot.id, None),
                "move_target": payload,
            }
        else:
            assignments[bot.id] = {
                "kind": "extractor",
                "mining_spot": mining_spot_base,
            }

    _cache["assignments"] = assignments
    _cache["target_by_battle"] = target_by_battle

    _apply_assignments(action, state, conf, assignments, enemy_in_capture)
    return action


def _rank_targets(state: GameState, path_cache: Dict[Tuple[int, int], Optional[float]]) -> List[BotState]:
    healers = [e for e in state.fleet_other if e.class_ == BotClass.Healer]
    extractor_enemies = [e for e in state.fleet_other if e.class_ == BotClass.Extractor]
    battles = [e for e in state.fleet_other if e.class_ == BotClass.Battle]

    result: List[BotState] = []

    for rank_class, rank_list in [(BotClass.Healer, healers), (BotClass.Extractor, extractor_enemies), (BotClass.Battle, battles)]:
        if not rank_list:
            continue
        best_by_path: Dict[int, BotState] = {}
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
            best_by_path[enemy.id] = enemy
            best_dist[enemy.id] = min_d

        sorted_enemies = sorted(rank_list, key=lambda e: best_dist[e.id])
        result.extend(sorted_enemies)

    return result


def _apply_assignments(action: FleetAction, state: GameState, conf,
                       assignments: Dict, enemy_in_capture: bool) -> None:
    payload = state.payload_pos()
    enemies_by_id = {e.id: e for e in state.fleet_other}

    for bot in state.fleet_me:
        bot_action = action.bots[bot.id]
        assignment = assignments.get(bot.id, {})

        kind = assignment.get("kind", bot.class_.name.lower())
        role = assignment.get("role", "combat")
        half_speed = assignment.get("half_speed", False)

        if kind == "healer" or bot.class_ == BotClass.Healer:
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
            nav_dir = navigate_to(bot.pos, move_target)
            if half_speed:
                nav_dir = nav_dir * 0.5
            bot_action.move_action = move_bot(nav_dir)

        elif kind == "battle" or bot.class_ == BotClass.Battle:
            target_id = assignment.get("target_id")
            target = enemies_by_id.get(target_id) if target_id is not None else None

            if target is not None:
                in_range = bot.pos.dist(target.pos) <= conf.bot.blaster_range
                los = line_of_sight(bot.pos, target.pos)
                can_fire = in_range and los and not _would_splash_friendly(bot, target, state, conf)
                bot_action.special_action = SpecialAction.Battle(fire=can_fire)
                bot_action.turn_action = turn_towards(target.pos)
            else:
                bot_action.special_action = SpecialAction.Battle(fire=False)
                bot_action.turn_action = turn_towards(payload)

            move_target = assignment.get("move_target", payload)
            nav_dir = navigate_to(bot.pos, move_target)
            if half_speed:
                nav_dir = nav_dir * 0.5
            bot_action.move_action = move_bot(nav_dir)

        else:
            mining_spot = assignment.get("mining_spot")
            if mining_spot is None:
                mining_spot = state.deposit_me.pos + Vec2(0.0, conf.deposit.radius + conf.bot.radius)
            bot_action.move_action = move_bot(navigate_to(bot.pos, mining_spot))
            bot_action.turn_action = turn_towards(state.deposit_me.pos)
            bot_action.special_action = SpecialAction.Extractor(mine=True)


def _execute_assignments(state: GameState, conf, in_endgame: bool) -> FleetAction:
    action = FleetAction.new()
    payload = state.payload_pos()

    assignments = _cache.get("assignments", {})
    target_by_battle: Dict[int, Optional[BotState]] = _cache.get("target_by_battle", {})
    enemies_by_id = {e.id: e for e in state.fleet_other}

    enemy_in_capture = any(
        e.pos.dist(payload) <= conf.payload.capture_radius for e in state.fleet_other
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
            continue

        if kind == "healer" or bot.class_ == BotClass.Healer:
            target_id = assignment.get("target_id")
            target = state.fleet_me.get(target_id) if target_id is not None else None

            if target is not None and bot.class_ == BotClass.Healer:
                in_range = bot.pos.dist(target.pos) <= conf.bot.base_heal_range
                in_arc = _is_in_heal_arc(bot, target, conf)
                bot_action.special_action = SpecialAction.Healer(
                    fire=in_range and in_arc,
                    target=target.id,
                )
            else:
                bot_action.special_action = SpecialAction.Healer(fire=False, target=0)

            turn_target = assignment.get("turn_target", payload)
            bot_action.turn_action = turn_towards(turn_target)

            move_target = assignment.get("move_target", payload)
            half_speed = assignment.get("half_speed", False)
            nav_dir = navigate_to(bot.pos, move_target)
            if half_speed:
                nav_dir = nav_dir * 0.5
            bot_action.move_action = move_bot(nav_dir)
            continue

        if kind == "battle" or bot.class_ == BotClass.Battle:
            target_id = assignment.get("target_id")
            target = enemies_by_id.get(target_id) if target_id is not None else None

            if target is not None:
                in_range = bot.pos.dist(target.pos) <= conf.bot.blaster_range
                los = line_of_sight(bot.pos, target.pos)
                can_fire = in_range and los and not _would_splash_friendly(bot, target, state, conf)
                bot_action.special_action = SpecialAction.Battle(fire=can_fire)
                bot_action.turn_action = turn_towards(target.pos)
            else:
                bot_action.special_action = SpecialAction.Battle(fire=False)
                bot_action.turn_action = turn_towards(payload)

            move_target = assignment.get("move_target", payload)
            half_speed = assignment.get("half_speed", False)
            nav_dir = navigate_to(bot.pos, move_target)
            if half_speed:
                nav_dir = nav_dir * 0.5
            bot_action.move_action = move_bot(nav_dir)
            continue

        bot_action.move_action = move_bot(navigate_to(bot.pos, payload))
        if bot.class_ == BotClass.Extractor:
            bot_action.special_action = SpecialAction.Extractor(mine=True)
        elif bot.class_ == BotClass.Healer:
            bot_action.special_action = SpecialAction.Healer(fire=False, target=0)
        else:
            bot_action.special_action = SpecialAction.Battle(fire=False)

    return action


def _compute_fabricator_next(state: GameState, conf, cache: Dict, in_endgame: bool) -> int:
    fleet_size = len(state.fleet_me)
    extractor_count = sum(1 for b in state.fleet_me if b.class_ == BotClass.Extractor)
    healer_count = sum(1 for b in state.fleet_me if b.class_ == BotClass.Healer)

    if fleet_size == 0:
        return int(BotClass.Extractor)

    rush_mode = (
        not in_endgame
        and state.fabricator_me.tokens >= conf.fabricator.rush_cost
        and fleet_size < BOTS_MAX // 2
    )

    if rush_mode and extractor_count < 2:
        return int(BotClass.Extractor)

    if rush_mode and healer_count < 2:
        return int(BotClass.Healer)

    if rush_mode and fleet_size < 14:
        return int(BotClass.Battle)

    if fleet_size < 10:
        return int(BotClass.Battle)

    last_alt = cache.get("last_alt", 0)
    alt_count = cache.get("alt_count", 0)
    if fleet_size > last_alt:
        alt_count += 1
        cache["alt_count"] = alt_count
    cache["last_alt"] = fleet_size

    return int(BotClass.Healer if alt_count % 2 == 1 else BotClass.Battle)


def _is_in_heal_arc(healer: BotState, target: BotState, conf) -> bool:
    to_target = target.pos - healer.pos
    if to_target.norm_sq() < 0.0001:
        return True
    angle_to_target = to_target.angle_deg()
    diff = abs(diff_degrees(angle_to_target, healer.angle))
    half_arc = conf.bot.base_heal_arc_deg / 2.0
    return diff <= half_arc + 1e-6


def _would_splash_friendly(firer: BotState, target: BotState, state: GameState, conf) -> bool:
    splash = conf.bot.base_blaster_splash_radius
    if splash <= 0:
        return False
    for ally in state.fleet_me:
        if ally.id == firer.id:
            continue
        if ally.pos.dist(target.pos) <= splash:
            return True
    return False


def _maybe_endgame_self_destruct(state: GameState, conf, action: FleetAction, cache: Dict) -> None:
    payload = state.payload_pos()
    threshold = conf.bot.blaster_range * 3.0
    extractors = [b for b in state.fleet_me if b.class_ == BotClass.Extractor]
    far_extractors = [e for e in extractors if e.pos.dist(payload) > threshold]
    if far_extractors:
        weakest = min(far_extractors, key=lambda b: b.health)
        action.bots[weakest.id].self_destruct = True
        cache["endgame_sd_done"] = True
