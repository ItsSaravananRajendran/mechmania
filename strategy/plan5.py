"""Plan 5 — Skirmisher Kiter.

See plans/plan5.md for the full spec (requirements R-P5.* and R-X*).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from . import (
    BotAction,
    BotClass,
    BotState,
    FleetAction,
    GameState,
    SpecialAction,
    Vec2,
    corridor_clear,
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


def plan5_strategy(state: GameState) -> FleetAction:
    budget = get_budget()
    conf = get_config()
    in_endgame = state.tick >= conf.max_ticks - conf.endgame_ticks

    last_tick = _cache.get("last_tick", -1)
    if last_tick >= 0 and state.tick - last_tick > 1:
        _cache["assignments"] = {}

    payload = state.payload_pos()
    capture_radius = conf.payload.capture_radius
    bot_radius = conf.bot.radius
    blaster_range = conf.bot.blaster_range
    splash_radius = conf.bot.base_blaster_splash_radius

    battles = [b for b in state.fleet_me if b.class_ == BotClass.Battle]
    healers = [h for h in state.fleet_me if h.class_ == BotClass.Healer]
    extractors = [e for e in state.fleet_me if e.class_ == BotClass.Extractor]

    friendly_ids = frozenset(b.id for b in state.fleet_me)
    last_friendly_ids = _cache.get("last_friendly_ids", frozenset())
    friendly_died = last_friendly_ids - friendly_ids

    last_assignment_tick = _cache.get("last_assignment_tick", -100)

    needs_recompute = (
        last_assignment_tick < 0
        or state.tick - last_assignment_tick >= 3
        or bool(friendly_died)
    )

    if budget.remaining <= 100_000:
        action = _execute_fallback(state, conf, in_endgame)
    elif needs_recompute:
        action = _recompute_assignments(state, conf, in_endgame)
        _cache["last_assignment_tick"] = state.tick
    else:
        action = _execute_assignments(state, conf, in_endgame)

    _cache["last_tick"] = state.tick
    _cache["last_friendly_ids"] = friendly_ids

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
    blaster_range = conf.bot.blaster_range
    splash_radius = conf.bot.base_blaster_splash_radius

    battles = [b for b in state.fleet_me if b.class_ == BotClass.Battle]
    healers = [h for h in state.fleet_me if h.class_ == BotClass.Healer]
    extractors = [e for e in state.fleet_me if e.class_ == BotClass.Extractor]

    enemies = list(state.fleet_other)
    enemies_by_id = {e.id: e for e in enemies}

    friendly_in_capture = [b for b in state.fleet_me if b.pos.dist(payload) <= capture_radius]
    in_capture_ids = set(b.id for b in friendly_in_capture)

    _screen_info = _compute_screen_geometry(state, conf)
    screen_centroid = _screen_info["screen_centroid"]
    screen_dir = _screen_info["screen_dir"]
    screen_perp = _screen_info["screen_perp"]
    screen_positions = _screen_info["screen_positions"]

    _cache["screen_centroid"] = screen_centroid
    _cache["screen_positions"] = screen_positions
    _cache["screen_perp"] = screen_perp
    _cache["screen_dir"] = screen_dir

    _update_engagement(state, conf)

    payload_battle_id: Optional[int] = None
    screen_battle_ids = []

    for i, battle in enumerate(battles):
        if i < len(screen_positions):
            move_target = screen_positions[i]
        else:
            move_target = payload

        if battle.id in in_capture_ids:
            if payload_battle_id is None:
                payload_battle_id = battle.id
            else:
                excess_offset = screen_dir * (capture_radius + bot_radius * 2)
                move_target = payload + excess_offset
            role = "payload"
        else:
            screen_battle_ids.append(battle.id)
            role = "screen"

        target_enemy = _pick_target_enemy(battle, enemies, enemies_by_id, conf, splash_radius)

        engagement = _cache.get("battle_engagement", {}).get(battle.id, {})
        engaged = engagement.get("engaged", False)
        fired_this_engagement = _cache.get("fired_engagement", {}).get(battle.id, False)
        retreat_info = engagement.get("retreat_info")

        if engaged and retreat_info:
            retreat_point, path_blocked = retreat_info
            if path_blocked:
                enemy_centroid = _cache.get("enemy_centroid")
                if enemy_centroid is not None:
                    to_enemy = enemy_centroid - battle.pos
                    if to_enemy.norm_sq() > 1e-8:
                        perp = Vec2(-to_enemy.y, to_enemy.x).normalize_or_zero()
                        retreat_point = battle.pos + perp * 1.5 * bot_radius

            _cache.setdefault("retreat_cache", {})[battle.id] = retreat_point

        assignments = _cache.setdefault("assignments", {})
        assignments[battle.id] = {
            "kind": "battle",
            "role": role,
            "target_id": target_enemy.id if target_enemy else None,
            "move_target": move_target,
            "screen_pos": screen_positions[i] if i < len(screen_positions) else None,
        }

    for h in healers:
        wounded = [b for b in state.fleet_me if b.id != h.id and b.health < conf.bot.health * 0.9]
        if wounded:
            target = min(wounded, key=lambda b: b.pos.dist(h.pos))
            target_id = target.id
            move_target = target.pos
        else:
            target_id = None
            if screen_centroid is not None and screen_dir is not None:
                move_target = screen_centroid - screen_dir * (3.0 * bot_radius)
            else:
                move_target = payload

        assignments = _cache.setdefault("assignments", {})
        assignments[h.id] = {
            "kind": "healer",
            "target_id": target_id,
            "move_target": move_target,
            "screen_centroid": screen_centroid,
        }

    for e in extractors:
        mining_spot = _compute_mining_spot(state, conf)
        assignments = _cache.setdefault("assignments", {})
        assignments[e.id] = {
            "kind": "extractor",
            "mining_spot": mining_spot,
        }

    _cache["payload_battle_id"] = payload_battle_id
    _cache["screen_battle_ids"] = screen_battle_ids
    _cache["screen_centroid_for_healer"] = screen_centroid

    _apply_assignments(action, state, conf, in_endgame)

    fired_cache = _cache.setdefault("fired_engagement", {})
    for b in battles:
        ba = action.bots[b.id]
        if ba.special_action.tag == 0:  # Battle
            if ba.special_action.payload.battle.fire:
                fired_cache[b.id] = True

    _enforce_no_overlap(state, conf, action)

    return action


def _execute_assignments(state: GameState, conf, in_endgame: bool) -> FleetAction:
    action = FleetAction.new()
    if in_endgame:
        _apply_endgame_posture(action, state, conf)
        return action
    payload = state.payload_pos()
    capture_radius = conf.payload.capture_radius
    bot_radius = conf.bot.radius
    blaster_range = conf.bot.blaster_range

    enemies = list(state.fleet_other)
    enemies_by_id = {e.id: e for e in enemies}

    screen_centroid = _cache.get("screen_centroid")
    screen_positions = _cache.get("screen_positions", [])
    retreat_cache = _cache.get("retreat_cache", {})
    battle_engagement = _cache.get("battle_engagement", {})
    fired_engagement = _cache.get("fired_engagement", {})

    for bot in state.fleet_me:
        bot_action = action.bots[bot.id]
        assignments = _cache.get("assignments", {})
        assignment = assignments.get(bot.id, {})

        kind = assignment.get("kind", bot.class_.name.lower())

        if kind == "battle" or bot.class_ == BotClass.Battle:
            role = assignment.get("role", "screen")
            target_id = assignment.get("target_id")
            target = enemies_by_id.get(target_id) if target_id is not None else None
            screen_pos = assignment.get("screen_pos")

            in_range = target is not None and bot.pos.dist(target.pos) <= blaster_range
            los = target is not None and line_of_sight(bot.pos, target.pos)

            engagement = battle_engagement.get(bot.id, {})
            engaged = engagement.get("engaged", False)
            already_fired = fired_engagement.get(bot.id, False)

            fire = in_range and los and not already_fired

            if engaged and target is not None:
                cached_retreat = retreat_cache.get(bot.id)
                if cached_retreat is not None:
                    move_target = cached_retreat
                    bot_action.move_action = move_bot(navigate_to(bot.pos, move_target))
                    bot_action.turn_action = turn_towards(target.pos)
                    bot_action.special_action = SpecialAction.Battle(fire=fire)
                    continue

            if target is not None:
                bot_action.turn_action = turn_towards(target.pos)
            else:
                bot_action.turn_action = turn_towards(payload)

            if role == "payload":
                move_target = payload
            elif screen_pos is not None:
                move_target = screen_pos
            else:
                move_target = assignment.get("move_target", payload)

            bot_action.move_action = move_bot(navigate_to(bot.pos, move_target))
            bot_action.special_action = SpecialAction.Battle(fire=fire)

        elif kind == "healer" or bot.class_ == BotClass.Healer:
            target_id = assignment.get("target_id")
            target = state.fleet_me.get(target_id) if target_id is not None else None

            if target is not None:
                bot_action.turn_action = turn_towards(target.pos)
                in_range = bot.pos.dist(target.pos) <= conf.bot.base_heal_range
                in_arc = _is_in_heal_arc(bot, target, conf)
                bot_action.special_action = SpecialAction.Healer(
                    fire=in_range and in_arc,
                    target=target_id,
                )
            else:
                sc = screen_centroid
                if sc is not None:
                    bot_action.turn_action = turn_towards(sc)
                else:
                    bot_action.turn_action = turn_towards(payload)
                bot_action.special_action = SpecialAction.Healer(fire=False, target=0)

            move_target = assignment.get("move_target", payload)
            bot_action.move_action = move_bot(navigate_to(bot.pos, move_target))

        elif kind == "extractor" or bot.class_ == BotClass.Extractor:
            mining_spot = assignment.get("mining_spot")
            if mining_spot is None:
                mining_spot = _compute_mining_spot(state, conf)
            bot_action.move_action = move_bot(navigate_to(bot.pos, mining_spot))
            bot_action.turn_action = turn_towards(state.deposit_me.pos)
            bot_action.special_action = SpecialAction.Extractor(mine=True)

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
    if in_endgame:
        _apply_endgame_posture(action, state, conf)
        return action
    payload = state.payload_pos()
    capture_radius = conf.payload.capture_radius
    bot_radius = conf.bot.radius
    blaster_range = conf.bot.blaster_range

    battles = [b for b in state.fleet_me if b.class_ == BotClass.Battle]
    healers = [h for h in state.fleet_me if h.class_ == BotClass.Healer]
    extractors = [e for e in state.fleet_me if e.class_ == BotClass.Extractor]

    screen_centroid = _cache.get("screen_centroid")
    screen_positions = _cache.get("screen_positions", [])
    retreat_cache = _cache.get("retreat_cache", {})
    battle_engagement = _cache.get("battle_engagement", {})
    fired_engagement = _cache.get("fired_engagement", {})
    assignments = _cache.get("assignments", {})

    screen_idx = 0
    for bot in battles:
        bot_action = action.bots[bot.id]
        assignment = assignments.get(bot.id, {})

        target_id = assignment.get("target_id")
        enemies_by_id = {e.id: e for e in state.fleet_other}
        target = enemies_by_id.get(target_id) if target_id is not None else None

        in_range = target is not None and bot.pos.dist(target.pos) <= blaster_range
        los = target is not None and line_of_sight(bot.pos, target.pos)

        engagement = battle_engagement.get(bot.id, {})
        engaged = engagement.get("engaged", False)
        already_fired = fired_engagement.get(bot.id, False)

        fire = in_range and los and not already_fired

        if engaged:
            cached_retreat = retreat_cache.get(bot.id)
            if cached_retreat is not None:
                bot_action.move_action = move_bot(navigate_to(bot.pos, cached_retreat))
                bot_action.turn_action = turn_towards(target.pos) if target else turn_towards(payload)
                bot_action.special_action = SpecialAction.Battle(fire=fire)
                continue

        if target is not None:
            bot_action.turn_action = turn_towards(target.pos)
        else:
            bot_action.turn_action = turn_towards(payload)

        if screen_idx < len(screen_positions):
            move_target = screen_positions[screen_idx]
        else:
            move_target = payload

        bot_action.move_action = move_bot(navigate_to(bot.pos, move_target))
        bot_action.special_action = SpecialAction.Battle(fire=fire)
        screen_idx += 1

    for h in healers:
        bot_action = action.bots[h.id]
        target_id = assignments.get(h.id, {}).get("target_id")
        target = state.fleet_me.get(target_id) if target_id is not None else None

        if target is not None:
            bot_action.turn_action = turn_towards(target.pos)
            in_range = h.pos.dist(target.pos) <= conf.bot.base_heal_range
            in_arc = _is_in_heal_arc(h, target, conf)
            bot_action.special_action = SpecialAction.Healer(
                fire=in_range and in_arc,
                target=target_id,
            )
            move_target = target.pos
        else:
            sc = screen_centroid
            if sc is not None:
                bot_action.turn_action = turn_towards(sc)
            else:
                bot_action.turn_action = turn_towards(payload)
            bot_action.special_action = SpecialAction.Healer(fire=False, target=0)
            move_target = sc if sc is not None else payload

        bot_action.move_action = move_bot(navigate_to(h.pos, move_target))

    for e in extractors:
        bot_action = action.bots[e.id]
        mining_spot = assignments.get(e.id, {}).get("mining_spot")
        if mining_spot is None:
            mining_spot = _compute_mining_spot(state, conf)
        bot_action.move_action = move_bot(navigate_to(e.pos, mining_spot))
        bot_action.turn_action = turn_towards(state.deposit_me.pos)
        bot_action.special_action = SpecialAction.Extractor(mine=True)

    _enforce_no_overlap(state, conf, action)
    return action


def _compute_fabricator_next(state: GameState, conf, cache: Dict) -> int:
    if not state.fleet_me.get(0):
        return int(BotClass.Extractor)

    extractor_count = sum(1 for b in state.fleet_me if b.class_ == BotClass.Extractor)
    battle_count = sum(1 for b in state.fleet_me if b.class_ == BotClass.Battle)

    if extractor_count == 0:
        return int(BotClass.Extractor)

    if battle_count < 4:
        return int(BotClass.Battle)

    if state.tick > 30:
        last_build = cache.get("last_build", BotClass.Battle)
        if last_build == BotClass.Battle:
            cache["last_build"] = BotClass.Healer
            return int(BotClass.Healer)
        else:
            cache["last_build"] = BotClass.Battle
            return int(BotClass.Battle)

    return int(BotClass.Battle)


def _compute_screen_geometry(state: GameState, conf) -> Dict:
    payload = state.payload_pos()
    bot_radius = conf.bot.radius
    blaster_range = conf.bot.blaster_range

    battles = [b for b in state.fleet_me if b.class_ == BotClass.Battle]
    capture_radius = conf.payload.capture_radius

    screen_battles = [b for b in battles if b.pos.dist(payload) > capture_radius]

    if screen_battles:
        avg_x = sum(b.pos.x for b in screen_battles) / len(screen_battles)
        avg_y = sum(b.pos.y for b in screen_battles) / len(screen_battles)
        screen_centroid = Vec2(avg_x, avg_y)
    else:
        screen_centroid = payload

    path = conf.payload_path
    if path and len(path) >= 2:
        seg_idx = min(int(state.capture * (len(path) - 1)), len(path) - 2)
        seg_dir = path[seg_idx + 1] - path[seg_idx]
        if seg_dir.norm_sq() > 1e-8:
            seg_dir = seg_dir.normalize_or_zero()
        else:
            seg_dir = Vec2(0.0, 1.0)
    else:
        seg_dir = Vec2(0.0, 1.0)

    perp = Vec2(-seg_dir.y, seg_dir.x)
    if perp.norm_sq() < 1e-8:
        perp = Vec2(1.0, 0.0)

    screen_dist = 2.0 * blaster_range
    screen_center = payload - seg_dir * screen_dist

    spacing = max(3.0 * bot_radius, 0.001)
    n_screen = len(screen_battles)
    if n_screen == 0:
        screen_positions = []
    elif n_screen == 1:
        screen_positions = [screen_center]
    else:
        total_span = (n_screen - 1) * spacing
        start_offset = -total_span / 2.0
        screen_positions = []
        for i in range(n_screen):
            offset = start_offset + i * spacing
            screen_positions.append(screen_center + perp * offset)

    return {
        "screen_centroid": screen_centroid,
        "screen_dir": seg_dir,
        "screen_perp": perp,
        "screen_positions": screen_positions,
        "screen_center": screen_center,
    }


def _update_engagement(state: GameState, conf) -> None:
    blaster_range = conf.bot.blaster_range
    payload = state.payload_pos()
    capture_radius = conf.payload.capture_radius

    battles = [b for b in state.fleet_me if b.class_ == BotClass.Battle]
    enemies = list(state.fleet_other)

    engagement = _cache.setdefault("battle_engagement", {})
    fired_cache = _cache.setdefault("fired_engagement", {})

    current_enemy_centroid = None
    if enemies:
        avg_x = sum(e.pos.x for e in enemies) / len(enemies)
        avg_y = sum(e.pos.y for e in enemies) / len(enemies)
        current_enemy_centroid = Vec2(avg_x, avg_y)
    _cache["enemy_centroid"] = current_enemy_centroid

    screen_battles = [b for b in battles if b.pos.dist(payload) > capture_radius]

    for battle in screen_battles:
        enemies_in_range = [e for e in enemies if battle.pos.dist(e.pos) <= blaster_range]

        if enemies_in_range:
            if not engagement.get(battle.id, {}).get("engaged", False):
                engagement[battle.id] = {"engaged": True, "retreat_computed": False}
                fired_cache[battle.id] = False

            if not engagement[battle.id].get("retreat_computed", False):
                enemy_centroid = Vec2(
                    sum(e.pos.x for e in enemies_in_range) / len(enemies_in_range),
                    sum(e.pos.y for e in enemies_in_range) / len(enemies_in_range),
                )

                healer = next((h for h in state.fleet_me if h.class_ == BotClass.Healer), None)
                if healer is None:
                    healer = next(iter(state.fleet_me), battle)
                to_healer = healer.pos - battle.pos
                if to_healer.norm_sq() > 1e-8:
                    retreat_dir = to_healer.normalize_or_zero()
                else:
                    retreat_dir = Vec2(0.0, 1.0)

                retreat_point = battle.pos + retreat_dir * (2.0 * blaster_range)

                path_clear = corridor_clear(battle.pos, retreat_point)
                path_blocked = not path_clear

                engagement[battle.id]["retreat_info"] = (retreat_point, path_blocked)
                engagement[battle.id]["retreat_computed"] = True

            engagement[battle.id]["ticks_out_of_range"] = 0
        else:
            entry = engagement.setdefault(battle.id, {})
            ticks_out = entry.get("ticks_out_of_range", 0) + 1
            entry["ticks_out_of_range"] = ticks_out
            if ticks_out >= 10:
                engagement[battle.id] = {"engaged": False}
                fired_cache[battle.id] = False
                _cache.setdefault("retreat_cache", {}).pop(battle.id, None)

    _cache["battle_engagement"] = engagement
    _cache["fired_engagement"] = fired_cache


def _pick_target_enemy(
    bot: BotState,
    enemies: list,
    enemies_by_id: dict,
    conf,
    splash_radius: float,
) -> Optional[BotState]:
    if not enemies:
        return None

    los_enemies = [e for e in enemies if line_of_sight(bot.pos, e.pos)]

    healer_candidates = []
    non_healer_candidates = []

    for e in los_enemies:
        if e.class_ == BotClass.Healer:
            others_nearby = any(
                e2.id != e.id and e.pos.dist(e2.pos) <= splash_radius
                for e2 in enemies
            )
            if not others_nearby:
                healer_candidates.append(e)
        else:
            non_healer_candidates.append(e)

    candidates = healer_candidates if healer_candidates else non_healer_candidates
    if not candidates:
        return None

    return min(candidates, key=lambda e: e.health)


def _compute_mining_spot(state: GameState, conf) -> Vec2:
    deposit = state.deposit_me.pos
    deposit_radius = conf.deposit.radius

    path = conf.payload_path
    if path and len(path) >= 2:
        seg_idx = min(int(state.capture * (len(path) - 1)), len(path) - 2)
        seg_dir = path[seg_idx + 1] - path[seg_idx]
        if seg_dir.norm_sq() > 1e-8:
            seg_dir = seg_dir.normalize_or_zero()
        else:
            seg_dir = Vec2(0.0, 1.0)
    else:
        seg_dir = Vec2(0.0, 1.0)

    standoff = 2.0 * deposit_radius
    return deposit - seg_dir * (2.0 * deposit_radius)


def _is_in_heal_arc(healer: BotState, target: BotState, conf) -> bool:
    to_target = target.pos - healer.pos
    if to_target.norm_sq() < 0.0001:
        return True
    angle_to_target = to_target.angle_deg()
    diff = abs(diff_degrees(angle_to_target, healer.angle))
    half_arc = conf.bot.base_heal_arc_deg / 2.0
    return diff <= half_arc + 1e-6


def _apply_assignments(action: FleetAction, state: GameState, conf, in_endgame: bool) -> None:
    payload = state.payload_pos()
    blaster_range = conf.bot.blaster_range
    enemies_by_id = {e.id: e for e in state.fleet_other}

    assignments = _cache.get("assignments", {})
    screen_centroid = _cache.get("screen_centroid_for_healer")
    retreat_cache = _cache.get("retreat_cache", {})
    battle_engagement = _cache.get("battle_engagement", {})

    if in_endgame:
        _apply_endgame_posture(action, state, conf)
        return

    for bot in state.fleet_me:
        bot_action = action.bots[bot.id]
        assignment = assignments.get(bot.id, {})

        kind = assignment.get("kind", bot.class_.name.lower())

        if kind == "extractor" or bot.class_ == BotClass.Extractor:
            mining_spot = assignment.get("mining_spot")
            if mining_spot is None:
                mining_spot = _compute_mining_spot(state, conf)
            bot_action.move_action = move_bot(navigate_to(bot.pos, mining_spot))
            bot_action.turn_action = turn_towards(state.deposit_me.pos)
            bot_action.special_action = SpecialAction.Extractor(mine=True)

        elif kind == "healer" or bot.class_ == BotClass.Healer:
            target_id = assignment.get("target_id")
            target = state.fleet_me.get(target_id) if target_id is not None else None

            if target is not None:
                bot_action.turn_action = turn_towards(target.pos)
                in_range = bot.pos.dist(target.pos) <= conf.bot.base_heal_range
                in_arc = _is_in_heal_arc(bot, target, conf)
                bot_action.special_action = SpecialAction.Healer(
                    fire=in_range and in_arc,
                    target=target_id,
                )
                move_target = target.pos
            else:
                sc = screen_centroid
                if sc is not None:
                    bot_action.turn_action = turn_towards(sc)
                else:
                    bot_action.turn_action = turn_towards(payload)
                bot_action.special_action = SpecialAction.Healer(fire=False, target=0)
                move_target = sc if sc is not None else payload

            bot_action.move_action = move_bot(navigate_to(bot.pos, move_target))

        elif kind == "battle" or bot.class_ == BotClass.Battle:
            role = assignment.get("role", "screen")
            target_id = assignment.get("target_id")
            target = enemies_by_id.get(target_id) if target_id is not None else None

            in_range = target is not None and bot.pos.dist(target.pos) <= blaster_range
            los = target is not None and line_of_sight(bot.pos, target.pos)

            engagement = battle_engagement.get(bot.id, {})
            engaged = engagement.get("engaged", False)

            fire = in_range and los

            if engaged:
                cached_retreat = retreat_cache.get(bot.id)
                if cached_retreat is not None:
                    bot_action.move_action = move_bot(navigate_to(bot.pos, cached_retreat))
                    bot_action.turn_action = turn_towards(target.pos) if target else turn_towards(payload)
                    bot_action.special_action = SpecialAction.Battle(fire=fire)
                    continue

            if target is not None:
                bot_action.turn_action = turn_towards(target.pos)
            else:
                bot_action.turn_action = turn_towards(payload)

            move_target = assignment.get("move_target", payload)
            bot_action.move_action = move_bot(navigate_to(bot.pos, move_target))
            bot_action.special_action = SpecialAction.Battle(fire=fire)

        else:
            bot_action.move_action = move_bot(navigate_to(bot.pos, payload))
            if bot.class_ == BotClass.Extractor:
                bot_action.special_action = SpecialAction.Extractor(mine=True)
            elif bot.class_ == BotClass.Healer:
                bot_action.special_action = SpecialAction.Healer(fire=False, target=0)
            else:
                bot_action.special_action = SpecialAction.Battle(fire=False)


def _apply_endgame_posture(action: FleetAction, state: GameState, conf) -> None:
    payload = state.payload_pos()
    bot_radius = conf.bot.radius
    blaster_range = conf.bot.blaster_range

    battles = [b for b in state.fleet_me if b.class_ == BotClass.Battle]
    healers = [h for h in state.fleet_me if h.class_ == BotClass.Healer]
    extractors = [e for e in state.fleet_me if e.class_ == BotClass.Extractor]

    enemies = list(state.fleet_other)
    enemy_centroid = None
    if enemies:
        avg_x = sum(e.pos.x for e in enemies) / len(enemies)
        avg_y = sum(e.pos.y for e in enemies) / len(enemies)
        enemy_centroid = Vec2(avg_x, avg_y)

    for b in battles:
        bot_action = action.bots[b.id]
        bot_action.move_action = move_bot(navigate_to(b.pos, payload))
        if enemy_centroid is not None:
            bot_action.turn_action = turn_towards(enemy_centroid)
        else:
            bot_action.turn_action = turn_towards(payload)
        target = _pick_endgame_target(b, enemies, conf)
        in_range = target is not None and b.pos.dist(target.pos) <= blaster_range
        los = target is not None and line_of_sight(b.pos, target.pos)
        bot_action.special_action = SpecialAction.Battle(fire=in_range and los)

    for h in healers:
        bot_action = action.bots[h.id]
        if battles:
            battle_centroid = Vec2(
                sum(b.pos.x for b in battles) / len(battles),
                sum(b.pos.y for b in battles) / len(battles),
            )
            bot_action.move_action = move_bot(navigate_to(h.pos, battle_centroid))
        else:
            bot_action.move_action = move_bot(navigate_to(h.pos, payload))

        if enemy_centroid is not None:
            bot_action.turn_action = turn_towards(enemy_centroid)
        else:
            bot_action.turn_action = turn_towards(payload)

        wounded = [b for b in state.fleet_me if b.id != h.id and b.health < conf.bot.health * 0.9]
        if wounded:
            target = min(wounded, key=lambda b: b.health)
            in_range = h.pos.dist(target.pos) <= conf.bot.base_heal_range
            in_arc = _is_in_heal_arc(h, target, conf)
            bot_action.special_action = SpecialAction.Healer(
                fire=in_range and in_arc,
                target=target.id,
            )
        else:
            bot_action.special_action = SpecialAction.Healer(fire=False, target=0)


def _pick_endgame_target(bot: BotState, enemies: list, conf) -> Optional[BotState]:
    if not enemies:
        return None
    blaster_range = conf.bot.blaster_range
    splash_radius = conf.bot.base_blaster_splash_radius

    in_range = [e for e in enemies if bot.pos.dist(e.pos) <= blaster_range and line_of_sight(bot.pos, e.pos)]
    if not in_range:
        return None

    healer_candidates = []
    non_healer_candidates = []

    for e in in_range:
        if e.class_ == BotClass.Healer:
            others_nearby = any(
                e2.id != e.id and e.pos.dist(e2.pos) <= splash_radius
                for e2 in enemies
            )
            if not others_nearby:
                healer_candidates.append(e)
        else:
            non_healer_candidates.append(e)

    candidates = healer_candidates if healer_candidates else non_healer_candidates
    if not candidates:
        return None

    return min(candidates, key=lambda e: e.health)


def _enforce_no_overlap(state: GameState, conf, action: FleetAction) -> None:
    bot_radius = conf.bot.radius
    epsilon = 0.5 * bot_radius

    bots = list(state.fleet_me)
    n = len(bots)
    for i in range(n):
        for j in range(i + 1, n):
            bi = bots[i]
            bj = bots[j]
            d = bi.pos.dist(bj.pos)
            if d < epsilon and d > 0.0001:
                diff = bj.pos - bi.pos
                if diff.norm_sq() > 1e-8:
                    push_dir = diff.normalize_or_zero()
                else:
                    push_dir = Vec2(1.0, 0.0)
                push_dist = epsilon - d + 0.1
                target_i = bi.pos - push_dir * (push_dist * 0.5)
                target_j = bj.pos + push_dir * (push_dist * 0.5)

                action.bots[bi.id].move_action = move_bot(navigate_to(bi.pos, target_i))
                action.bots[bj.id].move_action = move_bot(navigate_to(bj.pos, target_j))


def _maybe_endgame_self_destruct(state: GameState, conf, action: FleetAction, cache: Dict) -> None:
    extractors = [b for b in state.fleet_me if b.class_ == BotClass.Extractor]
    if not extractors:
        cache["endgame_sd_done"] = True
        return

    payload = state.payload_pos()
    blaster_range = conf.bot.blaster_range

    for e in extractors:
        d = path_length(e.pos, payload)
        if d is not None and d <= 4.0 * blaster_range:
            continue
        action.bots[e.id].self_destruct = True

    cache["endgame_sd_done"] = True
