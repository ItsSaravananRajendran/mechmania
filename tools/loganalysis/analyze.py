"""Behavioral analysis over a parsed `.mmgl` replay -- built to answer
"how does the opponent fleet actually behave", specifically:

- `hit_reactions`: when a bot takes damage, does it flee, keep fighting, or
  get left to die -- and how long until a healer reaches it.
- `healer_movement`: do healers camp a spot or chase around the map, and
  how much of the time do they actually have a heal target.
- `mining_deployment`: how many extractors are parked on a deposit and how
  many battle bots are riding shotgun for them, sampled over the match.
- `formation_shape`: the geometric shape (tight cluster / line / spread
  wedge) of the battle-bot group and the healer group, sampled over time.
- `payload_contest`: is the payload actually being fought over, and by
  which side -- this is the primary win condition, so a fleet that's
  winning the fight but ignoring the payload is a real finding.
- `economy`: token accumulation, build order, and deposit-slot occupancy
  (including denial -- sitting on slots you don't need, just to starve the
  other team's income).
- `engagements`: individual hits (from `hit_reactions`) clustered in time
  into discrete fights, so "how does the fleet behave when hit" has a
  fight-level answer too, not just a per-hit one.

Everything here reads a stream of `TickSnapshot`s (see `parser.py`) and
returns small, JSON-friendly dataclasses -- `report.py` turns those into
text; `cli.py` and `mcp_server.py` wire it to log files on disk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from .parser import (
    BotSnap,
    MatchResult,
    ROLE_BATTLE,
    ROLE_EXTRACTOR,
    ROLE_HEALER,
    Team,
    TickSnapshot,
)

# --------------------------------------------------------------------------
# Hit reactions
# --------------------------------------------------------------------------

# How many ticks after a hit to look at for "did it flee" / "did a healer
# reach it" -- a few times the healer's own beam interval is plenty; too
# short and a bot mid-turn reads as "didn't react", too long and unrelated
# later behavior gets blamed on this hit.
REACTION_WINDOW_TICKS = 30


@dataclass
class HitEvent:
    tick: int
    bot_id: int
    role: str
    damage: float
    health_after: float
    died: bool
    fled: Optional[bool]  # None when the bot died before the window closed
    speed_before: float
    speed_after: float
    healed_within_window: bool
    ticks_to_heal: Optional[int]


@dataclass
class HitReactionSummary:
    total_hits: int
    deaths: int
    fled_count: int
    held_ground_count: int
    healed_within_window_count: int
    avg_ticks_to_heal: Optional[float]
    by_role: Dict[str, int]
    events: List[HitEvent]


def _speed(b: BotSnap) -> float:
    return math.hypot(b.vx, b.vy)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _is_being_healed(b: BotSnap, healers: List[BotSnap]) -> bool:
    for h in healers:
        target = h.special.get(ROLE_HEALER, {}).get("healing")
        if isinstance(target, dict) and target.get("Some") == b.id:
            return True
    return False


def analyze_hit_reactions(
    ticks: List[TickSnapshot], team: Team, window: int = REACTION_WINDOW_TICKS
) -> HitReactionSummary:
    """Walk `ticks` looking for per-bot health drops on `team`, then look
    `window` ticks ahead to see whether that bot's speed picked up (fled),
    stayed put, died, or got a healer's beam on it."""

    # The engine reuses a bot id once it frees up, so grouping raw history
    # by id alone would splice a dead bot's timeline onto its eventual
    # replacement's. Instead, split each id's appearances into "lives"
    # bounded by `added`/`removed` events, and track each life's death tick
    # (if any -- the engine deletes a killed bot outright rather than ever
    # emitting a final `health: 0`, so this is the only way to tell "died
    # this tick" from "log ended while still alive") separately.
    lives: Dict[int, List[List[Tuple[int, BotSnap]]]] = {}
    life_death_tick: Dict[int, List[Optional[int]]] = {}

    def _current_life(bot_id: int) -> List[Tuple[int, BotSnap]]:
        segs = lives.setdefault(bot_id, [])
        deaths = life_death_tick.setdefault(bot_id, [])
        if not segs or deaths[-1] is not None:
            segs.append([])
            deaths.append(None)
        return segs[-1]

    for snap in ticks:
        for bid in snap.added.get(team, []):
            _current_life(bid)  # force a fresh life to start here
        for b in snap.bots(team):
            _current_life(b.id).append((snap.tick, b))
        for bid in snap.removed.get(team, []):
            if bid in life_death_tick and life_death_tick[bid]:
                life_death_tick[bid][-1] = snap.tick

    tick_index: Dict[int, TickSnapshot] = {s.tick: s for s in ticks}

    events: List[HitEvent] = []
    for bot_id, life_segments in lives.items():
      for life_idx, history in enumerate(life_segments):
        death_tick_for_life = life_death_tick[bot_id][life_idx]
        hit_indices = [
            i
            for i in range(1, len(history))
            if history[i - 1][1].health - history[i][1].health > 1e-6
        ]
        last_hit_index = hit_indices[-1] if hit_indices else None
        for i in hit_indices:
            prev_tick, prev = history[i - 1]
            cur_tick, cur = history[i]
            damage = prev.health - cur.health

            # The engine deletes a killed bot outright without ever logging
            # a final `health: 0`, so the last damage event recorded for a
            # bot that was later removed is the best evidence of its death
            # this log gives us, even when some ticks separate the two (the
            # bot may have sat at low health, invulnerable, before the
            # actual fatal hit came in unlogged).
            is_last_hit = i == last_hit_index
            died = cur.health <= 1e-6 or (is_last_hit and death_tick_for_life is not None)
            speed_before = _speed(prev)

            # Look ahead up to `window` ticks (bounded by data actually
            # present -- a hit near the end of the log just gets a shorter
            # window rather than raising).
            end_tick = cur_tick + window
            fled: Optional[bool] = None
            speed_after = _speed(cur)
            healed = False
            ticks_to_heal: Optional[int] = None
            if not died:
                future = [
                    (t, b)
                    for t, b in history[i:]
                    if t <= end_tick
                ]
                if len(future) >= 2:
                    speeds = [_speed(b) for _, b in future]
                    speed_after = max(speeds)
                    # "Fled" = meaningfully faster than it was moving right
                    # before the hit landed, at some point in the window.
                    fled = speed_after > speed_before * 1.5 + 0.001
                for t, _ in future:
                    snap = tick_index.get(t)
                    if snap is None:
                        continue
                    bot_now = snap.fleets.get(team, {}).get(bot_id)
                    if bot_now is None:
                        break
                    healers = snap.bots_by_role(team, ROLE_HEALER)
                    if _is_being_healed(bot_now, healers):
                        healed = True
                        ticks_to_heal = t - cur_tick
                        break

            events.append(
                HitEvent(
                    tick=cur_tick,
                    bot_id=bot_id,
                    role=cur.role,
                    damage=damage,
                    health_after=cur.health,
                    died=died,
                    fled=fled,
                    speed_before=speed_before,
                    speed_after=speed_after,
                    healed_within_window=healed,
                    ticks_to_heal=ticks_to_heal,
                )
            )

    by_role: Dict[str, int] = {}
    for e in events:
        by_role[e.role] = by_role.get(e.role, 0) + 1

    heal_times = [e.ticks_to_heal for e in events if e.ticks_to_heal is not None]
    return HitReactionSummary(
        total_hits=len(events),
        deaths=sum(1 for e in events if e.died),
        fled_count=sum(1 for e in events if e.fled is True),
        held_ground_count=sum(1 for e in events if e.fled is False),
        healed_within_window_count=sum(1 for e in events if e.healed_within_window),
        avg_ticks_to_heal=(sum(heal_times) / len(heal_times)) if heal_times else None,
        by_role=by_role,
        events=events,
    )


# --------------------------------------------------------------------------
# Healer movement
# --------------------------------------------------------------------------


@dataclass
class HealerProfile:
    bot_id: int
    ticks_observed: int
    avg_speed: float
    max_speed: float
    home_pos: Tuple[float, float]
    wander_radius: float  # avg distance from its own median position
    pct_ticks_with_target: float
    unique_targets_healed: int


@dataclass
class HealerMovementSummary:
    healer_count: int
    profiles: List[HealerProfile]


def analyze_healer_movement(
    ticks: List[TickSnapshot], team: Team
) -> HealerMovementSummary:
    positions: Dict[int, List[Tuple[float, float]]] = {}
    speeds: Dict[int, List[float]] = {}
    targets: Dict[int, List[Optional[int]]] = {}

    for snap in ticks:
        for b in snap.bots_by_role(team, ROLE_HEALER):
            positions.setdefault(b.id, []).append(b.pos)
            speeds.setdefault(b.id, []).append(_speed(b))
            target = b.special.get(ROLE_HEALER, {}).get("healing")
            tid = target.get("Some") if isinstance(target, dict) else None
            targets.setdefault(b.id, []).append(tid)

    profiles: List[HealerProfile] = []
    for bot_id, pts in positions.items():
        xs = sorted(p[0] for p in pts)
        ys = sorted(p[1] for p in pts)
        median = (xs[len(xs) // 2], ys[len(ys) // 2])
        wander = sum(_dist(p, median) for p in pts) / len(pts)
        spd = speeds[bot_id]
        tgt = targets[bot_id]
        with_target = sum(1 for t in tgt if t is not None)
        profiles.append(
            HealerProfile(
                bot_id=bot_id,
                ticks_observed=len(pts),
                avg_speed=sum(spd) / len(spd),
                max_speed=max(spd),
                home_pos=median,
                wander_radius=wander,
                pct_ticks_with_target=100.0 * with_target / len(tgt),
                unique_targets_healed=len({t for t in tgt if t is not None}),
            )
        )

    profiles.sort(key=lambda p: p.bot_id)
    return HealerMovementSummary(healer_count=len(profiles), profiles=profiles)


# --------------------------------------------------------------------------
# Mining deployment / guard coverage
# --------------------------------------------------------------------------


@dataclass
class MiningSample:
    tick: int
    deposit_team: Team  # "a" or "b" -- which deposit this counts around
    miners_present: int
    guards_present: int


@dataclass
class MiningDeploymentSummary:
    samples: List[MiningSample]
    avg_miners: Dict[Team, float]
    avg_guards: Dict[Team, float]
    max_miners: Dict[Team, int]
    max_guards: Dict[Team, int]


def analyze_mining_deployment(
    ticks: List[TickSnapshot],
    team: Team,
    extract_range: float,
    guard_range: float,
    sample_every: int = 20,
) -> MiningDeploymentSummary:
    """For each sampled tick, and for each of the two deposits, count how
    many of `team`'s extractors are within `extract_range` of it (mining
    or about to) and how many of `team`'s battle bots are within
    `guard_range` of it (escorting)."""

    samples: List[MiningSample] = []
    for snap in ticks:
        if snap.tick % sample_every != 0:
            continue
        miners = snap.bots_by_role(team, ROLE_EXTRACTOR)
        guards = snap.bots_by_role(team, ROLE_BATTLE)
        for dteam in ("a", "b"):
            dep = snap.deposits.get(dteam)
            if dep is None:
                continue
            dpos = (dep.x, dep.y)
            m = sum(1 for b in miners if _dist(b.pos, dpos) <= extract_range)
            g = sum(1 for b in guards if _dist(b.pos, dpos) <= guard_range)
            samples.append(
                MiningSample(
                    tick=snap.tick, deposit_team=dteam, miners_present=m, guards_present=g
                )
            )

    def _avg_max(dteam: Team, field_name: str) -> Tuple[float, int]:
        vals = [
            getattr(s, field_name) for s in samples if s.deposit_team == dteam
        ]
        if not vals:
            return 0.0, 0
        return sum(vals) / len(vals), max(vals)

    avg_miners: Dict[Team, float] = {}
    avg_guards: Dict[Team, float] = {}
    max_miners: Dict[Team, int] = {}
    max_guards: Dict[Team, int] = {}
    for dteam in ("a", "b"):
        am, mm = _avg_max(dteam, "miners_present")
        ag, mg = _avg_max(dteam, "guards_present")
        avg_miners[dteam] = am
        max_miners[dteam] = mm
        avg_guards[dteam] = ag
        max_guards[dteam] = mg

    return MiningDeploymentSummary(
        samples=samples,
        avg_miners=avg_miners,
        avg_guards=avg_guards,
        max_miners=max_miners,
        max_guards=max_guards,
    )


# --------------------------------------------------------------------------
# Formation shape
# --------------------------------------------------------------------------


@dataclass
class FormationSample:
    tick: int
    count: int
    centroid: Tuple[float, float]
    spread_radius: float  # avg dist from centroid
    aspect_ratio: float  # major/minor axis of the point spread (1 = round)
    orientation_deg: float  # angle of the major axis
    shape: str  # "cluster" | "line" | "wedge" | "n/a"


@dataclass
class FormationShapeSummary:
    role: str
    samples: List[FormationSample]
    dominant_shape: str


def _pca_axes(points: List[Tuple[float, float]]) -> Tuple[float, float, float]:
    """Return `(aspect_ratio, orientation_deg, spread_radius)` for a 2D
    point cloud via the eigenvalues/vectors of its covariance matrix --
    plain-Python 2x2 eigendecomposition, no numpy dependency."""

    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points) / n
    syy = sum((p[1] - my) ** 2 for p in points) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points) / n

    # Eigenvalues of [[sxx, sxy], [sxy, syy]].
    trace = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = max(trace * trace / 4 - det, 0.0)
    root = math.sqrt(disc)
    lam1 = trace / 2 + root  # major
    lam2 = trace / 2 - root  # minor

    if sxy == 0:
        angle = 0.0 if sxx >= syy else 90.0
    else:
        angle = math.degrees(math.atan2(lam1 - sxx, sxy))

    major = math.sqrt(max(lam1, 0.0))
    minor = math.sqrt(max(lam2, 0.0))
    aspect = (major / minor) if minor > 1e-6 else (major / 1e-6 if major > 1e-6 else 1.0)
    spread = sum(_dist(p, (mx, my)) for p in points) / n
    return aspect, angle % 180.0, spread


def _classify_shape(count: int, aspect_ratio: float, spread_radius: float) -> str:
    if count < 3:
        return "n/a"
    if spread_radius < 1.0:
        return "cluster"
    if aspect_ratio >= 3.0:
        return "line"
    if aspect_ratio >= 1.5:
        return "wedge"
    return "cluster"


def analyze_formation_shape(
    ticks: List[TickSnapshot], team: Team, role: str, sample_every: int = 20
) -> FormationShapeSummary:
    samples: List[FormationSample] = []
    for snap in ticks:
        if snap.tick % sample_every != 0:
            continue
        bots = snap.bots_by_role(team, role)
        if len(bots) < 2:
            samples.append(
                FormationSample(
                    tick=snap.tick,
                    count=len(bots),
                    centroid=(0.0, 0.0),
                    spread_radius=0.0,
                    aspect_ratio=1.0,
                    orientation_deg=0.0,
                    shape="n/a",
                )
            )
            continue
        pts = [b.pos for b in bots]
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        aspect, angle, spread = _pca_axes(pts)
        shape = _classify_shape(len(pts), aspect, spread)
        samples.append(
            FormationSample(
                tick=snap.tick,
                count=len(pts),
                centroid=(mx, my),
                spread_radius=spread,
                aspect_ratio=aspect,
                orientation_deg=angle,
                shape=shape,
            )
        )

    counts: Dict[str, int] = {}
    for s in samples:
        if s.shape == "n/a":
            continue
        counts[s.shape] = counts.get(s.shape, 0) + 1
    dominant = max(counts, key=counts.get) if counts else "n/a"

    return FormationShapeSummary(role=role, samples=samples, dominant_shape=dominant)


# --------------------------------------------------------------------------
# Payload contest
# --------------------------------------------------------------------------

# `capture` is the payload's own signed progress along `payload_path`,
# normalized by the path's total length (confirmed empirically: it moves
# by `payload.speed / path_length` per tick while being pushed, and a
# `winner: "A"` / `reason: "payload"` match ends with `capture == 1.0`).
# So: 0 is where it starts, 1 is team A's delivery (team A pushed it the
# whole way, unopposed), and more negative is the payload being pushed the
# other way, toward team B's delivery. A tick where it doesn't move at all
# means either nobody is near it, or both teams are and it's contested.
CAPTURE_STALL_EPS = 1e-9


@dataclass
class PayloadContestSample:
    tick: int
    capture: float


@dataclass
class PayloadContestSummary:
    samples: List[PayloadContestSample]
    initial_capture: float
    final_capture: float
    net_change: float
    ticks_pushed_toward_a_win: int  # capture increasing
    ticks_pushed_toward_b_win: int  # capture decreasing
    ticks_stalled: int  # unchanged -- unattended or evenly contested
    pct_stalled: float
    leader_at_end: str  # "a" | "b" | "even"


def analyze_payload_contest(ticks: List[TickSnapshot]) -> PayloadContestSummary:
    if not ticks:
        return PayloadContestSummary([], 0.0, 0.0, 0.0, 0, 0, 0, 0.0, "even")

    samples = [PayloadContestSample(tick=s.tick, capture=s.capture) for s in ticks]
    toward_a = toward_b = stalled = 0
    for i in range(1, len(ticks)):
        delta = ticks[i].capture - ticks[i - 1].capture
        if delta > CAPTURE_STALL_EPS:
            toward_a += 1
        elif delta < -CAPTURE_STALL_EPS:
            toward_b += 1
        else:
            stalled += 1

    total = max(len(ticks) - 1, 1)
    initial, final = ticks[0].capture, ticks[-1].capture
    net = final - initial
    leader = "even" if abs(net) < 1e-6 else ("a" if net > 0 else "b")

    return PayloadContestSummary(
        samples=samples,
        initial_capture=initial,
        final_capture=final,
        net_change=net,
        ticks_pushed_toward_a_win=toward_a,
        ticks_pushed_toward_b_win=toward_b,
        ticks_stalled=stalled,
        pct_stalled=100.0 * stalled / total,
        leader_at_end=leader,
    )


# --------------------------------------------------------------------------
# Economy: tokens, build order, deposit-slot denial
# --------------------------------------------------------------------------


@dataclass
class BuildEvent:
    tick: int
    bot_id: int
    role: str


@dataclass
class EconomySummary:
    token_avg: float
    token_max: float
    token_end: float
    builds: List[BuildEvent]
    build_counts_by_role: Dict[str, int]
    # Fraction (0..1) of the shared `extractor_cap` slots at each deposit
    # that `team` occupied, sampled over the match -- high occupancy at
    # the *enemy's* deposit with few of `team`'s own extractors actually
    # extracting nearby is a denial play, not a mining one.
    deposit_occupancy_avg: Dict[Team, float]
    deposit_occupancy_max: Dict[Team, float]


def analyze_economy(
    ticks: List[TickSnapshot],
    team: Team,
    extractor_cap: int,
    sample_every: int = 20,
) -> EconomySummary:
    token_samples: List[float] = []
    builds: List[BuildEvent] = []
    occ: Dict[Team, List[float]] = {"a": [], "b": []}

    fkey = "extractors_a" if team == "a" else "extractors_b"
    for snap in ticks:
        fab = snap.fabricators.get(team)
        if fab is not None:
            token_samples.append(fab.tokens)

        for bid in snap.added.get(team, []):
            bot = snap.fleets.get(team, {}).get(bid)
            if bot is not None:
                builds.append(BuildEvent(tick=snap.tick, bot_id=bid, role=bot.role))

        if snap.tick % sample_every != 0 or extractor_cap <= 0:
            continue
        for dteam in ("a", "b"):
            dep = snap.deposits.get(dteam)
            if dep is None:
                continue
            occupied = getattr(dep, fkey)
            occ[dteam].append(occupied / extractor_cap)

    build_counts: Dict[str, int] = {}
    for b in builds:
        build_counts[b.role] = build_counts.get(b.role, 0) + 1

    def _avg_max(vals: List[float]) -> Tuple[float, float]:
        if not vals:
            return 0.0, 0.0
        return sum(vals) / len(vals), max(vals)

    occ_avg: Dict[Team, float] = {}
    occ_max: Dict[Team, float] = {}
    for dteam in ("a", "b"):
        a, m = _avg_max(occ[dteam])
        occ_avg[dteam] = a
        occ_max[dteam] = m

    tok_avg, tok_max = _avg_max(token_samples)
    return EconomySummary(
        token_avg=tok_avg,
        token_max=tok_max,
        token_end=token_samples[-1] if token_samples else 0.0,
        builds=builds,
        build_counts_by_role=build_counts,
        deposit_occupancy_avg=occ_avg,
        deposit_occupancy_max=occ_max,
    )


# --------------------------------------------------------------------------
# Engagements: hits clustered into discrete fights
# --------------------------------------------------------------------------

ENGAGEMENT_GAP_TICKS = 60  # a quiet stretch this long ends a fight


@dataclass
class Engagement:
    start_tick: int
    end_tick: int
    hits: int
    bots_hit: int
    deaths: int
    fled_count: int
    held_ground_count: int


@dataclass
class EngagementSummary:
    engagements: List[Engagement]
    count: int
    avg_duration_ticks: float
    avg_hits: float
    avg_deaths: float
    deadliest: Optional[Engagement]


def analyze_engagements(
    hit_events: List[HitEvent], gap_ticks: int = ENGAGEMENT_GAP_TICKS
) -> EngagementSummary:
    ordered = sorted(hit_events, key=lambda e: e.tick)
    clusters: List[List[HitEvent]] = []
    for e in ordered:
        if clusters and e.tick - clusters[-1][-1].tick <= gap_ticks:
            clusters[-1].append(e)
        else:
            clusters.append([e])

    engagements: List[Engagement] = []
    for cluster in clusters:
        engagements.append(
            Engagement(
                start_tick=cluster[0].tick,
                end_tick=cluster[-1].tick,
                hits=len(cluster),
                bots_hit=len({e.bot_id for e in cluster}),
                deaths=sum(1 for e in cluster if e.died),
                fled_count=sum(1 for e in cluster if e.fled is True),
                held_ground_count=sum(1 for e in cluster if e.fled is False),
            )
        )

    if not engagements:
        return EngagementSummary([], 0, 0.0, 0.0, 0.0, None)

    durations = [e.end_tick - e.start_tick for e in engagements]
    return EngagementSummary(
        engagements=engagements,
        count=len(engagements),
        avg_duration_ticks=sum(durations) / len(durations),
        avg_hits=sum(e.hits for e in engagements) / len(engagements),
        avg_deaths=sum(e.deaths for e in engagements) / len(engagements),
        deadliest=max(engagements, key=lambda e: e.deaths),
    )


# --------------------------------------------------------------------------
# Top-level report
# --------------------------------------------------------------------------


@dataclass
class MatchReport:
    log_path: str
    team: Team
    ticks_analyzed: int
    match_result: Optional[MatchResult]
    hit_reactions: HitReactionSummary
    healer_movement: HealerMovementSummary
    mining_deployment: MiningDeploymentSummary
    battle_formation: FormationShapeSummary
    healer_formation: FormationShapeSummary
    payload_contest: PayloadContestSummary
    economy: EconomySummary
    engagements: EngagementSummary


def build_report(
    log_path: str,
    ticks: List[TickSnapshot],
    team: Team,
    extract_range: float,
    guard_range: float,
    extractor_cap: int = 8,
    sample_every: int = 20,
    match_result: Optional[MatchResult] = None,
) -> MatchReport:
    hit_reactions = analyze_hit_reactions(ticks, team)
    return MatchReport(
        log_path=log_path,
        team=team,
        ticks_analyzed=len(ticks),
        match_result=match_result,
        hit_reactions=hit_reactions,
        healer_movement=analyze_healer_movement(ticks, team),
        mining_deployment=analyze_mining_deployment(
            ticks, team, extract_range, guard_range, sample_every
        ),
        battle_formation=analyze_formation_shape(
            ticks, team, ROLE_BATTLE, sample_every
        ),
        healer_formation=analyze_formation_shape(
            ticks, team, ROLE_HEALER, sample_every
        ),
        payload_contest=analyze_payload_contest(ticks),
        economy=analyze_economy(ticks, team, extractor_cap, sample_every),
        engagements=analyze_engagements(hit_reactions.events),
    )
