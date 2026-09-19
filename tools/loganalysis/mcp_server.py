"""MCP server exposing `tools/loganalysis` so an assistant can query match
replay logs interactively instead of only reading a CLI dump.

Run it (needs the `mcp` package -- see `requirements.txt`; a venv with it
already installed lives at `.venv/` in the repo root):

    .venv/bin/python -m tools.loganalysis.mcp_server

Then point an MCP client at it over stdio. Every tool takes a `log_path`
relative to the current working directory (or absolute) -- `list_logs` is
the discovery entry point if you don't already have one.

Design notes for whoever's querying through this:

- Team "a" is this repo's own bot (`strategy/plan1`); team "b" is whoever
  it played against in that log. Analysis tools default to `team="b"` --
  the opponent -- since that's almost always what you want when asking
  "how did the *other* fleet behave"; pass `team="a"` to turn the same
  lens on ourselves.
- Summary tools (`get_match_report`, `get_hit_reactions`, ...) strip out
  the big per-tick/per-event lists so a single call stays small. Reach
  for the paired detail tool (`get_hit_events`, `get_payload_timeline`,
  `get_formation_samples`, `get_mining_samples`) when you need the raw
  series behind a summary stat, and use its `limit`/`offset` to page
  through it instead of asking for everything at once.
- Parsing a full log is the expensive part (a 9000-tick match takes
  under a second, but it's still real work) -- results are cached in
  memory per `(path, mtime)`, so asking several questions about the same
  log within one server session is cheap after the first.
"""

from __future__ import annotations

import dataclasses
import glob
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from tools.loganalysis.analyze import (  # noqa: E402
    MatchReport,
    analyze_economy,
    analyze_engagements,
    analyze_formation_shape,
    analyze_healer_movement,
    analyze_hit_reactions,
    analyze_mining_deployment,
    analyze_payload_contest,
    build_report,
)
from tools.loganalysis.parser import (  # noqa: E402
    MatchConfig,
    MatchResult,
    Team,
    TickSnapshot,
    iter_ticks,
    load_config,
    load_match_result,
)

mcp_server = MCPServer(
    "mm-loganalysis",
    instructions=(
        "Query MechMania .mmgl match replay logs. Start with list_logs to "
        "find logs, get_match_result for the quick outcome, then the "
        "analyze_* tools for behavioral detail (hit reactions, healer "
        "movement, mining/guard deployment, formation shape, payload "
        "contest, economy, engagements). Team 'b' is the opponent by "
        "default; team 'a' is this repo's own bot."
    ),
)


# --------------------------------------------------------------------------
# Loading + caching
# --------------------------------------------------------------------------

_CacheEntry = Tuple[float, MatchConfig, List[TickSnapshot], Optional[MatchResult]]
_cache: Dict[str, _CacheEntry] = {}


def _resolve(log_path: str) -> str:
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"no such log file: {log_path}")
    return os.path.abspath(log_path)


def _load(log_path: str) -> Tuple[MatchConfig, List[TickSnapshot], Optional[MatchResult]]:
    path = _resolve(log_path)
    mtime = os.path.getmtime(path)
    cached = _cache.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1], cached[2], cached[3]

    config = load_config(path)
    ticks = list(iter_ticks(path))
    result = load_match_result(path)
    _cache[path] = (mtime, config, ticks, result)
    return config, ticks, result


def _to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _paginate(items: List[Any], limit: int, offset: int) -> Dict[str, Any]:
    total = len(items)
    page = items[offset : offset + limit]
    return {
        "total": total,
        "offset": offset,
        "returned": len(page),
        "has_more": offset + len(page) < total,
        "items": [_to_jsonable(x) for x in page],
    }


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


@mcp_server.tool()
def list_logs(logs_dir: str = "logs", limit: int = 100) -> Dict[str, Any]:
    """List `.mmgl` replay logs available under `logs_dir`, newest first,
    with each one's outcome (winner/reason/tick) so you can pick which
    match to dig into without parsing every file."""

    paths = sorted(
        glob.glob(os.path.join(logs_dir, "*.mmgl")),
        key=os.path.getmtime,
        reverse=True,
    )[:limit]
    rows = []
    for p in paths:
        result = load_match_result(p)
        rows.append(
            {
                "path": p,
                "size_bytes": os.path.getsize(p),
                "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p))),
                "result": _to_jsonable(result) if result else None,
            }
        )
    return {"logs_dir": logs_dir, "count": len(rows), "logs": rows}


@mcp_server.tool()
def get_match_config(log_path: str) -> Dict[str, Any]:
    """Match config for `log_path`: bot stats (blaster_range, heal_range,
    extract_range, ...), payload/deposit/fabricator settings, map size,
    max_ticks/endgame_ticks. Useful context for interpreting the other
    tools' distances and counts."""

    config, _, _ = _load(log_path)
    return {
        "max_ticks": config.max_ticks,
        "endgame_ticks": config.endgame_ticks,
        "bot": config.bot,
        "payload": config.payload,
        "deposit": config.deposit,
        "fabricator": config.fabricator,
        "map_rows": len(config.raw.get("map", [])),
        "map_cols": len(config.raw["map"][0]) if config.raw.get("map") else 0,
    }


@mcp_server.tool()
def get_match_result(log_path: str) -> Optional[Dict[str, Any]]:
    """The engine's own verdict for `log_path`: winner, reason
    (elimination / payload / tie), tick it ended on, real time elapsed.
    `None` if the log has no trailing result comment (cut off mid-match)."""

    _, _, result = _load(log_path)
    return _to_jsonable(result) if result else None


# --------------------------------------------------------------------------
# Full report (compact -- big per-tick/per-event lists stripped; use the
# paired detail tool below to page through those)
# --------------------------------------------------------------------------


def _compact_report(report: MatchReport) -> Dict[str, Any]:
    d = _to_jsonable(report)
    d["hit_reactions"]["events"] = f"<{len(report.hit_reactions.events)} events -- use get_hit_events>"
    d["payload_contest"]["samples"] = f"<{len(report.payload_contest.samples)} samples -- use get_payload_timeline>"
    d["economy"]["builds"] = f"<{len(report.economy.builds)} builds -- use get_economy_builds>"
    d["mining_deployment"]["samples"] = f"<{len(report.mining_deployment.samples)} samples -- use get_mining_samples>"
    d["battle_formation"]["samples"] = f"<{len(report.battle_formation.samples)} samples -- use get_formation_samples>"
    d["healer_formation"]["samples"] = f"<{len(report.healer_formation.samples)} samples -- use get_formation_samples>"
    d["engagements"]["engagements"] = f"<{report.engagements.count} engagements -- use get_engagements>"
    return d


@mcp_server.tool()
def get_match_report(log_path: str, team: Team = "b", sample_every: int = 20) -> Dict[str, Any]:
    """The full behavioral report for one team in `log_path`: hit
    reactions, healer movement, mining/guard deployment, battle & healer
    formation shape, payload contest, economy, engagements -- everything
    in one call, with the big raw lists replaced by counts (use the
    matching detail tool, e.g. get_hit_events, to page through those)."""

    config, ticks, result = _load(log_path)
    report = build_report(
        log_path=log_path,
        ticks=ticks,
        team=team,
        extract_range=config.bot.get("base_extract_range", 5.0),
        guard_range=config.bot.get("blaster_range", 10.0),
        extractor_cap=config.deposit.get("extractor_cap", 8),
        sample_every=sample_every,
        match_result=result,
    )
    return _compact_report(report)


@mcp_server.tool()
def compare_teams(log_path: str, sample_every: int = 20) -> Dict[str, Any]:
    """A side-by-side summary of both teams in `log_path` -- the fastest
    way to spot asymmetries (e.g. one side fled hits far more, one side
    left its healers idle, one side never guarded its mining line) without
    two separate get_match_report calls and diffing them yourself."""

    config, ticks, result = _load(log_path)

    def _summary(team: Team) -> Dict[str, Any]:
        report = build_report(
            log_path=log_path,
            ticks=ticks,
            team=team,
            extract_range=config.bot.get("base_extract_range", 5.0),
            guard_range=config.bot.get("blaster_range", 10.0),
            extractor_cap=config.deposit.get("extractor_cap", 8),
            sample_every=sample_every,
            match_result=result,
        )
        hr, hm, md, eco, eg = (
            report.hit_reactions,
            report.healer_movement,
            report.mining_deployment,
            report.economy,
            report.engagements,
        )
        return {
            "hits_taken": hr.total_hits,
            "deaths": hr.deaths,
            "fled_pct": round(100 * hr.fled_count / hr.total_hits, 1) if hr.total_hits else None,
            "healed_in_window_pct": round(100 * hr.healed_within_window_count / hr.total_hits, 1)
            if hr.total_hits
            else None,
            "avg_ticks_to_heal": hr.avg_ticks_to_heal,
            "healer_count": hm.healer_count,
            "avg_healer_pct_with_target": round(
                sum(p.pct_ticks_with_target for p in hm.profiles) / len(hm.profiles), 1
            )
            if hm.profiles
            else None,
            "avg_miners_own_deposit": round(md.avg_miners[team], 2),
            "avg_guards_own_deposit": round(md.avg_guards[team], 2),
            "battle_formation_shape": report.battle_formation.dominant_shape,
            "healer_formation_shape": report.healer_formation.dominant_shape,
            "token_avg": round(eco.token_avg, 1),
            "token_end": round(eco.token_end, 1),
            "builds_by_role": eco.build_counts_by_role,
            "engagement_count": eg.count,
            "avg_deaths_per_engagement": round(eg.avg_deaths, 2),
        }

    pc = analyze_payload_contest(ticks)
    return {
        "log_path": log_path,
        "match_result": _to_jsonable(result) if result else None,
        "payload_contest": {
            "final_capture": pc.final_capture,
            "leader_at_end": pc.leader_at_end,
            "pct_stalled": round(pc.pct_stalled, 1),
        },
        "team_a": _summary("a"),
        "team_b": _summary("b"),
    }


# --------------------------------------------------------------------------
# Detail / paginated tools -- the raw series behind each summary
# --------------------------------------------------------------------------


@mcp_server.tool()
def get_hit_events(
    log_path: str,
    team: Team = "b",
    window: int = 30,
    only_deaths: bool = False,
    limit: int = 30,
    offset: int = 0,
) -> Dict[str, Any]:
    """Individual hit events for `team` in `log_path` -- tick, bot id/role,
    damage, whether it fled, whether a healer reached it, whether it died.
    Paginated (`limit`/`offset`); set `only_deaths=True` to see just the
    kills."""

    _, ticks, _ = _load(log_path)
    summary = analyze_hit_reactions(ticks, team, window=window)
    events = summary.events
    if only_deaths:
        events = [e for e in events if e.died]
    return _paginate(events, limit, offset)


@mcp_server.tool()
def get_payload_timeline(log_path: str, sample_every: int = 50) -> Dict[str, Any]:
    """The payload's `capture` value (its signed progress along
    payload_path -- see analyze_payload_contest's docstring for how the
    sign maps to who's winning) sampled every `sample_every` ticks, for
    plotting or spotting exactly when a push started or stalled."""

    _, ticks, _ = _load(log_path)
    summary = analyze_payload_contest(ticks)
    samples = [s for s in summary.samples if s.tick % sample_every == 0]
    return {
        "initial_capture": summary.initial_capture,
        "final_capture": summary.final_capture,
        "leader_at_end": summary.leader_at_end,
        "samples": _to_jsonable(samples),
    }


@mcp_server.tool()
def get_economy_builds(log_path: str, team: Team = "b", limit: int = 100, offset: int = 0) -> Dict[str, Any]:
    """The build order for `team` in `log_path`: every bot that entered
    the fleet, in build order, with its tick, id, and role -- paginated."""

    config, ticks, _ = _load(log_path)
    summary = analyze_economy(ticks, team, config.deposit.get("extractor_cap", 8))
    return _paginate(summary.builds, limit, offset)


@mcp_server.tool()
def get_formation_samples(
    log_path: str,
    team: Team = "b",
    role: str = "Battle",
    sample_every: int = 20,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """Per-sample formation-shape data (centroid, spread, aspect ratio,
    classified shape) for `team`'s `role` group (Battle/Healer/Extractor)
    over the match -- paginated."""

    _, ticks, _ = _load(log_path)
    summary = analyze_formation_shape(ticks, team, role, sample_every)
    return {"role": summary.role, "dominant_shape": summary.dominant_shape, **_paginate(summary.samples, limit, offset)}


@mcp_server.tool()
def get_mining_samples(
    log_path: str, team: Team = "b", sample_every: int = 20, limit: int = 50, offset: int = 0
) -> Dict[str, Any]:
    """Per-sample miner/guard counts near each deposit for `team` over the
    match -- paginated."""

    config, ticks, _ = _load(log_path)
    summary = analyze_mining_deployment(
        ticks,
        team,
        extract_range=config.bot.get("base_extract_range", 5.0),
        guard_range=config.bot.get("blaster_range", 10.0),
        sample_every=sample_every,
    )
    return _paginate(summary.samples, limit, offset)


@mcp_server.tool()
def get_engagements(
    log_path: str, team: Team = "b", window: int = 30, gap_ticks: int = 60, limit: int = 50, offset: int = 0
) -> Dict[str, Any]:
    """Every discrete fight for `team` in `log_path` -- hit events
    clustered by a `gap_ticks` quiet stretch -- with start/end tick, hits,
    distinct bots hit, deaths, and how many fled vs held ground.
    Paginated."""

    _, ticks, _ = _load(log_path)
    hit_summary = analyze_hit_reactions(ticks, team, window=window)
    summary = analyze_engagements(hit_summary.events, gap_ticks=gap_ticks)
    return _paginate(summary.engagements, limit, offset)


# --------------------------------------------------------------------------
# Cross-match aggregation
# --------------------------------------------------------------------------


@mcp_server.tool()
def aggregate_reports(log_paths: List[str], team: Team = "b", sample_every: int = 20) -> Dict[str, Any]:
    """Roll several matches' reports into one summary -- totals plus the
    frequency of each dominant battle-formation shape across matches --
    for spotting a *pattern* in how a team plays rather than one match's
    noise. Pass a list of `.mmgl` paths (e.g. from list_logs)."""

    per_match = []
    total_hits = total_deaths = total_fled = total_healed = 0
    shape_counts: Dict[str, int] = {}
    wins: Dict[str, int] = {}

    for path in log_paths:
        config, ticks, result = _load(path)
        report = build_report(
            log_path=path,
            ticks=ticks,
            team=team,
            extract_range=config.bot.get("base_extract_range", 5.0),
            guard_range=config.bot.get("blaster_range", 10.0),
            extractor_cap=config.deposit.get("extractor_cap", 8),
            sample_every=sample_every,
            match_result=result,
        )
        hr = report.hit_reactions
        total_hits += hr.total_hits
        total_deaths += hr.deaths
        total_fled += hr.fled_count
        total_healed += hr.healed_within_window_count
        shape_counts[report.battle_formation.dominant_shape] = (
            shape_counts.get(report.battle_formation.dominant_shape, 0) + 1
        )
        winner = result.winner if result else None
        wins[winner or "tie/unknown"] = wins.get(winner or "tie/unknown", 0) + 1
        per_match.append(
            {
                "log_path": path,
                "winner": winner,
                "hits": hr.total_hits,
                "deaths": hr.deaths,
                "battle_formation_shape": report.battle_formation.dominant_shape,
            }
        )

    return {
        "team": team,
        "matches": len(log_paths),
        "wins_by": wins,
        "totals": {
            "hits": total_hits,
            "deaths": total_deaths,
            "fled": total_fled,
            "healed_in_window": total_healed,
        },
        "battle_formation_shape_frequency": shape_counts,
        "per_match": per_match,
    }


def main() -> None:
    mcp_server.run(transport="stdio")


if __name__ == "__main__":
    main()
